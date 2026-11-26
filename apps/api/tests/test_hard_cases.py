"""三大硬性用例的自动化验收（pytest）。

运行（在 apps/api 下）：
    python -m pytest tests -q
    python -m pytest tests -q -s        # 额外打印每步的内存/耗时指标

设计约束
--------
1. **完全离线**：去重用本地确定性哈希向量（不调用 embedding/LLM），
   RAG 只用本地 PDF 与本地索引；仅 OCR 用例依赖本机 Tesseract，
   未安装时用 ``pytest.skip`` 明确跳过，不伪造识别结果。
2. 每个用例在 docstring 里写清前置条件 / 操作步骤 / 预期结果 / 通过标准 / 监控指标。
3. 断言既覆盖业务结果，也覆盖耗时、内存峰值、I/O 次数与缓存上界。
"""
from __future__ import annotations

import json
import os
import re
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import pytest

from app.citation_graph import build_result, demo_data, parse_citation_data
from app.graph_survey import CITE_RE, SECTIONS, generate, verify
from app.survey import _quote_in
from app.agents.dedup import (
    DEFAULT_DUP_TH,
    DEFAULT_REL_TH,
    DISTINCT,
    DUPLICATE,
    RELATED,
    PaperDoc,
    compare_papers,
    to_tool_envelope,
    version_advice,
)
from app.agents.tools.builtin import DedupArgs, canonical_id, semantic_dedup
from app.agents.tools.registry import RunContext, registry
from app.parsing.ocr import html_table_to_markdown
from app.parsing.pipeline import parse_pdf
from app.rag.ingest import ingest_pdf
from app.rag.memory import MemoryMonitor
from app.rag.retrieval import TieredLoader
from app.rag.store import ChunkIndex, JsonlStore, LRUCache

INGEST_TIME_LIMIT_S = float(os.getenv("RA_TEST_INGEST_SECONDS", "240"))
INGEST_MEMORY_LIMIT_MB = float(os.getenv("RA_TEST_MEMORY_MB", "1024"))
LOAD_MEMORY_LIMIT_MB = float(os.getenv("RA_TEST_LOAD_MEMORY_MB", "128"))
LEVELS = ("T0", "L0", "L1", "L2")


def _metrics(case: str, **values) -> None:
    rendered = []
    for key, value in values.items():
        rendered.append(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}")
    print(f"[metrics] {case} | " + " | ".join(rendered))


# ═══════════════════════════════════════════════════════════════════
# 用例一：v1/v7 语义去重与版本提示
# ═══════════════════════════════════════════════════════════════════
class TestDedupVersions:
    @pytest.fixture(autouse=True)
    def _force_offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """删掉可能存在的远端 embedding 配置，强制本地哈希向量，保证可重复。"""
        for name in (
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "EMBEDDING_API_KEY",
            "EMBEDDING_BASE_URL",
            "EMBEDDING_MODEL",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_v1_v7_duplicate_and_overwrite_advice(self) -> None:
        """用例 1-A：库中 v1，新上传 v7 → 判重复，并建议用新版本覆盖旧版本。

        前置条件：
            - 版本链语料 ``arxiv:1706.03762v1`` / ``v7`` 可读；
            - 未配置远端 embedding（本测试强制 local-hash），``use_llm=False``。
        操作步骤：
            1. 分别构造 v1、v7 的 PaperDoc；
            2. 调 ``compare_papers(v1, v7, use_llm=False)``；
            3. 读取 verdict / similarity / versions / version_advice / evidence。
        预期结果：
            - verdict=duplicate，similarity ≥ 0.82；
            - same_base=True，cmp=1（v7 比 v1 新）；
            - 建议文本包含“新上传版本更新”与“覆盖”；
            - 证据片段两侧均有原文引用，不凭空生成。
        通过标准：
            - 上述断言全部成立；embedding 引擎为 local-hash；
            - 单次比对耗时 < 10s，Python 堆峰值 < 64MB。
        监控指标：
            - elapsed_ms、peak_alloc_mb、similarity、verdict、cmp、evidence 条数。
        """
        a = PaperDoc.from_version_chain("arxiv:1706.03762v1")
        b = PaperDoc.from_version_chain("arxiv:1706.03762v7")
        assert a.abstract and b.abstract, "前置条件失败：版本链摘要为空"

        monitor = MemoryMonitor(trace=True, label="dedup_v1_v7")
        tracemalloc.reset_peak()
        started = time.perf_counter()
        result = compare_papers(a, b, use_llm=False, api_key="", base_url="")
        elapsed_ms = (time.perf_counter() - started) * 1000
        snap = monitor.snapshot("after_compare")

        _metrics(
            "dedup/v1→v7",
            elapsed_ms=elapsed_ms,
            peak_alloc_mb=snap.peak_mb or 0.0,
            similarity=result["similarity"],
            verdict=result["verdict"],
            cmp=result["versions"]["cmp"],
            evidence=len(result["evidence"]),
        )

        assert result["verdict"] == DUPLICATE
        assert result["similarity"] >= DEFAULT_DUP_TH
        assert result["versions"]["same_base"] is True
        assert result["versions"]["cmp"] == 1
        assert "新上传版本更新" in result["version_advice"]
        assert "覆盖" in result["version_advice"]
        assert result["engine"]["embedding"] == "local-hash"
        assert len(result["evidence"]) >= 2
        assert all(e.get("a_quote") and e.get("b_quote") for e in result["evidence"])

        assert elapsed_ms < 10_000, f"去重耗时 {elapsed_ms:.0f}ms 超过 10s"
        assert (snap.peak_mb or 0.0) < 64, "去重阶段堆峰值超过 64MB"

    def test_reverse_upload_keeps_newer_library_version(self) -> None:
        """用例 1-B：库中已是 v7，用户又传 v1 → 提示保留库中版本，仅存档旧版。

        前置条件：同 1-A。
        操作步骤：``compare_papers(v7, v1, use_llm=False)``，检查 cmp 与建议。
        预期结果：仍判重复；cmp=-1；建议包含“库中已有更新版本建议保留”。
        通过标准：verdict=duplicate 且建议文本命中上述关键字。
        监控指标：similarity、cmp、版本建议长度（只做正确性断言，量级与 1-A 相同）。
        """
        a = PaperDoc.from_version_chain("arxiv:1706.03762v1")
        b = PaperDoc.from_version_chain("arxiv:1706.03762v7")
        result = compare_papers(b, a, use_llm=False, api_key="", base_url="")

        assert result["verdict"] == DUPLICATE
        assert result["versions"]["cmp"] == -1
        assert "库中已有更新版本建议保留" in result["version_advice"]
        assert len(result["version_advice"]) > 20

    def test_semantic_duplicate_without_same_arxiv_id(self) -> None:
        """用例 1-C：doc_id 不同但内容一致 → 只能靠语义判定为重复。

        前置条件：``PaperDoc.from_corpus("doc_transformer")`` 可读。
        操作步骤：复制标题/摘要/正文，换成无关 doc_id 与 v1 版本后比对。
        预期结果：same_base=False，但 similarity ≥ 0.82，verdict=duplicate。
        通过标准：证明去重不是仅靠文件名/arXiv 主干号；“同内容不同 ID”不会漏判。
        监控指标：similarity、field_scores（title/abstract/body）。
        """
        base = PaperDoc.from_corpus("doc_transformer")
        assert base.abstract, "前置条件失败：语料缺少摘要"
        copy = PaperDoc(
            doc_id="local:copy-001",
            title=base.title,
            abstract=base.abstract,
            sections=list(base.sections),
            version="v1",
        )

        result = compare_papers(base, copy, use_llm=False, api_key="", base_url="")
        assert result["versions"]["same_base"] is False
        assert result["similarity"] >= DEFAULT_DUP_TH
        assert result["verdict"] == DUPLICATE
        assert min(result["field_scores"].values()) >= 0.99

    def test_unrelated_paper_is_distinct_and_not_version_advice(self) -> None:
        """用例 1-D：无关内容 → distinct，不给覆盖建议，不做逐段版本比对。

        前置条件：Transformer 语料与一份番茄炒蛋“摘要”互相无关。
        操作步骤：``compare_papers(transformer, recipe, use_llm=False, api_key="", base_url="")``。
        预期结果：verdict=distinct；similarity < 0.55；建议含“不涉及覆盖决策”；
                  diff_summary 说明不做版本级逐段比对。
        通过标准：上述四条全部成立，避免把不同论文误判为版本更新。
        监控指标：similarity、verdict、diff_summary 关键字。
        """
        transformer = PaperDoc.from_corpus("doc_transformer")
        recipe = PaperDoc.from_dict({
            "doc_id": "doc_recipe",
            "title": "家常番茄炒蛋的火候控制",
            "abstract": "讨论中式家常菜番茄炒蛋的火候与调味顺序。",
            "sections": [{"heading": "步骤", "text": "先炒蛋后炒番茄，控制油温六成热。"}],
        })

        result = compare_papers(transformer, recipe, use_llm=False, api_key="", base_url="")
        assert result["verdict"] == DISTINCT
        assert result["similarity"] < DEFAULT_REL_TH
        assert "不涉及覆盖决策" in result["version_advice"]
        assert "不做版本级逐段比对" in result["diff_summary"]

    def test_version_advice_matrix(self) -> None:
        """用例 1-E：版本建议矩阵（更新/更旧/相同/不可比较/非重复）。

        前置条件：仅调用纯函数 ``version_advice``，无 I/O。
        操作步骤：依次传入 (v1,v7)、(v7,v1)、(v7,v7)、(v1,"")、v1/v7+related。
        预期结果：分别给出覆盖、保留、无需覆盖、人工确认、独立入库。
        通过标准：五条文案各自命中唯一正确分支，不混淆“重复”和“相关”。
        监控指标：函数为 O(1)，不设资源阈值；主用例的资源断言见 1-A。
        """
        newer = version_advice("v1", "v7", DUPLICATE)
        older = version_advice("v7", "v1", DUPLICATE)
        same = version_advice("v7", "v7", DUPLICATE)
        unknown = version_advice("v1", "", DUPLICATE)
        related = version_advice("v1", "v7", RELATED)

        assert "新上传版本更新" in newer and "覆盖" in newer
        assert "库中已有更新版本建议保留" in older
        assert "版本相同" in same
        assert "版本不可比较" in unknown and "人工确认" in unknown
        assert "非重复文献" in related and "不涉及覆盖决策" in related

    def test_metadata_override_keeps_raw_semantic_score(self) -> None:
        """元数据规则可以抬判定，但必须保留原始语义分，不能伪装成语义命中。"""
        a = PaperDoc.from_dict({
            "doc_id": "arxiv:9999.00001v1", "title": "Cat Behaviour",
            "abstract": "A study of domestic cats and their sleeping habits.",
        })
        b = PaperDoc.from_dict({
            "doc_id": "arxiv:9999.00001v2", "title": "Diesel Engine Repair",
            "abstract": "A maintenance manual for heavy diesel engines.",
        })
        result = compare_papers(a, b, use_llm=False, api_key="", base_url="")

        assert result["metadata_override"] is True
        assert result["verdict"] == DUPLICATE
        assert result["similarity"] == DEFAULT_DUP_TH
        assert result["semantic_similarity"] < result["similarity"]
        assert any("元数据规则优先" in w for w in result["warnings"])

    def test_multidoc_dedup_is_metadata_only(self) -> None:
        """>2 篇时只做版本链分组，不伪造 0.96 相似度或差异内容。"""
        env = semantic_dedup(
            DedupArgs(doc_ids=[
                "arxiv:1706.03762v1", "arxiv:1706.03762v2", "arxiv:1706.03762v7",
            ]),
            RunContext(),
        )
        group = env["data"]["groups"][0]

        assert env["ok"] is True
        assert env["metrics"] == {"strategy": "version_rule_only", "semantic": False}
        assert group["canonical"] == "arxiv:1706.03762v7"
        assert group["similarity"] is None
        assert group["diff_summary"]["changed"] == []
        assert "未做语义比对" in group["diff_summary"]["note"]

    def test_canonical_and_canonical_versions_sort_numerically(self) -> None:
        versions = ["arxiv:1706.03762v9", "arxiv:1706.03762v10", "arxiv:1706.03762v2"]
        assert canonical_id("arxiv:1706.03762", versions) == "arxiv:1706.03762v10"

    def test_tool_envelope_forwards_warnings_and_engine(self) -> None:
        env = to_tool_envelope({
            "verdict": DUPLICATE, "similarity": 0.9, "version_advice": "x",
            "diff_summary": "y", "evidence": [], "warnings": ["降级为本地哈希"],
            "semantic_similarity": 0.9, "metadata_override": False,
            "engine": {"embedding": "local-hash"},
        })
        assert env["data"]["warnings"] == ["降级为本地哈希"]
        assert env["metrics"]["semantic"] is False


# ═══════════════════════════════════════════════════════════════════
# 用例二：扫描件 OCR 与结构化还原
# ═══════════════════════════════════════════════════════════════════
class TestScannedOcr:
    def test_scanned_pdf_ocr_and_reading_order(
        self, scanned_pdf: Path, require_tesseract: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """用例 2-A：无文本层扫描件 → OCR 出可追溯的段落块，顺序/坐标可还原。

        前置条件：
            - 本机安装 Tesseract（缺失则 skip，不伪造结果）；
            - ``scanned_pdf`` 为纯像素图片 PDF，文本为三行英文。
        操作步骤：
            1. 临时把 OCR_LANG 设为 eng，避免中文语言包干扰英文样张；
            2. ``parse_pdf(scanned_pdf, ocr="tesseract", dpi=200)``；
            3. 汇总所有 source=tesseract 的块，检查文本、置信度、页码、bbox、顺序。
        预期结果：
            - stats.pages=1、scanned_pages=1；
            - 至少 2 个 OCR 段落块，文本含 RESEARCH / ASSISTANT / OCR / TEST / 2026；
            - 每个块 confidence>0.3，均值≥0.5，source=tesseract，page=1；
            - bbox 为 PDF 坐标（pt）且落在页面范围内；
            - “RESEARCH ASSISTANT” 在 “STRUCTURED RESTORATION” 之前。
        通过标准：
            - 上述断言全部成立，且没有“无可用 OCR 后端”告警；
            - OCR 耗时 < 30s，堆峰值 < 256MB。
        监控指标：
            - elapsed_ms、peak_alloc_mb、blocks、mean_confidence、backend、page_size。
        """
        from app.config import get_settings

        monkeypatch.setenv("OCR_LANG", "eng")
        get_settings.cache_clear()
        try:
            monitor = MemoryMonitor(trace=True, label="ocr_scan")
            tracemalloc.reset_peak()
            started = time.perf_counter()
            result = parse_pdf(scanned_pdf, ocr="tesseract", dpi=200)
            elapsed_ms = (time.perf_counter() - started) * 1000
            snap = monitor.snapshot("after_ocr")
        finally:
            get_settings.cache_clear()

        ocr_blocks = [b for b in result.blocks if b.source == "tesseract"]
        text = " ".join(b.text.upper() for b in ocr_blocks)
        confidences = [b.confidence for b in ocr_blocks]
        page_size = result.layout_detail["page_sizes"][0]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

        _metrics(
            "ocr/scanned",
            elapsed_ms=elapsed_ms,
            peak_alloc_mb=snap.peak_mb or 0.0,
            blocks=len(ocr_blocks),
            mean_confidence=mean_conf,
            pages=result.stats.get("pages"),
            scanned_pages=result.stats.get("scanned_pages"),
        )

        assert result.stats["pages"] == 1
        assert result.stats["scanned_pages"] == 1
        assert len(ocr_blocks) >= 2, f"OCR 只还原出 {len(ocr_blocks)} 个块"
        for token in ("RESEARCH", "ASSISTANT", "OCR", "TEST", "2026", "STRUCTURED"):
            assert token in text, f"OCR 文本缺少 {token}: {text!r}"
        assert all(b.page == 1 for b in ocr_blocks)
        assert all(b.source == "tesseract" for b in ocr_blocks)
        assert min(confidences) > 0.3
        assert mean_conf >= 0.5
        for block in ocr_blocks:
            assert len(block.bbox) == 4
            assert 0 <= block.bbox[0] < block.bbox[2] <= page_size["width"] + 2
            assert 0 <= block.bbox[1] < block.bbox[3] <= page_size["height"] + 2

        research_at = text.find("RESEARCH ASSISTANT")
        structured_at = text.find("STRUCTURED")
        assert research_at != -1 and structured_at != -1
        assert research_at < structured_at, "OCR 块未保持自上而下的阅读顺序"
        assert not any("无可用 OCR" in w for w in result.warnings)

        assert elapsed_ms < 30_000, f"OCR 耗时 {elapsed_ms:.0f}ms 超过 30s"
        assert (snap.peak_mb or 0.0) < 256, "OCR 阶段堆峰值超过 256MB"

    def test_html_table_structure_to_markdown(self) -> None:
        """用例 2-B：结构化还原——OCR 表格的 HTML 结构转 Markdown 表格。

        前置条件：PaddleOCR / 表格识别后端输出 HTML 表格（纯函数，无外部依赖）。
        操作步骤：
            1. 传入带表头、加粗单元格、两行数据的 HTML 表格；
            2. 传入缺列的行，验证空单元格补齐；
            3. 传入空表格，验证不生成伪结构。
        预期结果：
            - 第一行为表头，第二行为 ``| --- | --- |`` 分隔行；
            - 单元格内标签被剥离，内容与顺序不变；
            - 缺列行补齐为空单元格，列数一致；
            - 空表格返回空字符串，不臆造表格。
        通过标准：
            - 逐行精确相等（不是“包含”），确保列对齐与表头语义没有漂移。
        监控指标：
            - 纯字符串处理，耗时应 < 100ms；主要质量指标为行内容精确匹配。
        """
        started = time.perf_counter()

        html = """
        <table>
          <tr><th>指标</th><th><b>数值</b></th></tr>
          <tr><td>准确率</td><td>98.5%</td></tr>
          <tr><td>耗时</td><td>120 ms</td></tr>
        </table>
        """
        lines = html_table_to_markdown(html).splitlines()
        assert lines == [
            "| 指标 | 数值 |",
            "| --- | --- |",
            "| 准确率 | 98.5% |",
            "| 耗时 | 120 ms |",
        ]

        ragged = html_table_to_markdown(
            "<table><tr><td>A</td></tr><tr><td>B</td><td>C</td></tr></table>"
        ).splitlines()
        assert ragged == ["| A |  |", "| --- | --- |", "| B | C |"]

        assert html_table_to_markdown("<table></table>") == ""

        elapsed_ms = (time.perf_counter() - started) * 1000
        _metrics("ocr/table_structure", elapsed_ms=elapsed_ms, rows=len(lines))
        assert elapsed_ms < 100

    def test_table_cell_escaping_and_line_breaks(self) -> None:
        md = html_table_to_markdown(
            "<table><tr><th>A|B</th><th>C&amp;D</th></tr>"
            "<tr><td>x<br>y</td><td>1</td></tr></table>"
        ).splitlines()
        assert md == [
            r"| A\|B | C&D |",
            "| --- | --- |",
            "| x y | 1 |",
        ]

    def test_ocr_tool_reports_real_confidences(
        self, scanned_pdf: Path, require_tesseract: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """工具层必须把真实 OCR 置信度透出，不能返回写死的 0.91。"""
        from app.config import get_settings

        monkeypatch.setenv("OCR_LANG", "eng")
        get_settings.cache_clear()
        try:
            env = registry.invoke(
                "extract_scanned_ocr",
                {"file_path": str(scanned_pdf), "engine": "tesseract"},
                RunContext(),
            )
        finally:
            get_settings.cache_clear()

        assert env["ok"] is True, env
        data = env["data"]
        assert data["demo"] is False
        assert data["blocks_recovered"] >= 2
        assert data["confidences"]
        assert all(0 < c <= 1 for c in data["confidences"].values())
        assert env["metrics"]["ocr_confidence_avg"] not in (None, 0.91)
        assert "RESEARCH" in " ".join(data["texts"].values()).upper()

    def test_ocr_tool_demo_is_labeled(self) -> None:
        """缺失文件只有在显式使用演示名时才回退语料，且不伪造置信度。"""
        env = registry.invoke(
            "extract_scanned_ocr", {"file_path": "doc_scan_sample.pdf"}, RunContext()
        )
        assert env["ok"] is True, env
        assert env["data"]["demo"] is True
        assert env["metrics"]["ocr_confidence_avg"] is None
        assert any("演示" in w for w in env["data"]["warnings"])


# ═══════════════════════════════════════════════════════════════════
# 用例三：541 页长文档切块、索引、分级加载与内存监控
# ═══════════════════════════════════════════════════════════════════
class TestLongDocument541:
    def test_t0_page_range_starts_at_chapter_heading(self, tmp_path: Path) -> None:
        """一章含多节时，T0 的 page_start 必须是章首页，而不是最后一节的起点。"""
        import pymupdf as pm

        pdf = tmp_path / "chapter.pdf"
        doc = pm.open()
        pages = [
            ("1 Long Chapter", "1.1 First Section", "First body paragraph."),
            (None, None, "Second page body paragraph."),
            (None, "1.2 Second Section", "Third page body paragraph."),
        ]
        for chapter, section, body in pages:
            page = doc.new_page(width=595, height=842)
            y = 100
            if chapter:
                page.insert_text((72, y), chapter, fontsize=20, fontname="hebo")
                y += 40
            if section:
                page.insert_text((72, y), section, fontsize=14, fontname="hebo")
                y += 40
            page.insert_text((72, y), body, fontsize=10, fontname="helv")
            page.insert_text((72, y + 18), "Additional body text for chunking.",
                             fontsize=10, fontname="helv")
        doc.save(str(pdf))
        doc.close()

        ingest_pdf(pdf, tmp_path / "out", page_batch=1, doc_id="t0_range")
        index = ChunkIndex.load(tmp_path / "out" / "index.json")
        t0 = index.by_level("T0")

        assert len(t0) == 1
        assert t0[0]["page_start"] == 1
        assert t0[0]["page_end"] == 3

    def test_jsonl_offsets_are_byte_accurate(self, tmp_path: Path) -> None:
        store = JsonlStore(tmp_path / "chunks.jsonl", "w")
        records = [{"text": "中文🙂"}, {"text": "a|b\nc"}, {"text": "last"}]
        spans = [store.append(r) for r in records]
        path = store.path
        store.close()

        for record, (offset, length) in zip(records, spans):
            assert store.read(offset, length) == record

        ro = JsonlStore(path, "r")
        with pytest.raises(ValueError):
            ro.append({"text": "nope"})
        assert list(ro.iter_all()) == records
        ro.close()

    def test_541_page_ingest_chunks_index_and_memory(self, ingested_long_doc) -> None:
        """用例 3-A：541 页 PDF 流式入库 → 四层切块 + 索引落盘 + 内存受控。

        前置条件：
            - 合成 541 页 PDF（每 10 页一章，共 55 章；20/14/10pt 标题-正文层级）；
            - 输出目录为空；psutil/tracemalloc 可用。
        操作步骤：
            1. ``ingest_pdf(page_batch=24, memory_limit_mb=1024, monitor=MemoryMonitor)``；
            2. 读取 report.json / params.json / chunks_*.jsonl / index.json；
            3. 校验页数、块数、四层块数、索引与实际文件大小。
        预期结果：
            - pages == 541，blocks > 0，chunks > 0；
            - T0/L0/L1/L2 四层均有块，T0 ≥ 2（章摘要层存在）；
            - index.json 可被 ChunkIndex 重新加载，chunks/by_level 与报告一致；
            - 四个 chunks_*.jsonl 文件非空且大小与报告一致；spill 临时文件已删除；
            - 峰值 RSS 与堆峰值均低于 1024MB，report.memory_limit.ok=true。
        通过标准：
            - 上述断言全部成立；入库总耗时 < 240s（可用 RA_TEST_INGEST_SECONDS 覆盖）。
        监控指标：
            - elapsed_s、peak_rss_mb、peak_alloc_mb、各阶段耗时、
              pages/blocks/chunks/by_level、index_bytes/store_bytes。
        """
        report = ingested_long_doc.report
        out: Path = ingested_long_doc.out
        monitor: MemoryMonitor = ingested_long_doc.monitor
        elapsed_s: float = ingested_long_doc.elapsed_s
        stages = {s.name: s.end.elapsed_ms for s in monitor.stages}

        _metrics(
            "rag/ingest541",
            elapsed_s=elapsed_s,
            peak_rss_mb=monitor.peak_rss_mb,
            peak_alloc_mb=monitor.peak_alloc_mb,
            pages=report["pages"],
            blocks=report["blocks"],
            chunks=report["chunks"],
            T0=report["by_level"].get("T0", 0),
            L0=report["by_level"].get("L0", 0),
            L1=report["by_level"].get("L1", 0),
            L2=report["by_level"].get("L2", 0),
            index_bytes=report["index_bytes"],
            store_bytes=sum(report["store_bytes"].values()),
            **{f"stage_{k}": v for k, v in stages.items()},
        )

        assert report["pages"] == 541
        assert report["blocks"] > 0
        assert report["chunks"] > 0
        assert sum(report["by_level"].values()) == report["chunks"]
        for level in LEVELS:
            assert report["by_level"].get(level, 0) > 0, f"{level} 层没有切出块"
        assert report["by_level"]["T0"] == 55, "541 页/每 10 页一章，应切出 55 个章摘要"
        assert report["index_bytes"] > 0
        assert sum(report["store_bytes"].values()) > report["index_bytes"]

        for level in LEVELS:
            chunk_file = out / f"chunks_{level}.jsonl"
            assert chunk_file.is_file() and chunk_file.stat().st_size > 0
            assert chunk_file.stat().st_size == report["store_bytes"][level]
        assert (out / "index.json").is_file()
        assert (out / "params.json").is_file()
        assert (out / "report.json").is_file()
        assert not (out / "_blocks_spill.jsonl").exists(), "spill 文件未清理"

        saved = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert saved["pages"] == 541
        assert saved["chunks"] == report["chunks"]

        index = ChunkIndex.load(out / "index.json")
        stats = index.stats()
        assert stats["chunks"] == report["chunks"]
        assert stats["by_level"] == report["by_level"]
        assert index.by_level("T0") and index.by_level("L2")

        stage_names = {s.name for s in monitor.stages}
        assert {"phase1_parse_spill", "phase2_headings", "phase2_chunk_emit"} <= stage_names

        assert monitor.peak_rss_mb > 0
        assert monitor.peak_rss_mb < INGEST_MEMORY_LIMIT_MB
        assert monitor.peak_alloc_mb < INGEST_MEMORY_LIMIT_MB
        assert report["memory_limit"]["ok"] is True
        assert elapsed_s < INGEST_TIME_LIMIT_S

    def test_tiered_loading_budget_io_cache_and_memory(self, ingested_long_doc) -> None:
        """用例 3-B：按需分级加载——预算内取到指定正文，且不全量读盘。

        前置条件：用例 3-A 已完成入库，``chunks_*.jsonl`` 与 index.json 可读。
        操作步骤：
            1. ``ChunkIndex.load`` 后构造 ``TieredLoader``（LRU 4MB）；
            2. 以第 42 章 T0 作为 seed，查询第 415 页唯一标记 ``C42P415``，
               token_budget=4000；
            3. 再以 token_budget=64 做一次加载，验证预算拒绝。
        预期结果：
            - used_tokens ≤ 4000；T0/L0/L1/L2 四层均被选中；
            - 至少一个 L1 项正文包含 C42P415（按需读盘命中）；
            - loaded_items < 总块数，loaded_bytes < 全部 store 字节（未全量加载）；
            - 缓存占用 ≤ 4MB 上限，且有 miss（确实发生了读盘）。
        通过标准：
            - 上述断言全部成立；加载耗时 < 5s，增量堆峰值 < 128MB；
            - 小预算加载 used_tokens ≤ 64，不超预算。
        监控指标：
            - elapsed_ms、peak_alloc_mb、used_tokens、tiers、loaded_items/bytes、cache。
        """
        out: Path = ingested_long_doc.out
        chunks_total = ingested_long_doc.report["chunks"]
        store_total = sum(ingested_long_doc.report["store_bytes"].values())

        index = ChunkIndex.load(out / "index.json")
        loader = TieredLoader(
            index=index,
            base_dir=out,
            cache=LRUCache(max_bytes=4 << 20),
        )
        seed = next(r for r in index.by_level("T0") if index.path_str_of(r).startswith("42 "))
        assert "42" in index.path_str_of(seed)

        monitor = MemoryMonitor(trace=True, label="tiered_load")
        tracemalloc.reset_peak()
        started = time.perf_counter()
        result = loader.load(
            "C42P415",
            seed_ids=[seed["id"]],
            token_budget=4000,
            top_chapters=2,
            top_sections=4,
            top_paras=6,
            detail_per_para=3,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        snap = monitor.snapshot("after_load")
        cache = result["cache"]

        _metrics(
            "rag/load_tiered",
            elapsed_ms=elapsed_ms,
            peak_alloc_mb=snap.peak_mb or 0.0,
            used_tokens=result["used_tokens"],
            T0=result["tiers"]["T0"],
            L0=result["tiers"]["L0"],
            L1=result["tiers"]["L1"],
            L2=result["tiers"]["L2"],
            loaded_items=result["io"]["loaded_items"],
            loaded_bytes=result["io"]["loaded_bytes"],
            cache_bytes=cache["bytes"],
            cache_misses=cache["misses"],
        )

        assert result["used_tokens"] <= 4000
        assert result["items"]
        assert result["tiers"]["T0"] >= 1
        assert result["tiers"]["L0"] >= 1
        assert result["tiers"]["L1"] >= 1
        assert result["tiers"]["L2"] >= 1
        assert any(
            item["tier"] == "L1" and "C42P415" in item["text"]
            for item in result["items"]
        ), "分级加载未命中目标正文"

        assert result["io"]["loaded_items"] >= len(result["items"])
        assert result["io"]["loaded_items"] < chunks_total, "加载项数接近全量，未实现按需读取"
        assert 0 < result["io"]["loaded_bytes"] < store_total, "读取字节接近全量正文"
        assert cache["bytes"] <= cache["max_bytes"] == 4 << 20
        assert cache["misses"] >= 1
        assert elapsed_ms < 5_000
        assert (snap.peak_mb or 0.0) < LOAD_MEMORY_LIMIT_MB

        small = loader.load(
            "42",
            seed_ids=[seed["id"]],
            token_budget=64,
            top_chapters=1,
            top_sections=1,
            top_paras=0,
            detail_per_para=0,
        )
        assert small["used_tokens"] <= 64
        assert len(small["items"]) <= 1
        if small["used_tokens"] == 0:
            assert small["dropped_by_budget"], "预算不足却没有记录被丢弃的块"


# ═══════════════════════════════════════════════════════════════════
# 用例四：20 篇引用图谱 + 图谱驱动综述 gold set
# ═══════════════════════════════════════════════════════════════════
#
# gold 真值：32 条引用边、3 篇基石论文、1 篇跨领域边缘论文、20 条摘要。
# 输入引用故意混用 id / 标题 / arXiv / DOI 四种写法，RecallOK 才有区分度；
# 三个指标（RecallOK / 引用准确率 / 幻觉率）打印在同一行，便于跨版本对比。

GOLD_TOPIC = "Transformer 与检索增强生成"
GOLD_EDGES: set[tuple[str, str]] = {
    # Transformer 主干
    ("bert", "att"), ("gpt", "att"), ("roberta", "bert"),
    ("t5", "att"), ("t5", "bert"), ("distilbert", "bert"),
    ("albert", "bert"), ("llama", "att"), ("llama", "gpt"),
    # 高效注意力
    ("longformer", "att"), ("longformer", "bert"),
    ("bigbird", "longformer"), ("bigbird", "bert"),
    ("informer", "att"), ("performer", "att"),
    ("flash", "longformer"), ("flash", "bigbird"),
    ("mamba", "att"), ("mamba", "flash"),
    # 检索增强
    ("dpr", "bert"), ("rag", "dpr"), ("rag", "bert"),
    ("realm", "dpr"), ("atlas", "rag"), ("atlas", "t5"),
    ("selfrag", "rag"),
    # 跨领域边缘论文
    ("pbridge", "longformer"), ("pbridge", "flash"), ("pbridge", "bigbird"),
    ("pbridge", "rag"), ("pbridge", "selfrag"), ("pbridge", "att"),
}
GOLD_IDS = {u for u, _ in GOLD_EDGES} | {v for _, v in GOLD_EDGES}
GOLD_KEYSTONES = {"bert", "att", "rag"}
GOLD_CROSS_DOMAIN = "pbridge"
GOLD_COMMUNITIES = 4

GOLD_ALIASES: dict[str, dict[str, str]] = {
    "att": {"arxiv_id": "1706.03762", "doi": "10.1000/att"},
    "bert": {"arxiv_id": "1810.04805", "doi": "10.1000/bert"},
    "gpt": {"arxiv_id": "1801.10198"},
    "roberta": {"arxiv_id": "1907.11692"},
    "t5": {"arxiv_id": "1910.10683"},
    "distilbert": {"arxiv_id": "1910.01108"},
    "albert": {"arxiv_id": "1909.11942"},
    "llama": {"arxiv_id": "2302.13971"},
    "longformer": {"arxiv_id": "2004.05150"},
    "bigbird": {"arxiv_id": "2007.14062"},
    "informer": {"arxiv_id": "2012.07436"},
    "performer": {"arxiv_id": "2009.14794"},
    "flash": {"arxiv_id": "2205.14135"},
    "mamba": {"arxiv_id": "2312.00752"},
    "dpr": {"arxiv_id": "2004.04906", "doi": "10.1000/dpr"},
    "rag": {"arxiv_id": "2005.11401", "doi": "10.1000/rag"},
    "realm": {"arxiv_id": "2002.08909"},
    "atlas": {"arxiv_id": "2208.03299"},
    "selfrag": {"arxiv_id": "2310.11511"},
    "pbridge": {"arxiv_id": "2401.00001"},
}

GOLD_ABSTRACTS: dict[str, str] = {
    "att": (
        "We propose a new simple network architecture, the Transformer, based solely on attention "
        "mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine "
        "translation tasks show that the model achieves better quality while being more parallelizable."
    ),
    "bert": (
        "We introduce a new language representation model called BERT, designed to pre-train deep "
        "bidirectional representations from unlabeled text. The pre-trained model can be fine-tuned "
        "with just one additional output layer to create state-of-the-art models for a wide range of tasks."
    ),
    "gpt": (
        "We show that a generative pre-training approach on a large unlabeled corpus improves language "
        "understanding across twelve tasks. The approach uses a transformer decoder with task-aware "
        "input transformations."
    ),
    "roberta": (
        "We present RoBERTa, a robustly optimized pretraining approach for BERT. We show that careful "
        "tuning of training data and hyperparameters improves downstream results."
    ),
    "t5": (
        "We propose a unified text-to-text framework that casts every language task as text generation. "
        "Experiments show that this framework transfers effectively across tasks."
    ),
    "distilbert": (
        "We introduce DistilBERT, a smaller distilled version of BERT. The model retains much of BERT "
        "performance while being faster and lighter."
    ),
    "albert": (
        "We present ALBERT, a lite BERT that shares parameters across layers. The design reduces memory "
        "while maintaining competitive performance."
    ),
    "llama": (
        "We introduce LLaMA, a collection of foundation language models trained on public data. We show "
        "that smaller models trained longer can match larger models."
    ),
    "longformer": (
        "We propose the Longformer, a modified Transformer architecture with an attention pattern that "
        "scales linearly with sequence length. The model enables processing of documents with thousands "
        "of tokens on current hardware."
    ),
    "bigbird": (
        "We propose BigBird, a sparse attention mechanism for long sequences. The model handles sequences "
        "far longer than standard transformers."
    ),
    "informer": (
        "We design Informer for long sequence time-series forecasting. The model reduces memory and "
        "computation while improving prediction efficiency."
    ),
    "performer": (
        "We introduce Performer, a linear attention architecture based on positive orthogonal random "
        "features. The approach scales to long sequences without quadratic cost."
    ),
    "flash": (
        "We present FlashAttention, an IO-aware exact attention algorithm. The method reduces memory "
        "access and speeds up long-sequence training."
    ),
    "mamba": (
        "We propose Mamba, a selective state space model for linear-time sequence modeling. The model "
        "scales to long sequences with efficient inference."
    ),
    "dpr": (
        "We present Dense Passage Retrieval, a dense dual-encoder framework for open-domain question "
        "answering. The retriever improves recall over sparse methods."
    ),
    "rag": (
        "We introduce RAG, a general-purpose fine-tuning approach that combines parametric and "
        "non-parametric memory for language generation. The model accesses an external document index "
        "and can be updated without re-training the generator."
    ),
    "realm": (
        "We introduce REALM, a retrieval-augmented language model pre-training method. The model learns "
        "to retrieve evidence documents while pre-training."
    ),
    "atlas": (
        "We present Atlas, a retrieval-augmented few-shot learning model. The model combines a retriever "
        "with a language model for knowledge-intensive tasks."
    ),
    "selfrag": (
        "We propose Self-RAG, a framework that learns to retrieve, generate, and critique its own outputs. "
        "The model adaptively decides when retrieval is needed."
    ),
    "pbridge": (
        "We present a bridging framework that connects sparse attention and retrieval-augmented "
        "generation for long-document question answering. The framework is evaluated on synthetic "
        "benchmarks and leaves open the question of stable citation impact."
    ),
}


def _gold_input() -> dict:
    """20 篇 gold 语料；引用混用 id / 标题 / arXiv / DOI，考验别名解析。"""
    data = demo_data()
    papers = {p["id"]: p for p in data["papers"]}
    for pid, node in papers.items():
        node["abstract"] = GOLD_ABSTRACTS[pid]
        node.update(GOLD_ALIASES.get(pid, {}))
    for pid, node in papers.items():
        targets = sorted(t for s, t in GOLD_EDGES if s == pid)
        node["references"] = [_gold_ref(t, papers[t], i % 4) for i, t in enumerate(targets)]
    return data


def _gold_ref(target: str, node: dict, mode: int):
    if mode == 0:
        return {"id": target}
    if mode == 1:
        return node["title"]
    if mode == 2:
        return {"arxiv_id": node["arxiv_id"]}
    if node.get("doi"):
        return {"doi": node["doi"]}
    return {"title": node["title"]}


def _ref_ok(node: dict, tail: str) -> bool:
    """参考文献行是否准确描述 gold 论文（作者/标题/年份/venue/arXiv/DOI）。"""
    checks = [
        node["title"] in tail,
        node["authors"][0].rstrip(".") in tail,
        str(node["year"]) in tail,
        node["venue"] in tail,
        f"arXiv:{node.get('arxiv_id')}" in tail if node.get("arxiv_id") else True,
        f"DOI:{node.get('doi')}" in tail if node.get("doi") else True,
    ]
    return all(checks)


def _citation_accuracy(markdown: str, meta: dict[str, dict]) -> tuple[int, int]:
    """正文引用标记中，能在参考文献里找到且元数据正确的比例。"""
    body, _, refs = markdown.partition("## 参考文献")
    ref_lines = {
        m.group(1): m.group(2).strip()
        for m in re.finditer(r"^- \*\*\[@([^\]\s]+)\]\*\* (.+)$", refs, re.M)
    }
    good = {
        pid for pid, tail in ref_lines.items()
        if pid in meta and _ref_ok(meta[pid], tail)
    }
    tokens = CITE_RE.findall(body)
    return sum(1 for t in tokens if t in good), len(tokens)


def _statement_verdict(
    stmt: str,
    meta: dict[str, dict],
    in_deg: Counter,
    out_deg: Counter,
    year_counts: Counter,
) -> bool | None:
    """单句是否可被 gold 材料证实；None 表示该句没有可核查的事实。"""
    ref = re.match(r"^- \*\*\[@([^\]\s]+)\]\*\* (.+)$", stmt)
    if ref:
        pid, tail = ref.group(1), ref.group(2)
        return pid in meta and _ref_ok(meta[pid], tail)

    ids = CITE_RE.findall(stmt)
    quotes = re.findall(r"“([^”]+)”", stmt)
    years = [int(y) for y in re.findall(r"(20\d{2})\s*年", stmt)]
    expected_counts = []
    m = re.search(r"材料：(\d+) 篇论文 / (\d+) 条引用 / (\d+) 个社区", stmt)
    if m:
        expected_counts += [
            (int(m.group(1)), len(GOLD_IDS)),
            (int(m.group(2)), len(GOLD_EDGES)),
            (int(m.group(3)), GOLD_COMMUNITIES),
        ]
    m = re.search(r"以 (\d+) 篇论文及其 (\d+) 条有向引用关系", stmt)
    if m:
        expected_counts += [(int(m.group(1)), len(GOLD_IDS)), (int(m.group(2)), len(GOLD_EDGES))]
    m = re.search(r"得到 (\d+) 个子领域", stmt)
    if m:
        expected_counts.append((int(m.group(1)), GOLD_COMMUNITIES))
    m = re.search(r"(\d+) 篇未提供摘要", stmt)
    if m:
        expected_counts.append((int(m.group(1)), 0))
    m = re.search(r"\*\*(\d{4}) 年\*\*（(\d+) 篇）", stmt)
    if m:
        expected_counts.append((int(m.group(2)), year_counts.get(int(m.group(1)), 0)))

    if not (ids or quotes or years or expected_counts):
        return None

    ok = all(pid in meta for pid in ids)
    if quotes:
        pool = (
            [meta[pid]["abstract"] for pid in set(ids) if pid in meta]
            or [node["abstract"] for node in meta.values()]
        )
        ok = ok and all(any(_quote_in(q.rstrip("…"), text) for text in pool) for q in quotes)
    ok = ok and all(year in year_counts for year in years)
    ok = ok and all(actual == expected for actual, expected in expected_counts)
    if len(set(ids)) == 1:
        pid = ids[0]
        m = re.search(r"被 (\d+) 篇引用", stmt)
        if m and int(m.group(1)) != in_deg.get(pid, 0):
            ok = False
        m = re.search(r"引用 (\d+) 篇", stmt)
        if m and int(m.group(1)) != out_deg.get(pid, 0):
            ok = False
    return ok


# ponytail: 幻觉率只核查可机验事实（引用/引句/年份/规模/度数），
# 不核查 PageRank 等派生数值；需要更严时接 NLI 逐句校验。
def _hallucination_audit(
    markdown: str, meta: dict[str, dict], edges: set[tuple[str, str]]
) -> tuple[int, int]:
    """返回（无依据陈述数，可核查陈述数）。

    核查面：编号/标题/arXiv/DOI 引用、摘要引句逐字回查、年份、材料规模、
    年度分组篇数、单篇被引/引用次数、参考文献元数据。
    """
    in_deg = Counter(v for _, v in edges)
    out_deg = Counter(u for u, _ in edges)
    year_counts = Counter(node["year"] for node in meta.values())
    body, _, ref_block = markdown.partition("## 参考文献")

    statements: list[str] = []
    for raw in body.splitlines():
        line = raw.strip().lstrip("> ").strip()
        if not line or line.startswith("#"):
            continue
        statements += [part.strip() for part in re.split(r"(?<=[。！？])", line) if part.strip()]
    statements += [
        line.strip() for line in ref_block.splitlines()
        if line.strip().startswith("- **[@")
    ]

    verifiable = [
        verdict for stmt in statements
        if (verdict := _statement_verdict(stmt, meta, in_deg, out_deg, year_counts)) is not None
    ]
    return verifiable.count(False), len(verifiable)


class TestGraphSurveyGoldSet:
    def test_20_paper_graph_recall_ok(self) -> None:
        """用例 4-A：20 篇 gold 图谱 → 边/节点/基石/边缘论文全召回。

        前置条件：
            - gold 真值：20 节点、32 条引用边、基石 {att,bert,rag}、跨领域 pbridge；
            - 引用混用 id / 标题 / arXiv / DOI 四种写法。
        操作步骤：
            1. ``parse_citation_data`` → ``build_result`` 构图；
            2. 对比恢复边集与 gold 边集，计算 RecallOK / precision；
            3. 检查 top-3 基石与边缘衍生论文。
        预期结果：
            - 20 节点、32 边，unknown_refs=0；
            - RecallOK ≥ 0.95 且 precision ≥ 0.95（本 gold 集应为 1.00 / 1.00）；
            - top-3 基石集合 = {att,bert,rag}，pbridge 进入边缘衍生列表。
        通过标准：上述全部成立；RecallOK 单独打印，便于跨版本对比。
        监控指标：RecallOK、precision、nodes、edges、keystones、peripheral。
        """
        data = _gold_input()
        parsed = parse_citation_data(data)
        result = build_result(data)

        nodes = {n["id"] for n in result["nodes"]}
        edges = {(e["source"], e["target"]) for e in result["edges"]}
        hit = len(edges & GOLD_EDGES)
        recall = hit / len(GOLD_EDGES)
        precision = hit / len(edges) if edges else 0.0
        keystones = {k["id"] for k in result["keystone"][:3]}
        peripheral = {p["id"] for p in result["peripheral"]}

        _metrics(
            "graph/gold20",
            RecallOK=recall,
            precision=precision,
            nodes=len(nodes),
            edges=len(edges),
            keystones=len(keystones),
            peripheral=len(peripheral),
        )

        assert parsed.unknown_refs == 0, "gold 引用存在无法解析的别名"
        assert result["meta"]["n_nodes"] == 20
        assert result["meta"]["n_edges"] == len(GOLD_EDGES) == 32
        assert nodes == GOLD_IDS
        assert edges == GOLD_EDGES
        assert recall >= 0.95 and precision >= 0.95
        assert keystones == GOLD_KEYSTONES
        assert GOLD_CROSS_DOMAIN in peripheral

    def test_graph_driven_survey_gold_metrics(self) -> None:
        """用例 4-B：图谱驱动综述 gold set → 引用准确率与幻觉率。

        前置条件：用例 4-A 的 gold 输入；20 篇均提供摘要，供引句逐字回查。
        操作步骤：
            1. ``generate(data, topic=...)`` 生成四节综述 + 参考文献；
            2. 引用准确率 = 正文 ``[@id]`` 中参考文献存在且元数据正确的占比；
            3. 幻觉率 = 无可核查依据的陈述 / 可核查陈述；
            4. 打印 RecallOK / 引用准确率 / 幻觉率三个数字。
        预期结果：
            - ``verify`` 无问题，四节齐全，无夸大措辞；
            - 3 篇基石与跨领域 pbridge 均进入正文引用；
            - 引用准确率 ≥ 0.95，幻觉率 ≤ 0.03（验收线同 ARCHITECTURE）。
        通过标准：上述全部成立；三个指标同屏输出，可直接做版本对比。
        监控指标：RecallOK、citation_accuracy、hallucination_rate、citation_tokens。
        """
        data = _gold_input()
        result = build_result(data)
        edges = {(e["source"], e["target"]) for e in result["edges"]}
        recall = len(edges & GOLD_EDGES) / len(GOLD_EDGES)
        meta = {p["id"]: p for p in data["papers"]}

        out = generate(data, topic=GOLD_TOPIC)
        markdown = out["markdown"]
        valid, tokens = _citation_accuracy(markdown, meta)
        hallucinated, verifiable = _hallucination_audit(markdown, meta, edges)
        citation_accuracy = valid / tokens if tokens else 0.0
        hallucination_rate = hallucinated / verifiable if verifiable else 0.0

        _metrics(
            "survey/gold20",
            RecallOK=recall,
            citation_accuracy=citation_accuracy,
            hallucination_rate=hallucination_rate,
            cited=len(out["cited"]),
            citation_tokens=tokens,
            hallucinated=hallucinated,
            verifiable=verifiable,
        )

        assert out["issues"] == [], out["issues"]
        assert out["stats"]["n_nodes"] == 20
        assert out["stats"]["n_edges"] == 32
        assert all(f"## {name}" in markdown for name in SECTIONS)
        assert set(out["cited"]) <= GOLD_IDS
        assert GOLD_KEYSTONES <= set(out["cited"])
        assert GOLD_CROSS_DOMAIN in out["cited"]
        assert verify(markdown, GOLD_IDS) == []
        assert tokens > 0 and verifiable > 0
        assert recall >= 0.95
        assert citation_accuracy >= 0.95
        assert hallucination_rate <= 0.03

        # 指标灵敏度：注入伪造引句/引用/作者后，三个数字必须真的变差。
        fake_quote = markdown.replace("We propose", "We invented", 1)
        assert _hallucination_audit(fake_quote, meta, edges)[0] > hallucinated
        fake_cite = markdown.replace("[@att]", "[@ghost]", 1)
        assert _hallucination_audit(fake_cite, meta, edges)[0] > hallucinated
        bad_ref = markdown.replace("Vaswani et al.", "Fake Author", 1)
        assert _citation_accuracy(bad_ref, meta)[0] < valid
