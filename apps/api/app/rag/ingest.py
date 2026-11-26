"""541 页级教材的流式入库。

整体是**两阶段 + 磁盘换内存**：

    阶段一：按页批（默认 24 页）解析 → 轻量分块 → **立即落盘 spill**，
            内存里只留字号直方图与标题候选，批结束即释放。
    阶段二：由直方图算正文字号 → 从候选里定标题 → 建标题树，
            再**流式回放 spill 文件**（一次一块），按"节"边界切 L0/L1/L2，
            章节 lead 与章摘要用有界缓冲，写满即不再接收。

这样峰值内存与**文档长度无关**，只与"页批大小 + 单块大小"同阶。
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

try:
    import pymupdf as pm
except ImportError:  # pragma: no cover
    import fitz as pm  # type: ignore

from app.parsing.extract import is_formula_candidate, match_caption
from app.parsing.headings import build_title_tree, classify_headings
from app.parsing.layout import Line, PageGeom, extract_page_geom, reading_order
from app.parsing.ocr import needs_ocr
from app.parsing.pipeline import _bbox_union, merge_lines
from app.rag.chunker import LeadBuffer, StreamingPacker, breadcrumb, make_chunk_id
from app.rag.memory import MemoryMonitor
from app.rag.params import ChunkParams, count_tokens, split_sentences
from app.rag.store import ChunkIndex, JsonlStore

NUM_PREFIX = re.compile(r"^\s*(?:\d{1,2}(?:\.\d{1,2}){0,3}|第[一二三四五六七八九十]+[章节]|[IVXL]{1,5})[.、\s]")
LIST_PREFIX = re.compile(r"^\s*(?:[•·▪◦‣\-–—*]|[（(]?\d{1,2}[)）.、]|[a-z][)）.])\s+\S")


# ══════════════════════════ 阶段一：页批解析 ══════════════════════════
def iter_page_batches(doc: Any, batch: int) -> Iterator[list[tuple[int, Any]]]:
    """按批产出页，调用方处理完即释放（不构建全量页列表）。"""
    buf: list[tuple[int, Any]] = []
    for i in range(doc.page_count):
        buf.append((i + 1, doc[i]))
        if len(buf) >= batch:
            yield buf
            buf = []
    if buf:
        yield buf


def _page_body_size(geom: PageGeom) -> float:
    """本页字号中位数（页眉页脚判定的参照，避免误杀页首的大号标题）。"""
    sizes = sorted(round(l.size, 1) for l in geom.lines)
    return sizes[len(sizes) // 2] if sizes else 0.0


def _is_header_footer(ln: Line, geom: PageGeom, body_local: float = 0.0) -> bool:
    """页眉/页脚判定：**在页首/页尾带内 + 短 + 字号不超过本页正文**。

    只看位置会把"恰好排在页首的章节标题"当成页眉丢掉
    （实测第 5 章标题在页首 63pt 处被静默丢弃，整章消失）。
    """
    in_band = ln.bbox[3] <= geom.height * 0.075 or ln.bbox[1] >= geom.height * 0.925
    if not in_band:
        return False
    if body_local and ln.size > body_local * 1.05:      # 大号字 = 标题，不是页眉
        return False
    return len(ln.text.strip()) < 40


def _light_type(text: str, lines: list[Line]) -> str:
    """阶段一的轻量分类（阶段二会用全局正文字号复核标题）。"""
    if match_caption(text):
        return "caption"
    if len(text) < 400 and LIST_PREFIX.match(text):
        return "list_item"
    font = lines[0].font if lines else ""
    if is_formula_candidate(text, font, max(l.size for l in lines), max(l.size for l in lines)):
        return "formula"
    return "paragraph"


def clean_text(text: str) -> str:
    """PDF 抽取常见噪点：不换行空格 \\xa0、软连字符、多余空白。"""
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ").replace("\xad", "")).strip()


def page_blocks(page: Any, page_no: int) -> tuple[list[dict[str, Any]], PageGeom]:
    geom = extract_page_geom(page, page_no - 1)
    body_local = _page_body_size(geom)
    by_block: dict[int, list[Line]] = {}
    for ln in reading_order(geom):
        if _is_header_footer(ln, geom, body_local):
            continue
        by_block.setdefault(ln.block_no, []).append(ln)

    # 关键：块内若**字号或字重发生变化**必须再切开。
    # 否则"正文末行 + 紧随其后的大号标题"会被 PyMuPDF 合成一个块，
    # 标题文本被正文污染（实测出现过"…应用部分。 第 1 章 线性代数基础"）。
    groups: list[list[Line]] = []
    for lines in by_block.values():
        run: list[Line] = []
        for ln in lines:
            if run and (round(run[-1].size, 1) != round(ln.size, 1) or run[-1].bold != ln.bold):
                groups.append(run)
                run = []
            run.append(ln)
        if run:
            groups.append(run)

    blocks: list[dict[str, Any]] = []
    for lines in groups:
        text = clean_text(merge_lines(lines))
        if not text:
            continue
        b_no = lines[0].block_no
        blocks.append({
            "page": page_no,
            "block_no": b_no,
            "bbox": _bbox_union([l.bbox for l in lines]),
            "text": text,
            "type": _light_type(text, lines),
            "line_keys": [[l.block_no, l.line_no] for l in lines],
            "size": round(max(l.size for l in lines), 1),
            "bold": any(l.bold for l in lines),
            "n_lines": len(lines),
        })
    return blocks, geom


def heading_candidates(blocks: list[dict[str, Any]], provisional_body: float) -> list[dict[str, Any]]:
    """宁可多选也不漏：阶段二会用最终正文字号严格复核，多选不产生错误结果。

    只收"短块"（≤120 字、≤2 行），内存占用与文档长度无关。
    """
    out: list[dict[str, Any]] = []
    for b in blocks:
        text = b["text"].strip()
        if not (4 <= len(text) <= 120) or b["n_lines"] > 2:
            continue
        if text.endswith(("。", ".", "；", ";", "，", ",", "：", ":")):
            continue
        if b["bold"] or NUM_PREFIX.match(text) or (provisional_body and b["size"] >= provisional_body * 1.08):
            out.append(b)
    return out


def weighted_mode(counter: Counter[float]) -> float:
    """按字符数加权的众数字号 = 正文字号（比中位数更抗标题干扰）。"""
    if not counter:
        return 10.0
    return counter.most_common(1)[0][0]


# ══════════════════════════ 阶段二：章节感知切块 ══════════════════════════
@dataclass
class SectionSink:
    """一个"节"的流式写出上下文：段落/句子打包器 + 有界 lead。"""

    doc_id: str
    params: ChunkParams
    stores: dict[str, JsonlStore]
    index: ChunkIndex
    seq: dict[str, int] = field(default_factory=lambda: {"T0": 0, "L0": 0, "L1": 0, "L2": 0})
    section_id: str = ""
    path: list[str] = field(default_factory=list)
    number: str = ""
    page_start: int = 0
    page_end: int = 0
    full_tokens: int = 0
    n_blocks: int = 0
    bbox: list[float] = field(default_factory=list)
    lead: LeadBuffer = field(default_factory=lambda: LeadBuffer(512))
    cur_l1_id: str = ""
    _para: StreamingPacker | None = None
    _sent: StreamingPacker | None = None

    def start(self, path: list[str], number: str, page: int, bbox: list[float]) -> None:
        self.path, self.number, self.page_start = path, number, page
        self.page_end, self.full_tokens, self.n_blocks = page, 0, 0
        self.bbox = bbox
        self.lead = LeadBuffer(self.params.section_lead)
        self.section_id = make_chunk_id(self.doc_id, "L0", self.seq["L0"])
        crumb = breadcrumb(path) if self.params.prefix_path else ""

        self._para = StreamingPacker(self.params.para_target, self.params.para_overlap,
                                     self.params.para_max, lambda units: self._emit("L1", units, crumb))
        self._sent = StreamingPacker(self.params.sent_target, self.params.sent_overlap,
                                     self.params.sent_max, lambda units: self._emit("L2", units, crumb))

    def add_block(self, block: dict[str, Any]) -> None:
        text = block["text"]
        meta = {"block_id": f"p{block['page']}b{block['block_no']}", "page": block["page"],
                "bbox": block.get("bbox") or [], "type": block.get("type", "paragraph")}
        self.full_tokens += count_tokens(text)
        self.n_blocks += 1
        self.page_end = max(self.page_end, block["page"])
        self.lead.add(text)
        if self._para:
            self._para.add(text, meta)
        if self._sent:
            for s in split_sentences(text):
                self._sent.add(s, meta)

    # ── 写出 ──
    def _emit(self, level: str, units: list[tuple[str, dict]], crumb: str) -> None:
        texts = [u[0] for u in units]
        body = " ".join(texts)
        chunk_text = f"【{crumb}】\n{body}" if crumb else body
        pages = [u[1].get("page", 0) for u in units]
        block_ids = [u[1].get("block_id") for u in units]
        bbox = next((u[1].get("bbox") for u in units if u[1].get("bbox")), [])

        cid = make_chunk_id(self.doc_id, level, self.seq[level])
        self.seq[level] += 1
        if level == "L1":
            self.cur_l1_id = cid
        self._write(level, cid, chunk_text, parent=self._parent_for(level),
                    page_start=min(pages), page_end=max(pages), block_ids=block_ids, bbox=bbox)

    def _parent_for(self, level: str) -> str:
        return {"L1": self.section_id, "L2": self.cur_l1_id or self.section_id}.get(level, "")

    def _write(self, level: str, cid: str, text: str, *, parent: str = "", page_start: int = 0,
               page_end: int = 0, block_ids: list[str] | None = None, bbox: list[float] | None = None,
               path: list[str] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        store = self.stores[level]
        offset, length = store.append({"id": cid, "text": text})
        use_path = list(path if path is not None else self.path)
        sid = cid if level in ("T0", "L0") else self.section_id
        meta = {
            "id": cid, "level": level, "doc_id": self.doc_id,
            "section_number": "" if level == "T0" else self.number, "section_id": sid,
            "parent_id": parent, "page_start": page_start or self.page_start,
            "page_end": page_end or self.page_end, "n_tokens": count_tokens(text),
            "block_ids": (block_ids or [])[:8],                       # 索引瘦身：只留前 8 个块 ID
            "bbox": (bbox or self.bbox) if self.params.keep_bbox else [],
            "store": level, "offset": offset, "length": length,
        }
        if extra:
            meta.update(extra)
        self.index.set_path(sid, use_path)      # 字典编码：同一小节的 chunk 共享一份路径
        self.index.add(meta)
        return meta

    def finish(self) -> dict[str, Any]:
        """节结束：flush 打包器 + 写出 L0 章节层块。"""
        if self._sent:
            self._sent.flush()
        if self._para:
            self._para.flush()
        cid = self.section_id
        self.seq["L0"] += 1
        body = self.lead.text()
        crumb = breadcrumb(self.path) if self.params.prefix_path else ""
        text = f"【{crumb}】\n{body}" if crumb else body
        return self._write("L0", cid, text, extra={
            "lead_tokens": self.lead.tokens, "full_tokens": self.full_tokens,
            "n_blocks": self.n_blocks, "summary_source": "extractive-lead"})


# ══════════════════════════ 主流程 ══════════════════════════
def ingest_pdf(
    pdf_path: str | Path,
    out_dir: str | Path,
    *,
    params: ChunkParams | None = None,
    page_batch: int = 24,
    doc_id: str | None = None,
    memory_limit_mb: float | None = None,
    max_pages: int = 0,
    monitor: MemoryMonitor | None = None,
) -> dict[str, Any]:
    params = params or ChunkParams()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc_id = doc_id or re.sub(r"\W+", "_", Path(pdf_path).stem)[:32]
    mon = monitor or MemoryMonitor()
    warnings: list[str] = []
    started = time.perf_counter()

    stores = {lv: JsonlStore(out / f"chunks_{lv}.jsonl", "w") for lv in ("T0", "L0", "L1", "L2")}
    index = ChunkIndex()
    spill = JsonlStore(out / "_blocks_spill.jsonl", "w")

    try:
        # ── 阶段一 ──
        size_hist: Counter[float] = Counter()
        candidates: list[dict[str, Any]] = []
        n_blocks = n_pages = 0
        scanned_pages: list[int] = []
        provisional = 0.0

        with mon.stage("phase1_parse_spill"):
            doc = pm.open(str(pdf_path))
            try:
                if max_pages and doc.page_count > max_pages:
                    raise ValueError(f"PDF 共 {doc.page_count} 页，超过上限 {max_pages} 页")
                for batch in iter_page_batches(doc, page_batch):
                    for page_no, page in batch:
                        n_pages += 1
                        if needs_ocr(page):
                            scanned_pages.append(page_no)
                            continue
                        blocks, geom = page_blocks(page, page_no)
                        for b in blocks:
                            spill.append(b)
                            for ln in geom.lines:
                                size_hist[round(ln.size, 1)] += len(ln.text)
                        n_blocks += len(blocks)
                        provisional = weighted_mode(size_hist) or provisional
                        candidates += heading_candidates(blocks, provisional)
                    del batch
                    if memory_limit_mb:
                        # 阶段一最容易堆积（解析器/页对象/候选标题），每批检查一次而不是结束后才看。
                        within, msg = mon.check(memory_limit_mb)
                        if not within:
                            raise MemoryError(f"入库中止：{msg}")
            finally:
                doc.close()
        spill.close()

        if scanned_pages:
            warnings.append(f"{len(scanned_pages)} 页无文本层（扫描页），已跳过；"
                            f"需 OCR 请先跑 app.parsing 的 OCR 链路：{scanned_pages[:5]}…")

        # ── 阶段二：标题树 ──
        body_size = weighted_mode(size_hist)
        by_page: dict[int, list[Line]] = {}
        for c in candidates:
            by_page.setdefault(c["page"], []).append(Line(
                page=c["page"], block_no=c["block_no"], line_no=c["line_keys"][0][1],
                bbox=tuple(c["bbox"]), text=c["text"], size=c["size"], bold=c["bold"]))
        cand_geoms = [PageGeom(page=p, width=595.0, height=842.0, lines=ls) for p, ls in sorted(by_page.items())]

        with mon.stage("phase2_headings"):
            heads = classify_headings(cand_geoms, body_size)
            tree = build_title_tree(heads)
            head_map: dict[tuple[int, int, int], dict[str, Any]] = {}
            for h in heads:
                head_map[(h.line.page, h.line.block_no, h.line.line_no)] = h
            # 节点 → 路径：以 (page, bbox) 为键做 O(1) 查表（不要靠标题文字去模糊匹配）
            path_of: dict[tuple[int, tuple], list[str]] = {}

            def walk(nodes: list[dict[str, Any]], prefix: list[str]) -> None:
                for n in nodes:
                    label = f"{n['number']} {n['text']}".strip() if n.get("number") else n["text"]
                    p = prefix + [label]
                    path_of[(int(n["page"]), tuple(n["bbox"]))] = p
                    walk(n.get("children", []), p)

            walk(tree, [])

        if memory_limit_mb:
            within, msg = mon.check(memory_limit_mb)
            if not within:
                raise MemoryError(f"入库中止：{msg}")

        # ── 阶段二：流式切块 ──
        sink = SectionSink(doc_id=doc_id, params=params, stores=stores, index=index)
        chapter_lead = LeadBuffer(params.summary_lead)
        chapter_started = False
        cur_chapter_path: list[str] = []
        chapter_id = ""
        chapter_sections = 0
        chapter_start_page = 0

        def flush_chapter() -> None:
            nonlocal chapter_lead, chapter_started, chapter_id, chapter_sections, chapter_start_page
            if not chapter_started:
                return
            cid = make_chunk_id(doc_id, "T0", sink.seq["T0"])
            sink.seq["T0"] += 1
            body = chapter_lead.text()
            crumb = breadcrumb(cur_chapter_path)
            text = f"【{crumb}】\n{body}" if crumb else body
            # 章首页必须是本章第一节的起点；sink.page_start 在多数章节里已指向最后一节。
            start_page = chapter_start_page or sink.page_start
            sink._write("T0", cid, text, path=cur_chapter_path, parent="",
                        page_start=start_page, page_end=sink.page_end,
                        block_ids=[], bbox=[], extra={"n_sections": chapter_sections,
                                                      "summary_source": "extractive-section-leads"})
            chapter_id = cid
            chapter_started = False
            chapter_start_page = 0
            chapter_lead = LeadBuffer(params.summary_lead)
            chapter_sections = 0

        with mon.stage("phase2_chunk_emit"):
            active = False
            for block in spill.iter_all():
                key = None
                lk = block.get("line_keys") or []
                if lk:
                    key = (block["page"], lk[0][0], lk[0][1])
                h = head_map.get(key) if key else None

                if h is not None:
                    # 命中标题 → 结束上一节
                    if active:
                        l0 = sink.finish()
                        chapter_lead.add(index.path_str_of(l0) + "：" + sink.lead.text()[:120])
                        chapter_sections += 1
                    path = path_of.get((h.line.page, tuple(h.line.bbox))) or \
                        [f"{h.number} {h.text}".strip() if h.number else h.text]
                    is_chapter = h.level <= 1
                    if is_chapter:
                        flush_chapter()
                        cur_chapter_path = path[:1]
                        chapter_started = True
                        chapter_start_page = block["page"]
                    sink.start(path=path, number=h.number or "", page=block["page"],
                               bbox=block.get("bbox") or [])
                    active = True
                    if params.heading_as_prefix and block["text"]:
                        sink.add_block(block)
                    continue

                if not active:      # 正文出现在任何标题之前（封面/前言）
                    sink.start(path=["前言"], number="", page=block["page"], bbox=block.get("bbox") or [])
                    active = True
                if not chapter_started:
                    cur_chapter_path = sink.path[:1]
                    chapter_started = True
                    chapter_start_page = block["page"]
                sink.add_block(block)

            if active:
                sink.finish()
                chapter_sections += 1
            flush_chapter()

    finally:
        spill.close()
        for st in stores.values():
            st.close()
    (out / "_blocks_spill.jsonl").unlink(missing_ok=True)   # spill 用完即弃

    # ── 落盘索引与报告 ──
    index.save(out / "index.json")
    (out / "params.json").write_text(json.dumps(
        {"params": params.__dict__, "page_batch": page_batch, "doc_id": doc_id,
         "body_font_size": body_size, "pdf": str(pdf_path), "max_pages": max_pages},
        ensure_ascii=False, indent=2),
        encoding="utf-8")

    stats = index.stats()
    report = {
        "doc_id": doc_id,
        "pdf": str(pdf_path),
        "pages": n_pages,
        "blocks": n_blocks,
        "chunks": stats["chunks"],
        "by_level": stats["by_level"],
        "tokens": stats["tokens"],
        "index_bytes": stats["index_bytes"],
        "store_bytes": {lv: st.size_bytes() for lv, st in stores.items()},
        "body_font_size": body_size,
        "headings": len(heads),
        "sections": stats["by_level"].get("L0", 0),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "memory": mon.as_dict(),
        "warnings": warnings,
        "max_pages": max_pages,
        "params": params.__dict__,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if memory_limit_mb:
        ok, msg = mon.check(memory_limit_mb)
        report["memory_limit"] = {"limit_mb": memory_limit_mb, "ok": ok, "msg": msg}
    return report
