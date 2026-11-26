"""内置工具集（11 个，覆盖六大模块的最小可用闭环）。

每个工具的入参都是 Pydantic 模型（自动生成 Function Calling schema），
返回值都是 ToolEnvelope。`parse_document` 已接入真实 PyMuPDF 流水线（app/parsing），
其余 I/O 密集部分（OCR 引擎、外部检索）在这里只做**契约演示**，
生产实现替换函数体即可，不影响上层 Agent。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.agents.tools import corpus as corpus
from app.agents.tools.registry import RunContext, Tool, fail, ok, registry
from app.schemas import EvidenceCard
from app.utils import short

# ═══════════════════════════ M1 解析模块 ═══════════════════════════


class ParseArgs(BaseModel):
    file_path: str = Field(..., description="待解析文件路径或 URI")
    force_ocr: bool = Field(False, description="是否强制走 OCR（扫描件设为 true）")
    page_range: list[int] | None = Field(None, description="页范围 [start, end]，长文档建议分片解析")


def parse_document(args: ParseArgs, ctx: RunContext) -> dict[str, Any]:
    """原生 PDF → DocIR（版式块 + 阅读序 + 标题树 + 坐标）。

    真实文件（存在且为 PDF）走 app/parsing 的 PyMuPDF+XY-Cut 流水线；
    仅不存在的 `.pdf` 演示文件名可回退内置语料。**非 PDF 不再静默回退**，
    图片/Word/文本请走 `/api/parse` 上传解析。
    """
    if not args.file_path:
        return fail("parse_document", "file_path 为空")
    path = Path(args.file_path)
    suffix = path.suffix.lower()
    if path.exists():
        if suffix == ".pdf":
            return _parse_real_pdf(path, args)
        return fail(
            "parse_document",
            f"暂不支持直接解析 {suffix or '无扩展名'} 文件",
            hint="PDF 可直接解析；图片/Word/文本请通过 /api/parse 上传（前端输入框附件按钮）",
        )
    if suffix != ".pdf":
        return fail(
            "parse_document",
            f"找不到文件：{args.file_path}",
            hint="仅 .pdf 演示文件名可回退内置语料；真实附件请上传",
        )
    stem = path.stem.replace(" ", "_")[:24]
    doc_id = stem if stem.startswith("doc_") else f"doc_{stem}"
    hit = doc_id in corpus.DOCS
    if not hit:
        return fail("parse_document", f"找不到 PDF：{args.file_path}",
                    hint="请确认路径，或通过前端附件按钮上传后解析")
    detail = corpus.DOCS.get(doc_id, {"pages": 1, "source": "native_pdf"})
    blocks = [b for b in corpus.BLOCKS if b["doc_id"] == doc_id]
    return ok(
        "parse_document",
        {
            "doc_id": doc_id,
            "parse_route": "ocr" if args.force_ocr else detail.get("source", "native_pdf"),
            "pages": detail.get("pages", 1),
            "blocks": len(blocks),
            "layout": [{"block_id": b["block_id"], "page": b["page"], "bbox": b["bbox"],
                        "section": b["section"]} for b in blocks],
            "note": "长文档请传 page_range 分片，避免单次任务超时",
        },
        metrics={"engine": "docling/mineru", "shardable": True},
    )


def _parse_real_pdf(path: Path, args: ParseArgs) -> dict[str, Any]:
    """真实 PDF：调 app/parsing 流水线，并把正文块提升为可被引用的证据卡。"""
    import re

    try:
        from app.parsing.pipeline import parse_pdf
    except Exception as e:  # noqa: BLE001
        return fail("parse_document", f"解析依赖不可用（pip install pymupdf）：{e}",
                    hint="离线演示可改用内置语料中的文件名，如 doc_transformer.pdf")

    assets = path.parent / f".{path.stem}_assets"
    res = parse_pdf(path, ocr="paddle" if args.force_ocr else "auto", dpi=200, assets_dir=assets)
    doc_id = "doc_" + re.sub(r"\W+", "_", path.stem)[:24]

    body = [b for b in res.blocks if b.type not in ("header", "footer")]
    cards = [
        EvidenceCard(
            doc_id=doc_id,
            block_id=b.id,
            quote=short(b.text or b.caption or b.type, 160),
            page=b.page,
            bbox=b.bbox,
            score=1.0,
            meta={"type": b.type, "level": b.level, "number": b.number,
                  "latex": b.latex, "markdown": (b.markdown or "")[:200] or None},
        )
        for b in body[:10] if (b.text or b.caption)
    ]
    return ok(
        "parse_document",
        {
            "doc_id": doc_id,
            "parse_route": "text" if not res.layout_detail.get("scanned_pages") else "text+ocr",
            "layout": res.layout,
            "pages": res.source.get("pages", 0),
            "blocks": len(body),
            "title_tree": [t.model_dump(mode="json") for t in res.title_tree],
            "block_index": [{"id": b.id, "type": b.type, "page": b.page, "bbox": b.bbox,
                             "level": b.level, "number": b.number,
                             "text": short(b.text, 80)} for b in body[:200]],
            "stats": res.stats,
            "warnings": res.warnings,
        },
        cited=cards,
        metrics={"engine": "pymupdf+xycut", "shardable": True,
                 "scanned_pages": res.layout_detail.get("scanned_pages", [])},
    )


class OCRArgs(BaseModel):
    file_path: str = Field(..., description="扫描件路径（PDF）")
    engine: str = Field("auto", description="OCR 引擎：auto | paddle | tesseract（mineru/commercial 尚未接入）")


_OCR_ENGINES = {"auto", "paddle", "tesseract"}
_LOW_CONF = 0.6


def extract_scanned_ocr(args: OCRArgs, ctx: RunContext) -> dict[str, Any]:
    """扫描件 → 版面恢复 + OCR + 低置信度标注（低置信块不进综述，只进检索）。

    有真实文件时调用 ``parse_pdf``，置信度/低置信块全部来自实际 OCR 结果；
    仅当显式传入内置演示文件名 ``doc_scan_sample.pdf`` 时才返回语料，并标注 ``demo=True``。
    """
    if not args.file_path:
        return fail("extract_scanned_ocr", "file_path 为空")
    engine = (args.engine or "auto").strip().lower()
    if engine not in _OCR_ENGINES:
        return fail("extract_scanned_ocr", f"OCR 引擎 {engine} 未接入",
                    hint="当前可用：auto / paddle / tesseract")
    path = Path(args.file_path)
    if path.suffix.lower() != ".pdf":
        return fail("extract_scanned_ocr", f"仅支持 PDF：{args.file_path}",
                    hint="图片/Word 请先走 /api/parse 上传转换")

    if path.exists():
        return _ocr_real_pdf(path, engine)

    # 演示语料：必须显式命名，避免把任意不存在的路径伪装成 OCR 结果
    demo_id = path.stem if path.stem.startswith("doc_") else f"doc_{path.stem}"
    if demo_id != "doc_scan_sample" or demo_id not in corpus.DOCS:
        return fail("extract_scanned_ocr", f"找不到文件：{args.file_path}",
                    hint="真实扫描件请传绝对/相对路径；离线演示请用 doc_scan_sample.pdf")
    blocks = [b for b in corpus.BLOCKS if b["doc_id"] == demo_id]
    low_conf = [b["block_id"] for b in blocks if "OCR低置信" in b["text"] or "【" in b["text"]]
    return ok(
        "extract_scanned_ocr",
        {
            "doc_id": demo_id,
            "demo": True,
            "quality_grade": "scanned",
            "engine": "demo-corpus",
            "pages": corpus.DOCS[demo_id].get("pages", 0),
            "blocks_recovered": len(blocks),
            "low_confidence_blocks": low_conf,
            "low_confidence_threshold": _LOW_CONF,
            "confidences": {},                    # 语料没有逐块置信度，不伪造数字
            "texts": {b["block_id"]: b["text"] for b in blocks},
            "warnings": ["使用内置演示语料，未执行真实 OCR；置信度不可用"],
        },
        metrics={"ocr_confidence_avg": None, "need_hitl": bool(low_conf), "demo": True},
    )


def _ocr_real_pdf(path: Path, engine: str) -> dict[str, Any]:
    import re

    try:
        from app.parsing.pipeline import parse_pdf
    except Exception as e:  # noqa: BLE001
        return fail("extract_scanned_ocr", f"OCR 依赖不可用（pip install pymupdf）：{e}")

    assets = path.parent / f".{path.stem}_ocr_assets"
    try:
        res = parse_pdf(path, ocr=engine, dpi=200, assets_dir=assets)
    except Exception as e:  # noqa: BLE001 —— 损坏/加密 PDF 必须给出可读错误
        return fail("extract_scanned_ocr", f"OCR 失败：{type(e).__name__}: {e}",
                    hint="确认 PDF 未加密、未损坏；扫描页可先用 /api/parse 预览")

    scanned = [int(p) for p in (res.layout_detail.get("scanned_pages") or [])]
    selected = [b for b in res.blocks if b.page in set(scanned)] if scanned else []
    texts = {b.id: b.text for b in selected if b.text}
    confs = [b.confidence for b in selected if b.text and b.confidence > 0]
    low_conf = [b.id for b in selected if not b.text or b.confidence < _LOW_CONF]
    backends = sorted({b.source for b in selected if b.source and b.source != "none"})
    mean_conf = round(sum(confs) / len(confs), 3) if confs else None
    doc_id = "doc_" + re.sub(r"\W+", "_", path.stem)[:24]
    return ok(
        "extract_scanned_ocr",
        {
            "doc_id": doc_id,
            "demo": False,
            "quality_grade": "scanned" if scanned else "native",
            "engine": ",".join(backends) or engine,
            "pages": res.stats.get("pages", 0),
            "scanned_pages": scanned,
            "blocks_recovered": len(texts),
            "low_confidence_blocks": low_conf,
            "low_confidence_threshold": _LOW_CONF,
            "confidences": {b.id: b.confidence for b in selected if b.text},
            "texts": texts,
            "warnings": res.warnings,
        },
        metrics={"ocr_confidence_avg": mean_conf, "need_hitl": bool(low_conf),
                 "pages": res.stats.get("pages", 0), "scanned_pages": len(scanned)},
    )


class DedupArgs(BaseModel):
    doc_ids: list[str] = Field(..., description="待去重文档 ID 列表（如 arXiv v1..v7）")


def semantic_dedup(args: DedupArgs, ctx: RunContext) -> dict[str, Any]:
    """三级去重：精确哈希 → MinHash/pg_trgm → 语义相似 + **版本链**优先。

    关键：版本规则（同 arXiv ID）优先于聚类，避免把 v1/v7 判成两篇（重复计数）
    或把同主题不同论文误合并（丢文献）。这是两类错误的唯一平衡点。
    """
    # 恰好两篇且都能取到内容 → 走真实语义比对（相似度依据 + 版本建议 + 逐段差异）
    ids = [d for d in args.doc_ids if d]
    warnings: list[str] = []
    if len(ids) == 2:
        try:
            docs = [_resolve_for_dedup(i) for i in ids]
            if all(docs):
                from app.agents.dedup import compare_papers, to_tool_envelope

                res = compare_papers(docs[0], docs[1])
                env = to_tool_envelope(res)
                env["data"].update({
                    "input": 2,
                    "groups": [{
                        "canonical": _newer(ids[0], ids[1]) or ids[1],
                        "versions": sorted(ids),
                        "same_paper": res["verdict"] == "duplicate",
                        "similarity": res["similarity"],
                        "semantic_similarity": res.get("semantic_similarity"),
                        "diff_summary": res["diff_summary"],
                        "diff_detail": res["diff_detail"],
                    }],
                    "deduped_count": 1 if res["verdict"] == "duplicate" else 2,
                })
                return env
            warnings.append("两篇文档中有无法解析的内容，回退到元数据版本链分组")
        except Exception as e:  # noqa: BLE001 —— 语义比对失败不影响原去重聚类
            warnings.append(f"语义比对失败（{type(e).__name__}），回退到元数据版本链分组")

    from app.agents.dedup import base_id

    clusters: dict[str, list[str]] = {}
    for d in args.doc_ids:
        base = base_id(d) or d
        clusters.setdefault(base, []).append(d)
    groups = [
        {
            "canonical": canonical_id(base, versions),
            "versions": sorted(versions),
            "same_paper": True,               # 元数据层判定：同一 arXiv 主干
            "similarity": None,               # 未做语义比对，不编造分数
            "semantic_similarity": None,
            "diff_summary": diff_summary(base, versions),
        }
        for base, versions in clusters.items()
    ]
    return ok(
        "semantic_dedup",
        {"input": len(args.doc_ids), "groups": groups, "deduped_count": len(groups),
         "warnings": warnings},
        metrics={"strategy": "version_rule_only", "semantic": False},
    )


def _resolve_for_dedup(doc_id: str):
    """把 doc_id 解析成可比对的内容视图（版本链 / 语料文档），取不到返回 None。"""
    from app.agents.dedup import PaperDoc

    did = (doc_id or "").strip()
    if did.lower().startswith("arxiv:"):
        return PaperDoc.from_version_chain(did)
    if did in corpus.DOCS:
        return PaperDoc.from_corpus(did)
    return None


def _newer(a: str, b: str) -> str:
    """返回更新的那个版本号；不可比较时返回后者（新上传）。"""
    from app.agents.dedup import compare_versions

    cmp = compare_versions(a, b)
    return a if cmp is not None and cmp < 0 else b


def canonical_id(base: str, versions: list[str]) -> str:
    """保留最新版作为 canonical，其余作为历史版本（不是简单丢弃）。

    必须按版本元组比较：字典序会把 v10 排在 v9 前面。
    """
    from app.agents.dedup import parse_version

    def key(v: str) -> tuple[bool, tuple[int, ...]]:
        kind, nums = parse_version(v)
        return (kind != "none", nums)

    return max(versions, key=key) if versions else base


def diff_summary(base: str, versions: list[str]) -> dict[str, Any]:
    """元数据层分组没有正文差异可陈述，只报版本范围，不伪造 `changed`。"""
    from app.agents.dedup import parse_version

    def key(v: str) -> tuple[bool, tuple[int, ...]]:
        kind, nums = parse_version(v)
        return (kind != "none", nums)

    ordered = sorted(versions, key=key)
    if len(ordered) < 2:
        return {"from": ordered[0] if ordered else base, "to": None,
                "changed": [], "delta_chars": 0, "note": "仅一条记录，无版本差异"}
    return {"from": ordered[0], "to": ordered[-1], "changed": [], "delta_chars": 0,
            "note": "仅基于版本链元数据分组，未做语义比对"}


# ═══════════════════════════ M2 检索 / RAG ═══════════════════════════


class SearchArgs(BaseModel):
    query: str = Field(..., description="检索 Query（中英均可）")
    top_k: int = Field(6, description="返回条数")
    doc_scope: list[str] | None = Field(None, description="限定文档范围")


def hybrid_search(args: SearchArgs, ctx: RunContext) -> dict[str, Any]:
    """混合检索（BM25 + 向量）→ Rerank → EvidenceCard 列表（含 page/bbox）。"""
    cards = corpus.hybrid_search(args.query, top_k=args.top_k, doc_scope=args.doc_scope)
    if not cards:
        return fail("hybrid_search", f"未召回任何结果：{args.query}", hint="换个更短的主题词或放宽 top_k")
    return ok(
        "hybrid_search",
        {"query": args.query, "hits": [c.model_dump(mode="json") for c in cards]},
        cited=cards,
        metrics={"top_k": args.top_k, "strategy": "bm25+dense+rerank"},
    )


class FetchPaperArgs(BaseModel):
    query: str = Field(..., description="检索关键词")
    source: str = Field("openalex", description="openalex | crossref | s2 | arxiv")
    limit: int = Field(5, description="返回条数")


def fetch_paper(args: FetchPaperArgs, ctx: RunContext) -> dict[str, Any]:
    """外部学术库检索（生产：OpenAlex/Crossref/Semantic Scholar 三源互补）。"""
    kw = args.query.lower()
    hits = [p for p in corpus.PAPER_DB if kw[:4] in p["title"].lower() or any(
        k in p["title"].lower() for k in ("transformer", "bert", "longformer", "linformer") if k in kw)]
    return ok(
        "fetch_paper",
        {"source": args.source, "papers": (hits or corpus.PAPER_DB)[: args.limit]},
        metrics={"rate_limited": False},
    )


class CompareArgs(BaseModel):
    topic_a: str = Field(..., description="对比对象 A")
    topic_b: str = Field(..., description="对比对象 B")
    aspects: list[str] = Field(..., description="对比维度列表")


def compare_claims(args: CompareArgs, ctx: RunContext) -> dict[str, Any]:
    """多文对照：逐维度取双方证据，**每个结论必须带 doc_id#block_id**。"""
    matrix: list[dict[str, Any]] = []
    cited: list[EvidenceCard] = []
    for asp in args.aspects:
        row: dict[str, Any] = {"aspect": asp}
        for key, topic in (("a", args.topic_a), ("b", args.topic_b)):
            cards = corpus.hybrid_search(f"{topic} {asp}", top_k=2)
            cards = [c for c in cards if topic.lower() in c.doc_id or topic.lower() in c.meta.get("title", "").lower()] or cards
            top = cards[0] if cards else None
            row[key] = {"claim": short(top.quote, 110) if top else "库内无可靠证据",
                        "cite": top.cite_key if top else None}
            if top:
                cited.append(top)
        matrix.append(row)
    return ok("compare_claims", {"topic_a": args.topic_a, "topic_b": args.topic_b, "matrix": matrix},
              cited=cited, metrics={"evidence_per_cell": 1})


# ═══════════════════════════ M3 翻译 ═══════════════════════════


class TranslateArgs(BaseModel):
    doc_id: str = Field(..., description="文档 ID")
    block_ids: list[str] = Field(..., description="待翻译的 block_id 列表")
    target_lang: str = Field("zh", description="目标语言：zh | en")
    use_glossary: bool = Field(True, description="是否套用用户术语表/翻译记忆")


def translate_block(args: TranslateArgs, ctx: RunContext) -> dict[str, Any]:
    """版式感知翻译：**锚点是 block_id + bbox**，绝不是字符 offset（PDF 不可靠）。"""
    from app.memory import memory  # 延迟导入避免循环依赖

    glossary = memory.get_glossary(ctx.user_id) if args.use_glossary else {}
    trans: list[dict[str, Any]] = []
    cited: list[EvidenceCard] = []
    for bid in args.block_ids:
        blk = corpus.get_block(bid, args.doc_id)
        if not blk:
            trans.append({"block_id": bid, "error": "block 不存在"})
            continue
        target = blk["text"]
        for en, zh in glossary.items():
            if args.target_lang == "zh":
                target = target.replace(en, zh)
        trans.append({
            "block_id": bid,
            "source_lang": "en" if args.target_lang == "zh" else "zh",
            "target_lang": args.target_lang,
            "source_text": blk["text"],
            "target_text": target,
            "page": blk["page"],
            "bbox": blk["bbox"],                      # 双向联动：原文↔译文的高亮锚点
            "alignment": [{"source_span": [0, len(blk["text"])], "target_span": [0, len(target)]}],
        })
        cited.append(EvidenceCard(doc_id=blk["doc_id"], block_id=bid, quote=short(blk["text"], 120),
                                  page=blk["page"], bbox=blk["bbox"], score=1.0))
    return ok("translate_block", {"doc_id": args.doc_id, "translations": trans, "glossary_used": bool(glossary)},
              cited=cited, metrics={"anchors": "block_id+bbox", "count": len(trans)})


# ═══════════════════════════ M5 图谱 ═══════════════════════════


class GraphArgs(BaseModel):
    seed_doc_ids: list[str] = Field(..., description="种子文献 ID")
    depth: int = Field(2, description="扩展深度")


def build_citation_graph(args: GraphArgs, ctx: RunContext) -> dict[str, Any]:
    """异构引用图：节点=论文/作者/概念，边=cites/co-cited；输出社区与主路径。"""
    nodes = [{"id": d, "type": "paper", "title": corpus.DOCS[d]["title"], "year": corpus.DOCS[d]["year"]}
             for d in corpus.DOCS]
    edges = [{"source": s, "target": t, "rel": r} for s, t, r in corpus.CITATION_EDGES]
    seeds = [n["id"] for n in nodes if n["id"] in args.seed_doc_ids] or nodes[0]["id"]
    communities = {n["id"]: 0 for n in nodes}
    main_path = ["doc_transformer", "doc_bert", "doc_survey_2020"]
    return ok(
        "build_citation_graph",
        {"seeds": seeds, "nodes": nodes, "edges": edges, "communities": communities,
         "main_path": main_path, "depth": args.depth},
        metrics={"nodes": len(nodes), "edges": len(edges), "algorithm": "leiden+main_path"},
    )


# ═══════════════════════════ M6 写作 / 可视化 ═══════════════════════════


class WriteSectionArgs(BaseModel):
    title: str = Field(..., description="章节标题")
    aspects: list[str] = Field(..., description="本章要覆盖的要点")
    max_words: int = Field(600, description="字数上限")


def write_section(args: WriteSectionArgs, ctx: RunContext) -> dict[str, Any]:
    """大纲驱动写作：**引用必须来自 ctx 中的真实证据**，禁止模型凭空生成参考文献。"""
    pool: list[dict[str, Any]] = list(ctx.extra.get("evidence_pool", []))[:8]
    body: list[str] = []
    cited: list[EvidenceCard] = []
    for i, asp in enumerate(args.aspects):
        ev = pool[i % len(pool)] if pool else None
        if ev:
            body.append(f"- {asp}：{short(str(ev.get('quote')), 90)} [[ev:{ev.get('doc_id')}#{ev.get('block_id')}]]")
            cited.append(EvidenceCard(**{k: v for k, v in ev.items() if k in EvidenceCard.model_fields}))
        else:
            body.append(f"- {asp}：（无证据支撑，已留空，待补充来源）")
    md = f"## {args.title}\n" + "\n".join(body)
    return ok("write_section", {"title": args.title, "markdown": md,
                                "citation_anchors": [c.cite_key for c in cited],
                                "unsupported_points": [a for a, e in zip(args.aspects, [pool[i % len(pool)] if pool else None for i in range(len(args.aspects))]) if not e]},
              cited=cited, metrics={"anchors": len(cited), "max_words": args.max_words})


class ChartArgs(BaseModel):
    chart_type: str = Field("bar", description="bar | line | scatter")
    x: list[str] = Field(..., description="横轴类目")
    y: list[float] = Field(..., description="纵轴数值")
    title: str = Field("", description="图表标题")


def render_chart(args: ChartArgs, ctx: RunContext) -> dict[str, Any]:
    """输出 Vega-Lite spec（前端直接渲染，避免后端出图的性能与样式耦合）。"""
    if len(args.x) != len(args.y):
        return fail("render_chart", "x 与 y 长度不一致", hint="请保证每个类目都有对应数值")
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": args.chart_type,
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        },
        "data": {"values": [{"category": x, "value": y} for x, y in zip(args.x, args.y)]},
        "title": args.title,
    }
    return ok("render_chart", {"spec": spec}, metrics={"renderer": "vega-lite"})


class ExportArgs(BaseModel):
    title: str = Field(..., description="论文/报告标题")
    sections: list[str] = Field(default_factory=list, description="章节 markdown（来自 write_section）")


def export_latex(args: ExportArgs, ctx: RunContext) -> dict[str, Any]:
    """导出可编译 LaTeX + BibTeX，引用锚点反查真实条目。"""
    body = "\n".join(f"\\section{{{s.splitlines()[0].lstrip('# ').strip()}}}\n{s}" if s else "" for s in args.sections)
    latex = f"\\documentclass{{article}}\n\\title{{{args.title}}}\n\\begin{{document}}\n\\maketitle\n{body}\n\\bibliography{{refs}}\n\\end{{document}}"
    bib = "\n".join(
        f"@article{{{d},\n  title={{{corpus.DOCS[d]['title']}}},\n  year={{{corpus.DOCS[d]['year']}}}\n}}"
        for d in corpus.DOCS
    )
    return ok("export_latex", {"latex": latex, "bibtex": bib},
              metrics={"unresolved_citations": latex.count("[[ev:")})


# ═══════════════════════════ 注册 ═══════════════════════════
# scopes 用于"工具授权"，任何 Agent 只能看到自己范围内的工具
AGENT_SCOPES: dict[str, tuple[str, ...]] = {
    "parse": ("m1",),
    "retrieval": ("m4", "shared"),
    "rag": ("m2", "shared"),
    "translation": ("m3", "shared"),
    "graph": ("m5", "shared"),
    "writing": ("m6", "shared"),
}


def register_all() -> None:
    tools = [
        Tool("parse_document", "解析 PDF/文档为 DocIR（版式块、阅读序、表格公式）", ParseArgs, parse_document, ("m1",)),
        Tool("extract_scanned_ocr", "扫描件质量分级 + OCR + 版面恢复 + 低置信标注", OCRArgs, extract_scanned_ocr, ("m1",)),
        Tool("semantic_dedup", "三级语义去重 + arXiv 版本链判定 + diff 摘要", DedupArgs, semantic_dedup, ("m1", "m4")),
        Tool("hybrid_search", "混合检索：BM25 + 向量 + 精排，返回带 page/bbox 的证据卡", SearchArgs, hybrid_search, ("m2", "m4", "shared")),
        Tool("fetch_paper", "从 OpenAlex/Crossref/S2 检索外部文献元数据", FetchPaperArgs, fetch_paper, ("m4", "m5", "shared")),
        Tool("compare_claims", "两个对象在多个维度上的多文对照，每格结论都附引用锚点", CompareArgs, compare_claims, ("m2",)),
        Tool("translate_block", "按 block 翻译并保持 bbox 锚点，支持术语表", TranslateArgs, translate_block, ("m3",)),
        Tool("build_citation_graph", "构建引用/共引异构图，输出社区与主路径", GraphArgs, build_citation_graph, ("m5",)),
        Tool("write_section", "按大纲撰写章节，引用锚点强制来自已有证据池", WriteSectionArgs, write_section, ("m6",)),
        Tool("render_chart", "生成 Vega-Lite 图表规范", ChartArgs, render_chart, ("m6",)),
        Tool("export_latex", "导出 LaTeX 正文与 BibTeX 参考文献", ExportArgs, export_latex, ("m6",)),
    ]
    for t in tools:
        registry.register(t)


register_all()
