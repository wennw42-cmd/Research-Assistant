"""解析流水线：PDF → DocIR(JSON)。

顺序：页面几何 → 分栏与阅读顺序(XY-Cut) → 标题层级 → 块分类(公式/表格/图/参考文献)
→ 扫描页 OCR 兜底 → 组装 ParseResult。每一步都保留原始文本与坐标，不生成任何推测内容。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

try:
    import pymupdf as pm
except ImportError:  # pragma: no cover
    import fitz as pm  # type: ignore

from app.parsing.docir import Block, ParseResult
from app.parsing.extract import (
    find_tables,
    is_formula_candidate,
    latex_approx,
    match_caption,
    nearest_caption,
    reference_fields,
    rows_to_markdown,
    split_reference_items,
)
from app.parsing.headings import body_font_size, build_title_tree, classify_headings
from app.parsing.layout import Line, PageGeom, analyse_layout, extract_page_geom, reading_order, xy_cut
from app.parsing.ocr import DEFAULT_DPI, crop_region, needs_ocr, ocr_page

LATEX_COVERAGE_MIN = 0.6


# ────────────────────────── 文本合并 ──────────────────────────
def merge_lines(lines: list[Line]) -> str:
    """按阅读顺序合并行内文本；断词连字符 '-' 处直接拼接。"""
    out: list[str] = []
    for ln in lines:
        t = ln.text.strip()
        if not t:
            continue
        if out and out[-1].endswith("-"):
            out[-1] = out[-1][:-1] + t
        else:
            out.append(t)
    return " ".join(out)


def _bbox_union(boxes: list[tuple[float, float, float, float]]) -> list[float]:
    return [round(min(b[0] for b in boxes), 2), round(min(b[1] for b in boxes), 2),
            round(max(b[2] for b in boxes), 2), round(max(b[3] for b in boxes), 2)]


def _pseudo(page: int, bbox: tuple[float, float, float, float], tag: int) -> Line:
    return Line(page=page, block_no=tag, line_no=0, bbox=tuple(round(v, 2) for v in bbox), text="")


# ────────────────────────── 主流程 ──────────────────────────
def parse_pdf(
    path: str | Path,
    *,
    ocr: str = "auto",
    dpi: int = DEFAULT_DPI,
    assets_dir: str | Path | None = None,
    keep_unknown_images: bool = True,
) -> ParseResult:
    path = Path(path)
    doc = pm.open(str(path))
    try:
        warnings: list[str] = []
        geoms: list[PageGeom] = []
        scanned: list[int] = []

        for i, page in enumerate(doc):
            geom = extract_page_geom(page, i)
            if needs_ocr(page):
                scanned.append(i + 1)
            geoms.append(geom)

        body_size = body_font_size(geoms)
        heads = classify_headings(geoms, body_size)
        head_keys = {(h.line.page, h.line.block_no, h.line.line_no): h for h in heads}
        title_tree = build_title_tree(heads)

        blocks: list[Block] = []
        seq = 0
        for i, page in enumerate(doc):
            geom = geoms[i]
            page_no = i + 1
            if page_no in scanned:
                items, backend = ocr_page(page, dpi=dpi, backend=ocr, assets_dir=assets_dir, page_no=page_no)
                if backend == "none":
                    warnings.append(f"第 {page_no} 页为扫描页且无可用 OCR 后端，已裁图保留原区域（type=unknown）")
                for it in items:
                    seq += 1
                    blocks.append(Block(
                        id=f"p{page_no}b{seq}",
                        type=it.get("type", "paragraph"),  # type: ignore[arg-type]
                        page=page_no,
                        bbox=it.get("bbox", []),
                        text=it.get("text", ""),
                        markdown=it.get("markdown"),
                        latex=it.get("latex"),
                        latex_approx=bool(it.get("latex_approx", False)),
                        image_path=it.get("image_path"),
                        confidence=float(it.get("confidence", 0.0)),
                        source=str(it.get("source", "ocr")),
                        meta=it.get("meta", {}),
                    ))
                continue
            blocks += _parse_text_page(page, geom, body_size, head_keys, assets_dir,
                                       keep_unknown_images, scanned_ok=True)

        blocks += _extract_references(blocks, warnings)
        layout, layout_detail = analyse_layout(geoms)
        layout_detail["scanned_pages"] = scanned
        layout_detail["body_font_size"] = body_size

        stats = {
            "pages": len(geoms),
            "scanned_pages": len(scanned),
            "blocks": len(blocks),
            "by_type": _count_by_type(blocks),
            "tables": len([b for b in blocks if b.type == "table"]),
            "formulas": len([b for b in blocks if b.type == "formula"]),
            "unknown_with_image": len([b for b in blocks if b.type == "unknown" and b.image_path]),
            "title_nodes": _count_nodes(title_tree),
        }
        return ParseResult(
            source={"path": str(path), "sha1": _sha1(path), "pages": len(geoms),
                    "parser": "pymupdf+xycut", "dpi": dpi},
            layout=layout,  # type: ignore[arg-type]
            layout_detail=layout_detail,
            title_tree=title_tree,  # type: ignore[arg-type]
            blocks=blocks,
            stats=stats,
            warnings=warnings,
        )

    finally:
        doc.close()


def _count_by_type(blocks: list[Block]) -> dict[str, int]:
    out: dict[str, int] = {}
    for b in blocks:
        out[b.type] = out.get(b.type, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _count_nodes(nodes: list[dict[str, Any]]) -> int:
    return sum(1 + _count_nodes(n.get("children", [])) for n in nodes)


def _sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ────────────────────────── 单页（有文本层）──────────────────────────
def _parse_text_page(page: Any, geom: PageGeom, body_size: float,
                     head_keys: dict[tuple[int, int, int], Any],
                     assets_dir: str | Path | None,
                     keep_unknown_images: bool, scanned_ok: bool) -> list[Block]:
    ordered = reading_order(geom)
    if not ordered:
        return []

    # 按 PyMuPDF block_no 聚合成段落块（保持块完整性）
    groups: dict[int, list[Line]] = {}
    for ln in ordered:
        groups.setdefault(ln.block_no, []).append(ln)

    tables = find_tables(page)
    # 参与排序的"项"：段落块 + 表格 + 图片
    items: list[Line] = []
    meta: dict[int, dict[str, Any]] = {}
    for b_no, lines in groups.items():
        it = _pseudo(geom.page, tuple(_bbox_union([l.bbox for l in lines])), b_no)  # type: ignore[arg-type]
        items.append(it)
        meta[id(it)] = {"kind": "text", "lines": lines}
    for t in tables:
        it = _pseudo(geom.page, t["bbox"], -1 - len(meta))
        items.append(it)
        meta[id(it)] = {"kind": "table", "table": t}
    for img in _dedup_images(geom.image_bboxes):
        it = _pseudo(geom.page, img, -1000 - len(meta))
        items.append(it)
        meta[id(it)] = {"kind": "figure", "bbox": img}

    ordered_items = xy_cut(items)
    blocks: list[Block] = []
    seq = 0
    text_blocks: list[Block] = []  # 供图注/表注就近匹配

    for it in ordered_items:
        m = meta[id(it)]
        seq += 1
        bid = f"p{geom.page}b{seq}"
        if m["kind"] == "table":
            rows = m["table"]["rows"]
            blocks.append(Block(
                id=bid, type="table", page=geom.page, bbox=list(it.bbox),
                text=" | ".join(rows[0][:4]),
                markdown=rows_to_markdown(rows),
                source=m["table"]["source"],
                confidence=0.95 if m["table"]["source"] == "find_tables" else 0.6,
                meta={"n_rows": len(rows), "header_external": m["table"]["header_external"]},
            ))
            continue
        if m["kind"] == "figure":
            blocks.append(Block(id=bid, type="figure", page=geom.page, bbox=list(it.bbox),
                                text="", confidence=0.9,
                                meta={"caption": None}))
            continue

        lines = m["lines"]
        text = merge_lines(lines)
        if not text:
            continue
        bbox = _bbox_union([l.bbox for l in lines])
        block = _classify(id=bid, page=geom.page, bbox=bbox, text=text, lines=lines,
                          geom=geom, body_size=body_size, head_keys=head_keys,
                          assets_dir=assets_dir, page_obj=page,
                          keep_unknown_images=keep_unknown_images)
        blocks.append(block)
        text_blocks.append(block)

    # 图注/表注回填
    dumps = [t.model_dump() for t in text_blocks]
    for b in blocks:
        if not b.bbox:
            continue
        if b.type == "figure":
            cap = nearest_caption(dumps, tuple(b.bbox), "figure")
            if cap:
                b.caption = cap.get("text", "")
                b.text = b.caption
                b.meta["caption_id"] = cap.get("id")
        elif b.type == "table":
            cap = nearest_caption(dumps, tuple(b.bbox), "table")
            if cap:
                b.caption = cap.get("text", "")
                b.meta["caption_id"] = cap.get("id")
                b.meta["table_no"] = (cap.get("meta") or {}).get("table_no")
    return blocks


def _dedup_images(bboxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for b in bboxes:
        if all(abs(b[0] - o[0]) > 4 or abs(b[1] - o[1]) > 4 for o in out):
            out.append(b)
    return out


def _classify(*, id: str, page: int, bbox: list[float], text: str, lines: list[Line],
              geom: PageGeom, body_size: float, head_keys: dict[tuple[int, int, int], Any],
              assets_dir: str | Path | None, page_obj: Any,
              keep_unknown_images: bool) -> Block:
    # 页眉 / 页脚：只看位置会把"恰好排在页首的章节标题"误判成页眉，必须加字号护栏
    size = max((l.size for l in lines), default=body_size)
    if bbox[3] <= geom.height * 0.09 and size <= body_size * 1.05:
        return Block(id=id, type="header", page=page, bbox=bbox, text=text, confidence=0.8)
    if bbox[1] >= geom.height * 0.91 and size <= body_size * 1.05:
        return Block(id=id, type="footer", page=page, bbox=bbox, text=text, confidence=0.8)

    # 标题（首页无编号且字号显著大于正文 → 视为文章标题）
    matched = [
        (ln, head_keys[(ln.page, ln.block_no, ln.line_no)])
        for ln in lines if (ln.page, ln.block_no, ln.line_no) in head_keys
    ]
    if matched:
        first_ln, first = matched[0]
        is_title = (first_ln.page == 1 and not first.number and body_size
                    and first_ln.size >= body_size * 1.5)
        # PDF 常把编号和标题拆成同一行的两段（"1" / "Introduction"），合并回去
        parts = [first.text]
        level = first.level
        for _ln, h in matched[1:]:
            if h.number:
                break
            parts.append(h.text)
            level = min(level, h.level)
        return Block(id=id, type="title" if is_title else "heading",
                     page=page, bbox=bbox, text=" ".join(p for p in parts if p),
                     level=level, number=first.number or None, confidence=0.9)

    # 图注 / 表注
    cap = match_caption(text)
    if cap:
        kind, num = cap
        return Block(id=id, type="figure_caption" if kind == "figure" else "table_caption",
                     page=page, bbox=bbox, text=text, confidence=0.95,
                     meta={"figure_no" if kind == "figure" else "table_no": num})

    # 列表项
    if len(text) < 400 and any(_list_line(l.text) for l in lines):
        return Block(id=id, type="list_item", page=page, bbox=bbox, text=text, confidence=0.75)

    # 公式
    font = lines[0].font if lines else ""
    if is_formula_candidate(text, font, size, body_size):
        latex, coverage = latex_approx(text)
        if coverage >= LATEX_COVERAGE_MIN:
            return Block(id=id, type="formula", page=page, bbox=bbox, text=text,
                         latex=latex, latex_approx=True,
                         confidence=round(min(0.95, 0.5 + coverage * 0.45), 3),
                         source="pymupdf+heuristic",
                         meta={"symbol_coverage": round(coverage, 3), "math_font": font})
        if assets_dir and keep_unknown_images:
            img = crop_region(page_obj, tuple(bbox), Path(assets_dir) / f"{id}_formula.png")
            return Block(id=id, type="unknown", page=page, bbox=bbox, text=text,
                         image_path=img, confidence=0.3,
                         source="pymupdf+heuristic",
                         meta={"reason": f"符号转写覆盖率 {coverage:.2f} < {LATEX_COVERAGE_MIN}，保留原图区域"})
        return Block(id=id, type="unknown", page=page, bbox=bbox, text=text, confidence=0.3,
                     source="pymupdf+heuristic",
                     meta={"reason": "公式符号无法可靠转写为 LaTeX"})

    return Block(id=id, type="paragraph", page=page, bbox=bbox, text=text, confidence=0.9)


def _list_line(t: str) -> bool:
    s = t.strip()
    if not s:
        return False
    if s[0] in "•·▪◦‣-*":
        return True
    import re

    return bool(re.match(r"^(?:\d{1,2}[)）.、]|[a-z][)）.])\s+\S", s))


# ────────────────────────── 参考文献 ──────────────────────────
REF_HEAD_TEXT = ("references", "bibliography", "参考文献", "参考书目")


def _extract_references(blocks: list[Block], warnings: list[str]) -> list[Block]:
    """把参考文献段落切成条目块；只抽 ID / DOI / arXiv / URL 这些可正则验证的字段。"""
    head_idx = -1
    head_level = 9
    for i, b in enumerate(blocks):
        if b.type in ("heading", "title") and (b.text or "").strip().lower().rstrip(":：") in REF_HEAD_TEXT:
            head_idx, head_level = i, b.level or 9
            break
    if head_idx < 0:
        return []

    ref_blocks: list[Block] = []
    last: Block | None = None
    for b in blocks[head_idx + 1:]:
        # 参考文献之后的下一个标题（附录/Attention Visualizations 等）就是终点
        if b.type in ("heading", "title"):
            break
        if b.type in ("header", "footer", "figure", "table"):
            continue
        if not b.text.strip():
            continue
        x0 = (b.bbox or [0.0])[0]
        for j, item in enumerate(split_reference_items(b.text)):
            fields = reference_fields(item)
            # PyMuPDF 常把条目折行拆成独立块；悬挂缩进的续行并回上一条
            # （续行可能以 "(8):1735–1780" 的卷期号开头，不能只看编号）
            if last is not None and x0 > (last.bbox or [0.0])[0] + 4:
                last.text = f"{last.text.rstrip()} {item}"
                for key in ("arxiv", "doi", "url"):
                    if fields[key] and not last.meta.get(key):
                        last.meta[key] = fields[key]
                continue
            block = Block(
                id=f"{b.id}r{j}", type="reference", page=b.page, bbox=b.bbox,
                text=item, confidence=0.9 if fields["ref_id"] else 0.7,
                source="regex",
                meta={**fields, "note": "条目坐标沿用所属段落 bbox（行级坐标需 OCR 支持）"},
            )
            ref_blocks.append(block)
            last = block
    if not ref_blocks:
        warnings.append("检测到参考文献标题但未切出条目，请检查版式")
    return ref_blocks
