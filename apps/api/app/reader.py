"""解析结果 → 学术阅读器文档（引用映射 / 跳转锚点 / 标题树 / 双栏几何）。

输入是 `app.parsing` 产出的 ParseResult（或其 JSON），输出是前端直接消费的
ReaderDocument。核心约束与解析层一致：**只搬运已有的文本与坐标，不补全**。

    python -m app.reader parse.json -o reader.json
    python -m app.reader --demo --format js -o ../web/reader/demo-data.js
    python -m app.reader --selftest

四件事：

1. **引用映射**：正文 `[11]` / `[2, 3]` / `[4–6]` → 参考文献块。只按参考文献
   自带的编号（`meta.ref_id` 或条目开头的 `[n]`/`(n)`）建表；编号缺失就保持
   `number=None`，绝不按出现顺序臆造编号。重复编号标记 ambiguous，不随便选一个。
2. **双向锚点**：每个 marker 有 occurrence id，每条参考文献有 ref id；
   occurrence → targets，reference → cite_ids（回链）。
3. **标题树**：从 heading 块重建（而不是用没有块 ID 的 title_tree），节点带
   `anchor`，供前端滚动联动。参考文献标题映射到文末引用区。
4. **双栏几何**：每页记录 width/height/columns/gutters；每个块给 `column`
   与 `span`，前端按"栏带（band）"重排。坐标是原 PDF pt，不缩放文本。

输出里的 `segments` 把正文切成 text/cite 片段，前端不需要再写一遍引用正则。
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

VERSION = 1
MAX_RANGE = 50                      # [1-9999] 这类异常范围不展开，标 invalid
REF_HEAD_TEXT = {"references", "bibliography", "参考文献", "参考书目", "参考引用"}

# [1] / [1, 2] / [1-3] / [1–3] / 全角括号；只收纯数字内容，避免命中 LaTeX [x]
CITE_RE = re.compile(r"[\[［]\s*(\d{1,4}(?:\s*[-–—,，、;；]\s*\d{1,4})*)\s*[\]］]")
RANGE_RE = re.compile(r"^(\d{1,4})\s*[-–—]\s*(\d{1,4})$")
REF_NUM_RE = re.compile(r"^\s*[\[［(（]\s*(\d{1,4})\s*[\]］)）]")
SKIP_TYPES = {"header", "footer", "reference"}


# ══════════════════════════════ 基础 ══════════════════════════════
def _as_dict(parsed: Any) -> dict[str, Any]:
    if hasattr(parsed, "model_dump"):
        return parsed.model_dump(mode="json")
    if isinstance(parsed, dict):
        return parsed
    raise TypeError("parsed 必须是 ParseResult 或 dict")


def _num(value: Any) -> str | None:
    """把编号规范成十进制字符串：01 → 1；非数字原样保留。"""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return str(int(s)) if s.isdigit() else s


def _bbox(b: dict[str, Any]) -> list[float] | None:
    bb = b.get("bbox") or []
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return None
    try:
        out = [float(x) for x in bb]
    except (TypeError, ValueError):
        return None
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def _meta(b: dict[str, Any]) -> dict[str, Any]:
    m = b.get("meta") or {}
    return m if isinstance(m, dict) else {}


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# ══════════════════════════════ 页面几何 ══════════════════════════════
def _page_geometry(parsed: dict[str, Any], blocks: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    detail = parsed.get("layout_detail") or {}
    sizes = {int(x["page"]): x for x in (detail.get("page_sizes") or [])
             if isinstance(x, dict) and x.get("page")}
    cols = detail.get("page_columns") or []
    gutters_in = {int(x["page"]): x.get("gutters") or []
                  for x in (detail.get("gutters") or [])
                  if isinstance(x, dict) and x.get("page")}
    max_page = max([int(b.get("page") or 1) for b in blocks] + [1])
    warnings: list[str] = []
    estimated_any = False
    pages: dict[int, dict[str, Any]] = {}

    for p in range(1, max_page + 1):
        size = sizes.get(p)
        if size:
            width, height = float(size["width"]), float(size["height"])
            estimated = False
        else:
            bbs = [_bbox(b) for b in blocks if int(b.get("page") or 1) == p]
            bbs = [bb for bb in bbs if bb]
            width = (max((bb[2] for bb in bbs), default=612.0) * 1.06) or 612.0
            height = (max((bb[3] for bb in bbs), default=792.0) * 1.04) or 792.0
            estimated = True
            estimated_any = True

        if p - 1 < len(cols) and cols[p - 1]:
            columns = max(1, int(cols[p - 1]))
        else:
            columns = 2 if parsed.get("layout") in ("double", "multi") else 1
            estimated = True

        gutters = [list(map(float, g)) for g in (gutters_in.get(p) or [])
                   if isinstance(g, (list, tuple)) and len(g) == 2]
        if columns >= 2 and not gutters:
            mid = width / 2
            gutters = [[round(mid - 6, 2), round(mid + 6, 2)]]
            estimated = True
            warnings.append(f"第 {p} 页缺少栏间距，已按页宽中点近似")
        if len(gutters) != max(0, columns - 1) and columns >= 2:
            warnings.append(f"第 {p} 页栏数={columns} 与栏间距数={len(gutters)} 不一致，按坐标就近归栏")

        pages[p] = {"page": p, "width": round(width, 2), "height": round(height, 2),
                    "columns": columns, "gutters": gutters, "estimated": estimated}
        estimated_any = estimated_any or estimated

    if estimated_any:
        warnings.append("部分页面缺少原始尺寸/栏距，已按块 bbox 估算（estimated=true）")
    return pages, warnings


def _place(bb: list[float] | None, page: dict[str, Any]) -> tuple[int, bool]:
    """返回 (column, span)。跨栏或宽度 ≥70% 页宽 → span=True，column=-1。"""
    if not bb or page["columns"] <= 1:
        return 0, True
    x0, _y0, x1, _y1 = bb
    mids = [(g[0] + g[1]) / 2 for g in page["gutters"]]
    if any(x0 < m < x1 for m in mids) or (x1 - x0) >= page["width"] * 0.7:
        return -1, True
    cx = (x0 + x1) / 2
    return sum(1 for m in mids if cx > m), False


# ══════════════════════════════ 引用 ══════════════════════════════
def _expand_ids(inner: str) -> tuple[list[str], bool]:
    """'2, 4-6' → ['2','4','5','6']；范围过大返回 invalid。"""
    ids: list[str] = []
    invalid = False
    for part in re.split(r"[,，、;；]", inner):
        part = part.strip()
        if not part:
            continue
        m = RANGE_RE.match(part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            if b - a + 1 > MAX_RANGE:
                invalid = True
                continue
            for n in range(a, b + 1):
                s = str(n)
                if s not in ids:
                    ids.append(s)
        elif part.isdigit():
            s = str(int(part))
            if s not in ids:
                ids.append(s)
        else:
            invalid = True
    return ids, invalid


def _scan_block(b: dict[str, Any], by_number: dict[str, list[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = b.get("text") or ""
    segments: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []
    last = 0
    for m in CITE_RE.finditer(text):
        if m.start() > last:
            segments.append({"kind": "text", "text": text[last:m.start()]})
        raw = m.group(0)
        ids, invalid = _expand_ids(m.group(1))
        targets: list[str] = []
        for n in ids:
            for rid in by_number.get(n, []):
                if rid not in targets:
                    targets.append(rid)
        unresolved = [n for n in ids if not by_number.get(n)]
        ambiguous = [n for n in ids if len(by_number.get(n, [])) > 1]
        if invalid or not ids:
            status = "invalid"
        elif not targets:
            status = "missing"
        elif ambiguous:
            status = "ambiguous"
        elif unresolved:
            status = "partial"
        else:
            status = "resolved"

        occ = {
            "id": f"c-{b['id']}-{m.start()}",
            "block_id": b["id"], "start": m.start(), "end": m.end(), "raw": raw,
            "ref_ids": ids, "targets": targets,
            "unresolved_ids": unresolved, "ambiguous_ids": ambiguous,
            "status": status,
        }
        occurrences.append(occ)
        segments.append({"kind": "cite", "text": raw, "occurrence_id": occ["id"],
                         "ref_ids": ids, "targets": targets, "status": status})
        last = m.end()
    if last < len(text):
        segments.append({"kind": "text", "text": text[last:]})
    return segments, occurrences


def _reference_number(b: dict[str, Any]) -> str | None:
    n = _meta(b).get("ref_id")
    if n is not None:
        return _num(n)
    m = REF_NUM_RE.match(b.get("text") or "")
    return _num(m.group(1)) if m else None


def _build_references(blocks: list[dict[str, Any]], warnings: list[str]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    refs: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for i, b in enumerate([x for x in blocks if x.get("type") == "reference"]):
        bid = str(b.get("id") or f"ref{i}")
        rid = f"ref-{bid}"
        while rid in used_ids:
            rid = f"{rid}-{i}"
        used_ids.add(rid)
        meta = _meta(b)
        refs.append({
            "id": rid, "block_id": bid, "number": _reference_number(b),
            "anchor": rid, "page": int(b.get("page") or 0), "bbox": _bbox(b) or [],
            "text": b.get("text") or "",
            "doi": meta.get("doi"), "arxiv": meta.get("arxiv"), "url": meta.get("url"),
            "ambiguous": False, "cite_ids": [],
        })

    by_number: dict[str, list[str]] = {}
    for r in refs:
        if r["number"]:
            by_number.setdefault(r["number"], []).append(r["id"])
    for num, ids in sorted(by_number.items(), key=lambda kv: (len(kv[0]), kv[0])):
        if len(ids) > 1:
            for rid in ids:
                next(r for r in refs if r["id"] == rid)["ambiguous"] = True
            warnings.append(f"参考文献编号 {num} 重复（{len(ids)} 条），正文引用标记为 ambiguous")
    return refs, by_number


# ══════════════════════════════ 标题树 ══════════════════════════════
def _heading_level(b: dict[str, Any]) -> int:
    lv = b.get("level")
    if isinstance(lv, int) and 1 <= lv <= 6:
        return lv
    num = b.get("number")
    return min(4, str(num).count(".") + 1) if num else 1


def _is_refs_heading(b: dict[str, Any]) -> bool:
    t = _norm_text(b.get("text") or "").rstrip(":：").lower()
    return t in REF_HEAD_TEXT


def _build_toc(blocks: list[dict[str, Any]], doc_title_id: str | None,
               refs_heading_id: str | None) -> list[dict[str, Any]]:
    root: list[dict[str, Any]] = []
    stack: list[tuple[int, dict[str, Any]]] = []
    for i, b in enumerate(blocks):
        if b.get("type") not in ("heading", "title"):
            continue
        if b.get("id") == doc_title_id:
            continue
        # 页码水印、行号这类"标题"没有字母，进目录只会制造噪声
        text = (b.get("text") or "").strip()
        if not re.search(r"[A-Za-z一-鿿]", text):
            continue
        if text.lower().startswith("arxiv:") or len(text) > 120:
            continue
        if b.get("id") == refs_heading_id:
            anchor, kind = "reader-references", "references"
        else:
            anchor, kind = f"blk-{b['id']}", "heading"
        level = 1 if b.get("type") == "title" else _heading_level(b)
        number = _num(b.get("number"))
        if number and text.startswith(number):  # pipeline 已把编号并进正文
            text = text[len(number):].lstrip(" .、:：\t")
        node = {"id": f"toc-{i}", "text": text, "level": level,
                "number": number, "page": int(b.get("page") or 0),
                "anchor": anchor, "kind": kind, "children": []}
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            root.append(node)
        stack.append((level, node))
    return root


def _count_toc(nodes: Iterable[dict[str, Any]]) -> int:
    return sum(1 + _count_toc(n.get("children", [])) for n in nodes)


# ══════════════════════════════ 主入口 ══════════════════════════════
def build_reader(parsed: Any, *, title: str | None = None) -> dict[str, Any]:
    data = _as_dict(parsed)
    # 外部传入的 JSON 可能缺 block id；补一个稳定的本地 id，后续锚点/occurrence 才能建
    raw_blocks: list[dict[str, Any]] = []
    for i, b in enumerate(data.get("blocks") or []):
        if not isinstance(b, dict):
            continue
        raw_blocks.append(b if b.get("id") else {**b, "id": f"b{i}"})
    if not raw_blocks:
        raise ValueError("解析结果没有 blocks")

    pages, geo_warnings = _page_geometry(data, raw_blocks)
    references, by_number = _build_references(raw_blocks, geo_warnings)
    warnings: list[str] = list(data.get("warnings") or []) + geo_warnings

    title_blocks = [b for b in raw_blocks if b.get("type") == "title" and (b.get("text") or "").strip()]
    doc_title_id = title_blocks[0]["id"] if title_blocks else None
    doc_title = title or (title_blocks[0].get("text") if title_blocks else None)

    refs_heading = next((b for b in raw_blocks
                         if b.get("type") in ("heading", "title") and _is_refs_heading(b)), None)
    # 解析层把参考文献条目另存为 reference 块，但**原文段落仍留在 blocks 里**。
    # 若正文条目不排除，文献区会被渲染两遍，且条目里的 [1] 会被误当成正文引用。
    # 这里按解析层相同的规则圈出文献区：标题之后、下一个同级/更高级标题之前。
    have_refs = bool(references)
    refs_heading_id = refs_heading.get("id") if (refs_heading and have_refs) else None
    skip_from = skip_to = -1
    if have_refs and refs_heading is not None:
        start = next((i for i, b in enumerate(raw_blocks)
                      if b.get("id") == refs_heading.get("id")), -1)
        if start >= 0:
            level = refs_heading.get("level") or 9
            end = len(raw_blocks)
            for j in range(start + 1, len(raw_blocks)):
                nb = raw_blocks[j]
                if nb.get("type") in ("heading", "title") and (nb.get("level") or 9) <= level:
                    end = j
                    break
            skip_from, skip_to = start + 1, end

    # 表格/插图块本身就带 caption，解析层又会单独产出 caption 块；同一句只留一份。
    attached_captions = {
        _norm_text(b.get("caption") or "") for b in raw_blocks
        if b.get("type") in ("table", "figure") and (b.get("caption") or "").strip()
    }

    body: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []
    for i, b in enumerate(raw_blocks):
        if skip_from <= i < skip_to:
            continue
        btype = b.get("type")
        if btype in SKIP_TYPES:
            continue
        if b.get("id") == refs_heading_id:
            continue
        text = b.get("text") or ""
        if btype in ("table_caption", "figure_caption") and _norm_text(text) in attached_captions:
            continue
        if not (text.strip() or b.get("markdown") or b.get("latex") or b.get("image_path")):
            continue
        bb = _bbox(b)
        page = pages.get(int(b.get("page") or 1)) or pages[1]
        column, span = _place(bb, page)
        segments, occs = _scan_block(b, by_number)
        occurrences.extend(occs)

        out = {
            "id": str(b.get("id") or f"b{len(body)}"),
            "anchor": f"blk-{b.get('id') or f'b{len(body)}'}",
            "type": btype or "paragraph",
            "page": int(b.get("page") or 1),
            "bbox": bb or [],
            "column": column,
            "span": span,
            "rank": len(body),
            "level": b.get("level"),
            "number": _num(b.get("number")),
            "text": text,
            "latex": b.get("latex"),
            "latex_approx": bool(b.get("latex_approx")),
            "markdown": b.get("markdown"),
            "caption": b.get("caption"),
            "image_path": b.get("image_path"),
            "confidence": b.get("confidence"),
            "source": b.get("source"),
        }
        if segments:
            out["segments"] = segments
        body.append({k: v for k, v in out.items() if v is not None})

    # 回链：reference → 正文 occurrence
    ref_index = {r["id"]: r for r in references}
    for occ in occurrences:
        for rid in occ["targets"]:
            if rid in ref_index and occ["id"] not in ref_index[rid]["cite_ids"]:
                ref_index[rid]["cite_ids"].append(occ["id"])

    toc = _build_toc(raw_blocks, doc_title_id, refs_heading_id)

    unresolved = sorted({n for occ in occurrences for n in occ["unresolved_ids"]},
                        key=lambda s: (len(s), s))
    if unresolved:
        warnings.append(f"正文引用了参考文献表中不存在的编号：{', '.join(unresolved)}")
    invalid = [occ["raw"] for occ in occurrences if occ["status"] == "invalid"]
    if invalid:
        warnings.append(f"无法展开的引用标记：{', '.join(invalid[:5])}")

    counts = {k: sum(1 for o in occurrences if o["status"] == k)
              for k in ("resolved", "partial", "missing", "ambiguous", "invalid")}
    return {
        "version": VERSION,
        "meta": {
            "title": doc_title or "未命名文献",
            "source": data.get("source") or {},
            "layout": data.get("layout") or "single",
            "pages": len(pages),
        },
        "pages": [pages[p] for p in sorted(pages)],
        "blocks": body,
        "toc": toc,
        "references": references,
        "citations": {
            "occurrences": occurrences,
            "by_number": by_number,
            "unresolved_numbers": unresolved,
            "stats": {"total": len(occurrences), **counts},
        },
        "stats": {
            "blocks": len(body),
            "references": len(references),
            "toc_nodes": _count_toc(toc),
            "pages": len(pages),
            "citations": len(occurrences),
        },
        "warnings": warnings,
    }


# ══════════════════════════════ 演示数据 ══════════════════════════════
def demo_parse_result() -> dict[str, Any]:
    """一份仿 ParseResult 的双栏样例（含引用标记、缺失编号、跨栏表格）。"""

    def bb(x0: float, y0: float, x1: float, y1: float) -> list[float]:
        return [x0, y0, x1, y1]

    blocks: list[dict[str, Any]] = [
        {"id": "p1b1", "type": "title", "page": 1, "bbox": bb(120, 48, 475, 82),
         "text": "A Demo Paper: Attention-Based Sequence Models", "confidence": 0.99},
        {"id": "p1b2", "type": "paragraph", "page": 1, "bbox": bb(60, 112, 280, 190),
         "text": "We propose a compact attention model [1]. Prior sequence models rely on recurrence, "
                 "which limits parallel training.", "confidence": 0.95},
        {"id": "p1b3", "type": "heading", "page": 1, "bbox": bb(60, 210, 280, 226),
         "text": "1 Introduction", "level": 1, "number": "1", "confidence": 0.95},
        {"id": "p1b4", "type": "paragraph", "page": 1, "bbox": bb(60, 236, 280, 326),
         "text": "Attention scales with sequence length rather than depth [2–3]. "
                 "This property is central to our design.", "confidence": 0.95},
        {"id": "p1b5", "type": "paragraph", "page": 1, "bbox": bb(315, 112, 535, 190),
         "text": "The method builds on prior encoder-decoder work [2, 3].",
         "confidence": 0.95},
        {"id": "p1b6", "type": "heading", "page": 1, "bbox": bb(315, 210, 535, 226),
         "text": "2 Method", "level": 1, "number": "2", "confidence": 0.95},
        {"id": "p1b7", "type": "paragraph", "page": 1, "bbox": bb(315, 236, 535, 300),
         "text": "Each layer applies scaled dot-product attention. We compare against the "
                 "large-batch baseline [9].", "confidence": 0.95},
        {"id": "p1b8", "type": "heading", "page": 1, "bbox": bb(315, 316, 535, 332),
         "text": "2.1 Scaled Attention", "level": 2, "number": "2.1", "confidence": 0.95},
        {"id": "p1b9", "type": "paragraph", "page": 1, "bbox": bb(315, 342, 535, 410),
         "text": "The softmax temperature follows the original formulation [1].",
         "confidence": 0.95},
        {"id": "p1b10c", "type": "table_caption", "page": 1, "bbox": bb(60, 436, 535, 448),
         "text": "Table 1: Results on the demo benchmark [3].", "confidence": 0.9},
        {"id": "p1b10", "type": "table", "page": 1, "bbox": bb(60, 450, 535, 540),
         "text": "Table 1: Results on the demo benchmark [3].",
         "markdown": "| Model | BLEU |\n| --- | --- |\n| Ours | 28.4 |\n| Baseline | 25.2 |",
         "caption": "Table 1: Results on the demo benchmark [3].",
         "confidence": 0.92, "source": "find_tables"},
        {"id": "p2b1", "type": "heading", "page": 2, "bbox": bb(60, 60, 280, 76),
         "text": "3 Results", "level": 1, "number": "3", "confidence": 0.95},
        {"id": "p2b2", "type": "paragraph", "page": 2, "bbox": bb(60, 86, 280, 160),
         "text": "The proposed model improves BLEU over the baseline [2].",
         "confidence": 0.95},
        {"id": "p2b3", "type": "heading", "page": 2, "bbox": bb(60, 190, 280, 206),
         "text": "References", "level": 1, "number": "", "confidence": 0.95},
        {"id": "p2b4", "type": "reference", "page": 2, "bbox": bb(60, 220, 280, 258),
         "text": "[1] A. Author. A Demo Paper on Attention. arXiv:1706.03762, 2017.",
         "meta": {"ref_id": "1", "arxiv": "arXiv:1706.03762"}, "confidence": 0.9},
        {"id": "p2b5", "type": "reference", "page": 2, "bbox": bb(315, 220, 535, 258),
         "text": "[2] B. Author. Sequence Models Revisited. DOI:10.1000/demo.2018, 2018.",
         "meta": {"ref_id": "2", "doi": "10.1000/demo.2018"}, "confidence": 0.9},
        {"id": "p2b6", "type": "reference", "page": 2, "bbox": bb(60, 264, 280, 302),
         "text": "[3] C. Author. Attention Baselines. https://example.org/p3, 2019.",
         "meta": {"ref_id": "3", "url": "https://example.org/p3"}, "confidence": 0.9},
        # 解析层会保留文献区的原文段落；阅读器必须排除，否则重复渲染且误判引用
        {"id": "p2b7", "type": "paragraph", "page": 2, "bbox": bb(315, 308, 535, 346),
         "text": "Raw source paragraph of a reference entry [2]; must not be body text.",
         "confidence": 0.9},
    ]
    return {
        "source": {"path": "demo.pdf", "sha1": "demo", "pages": 2, "parser": "demo"},
        "layout": "double",
        "layout_detail": {
            "page_columns": [2, 2],
            "page_sizes": [{"page": 1, "width": 595.0, "height": 842.0},
                           {"page": 2, "width": 595.0, "height": 842.0}],
            "gutters": [{"page": 1, "gutters": [[290.0, 305.0]]},
                        {"page": 2, "gutters": [[290.0, 305.0]]}],
            "pages_with_text": 2,
        },
        "title_tree": [],
        "blocks": blocks,
        "stats": {},
        "warnings": [],
    }


# ══════════════════════════════ 自检 ══════════════════════════════
def _selftest() -> int:
    parsed = demo_parse_result()
    doc = build_reader(parsed)
    checks: list[tuple[str, bool, str]] = []

    refs = {r["number"]: r for r in doc["references"]}
    checks.append(("3 条参考文献", len(doc["references"]) == 3, str(sorted(refs))))
    checks.append(("编号映射 1/2/3", set(doc["citations"]["by_number"]) == {"1", "2", "3"},
                   str(doc["citations"]["by_number"])))

    occs = doc["citations"]["occurrences"]
    by_block = {}
    for o in occs:
        by_block.setdefault(o["block_id"], []).append(o)

    first = by_block["p1b2"][0]
    checks.append(("[1] → 第一条参考文献",
                   first["ref_ids"] == ["1"] and first["targets"] == [refs["1"]["id"]]
                   and first["status"] == "resolved", json.dumps(first, ensure_ascii=False)))
    multi = by_block["p1b5"][0]
    checks.append(("[2, 3] 展开为两条",
                   multi["ref_ids"] == ["2", "3"] and len(multi["targets"]) == 2
                   and multi["status"] == "resolved", multi["raw"]))
    rng = by_block["p1b4"][0]
    checks.append(("[2–3] 范围展开", rng["ref_ids"] == ["2", "3"], rng["raw"]))
    miss = by_block["p1b7"][0]
    checks.append(("[9] 未命中 → missing",
                   miss["status"] == "missing" and miss["unresolved_ids"] == ["9"], miss["status"]))
    checks.append(("回链 reference→occurrence",
                   len(refs["1"]["cite_ids"]) >= 2, str(refs["1"]["cite_ids"])))

    # 片段拼回去必须与原文逐字一致，偏移必须能切开原文
    ok_seg = ok_off = True
    for b in doc["blocks"]:
        if b.get("segments"):
            ok_seg &= "".join(s["text"] for s in b["segments"]) == b["text"]
    for o in occs:
        text = next(b["text"] for b in doc["blocks"] if b["id"] == o["block_id"])
        ok_off &= text[o["start"]:o["end"]] == o["raw"]
    checks.append(("segments 无损", ok_seg, "text/cite 片段拼接 == text"))
    checks.append(("occurrence 偏移可回切", ok_off, "text[start:end] == raw"))
    checks.append(("图注不重复", len(doc["blocks"]) == 12
                   and all(b["id"] != "p1b10c" for b in doc["blocks"]),
                   f"blocks={len(doc['blocks'])}"))
    checks.append(("文献区原文不重复进入正文",
                   all(b["id"] != "p2b7" for b in doc["blocks"])
                   and doc["stats"]["citations"] == 7,
                   f"blocks={doc['stats']['blocks']} cites={doc['stats']['citations']}"))

    # 标题树
    toc = doc["toc"]
    flat = []

    def walk(nodes, depth=0):
        for n in nodes:
            flat.append((n["text"], n["level"], n["anchor"], depth))
            walk(n.get("children", []), depth + 1)

    walk(toc)
    intro = next(x for x in flat if x[0] == "1 Introduction")
    method = next(x for x in flat if x[0] == "2 Method")
    sub = next(x for x in flat if x[0] == "2.1 Scaled Attention")
    checks.append(("标题树 2.1 嵌套在 2 下", sub[3] == method[3] + 1,
                   f"method_depth={method[3]} sub_depth={sub[3]}"))
    checks.append(("标题 anchor 指向块", intro[2] == "blk-p1b3", intro[2]))
    refs_node = [x for x in flat if x[2] == "reader-references"]
    checks.append(("References 标题 → 文末引用区",
                   bool(refs_node) and refs_node[0][0] == "References", str(refs_node[:1])))

    # 双栏几何
    blocks = {b["id"]: b for b in doc["blocks"]}
    checks.append(("左栏/右栏/跨栏",
                   blocks["p1b2"]["column"] == 0 and not blocks["p1b2"]["span"]
                   and blocks["p1b5"]["column"] == 1 and not blocks["p1b5"]["span"]
                   and blocks["p1b10"]["span"],
                   str({k: (blocks[k]["column"], blocks[k]["span"])
                        for k in ("p1b2", "p1b5", "p1b10")})))
    checks.append(("页面尺寸与栏距", doc["pages"][0]["width"] == 595.0
                   and doc["pages"][0]["gutters"] == [[290.0, 305.0]],
                   json.dumps(doc["pages"][0], ensure_ascii=False)))

    # 重复编号 → ambiguous，而不是随便选一条
    dup = copy.deepcopy(parsed)
    extra = copy.deepcopy(next(b for b in reversed(dup["blocks"]) if b["type"] == "reference"))
    extra["id"] = "p2b7"
    extra["text"] = "[2] D. Author. Duplicate Number Entry. 2020."
    extra["meta"] = {"ref_id": "2"}
    dup["blocks"].append(extra)
    doc2 = build_reader(dup)
    occ2 = next(o for o in doc2["citations"]["occurrences"] if "2" in o["ref_ids"])
    checks.append(("重复编号 → ambiguous",
                   len(doc2["citations"]["by_number"]["2"]) == 2
                   and occ2["status"] == "ambiguous"
                   and all(r["ambiguous"] for r in doc2["references"] if r["number"] == "2"),
                   f"targets={len(occ2['targets'])} status={occ2['status']}"))

    # 无编号条目：保留但不进映射
    unnum = copy.deepcopy(parsed)
    extra = copy.deepcopy(next(b for b in reversed(unnum["blocks"]) if b["type"] == "reference"))
    extra.update({"id": "p2b8", "text": "Anonymous. Undated note.", "meta": {}})
    unnum["blocks"].append(extra)
    doc3 = build_reader(unnum)
    checks.append(("无编号条目保留但不臆造编号",
                   len(doc3["references"]) == 4
                   and sum(1 for r in doc3["references"] if r["number"] is None) == 1,
                   str([(r["number"], r["id"]) for r in doc3["references"]])))

    # 缺少页面尺寸时用 bbox 估算，不崩
    thin = copy.deepcopy(parsed)
    thin.pop("layout_detail", None)
    doc4 = build_reader(thin)
    checks.append(("缺几何 → 估算并告警",
                   doc4["pages"][0]["estimated"] and bool(doc4["warnings"]),
                   f"width={doc4['pages'][0]['width']} warnings={len(doc4['warnings'])}"))

    # 外部 JSON 缺 block id：补 id 后仍能建锚点
    noid = copy.deepcopy(parsed)
    del noid["blocks"][1]["id"]
    doc5 = build_reader(noid)
    checks.append(("缺 block id 自动补",
                   all(b["id"] and b["anchor"].startswith("blk-") for b in doc5["blocks"]),
                   str(doc5["blocks"][1]["id"])))

    # 配置了 page_sizes 时 reading 输出可直接 JSON 序列化
    checks.append(("JSON 可序列化", json.loads(json.dumps(doc, ensure_ascii=False))["version"] == 1,
                   str(doc["stats"])))

    print("─" * 88)
    passed = 0
    for name, ok, detail in checks:
        passed += bool(ok)
        print(f"[{'OK' if ok else 'XX'}] {name:<28} {detail}")
    print("─" * 88)
    print(f"selftest: {passed}/{len(checks)} 通过")
    return 0 if passed == len(checks) else 1


# ══════════════════════════════ CLI ══════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ParseResult → 学术阅读器文档")
    ap.add_argument("parse", nargs="?", help="parse_pdf 输出的 JSON")
    ap.add_argument("--demo", action="store_true", help="使用内置双栏样例")
    ap.add_argument("--format", choices=["json", "js"], default="json",
                    help="js = window.__READER_DOC__ = {...}（浏览器直接加载）")
    ap.add_argument("-o", "--out", help="输出文件；省略则打印")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    if args.selftest:
        return _selftest()
    if args.demo:
        parsed = demo_parse_result()
    elif args.parse:
        parsed = json.loads(Path(args.parse).read_text(encoding="utf-8"))
    else:
        ap.print_help()
        return 2

    doc = build_reader(parsed)
    text = json.dumps(doc, ensure_ascii=False, indent=2)
    if args.format == "js":
        text = f"window.__READER_DOC__ = {text};\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"written: {args.out}  blocks={doc['stats']['blocks']} "
              f"refs={doc['stats']['references']} cites={doc['stats']['citations']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
