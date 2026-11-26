"""版式分析：单/双栏判定 + 阅读顺序（XY-Cut）。

为什么不用"按 y 排序"直接出结论：双栏论文右栏的顶部 y 往往小于左栏的底部 y，
单纯 y 排序会把两栏交错成一段读不通的正文。

XY-Cut 是学术排版的事实标准做法：先在 y 轴找空白带切成"行带"，再在 x 轴找
空白带切成"列"，递归下去直到不可再切。因为我们在**文本行**（细长矩形）上做切分，
不会遇到"跨行大块"把行带连通的经典失效场景。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # PyMuPDF >= 1.24 推荐 import pymupdf；fitz 为兼容别名
    import pymupdf as pm
except ImportError:  # pragma: no cover
    import fitz as pm  # type: ignore


@dataclass
class Line:
    """一个文本行（排版分析的最小单位）。"""

    page: int
    block_no: int
    line_no: int
    bbox: tuple[float, float, float, float]
    text: str
    font: str = ""
    size: float = 0.0
    bold: bool = False
    italic: bool = False
    column: int = 0


@dataclass
class PageGeom:
    page: int
    width: float
    height: float
    columns: int = 1
    gutters: list[tuple[float, float]] = field(default_factory=list)  # 栏间空白带
    lines: list[Line] = field(default_factory=list)
    image_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


# ────────────────────────── 抽取 ──────────────────────────
def extract_page_geom(page: Any, page_index: int) -> PageGeom:
    rect = page.rect
    geom = PageGeom(page=page_index + 1, width=float(rect.width), height=float(rect.height))

    raw = page.get_text("dict")
    for b_no, b in enumerate(raw.get("blocks", [])):
        if b.get("type") != 0:  # 1 = 图片
            if b.get("type") == 1:
                geom.image_bboxes.append(tuple(round(v, 2) for v in b["bbox"]))
            continue
        for l_no, l in enumerate(b.get("lines", [])):
            text = "".join(s.get("text", "") for s in l.get("spans", [])).strip()
            if not text:
                continue
            spans = [s for s in l.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            biggest = max(spans, key=lambda s: len(s.get("text", "")))
            size = float(biggest.get("size", 0.0))
            flags = int(biggest.get("flags", 0))
            geom.lines.append(Line(
                page=page_index + 1,
                block_no=b_no,
                line_no=l_no,
                bbox=tuple(round(v, 2) for v in l["bbox"]),  # type: ignore[arg-type]
                text=text,
                font=str(biggest.get("font", "")),
                size=size,
                bold=bool(flags & 2**4),
                italic=bool(flags & 2**1),
            ))

    for img in _safe_image_info(page):
        geom.image_bboxes.append(tuple(round(v, 2) for v in img["bbox"]))
    geom.columns, geom.gutters = detect_columns(geom)
    for ln in geom.lines:
        ln.column = _line_column(ln, geom)
    return geom


def _safe_image_info(page: Any) -> list[dict[str, Any]]:
    try:
        return list(page.get_image_info())
    except Exception:  # noqa: BLE001
        return []


# ────────────────────────── 分栏判定 ──────────────────────────
def detect_columns(geom: PageGeom, min_gutter: float = 14.0) -> tuple[int, list[tuple[float, float]]]:
    """x 轴投影找空白带：正文行的水平覆盖在栏间会形成连续空隙。

    只在页面中段 15%~85% 宽度内找，避免把页边距误判成栏间距。
    """
    if len(geom.lines) < 4:
        return 1, []

    lo, hi = geom.width * 0.15, geom.width * 0.85
    bins = 100
    step = geom.width / bins
    cover = [0.0] * bins
    for ln in geom.lines:
        x0, x1 = max(0.0, ln.bbox[0]), min(geom.width, ln.bbox[2])
        i0, i1 = int(x0 / step), min(bins - 1, int(x1 / step))
        for i in range(i0, i1 + 1):
            cover[i] += x1 - x0 if i1 == i0 else step

    gutters: list[tuple[float, float]] = []
    start: int | None = None
    for i in range(bins):
        center = (i + 0.5) * step
        empty = cover[i] <= 1e-6 and lo <= center <= hi
        if empty and start is None:
            start = i
        elif not empty and start is not None:
            g0, g1 = start * step, i * step
            if g1 - g0 >= min_gutter:
                gutters.append((round(g0, 2), round(g1, 2)))
            start = None
    if start is not None:
        g0, g1 = start * step, bins * step
        if g1 - g0 >= min_gutter and lo <= g0 <= hi:
            gutters.append((round(g0, 2), round(g1, 2)))

    # 合并过近的空白带（同一栏内的大空格）
    merged: list[tuple[float, float]] = []
    for g in gutters:
        if merged and g[0] - merged[-1][1] < 6:
            merged[-1] = (merged[-1][0], g[1])
        else:
            merged.append(g)
    return len(merged) + 1, merged


def _line_column(ln: Line, geom: PageGeom) -> int:
    col = 0
    for g0, g1 in geom.gutters:
        if ln.bbox[2] <= g0 + 1:
            return col
        col += 1
    return col


# ────────────────────────── XY-Cut 阅读顺序 ──────────────────────────
def _connected_runs(items: list[Line], axis: int, tol: float) -> list[list[Line]]:
    """沿某轴的投影连通分量：axis=1 按 y（行带），axis=0 按 x（列带）。"""
    if not items:
        return []
    ordered = sorted(items, key=lambda it: (it.bbox[axis], it.bbox[axis + 2]))
    runs: list[list[Line]] = [[ordered[0]]]
    cur_end = ordered[0].bbox[axis + 2]
    for it in ordered[1:]:
        if it.bbox[axis] - cur_end > tol:
            runs.append([it])
            cur_end = it.bbox[axis + 2]
        else:
            runs[-1].append(it)
            cur_end = max(cur_end, it.bbox[axis + 2])
    return runs


def xy_cut(items: list[Line], y_tol: float = 6.0, x_tol: float = 14.0, depth: int = 0) -> list[Line]:
    """递归 XY-Cut：先切行带（y 方向），行带内再切列（x 方向）。"""
    if len(items) <= 1 or depth > 8:
        return sorted(items, key=lambda it: (it.bbox[1], it.bbox[0]))

    rows = _connected_runs(items, axis=1, tol=y_tol)
    if len(rows) > 1:
        out: list[Line] = []
        for r in rows:
            out.extend(xy_cut(r, y_tol, x_tol, depth + 1))
        return out

    cols = _connected_runs(items, axis=0, tol=x_tol)
    if len(cols) > 1:
        out = []
        for c in cols:
            out.extend(xy_cut(c, y_tol, x_tol, depth + 1))
        return out

    return sorted(items, key=lambda it: (it.bbox[1], it.bbox[0]))


def reading_order(geom: PageGeom) -> list[Line]:
    """返回本页按人类阅读顺序排列的文本行。"""
    body = [ln for ln in geom.lines]
    if not body:
        return []
    ordered = xy_cut(body)

    # 页眉/页脚不参与正文顺序，按出现位置吸附到首尾
    top_y = geom.height * 0.09
    bot_y = geom.height * 0.91
    head = [ln for ln in ordered if ln.bbox[3] <= top_y]
    foot = [ln for ln in ordered if ln.bbox[1] >= bot_y]
    mid = [ln for ln in ordered if ln not in head and ln not in foot]
    return head + mid + foot


def analyse_layout(geoms: list[PageGeom]) -> tuple[str, dict[str, Any]]:
    """汇总整篇版式：单栏 / 双栏 / 多栏 / 混合。"""
    cols = [g.columns for g in geoms if g.lines]
    detail = {
        "page_columns": [g.columns for g in geoms],
        # 阅读器还原双栏需要用页面尺寸把 bbox 归一化；这里只记录解析时已有的几何，
        # 不改变判定逻辑，也不影响旧版消费方（新增字段）。
        "page_sizes": [{"page": g.page, "width": round(g.width, 2),
                        "height": round(g.height, 2)} for g in geoms],
        "gutters": [{"page": g.page, "gutters": g.gutters} for g in geoms],
        "pages_with_text": sum(1 for g in geoms if g.lines),
    }
    if not cols:
        return "empty", detail
    # 跳过封面/首页（常为单栏大标题）后统计
    body = cols[1:] or cols
    two = sum(1 for c in body if c >= 2)
    ratio = two / len(body)
    if ratio >= 0.8:
        label = "double" if max(body) == 2 else "multi"
    elif ratio <= 0.2:
        label = "single"
    else:
        label = "mixed"
    return label, detail
