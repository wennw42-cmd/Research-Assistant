"""公式 / 表格 / 图注 / 参考文献抽取。

**反臆造原则**（本模块最重要的约束）：
- 公式：只把**真实抽取到的字形**按符号表转写成 LaTeX，并打 `latex_approx=True`；
  转写覆盖率 < 60% 时降级为 `type="unknown"` + 裁图保留原区域，绝不猜公式语义。
- 表格：优先用 PyMuPDF 的 `find_tables()`（矢量线框 + 文本对齐），拿不到才用对齐启发式并标注来源。
- 参考文献：只做**条目切分 + ID/DOI/arXiv 正则抽取**，不解析作者/年份/期刊字段（那是幻觉高发区）。
"""
from __future__ import annotations

import re
from typing import Any

from app.parsing.layout import Line, PageGeom

# ────────────────────────── 公式 ──────────────────────────
MATH_FONTS = ("math", "cmmi", "cmsy", "cmex", "mtmi", "mtsy", "symbol",
              "cambria math", "latin modern math", "asana", "stix", "texgyre")

SYMBOL2LATEX: dict[str, str] = {
    # 希腊字母
    "α": r"\alpha ", "β": r"\beta ", "γ": r"\gamma ", "δ": r"\delta ", "ε": r"\varepsilon ",
    "ζ": r"\zeta ", "η": r"\eta ", "θ": r"\theta ", "ι": r"\iota ", "κ": r"\kappa ",
    "λ": r"\lambda ", "μ": r"\mu ", "ν": r"\nu ", "ξ": r"\xi ", "π": r"\pi ",
    "ρ": r"\rho ", "σ": r"\sigma ", "τ": r"\tau ", "υ": r"\upsilon ", "φ": r"\varphi ",
    "χ": r"\chi ", "ψ": r"\psi ", "ω": r"\omega ",
    "Γ": r"\Gamma ", "Δ": r"\Delta ", "Θ": r"\Theta ", "Λ": r"\Lambda ", "Ξ": r"\Xi ",
    "Π": r"\Pi ", "Σ": r"\Sigma ", "Φ": r"\Phi ", "Ψ": r"\Psi ", "Ω": r"\Omega ",
    # 运算符与关系符
    "∑": r"\sum ", "∏": r"\prod ", "∫": r"\int ", "∬": r"\iint ", "∮": r"\oint ",
    "√": r"\sqrt", "∞": r"\infty ", "∂": r"\partial ", "∇": r"\nabla ",
    "±": r"\pm ", "∓": r"\mp ", "×": r"\times ", "÷": r"\div ", "⋅": r"\cdot ",
    "≤": r"\le ", "≥": r"\ge ", "≠": r"\ne ", "≈": r"\approx ", "≡": r"\equiv ",
    "∼": r"\sim ", "≪": r"\ll ", "≫": r"\gg ", "∈": r"\in ", "∉": r"\notin ",
    "⊂": r"\subset ", "⊆": r"\subseteq ", "∪": r"\cup ", "∩": r"\cap ", "∅": r"\emptyset ",
    "∀": r"\forall ", "∃": r"\exists ", "→": r"\to ", "←": r"\leftarrow ",
    "↔": r"\leftrightarrow ", "⇒": r"\Rightarrow ", "⇔": r"\Leftrightarrow ",
    "⊕": r"\oplus ", "⊗": r"\otimes ", "⊥": r"\perp ", "∥": r"\parallel ",
    "°": r"^{\circ}", "′": r"'", "″": r"''", "ℓ": r"\ell ", "‖": r"\| ",
}

SUPER = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
         "⁷": "7", "⁸": "8", "⁹": "9", "⁺": "+", "⁻": "-", "ⁿ": "n", "ⁱ": "i",
         "ᵀ": "T", "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e", "ᵍ": "g",
         "ʰ": "h", "ʲ": "j", "ᵏ": "k", "ˡ": "l", "ᵐ": "m", "ᵒ": "o", "ᵖ": "p",
         "ʳ": "r", "ˢ": "s", "ᵗ": "t", "ᵘ": "u", "ᵛ": "v", "ˣ": "x"}
SUB = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5",
       "₆": "6", "₇": "7", "₈": "8", "₉": "9"}

# 「普通字符」= ASCII 字母数字 / 常见标点 / 花括号 / CJK。
# 关键点：**花括号是我们自己产生的分组符号**，绝不能计入"未映射"，
# 否则替换越成功覆盖率反而越低（这正是第一版把 QK^{T} 判成 0.33 的原因）。
_PLAIN = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " ,.;:()[]<>/+-=|'\"*!?%&{}"
)


def is_special_char(c: str) -> bool:
    """是否属于"需要 LaTeX 转写"的符号：排除 ASCII、常见标点、CJK 与假名。"""
    if c in _PLAIN:
        return False
    o = ord(c)
    if 0x4E00 <= o <= 0x9FFF:      # CJK 统一表意文字（中文正文不算公式符号）
        return False
    if 0x3040 <= o <= 0x30FF:      # 日文假名
        return False
    if 0xAC00 <= o <= 0xD7AF:      # 韩文音节
        return False
    return True


def math_font(font: str) -> bool:
    f = (font or "").lower()
    return any(k in f for k in MATH_FONTS)


def special_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if is_special_char(c)) / len(chars)


def math_operator_char(c: str) -> bool:
    return c in "=<>≤≥±×÷∈∑∫√∞∂∇∀∃→←↔⇒≈≡≠∪∩⊂⊆∅⊕⊗"


# PDF 文本层里 `^` / `_` 就是上下标的字面写法（MathType/Word 导出尤甚），
# 先按 LaTeX 分组还原，这是**忠实还原**而非猜测；√ 后紧跟的项补上花括号。
_CARET_RE = re.compile(r"([A-Za-z0-9\)\]\}])\^(\{[^}]+\}|[A-Za-z0-9+\-]+)")
_UNDER_RE = re.compile(r"([A-Za-z0-9\)\]\}])_(\{[^}]+\}|[A-Za-z0-9+\-]+)")
_SQRT_RE = re.compile(r"\\sqrt\s*(\{[^}]+\}|[A-Za-z0-9]+(?:_\{[A-Za-z0-9]+\})?)")


def _unwrap(s: str) -> str:
    """只在**整体**被花括号包裹时脱壳，避免把 d_{k} 削成 d_{k 造成括号失配。"""
    return s[1:-1] if s.startswith("{") and s.endswith("}") and len(s) > 1 else s


def latex_approx(text: str) -> tuple[str, float]:
    """把真实抽取到的 Unicode 数学符号转写成 LaTeX，返回 (latex, 覆盖率)。

    覆盖率 = 已映射特殊字符 / 特殊字符总数；覆盖不足说明存在无法可靠转写的符号。
    """
    out: list[str] = []
    mapped = total = 0

    work, n1 = _CARET_RE.subn(lambda m: f"{m.group(1)}^{{{_unwrap(m.group(2))}}}", text)
    work, n2 = _UNDER_RE.subn(lambda m: f"{m.group(1)}_{{{_unwrap(m.group(2))}}}", work)
    mapped += n1 + n2
    total += n1 + n2

    i = 0
    while i < len(work):
        c = work[i]
        if c in SYMBOL2LATEX:
            out.append(SYMBOL2LATEX[c])
            mapped += 1
            total += 1
        elif c in SUPER:
            j = i
            buf = ""
            while j < len(work) and work[j] in SUPER:
                buf += SUPER[work[j]]
                j += 1
            out.append("^{" + buf + "}")
            mapped += 1
            total += 1
            i = j - 1
        elif c in SUB:
            j = i
            buf = ""
            while j < len(work) and work[j] in SUB:
                buf += SUB[work[j]]
                j += 1
            out.append("_{" + buf + "}")
            mapped += 1
            total += 1
            i = j - 1
        else:
            if is_special_char(c):
                total += 1
                out.append(c)  # 无法可靠转写：原样保留，不做猜测
            else:
                out.append(c)
        i += 1
    latex = re.sub(r"\s{2,}", " ", "".join(out)).strip()
    latex = _SQRT_RE.sub(lambda m: r"\sqrt{" + _unwrap(m.group(1)) + "}", latex)
    coverage = 1.0 if total == 0 else mapped / total
    return latex, coverage


def is_formula_candidate(text: str, font: str, size: float, body_size: float) -> bool:
    """公式候选：数学字体，或孤立短行且符号密度高。"""
    if math_font(font):
        return True
    if len(text) > 200:
        return False
    words = [w for w in text.split() if any(ch.isalnum() for ch in w)]
    if len(words) <= 6 and special_ratio(text) >= 0.12:
        return True
    return special_ratio(text) >= 0.25


# ────────────────────────── 图注 / 表注 ──────────────────────────
FIG_CAP_RE = re.compile(r"^\s*(fig(?:ure)?\.?|图|図)\s*([0-9A-Za-z]+(?:[.\-][0-9A-Za-z]+)?)\s*[:.：、]", re.I)
TAB_CAP_RE = re.compile(r"^\s*(tab(?:le)?\.?|表)\s*([0-9A-Za-z]+(?:[.\-][0-9A-Za-z]+)?)\s*[:.：、]", re.I)
LIST_RE = re.compile(r"^\s*(?:[•·▪◦‣\-–—*]|[（(]?\d{1,2}[)）.、]|[a-z][)）.])\s+\S")


def match_caption(text: str) -> tuple[str, str] | None:
    """返回 ("figure"|"table", 编号)。"""
    m = FIG_CAP_RE.match(text)
    if m:
        return "figure", m.group(2)
    m = TAB_CAP_RE.match(text)
    if m:
        return "table", m.group(2)
    return None


# ────────────────────────── 表格 ──────────────────────────
try:  # PyMuPDF >= 1.24 推荐 import pymupdf
    import pymupdf as _pm
except ImportError:  # pragma: no cover
    import fitz as _pm  # type: ignore


def _rect(vals: Any) -> Any:
    return _pm.Rect(*[float(v) for v in vals[:4]])


def _page_words(page: Any) -> list[tuple]:
    try:
        return list(page.get_text("words"))
    except Exception:  # noqa: BLE001
        return []


def _cells_from_geometry(table: Any, words: list[tuple]) -> list[list[str]]:
    """用单元格 bbox 与 **words 的空间包含** 填格。

    为什么不用 table.extract()：它会按 span 顺序拼接，实测把 `EN-DE` 拼成 `ENDE\\n-`、
    `28.4` 拼成 `284\\n.`（分隔符被挤到下一行）。按几何取词则原样保留抽取到的文本。
    """
    if not words:
        return []
    rows_out: list[list[str]] = []
    for row in getattr(table, "rows", []):
        cells_out: list[str] = []
        for cell in getattr(row, "cells", []):
            # 版本差异：新版 PyMuPDF 的 row.cells 是 bbox 元组，旧版是带 .bbox 的对象
            cb = getattr(cell, "bbox", None)
            if cb is None and isinstance(cell, (tuple, list)) and len(cell) >= 4:
                cb = tuple(cell[:4])
            if not cb:
                return []
            rect = _rect(cb)
            picked = []
            for w in words:
                wr = _rect(w[:4])
                area = wr.get_area()  # 必须先取面积：Rect.intersect() 是原地修改
                if area <= 0:
                    continue
                if wr.intersect(rect).get_area() / area > 0.55:
                    picked.append(w)
            picked.sort(key=lambda w: (round(w[1] / 4), w[0]))
            cells_out.append(" ".join(w[4] for w in picked).strip())
        if any(cells_out):
            rows_out.append(cells_out)
    return rows_out


def _cells_from_extract(table: Any) -> list[list[str]]:
    try:
        return [[(c or "").replace("\n", " ").strip() for c in row] for row in table.extract()]
    except Exception:  # noqa: BLE001
        return []


def find_tables(page: Any) -> list[dict[str, Any]]:
    """优先 PyMuPDF find_tables()（几何填格）→ extract() → 对齐启发式。"""
    found: list[dict[str, Any]] = []
    try:
        tabs = page.find_tables()
        words = _page_words(page)
        for t in getattr(tabs, "tables", []):
            try:
                rows = _cells_from_geometry(t, words) or _cells_from_extract(t)
            except Exception:  # noqa: BLE001 —— 单个表格失败不影响本页其它表格
                rows = []
            rows = [r for r in rows if any(c for c in r)]
            if len(rows) < 2:
                continue
            found.append({
                "bbox": tuple(round(v, 2) for v in t.bbox),
                "rows": rows,
                "source": "find_tables",
                "header_external": bool(getattr(getattr(t, "header", None), "external", False)),
            })
    except Exception:  # noqa: BLE001
        pass
    if not found:
        found = heuristic_tables(page)
    return found


def heuristic_tables(page: Any) -> list[dict[str, Any]]:
    """文本对齐启发式：连续多行出现 **列位置一致** 的水平空白带 → 视为表格。

    只处理最朴素的情况（≥3 行、≥2 列、列间隙稳定），拿不准就不报。
    """
    try:
        words = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,word_no)
    except Exception:  # noqa: BLE001
        return []
    rows: dict[int, list[Any]] = {}
    for w in words:
        rows.setdefault(round(w[1] / 4), []).append(w)
    if len(rows) < 3:
        return []

    cells: list[list[str]] = []
    for key in sorted(rows):
        ws = sorted(rows[key], key=lambda w: w[0])
        row: list[str] = []
        cur: list[str] = []
        prev_x1: float | None = None
        for w in ws:
            if prev_x1 is not None and w[0] - prev_x1 > 24:  # 列间隙
                row.append(" ".join(cur))
                cur = []
            cur.append(w[4])
            prev_x1 = w[1 + 2]
        if cur:
            row.append(" ".join(cur))
        if len(row) >= 2:
            cells.append(row)
    if len(cells) < 3 or len({len(r) for r in cells}) > 2:
        return []
    bbox = (
        min(w[0] for ws in rows.values() for w in ws),
        min(w[1] for ws in rows.values() for w in ws),
        max(w[2] for ws in rows.values() for w in ws),
        max(w[3] for ws in rows.values() for w in ws),
    )
    return [{"bbox": tuple(round(v, 2) for v in bbox), "rows": cells,
             "source": "heuristic", "header_external": False}]


def rows_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    norm = [list(r) + [""] * (width - len(r)) for r in rows]

    def esc(s: str) -> str:
        return (s or "").replace("|", r"\|").replace("\n", " ").strip()

    head = "| " + " | ".join(esc(c) for c in norm[0]) + " |"
    sep = "| " + " | ".join("---" for _ in range(width)) + " |"
    body = ["| " + " | ".join(esc(c) for c in r) + " |" for r in norm[1:]]
    return "\n".join([head, sep, *body])


# ────────────────────────── 参考文献 ──────────────────────────
REF_HEAD_RE = re.compile(r"^(references|bibliography|参考文献|参考书目)\s*[:：]?$", re.I)
REF_ITEM_SPLIT = re.compile(r"(?=(?:\[\d+\]|\(\d+\)|\b\d{1,3}\.\s+[A-Z（(]))")
ARXIV_RE = re.compile(r"(?:arxiv\s*[:.]?\s*)?(\d{4}\.\d{4,5})(?:v(\d+))?", re.I)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;)\]]+)")
URL_RE = re.compile(r"https?://[^\s,;)\]]+")
REF_ID_RE = re.compile(r"^\s*\[(\d+)\]|\((\d+)\)")


def split_reference_items(text: str) -> list[str]:
    parts = [p.strip() for p in REF_ITEM_SPLIT.split(text) if p.strip()]
    return parts if len(parts) > 1 else ([text.strip()] if text.strip() else [])


def reference_fields(text: str) -> dict[str, Any]:
    """只抽**可正则验证**的字段；作者/年份/期刊不解析（避免幻觉）。"""
    arxiv = ARXIV_RE.search(text)
    doi = DOI_RE.search(text)
    url = URL_RE.search(text)
    m = REF_ID_RE.match(text)
    return {
        "ref_id": (m.group(1) or m.group(2)) if m else None,
        "arxiv": (f"arXiv:{arxiv.group(1)}" + (f"v{arxiv.group(2)}" if arxiv.group(2) else "")) if arxiv else None,
        "doi": doi.group(1) if doi else None,
        "url": url.group(0) if url else None,
    }


def nearest_caption(blocks: list[dict[str, Any]], target: tuple[float, ...], kind: str,
                    max_gap: float = 34.0) -> dict[str, Any] | None:
    """在目标区域上/下方找匹配的图注/表注块。"""
    best: tuple[float, dict[str, Any]] | None = None
    for b in blocks:
        m = match_caption(b.get("text", ""))
        if not m or m[0] != kind:
            continue
        bb = b.get("bbox") or [0, 0, 0, 0]
        gap = target[1] - bb[3] if bb[3] <= target[1] else bb[1] - target[3]
        if 0 <= gap <= max_gap and (best is None or gap < best[0]):
            best = (gap, b)
    return best[1] if best else None


def lines_of(geom: PageGeom) -> list[Line]:
    return geom.lines
