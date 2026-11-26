"""标题层级树（H1-H4）：字号 + 字重 + 章节编号三重判据，保留原始编号。

判据优先级：编号正则 > 字号倍数 > 字重/全大写。
只输出**真实存在于版面上的**标题：既没编号、字号又接近正文的一律不升格为标题。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.parsing.layout import Line, PageGeom

# 编号：1 / 1.1 / 1.1.1 / 1.1.1.1 / 第一章 / 1.1节 / I. / A.
NUM_RE = re.compile(
    r"^\s*(?:"
    r"(?P<dot>\d{1,2}(?:\.\d{1,2}){0,3})[.、]?\s+"          # 1 / 2.1 / 3.2.1
    r"|(?P<cn>第[一二三四五六七八九十百]+[章节篇部])\s*"      # 第三章
    r"|(?P<rom>[IVXL]{1,5})[.、]\s+"                         # IV.
    r"|(?P<alpha>[A-Z])[.、)]\s+"                            # A. （附录级别）
    r")"
)
REF_NAMES = re.compile(
    r"^(references|bibliography|参考文献|参考书目|acknowledge?ments?|致谢|"
    r"appendix|附录|abstract|摘要)\s*[:：]?\s*$", re.I)


@dataclass
class Heading:
    line: Line
    level: int
    number: str
    text: str


def body_font_size(geoms: list[PageGeom]) -> float:
    """正文字号 = 按字符数加权的众数字号（比单纯取中位数更抗标题干扰）。"""
    counter: Counter[float] = Counter()
    for g in geoms:
        for ln in g.lines:
            counter[round(ln.size, 1)] += len(ln.text)
    if not counter:
        return 10.0
    return counter.most_common(1)[0][0]


def _plausible_number(number: str) -> bool:
    """章节编号的每一段都该是小整数；"41.29"、"28.4" 多半是表格里的数字。"""
    try:
        return all(1 <= int(p) <= 20 for p in number.split("."))
    except ValueError:
        return False


def _level_by_number(number: str) -> int:
    if not number:
        return 4
    if re.match(r"^第[一二三四五六七八九十百]+[章节篇部]$", number):
        return 1
    if re.match(r"^[IVXL]{1,5}$", number):
        return 1
    if re.match(r"^[A-Z]$", number):
        return 2
    return min(4, number.count(".") + 1)


def classify_headings(geoms: list[PageGeom], body_size: float) -> list[Heading]:
    heads: list[Heading] = []
    consumed: set[int] = set()
    for g in geoms:
        order: list[Line] = []
        seen: set[int] = set()
        for ln in g.lines:
            if id(ln) not in seen:
                order.append(ln)
                seen.add(id(ln))

        for ln in g.lines:
            if id(ln) in consumed:
                continue
            text = ln.text.strip()
            if not text or len(text) > 120:
                continue
            ratio = ln.size / body_size if body_size else 1.0
            m = NUM_RE.match(text)
            is_ref_name = bool(REF_NAMES.match(text))

            # 编号单独占一行时（PDF 常见："1" 与 "Introduction" 同一基线），
            # 先按编号定级，正文标题行由 pipeline 合并回同一块。
            solo = re.fullmatch(r"(\d{1,2}(?:\.\d{1,2}){0,3})", text)
            if solo and (ln.bold or ratio >= 1.02) and _plausible_number(solo.group(1)):
                number = solo.group(1)
                label = text
                # 同一基线上紧跟的短行就是标题本身
                for other in g.lines:
                    if other is ln or id(other) in consumed or other.block_no != ln.block_no:
                        continue
                    if abs(other.bbox[1] - ln.bbox[1]) > 2:
                        continue
                    ot = other.text.strip()
                    if not ot or len(ot) > 90:
                        continue
                    label = f"{number} {ot}"
                    consumed.add(id(other))
                    break
                heads.append(Heading(line=ln, level=_level_by_number(number),
                                     number=number, text=label))
                continue

            # 行尾无句号 + 短行 + （编号 或 明显大号 或 加粗大字）
            ends_clean = not text.endswith(("。", ".", "；", ";", "，", ","))
            # 表格行也常以数字开头（"23.75 ByteNet"），字号/字重没优势就不当标题
            num = (m.group("dot") or m.group("cn") or m.group("rom") or m.group("alpha") or "") if m else ""
            numbered = (bool(m) and len(text) < 90 and (ln.bold or ratio >= 1.02)
                        and (not m.group("dot") or _plausible_number(num)))
            by_size = ratio >= 1.12 and len(text) < 90
            by_weight = (ln.bold or text.isupper()) and ratio >= 1.02 and len(text) < 60
            if not ends_clean:
                continue
            if not (numbered or is_ref_name or by_size or by_weight):
                continue

            if numbered:
                number = (m.group("dot") or m.group("cn") or m.group("rom") or m.group("alpha") or "").strip()
                level = _level_by_number(number)
            elif is_ref_name:
                number, level = "", 1
            elif ratio >= 1.55:
                number, level = "", 1
            elif ratio >= 1.32:
                number, level = "", 2
            elif ratio >= 1.16:
                number, level = "", 3
            else:
                number, level = "", 4

            body = text[len(number):].strip(" .、:：\t") if number else text
            heads.append(Heading(line=ln, level=level, number=number, text=body or text))
    return heads


def build_title_tree(heads: list[Heading]) -> list[dict[str, Any]]:
    """按层级压栈成树；parent 缺失时按同级归拢（不臆造父节点）。"""
    root: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []  # 每层一个节点

    for h in heads:
        node = {
            "text": h.text,
            "level": h.level,
            "number": h.number or None,
            "page": h.line.page,
            "bbox": list(h.line.bbox),
            "children": [],
        }
        while stack and stack[-1]["level"] >= h.level:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            root.append(node)
        stack.append(node)
    return root
