"""学术段落的英译中：占位符保护、长句拆分、段落级双语对齐。

流水线
------
    原文 → 保护（公式/代码/变量/文献标记 → ⟦PH0⟧） → 分段 → 切句/拆长句
         → 翻译（LLM 或翻译记忆） → 逐片段校验占位符 → 还原 → 拼接
         → {"pairs": [{"en": ..., "zh": ...}]}

几条硬规则
----------
1. **先保护、后翻译、再还原**。占位符丢失、重复、臆造都视为翻译失败：
   自动带修复提示重试一次，仍失败则抛 ``TranslationError``，绝不静默丢公式。
2. **段落一一对应**。按空行切段，但围栏代码块内部的空行不切；``en`` 保留
   原文（块级公式/代码的换行原样保留），顺序与输入一致。
3. **长句拆分但保逻辑**。先按句切；单句超过 ``max_unit_chars`` 时只在
   ``;`` / ``, which`` / ``, and`` 这类从属或并列边界处递归拆分，
   连接词留在后一片段，译后用 ``；``/``，`` 拼接。拆点不会落在占位符内。
4. **术语一致**。可传入 ``glossary={en: zh}``，每次翻译都会随提示词下发。

用法::

    python -m app.translate --selftest
    python -m app.translate --demo
    python -m app.translate --text "..." --out zh.json
    python -m app.translate --file paper.txt --glossary terms.json
    Get-Content paper.txt -Raw | python -m app.translate

生产翻译使用项目现有的 OpenAI 兼容 Provider（``app.providers.llm``）；
未配置 Key 时 ``get_llm()`` 会退化为 MockLLM，此时本模块明确报错，
避免把 mock 文本当成译文。``--demo`` 走内置翻译记忆，离线可跑。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

# ══════════════════════════════ 占位符 ══════════════════════════════
PH_OPEN, PH_CLOSE = "⟦", "⟧"

# 宽容匹配：模型偶尔会把 ⟦⟧ 改成 []/【】/〔〕/()，只要编号在就还能还原。
_PH_FIND_RE = re.compile(
    r"⟦\s*PH\s*(\d+)\s*⟧"
    r"|\[\s*PH\s*(\d+)\s*\]"
    r"|【\s*PH\s*(\d+)\s*】"
    r"|〔\s*PH\s*(\d+)\s*〕"
    r"|[（(]\s*PH\s*(\d+)\s*[）)]"
)


def _ph(pid: int) -> str:
    return f"{PH_OPEN}PH{pid}{PH_CLOSE}"


def _pid_from_match(m: re.Match[str]) -> int:
    return next(int(g) for g in m.groups() if g is not None)


class PlaceholderError(ValueError):
    """占位符丢失/重复/臆造。"""


class TranslationError(RuntimeError):
    """某一段/片段的翻译失败（保留原文上下文，便于定位）。"""


# ══════════════════════════════ 保护规则 ══════════════════════════════
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    kind: str
    priority: int


# 顺序不重要（按 start/长度/优先级选不重叠区间），但优先级决定同起点时谁赢。
_FENCED = re.compile(r"```[^\n]*\n.*?```|~~~[^\n]*\n.*?~~~", re.S)
_DISPLAY_MATH = re.compile(
    r"\$\$.*?\$\$"
    r"|\\\[.*?\\\]"
    r"|\\begin\{(equation|align|gather|multline|eqnarray)\*?\}.*?\\end\{\1\*?\}",
    re.S,
)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_INLINE_MATH = re.compile(
    r"(?<!\$)\$(?!\$)(?:\\.|[^$\n])+?\$(?!\$)"
    r"|\\\((?:\\.|[^\n])*?\\\)"
)
_LATEX_CITE = re.compile(
    r"\\(?:cite|citep|citet|parencite|textcite|autocite|ref|eqref|autoref)\*?"
    r"(?:\[[^\]]*\])*\{[^{}]*\}"
)
_NUM_CITE = re.compile(r"[\[［]\s*\d{1,4}(?:\s*[-–—,，、;；]\s*\d{1,4})*\s*[\]］]")
# 作者-年份引用，如 (Smith et al., 2020)、（王小明等，2021）。
# 只保护“作者 + 逗号 + 年份”本身，不保护外层括号：中译英时括号可随目标语言调整，
# 作者名与年份仍逐字保留；英译中时外层括号由译文自然写成全角。
# 只匹配这类括号，避免把普通括注当成引用保护。
_AUTHOR_YEAR = re.compile(
    r"(?<=[（(])\s*"
    r"(?:[A-Z][A-Za-z'’\-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'’\-]+"
    r"|&\s+[A-Z][A-Za-z'’\-]+))?|[\u4e00-\u9fff]{2,8}(?:等)?)"
    r"\s*[,，]\s*(?:19|20)\d{2}[a-z]?"
    r"(?:\s*[;；]\s*"
    r"(?:[A-Z][A-Za-z'’\-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'’\-]+"
    r"|&\s+[A-Z][A-Za-z'’\-]+))?|[\u4e00-\u9fff]{2,8}(?:等)?)"
    r"\s*[,，]\s*(?:19|20)\d{2}[a-z]?)*"
    r"\s*(?=[）)])"
)
_URL = re.compile(r"https?://[^\s<>\"'）)]+|www\.[^\s<>\"'）)]+")
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_BIG_O = re.compile(r"\b[OΘΩ]\((?:[^()\n]|\([^()\n]*\)){1,60}\)")
_ASSIGN = re.compile(
    r"(?<![\w$])[A-Za-zα-ωΑ-Ω][A-Za-z0-9α-ωΑ-Ω_]*"
    r"\s*(?:=|≈|≤|≥|<|>)\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])"
)
_VAR = re.compile(r"(?<![\w$])(?:[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)(?![\w$])")
# 裸小写单字母变量，如“设 x 为输入”“n 个样本”。只保护紧邻中文/中文标点的单字母，
# 避免把英文冠词、缩写、单词首字母，以及 X射线/A股/C语言 这类专名拆开。
_BARE_VAR = re.compile(
    r"(?<![\w$])([a-z])(?=\s*[一-鿿，。；：、（）“”‘’%]|$)"
)

_PATTERNS: list[tuple[str, int, re.Pattern[str]]] = [
    ("ph_literal", 110, _PH_FIND_RE),   # 原文里本来就有的 ⟦PH0⟧ / [PH0]，先保护掉防碰撞
    ("code_block", 100, _FENCED),
    ("math_block", 95, _DISPLAY_MATH),
    ("inline_code", 90, _INLINE_CODE),
    ("math_inline", 85, _INLINE_MATH),
    ("latex_cite", 80, _LATEX_CITE),
    ("author_year", 76, _AUTHOR_YEAR),
    ("citation", 75, _NUM_CITE),
    ("url", 70, _URL),
    ("doi", 68, _DOI),
    ("email", 66, _EMAIL),
    ("big_o", 62, _BIG_O),
    ("assign", 58, _ASSIGN),
    ("variable", 55, _VAR),
    ("bare_variable", 54, _BARE_VAR),
]


def _looks_like_math(token: str) -> bool:
    """排除 $5 / $1,000 这类货币写法；$x$、$d_{model}$、$O(n^2)$ 保留。"""
    if token.startswith("\\("):
        return True
    inner = token[1:-1]
    if not inner.strip():
        return False
    if re.fullmatch(r"[\d.,\s]+", inner):
        return False
    if re.search(r"[\\_^={}]", inner):
        return True
    if " " in inner and not re.search(r"[=<>+\-*/^_|]", inner):
        return False
    return True


def _collect_spans(text: str) -> list[Span]:
    found: list[Span] = []
    for kind, prio, pat in _PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(0)
            if kind == "math_inline" and not _looks_like_math(raw):
                continue
            found.append(Span(m.start(), m.end(), raw, kind, prio))
    # 起点优先；同起点取更长者；再同长取优先级高者。之后贪心选不重叠区间。
    found.sort(key=lambda s: (s.start, -(s.end - s.start), -s.priority))
    chosen: list[Span] = []
    last = -1
    for s in found:
        if s.start >= last:
            chosen.append(s)
            last = s.end
    return chosen


def protect(text: str) -> tuple[str, dict[int, str]]:
    """把受保护片段替换成占位符，返回 (保护后文本, 编号→原文)。"""
    parts: list[str] = []
    mapping: dict[int, str] = {}
    pos = 0
    for i, sp in enumerate(_collect_spans(text)):
        parts.append(text[pos:sp.start])
        parts.append(_ph(i))
        mapping[i] = sp.text
        pos = sp.end
    parts.append(text[pos:])
    return "".join(parts), mapping


def check_placeholders(text: str, expected: Mapping[int, str]) -> "PlaceholderIssue":
    counts = Counter(_pid_from_match(m) for m in _PH_FIND_RE.finditer(text))
    exp = set(expected)
    found = set(counts)
    return PlaceholderIssue(
        missing=tuple(sorted(exp - found)),
        duplicated=tuple(sorted(i for i, c in counts.items() if c > 1)),
        unknown=tuple(sorted(found - exp)),
    )


@dataclass(frozen=True)
class PlaceholderIssue:
    missing: tuple[int, ...] = ()
    duplicated: tuple[int, ...] = ()
    unknown: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        return not (self.missing or self.duplicated or self.unknown)

    def describe(self) -> str:
        bits: list[str] = []
        if self.missing:
            bits.append("遗漏 " + ", ".join(_ph(i) for i in self.missing))
        if self.duplicated:
            bits.append("重复 " + ", ".join(_ph(i) for i in self.duplicated))
        if self.unknown:
            bits.append("臆造 " + ", ".join(_ph(i) for i in self.unknown))
        return "；".join(bits) or "无"


def restore(text: str, mapping: Mapping[int, str]) -> str:
    """按编号还原；未知编号直接报错，避免把模型编造的 PH 留在正文里。"""
    def sub(m: re.Match[str]) -> str:
        pid = _pid_from_match(m)
        if pid not in mapping:
            raise PlaceholderError(f"译文出现未知占位符 {_ph(pid)}")
        return mapping[pid]

    return _PH_FIND_RE.sub(sub, text)


# ══════════════════════════════ 分段 / 切句 / 拆长句 ══════════════════════════════
def split_paragraphs(text: str) -> list[str]:
    """按空行切段；围栏代码块内部的空行不切。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    paras: list[str] = []
    buf: list[str] = []
    fence = False
    for line in text.split("\n"):
        if re.match(r"^\s*(```|~~~)", line):
            fence = not fence
        if not fence and not line.strip():
            if buf:
                paras.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        paras.append("\n".join(buf).strip())
    return [p for p in paras if p]


_ABBREV = {
    "e.g.", "i.e.", "et al.", "fig.", "figs.", "eq.", "eqs.", "sec.", "secs.",
    "vs.", "no.", "nos.", "dr.", "prof.", "al.", "approx.", "cf.", "resp.",
    "ref.", "refs.", "u.s.",
}


def split_sentences(text: str) -> list[str]:
    """句末切分，避开小数、缩写（Fig./et al./e.g.）与占位符内部。"""
    out: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in ".!?":
            i += 1
            continue
        if ch == "." and 0 < i < n - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            i += 1
            continue
        if ch == "." and i + 1 < n and text[i + 1] == ".":     # 省略号
            i += 1
            continue
        # 回看当前“词”，判断是否缩写
        j = i
        while j > 0 and not text[j - 1].isspace() and text[j - 1] not in "\"'“”‘’([":
            j -= 1
        word = text[j:i + 1].lower()
        if ch == "." and word in _ABBREV:
            i += 1
            continue
        k = i + 1
        while k < n and text[k].isspace():
            k += 1
        if k >= n:
            out.append(text[start:i + 1].strip())
            start = i + 1
            break
        nxt = text[k]
        if nxt.islower() or nxt in ",;:":
            i += 1
            continue
        out.append(text[start:i + 1].strip())
        start = k
        i = k
    if start < n:
        tail = text[start:].strip()
        if tail:
            out.append(tail)
    return [s for s in out if s]


# 只在“真正能承载逻辑”的边界拆：分号、逗号+从属连词/关系代词、破折号。
_CLAUSE_RE = re.compile(
    r"(?P<sep>;)\s+"
    r"|,\s*(?P<conj>which|where|while|whereas|although|though|because|since|if|when|that|and|but|so|yet|or|who|whom|whose|thereby|thus|hence|therefore)\b"
    r"|(?P<dash>\s+[—–]\s+)"
)
_CONTRAST = {"although", "though", "whereas", "while", "but", "yet"}
_CAUSAL = {"because", "since", "therefore", "thus", "hence", "thereby"}


def _boundary(m: re.Match[str]) -> tuple[int, str]:
    if m.group("conj"):
        # 连接词留给后一片段：英文 and/which 需要在中文里显式译出
        return m.start("conj"), ("；" if m.group("conj").lower() in _CONTRAST | _CAUSAL else "，")
    if m.group("sep"):
        return m.end("sep"), "；"
    return m.end("dash"), "，"


def split_long(text: str, max_chars: int, min_chars: int = 40) -> list[tuple[str, str]]:
    """递归把长句拆成 (片段, 与下一片段的连接符)，尽量在句中断点附近均分。"""
    text = text.strip()
    if len(text) <= max_chars:
        return [(text, "")] if text else []

    candidates: list[tuple[int, str]] = []
    for m in _CLAUSE_RE.finditer(text):
        pos, joiner = _boundary(m)
        if min_chars <= pos <= len(text) - min_chars:
            candidates.append((pos, joiner))

    if candidates:
        mid = len(text) / 2
        pos, joiner = min(candidates, key=lambda c: abs(c[0] - mid))
    else:
        cut = text.rfind(" ", min_chars, max_chars + 1)
        if cut < 0:
            cut = max(min_chars, min(max_chars, len(text) - 1))
        pos, joiner = cut, "，"

    left = split_long(text[:pos].strip(), max_chars, min_chars)
    right = split_long(text[pos:].strip(), max_chars, min_chars)
    if not left or not right:
        return [(text, "")]
    left[-1] = (left[-1][0], joiner)
    return left + right


# ══════════════════════════════ 翻译单元 ══════════════════════════════
@dataclass(frozen=True)
class Unit:
    raw: str                         # 还原后的片段原文
    protected: str                   # 送翻译的占位符文本
    placeholders: dict[int, str]     # 本片段实际出现的占位符
    fragment: bool                   # 是否是长句的前半段（不应补句号）
    joiner: str                      # 与下一片段的连接符
    context: str                     # 整段原文，供片段消歧


class Translator(Protocol):
    def translate(
        self,
        unit: Unit,
        glossary: Mapping[str, str] | None = None,
        repair: PlaceholderIssue | None = None,
    ) -> str:
        ...


def _prepare_paragraph(raw: str) -> tuple[str, str, dict[int, str]]:
    """返回 (清理后的原文, 保护后文本, 占位符表)。

    普通段落把 PDF 的硬换行折成空格；含块级公式/代码的段落保留换行，
    否则 ``` / $$ 会被折到行内，Markdown 与公式都不再成立。
    """
    protected, mapping = protect(raw)
    if not (_FENCED.search(raw) or _DISPLAY_MATH.search(raw)):
        protected = re.sub(r"[ \t]*\n[ \t]*", " ", protected)
        protected = re.sub(r"[ \t]{2,}", " ", protected)
    protected = protected.strip()
    clean = restore(protected, mapping)
    return clean, protected, mapping


def _make_units(protected: str, mapping: dict[int, str], max_chars: int, context: str) -> list[Unit]:
    units: list[Unit] = []
    for sent in split_sentences(protected):
        pieces = split_long(sent, max_chars)
        for idx, (piece, joiner) in enumerate(pieces):
            pids: list[int] = []
            for m in _PH_FIND_RE.finditer(piece):
                pid = _pid_from_match(m)
                if pid not in pids:
                    pids.append(pid)
            ph = {pid: mapping[pid] for pid in pids}
            units.append(Unit(
                raw=restore(piece, ph),
                protected=piece,
                placeholders=ph,
                # 只有“后面还有片段”的前半句才禁止补句号；最后一片是句子结尾
                fragment=len(pieces) > 1 and idx < len(pieces) - 1,
                joiner=joiner,
                context=context,
            ))
    return units


def _has_words(protected: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", _PH_FIND_RE.sub("", protected)))


def _ends_terminal(text: str) -> bool:
    return bool(re.search(r"[。！？!?…][”’\"'）)\]]?$", text.strip()))


def _translate_unit(unit: Unit, translator: Translator, glossary: Mapping[str, str]) -> str:
    repair: PlaceholderIssue | None = None
    last_error = "空译文"
    for _ in range(2):
        out = (translator.translate(unit, glossary, repair=repair) or "").strip()
        if not out:
            last_error = "空译文"
            continue
        issue = check_placeholders(out, unit.placeholders)
        if issue.ok:
            return restore(out, unit.placeholders)
        last_error = issue.describe()
        repair = issue
    raise TranslationError(f"片段翻译失败（{last_error}）：{unit.raw[:100]}")


def _join_units(units: Sequence[Unit], translated: Sequence[str]) -> str:
    parts: list[str] = []
    for i, (u, zh) in enumerate(zip(units, translated)):
        zh = zh.strip()
        if u.joiner:
            # 片段自带的句末标点交给连接符处理，避免“，，”或“；。”
            zh = re.sub(r"[，,；;。.！!？?]\s*$", "", zh)
            if not zh:
                zh = "……"
            parts.append(zh)
            parts.append(u.joiner)
        else:
            if (i < len(units) - 1 and not u.fragment
                    and re.search(r"[.!?][\"'”’)]?$", u.raw) and not _ends_terminal(zh)):
                zh += "。"       # 整句缺少句末标点
            parts.append(zh)
    text = "".join(parts).strip()
    if units and re.search(r"[.!?][\"'”’)]?$", units[-1].raw) and not _ends_terminal(text):
        text += "。"
    return text


def translate_paragraphs(
    text: str,
    translator: Translator,
    *,
    glossary: Mapping[str, str] | None = None,
    max_unit_chars: int = 240,
) -> dict[str, Any]:
    """英译中主入口，返回 ``{"pairs": [{"en": ..., "zh": ...}]}``。"""
    glossary = dict(glossary or {})
    max_unit_chars = max(80, int(max_unit_chars))
    pairs: list[dict[str, str]] = []

    for raw in split_paragraphs(text):
        clean, protected, mapping = _prepare_paragraph(raw)
        if not protected or not _has_words(protected):
            pairs.append({"en": clean, "zh": clean})   # 纯公式/代码段：原样返回
            continue
        units = _make_units(protected, mapping, max_unit_chars, clean)
        translated = [_translate_unit(u, translator, glossary) for u in units]
        pairs.append({"en": clean, "zh": _join_units(units, translated)})
    return {"pairs": pairs}


# ══════════════════════════════ 输出 ══════════════════════════════
def _md_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = ["## Chinese Translation / 中文译文", "", str(result.get("translation") or "").strip(), ""]
    lines.extend(["## Glossary / 术语表", ""])
    glossary = result.get("glossary") or []
    if glossary:
        lines.extend(["| English | 中文术语 | 说明 |", "|---|---|---|"])
        for term in glossary:
            lines.append(
                f"| {_md_cell(term.get('en'))} | {_md_cell(term.get('zh'))} | "
                f"{_md_cell(term.get('note'))} |"
            )
    else:
        lines.append("（未设置术语。）")
    stats = result.get("stats") or {}
    if stats:
        lines.extend([
            "",
            "---",
            f"段落 {stats.get('paragraphs', 0)} · 术语 {stats.get('terms', 0)} · "
            f"原文字符 {stats.get('source_chars', 0)} · 译文字符 {stats.get('translation_chars', 0)}",
        ])
    return "\n".join(lines).rstrip() + "\n"


# ══════════════════════════════ LLM 后端 ══════════════════════════════
SYSTEM_PROMPT = """你是学术翻译专家，把英文论文片段译成规范、准确的中文。

硬性规则：
1. 术语采用中文学术界通行译法；同一术语在同一段内保持一致。模型名/方法名（如 Transformer、BERT）保留英文原名。
2. 文本中的 ⟦PH数字⟧ 是公式、代码、变量名或文献标记的占位符。必须原样保留，包括编号与括号；不得翻译、改写、增删、调换顺序，也不得在译文中自行补写其内容。
3. 只翻译英文，不解释、不总结、不扩写、不添加原文没有的信息。
4. 输出纯译文，不要引号、代码块、"译文："等前后缀。
5. 标注【长句片段】时，它只是同一句的一部分：不要补句号，保留 and/but/which/because 等逻辑连接词，使其能与相邻片段自然拼接。
6. 占位符外层的括号/引号属于原文结构：按中文标点习惯处理，但不要额外再加一层括号。"""


def _clean_model_output(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(?:译文|翻译)\s*[:：]\s*", "", text)
    return text.strip()


def _compose_user(unit: Unit, glossary: Mapping[str, str], repair: PlaceholderIssue | None) -> str:
    parts: list[str] = []
    if glossary:
        parts.append("【术语表（必须遵守）】")
        parts.extend(f"- {en} → {zh}" for en, zh in glossary.items())
    if unit.fragment and unit.context and unit.context != unit.raw:
        parts.append("【段落上下文（仅供理解，不要翻译）】")
        parts.append(unit.context)
    parts.append("【长句片段：不要补句号，保留连接词】" if unit.fragment else "【完整句子】")
    parts.append("【待译片段】")
    parts.append(unit.protected)
    if repair and not repair.ok:
        parts.append("【修正要求】")
        parts.append(
            f"上一版译文占位符有误：{repair.describe()}。"
            "请重新输出完整译文，所有占位符必须原样出现且只出现一次。"
        )
    return "\n".join(parts)


class LLMTranslator:
    """生产后端：复用项目现有的 OpenAI 兼容 Provider。"""

    def __init__(self, llm: Any | None = None) -> None:
        from app.providers.llm import get_llm  # 延迟导入：离线自检不依赖 openai

        self.llm = llm or get_llm()
        if getattr(self.llm, "model_name", "") == "mock-llm":
            raise RuntimeError(
                "检测到 MockLLM，无法真实翻译。请配置 LLM_PROVIDER=openai_compatible 与 LLM_API_KEY；"
                "离线查看效果请用 --demo。"
            )

    def translate(
        self,
        unit: Unit,
        glossary: Mapping[str, str] | None = None,
        repair: PlaceholderIssue | None = None,
    ) -> str:
        resp = self.llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": _compose_user(unit, glossary or {}, repair)}],
            temperature=0.1,
            max_tokens=1536,
        )
        text = _clean_model_output(getattr(resp, "content", "") or "")
        if not text:
            raise TranslationError(f"模型返回空译文：{unit.raw[:80]}")
        return text


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class CannedTranslator:
    """离线演示/回归用的翻译记忆：精确匹配片段，再按源顺序回填占位符。

    只用于内置 DEMO_TEXT。生产文本请用 LLMTranslator。
    """

    def __init__(self, tm: Mapping[str, str]) -> None:
        self.tm = {_norm(k): v for k, v in tm.items()}

    def translate(
        self,
        unit: Unit,
        glossary: Mapping[str, str] | None = None,
        repair: PlaceholderIssue | None = None,
    ) -> str:
        zh = self.tm.get(_norm(unit.raw))
        if zh is None:
            raise TranslationError(f"演示翻译记忆缺少片段：{unit.raw[:100]!r}")
        for pid, orig in unit.placeholders.items():
            if orig not in zh:
                raise TranslationError(f"演示译文未保留受保护内容 {orig!r}：{unit.raw[:80]!r}")
            # 同一原文多次出现时按源顺序逐个替换
            zh = zh.replace(orig, _ph(pid), 1)
        return zh


# ══════════════════════════════ 演示材料 ══════════════════════════════
DEMO_TEXT = r"""The Transformer replaces recurrence with self-attention [1]. For a sequence of length $n$, the computational complexity of self-attention is $O(n^2 d)$, where $d$ is the model dimension. In our implementation we set `d_model = 512` and use $h = 8$ attention heads.

Although the model achieves strong results on the WMT 2014 English-to-German translation task, its quadratic memory footprint becomes prohibitive for long documents, and prior work has therefore explored sparse attention patterns [2, 3], which reduce the cost at the price of a lower effective capacity.

The scaled dot-product attention is defined as
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
$$
where $d_k$ is the dimension of the keys [4–6].

We use the following configuration:
```python
config = {"d_model": 512, "num_heads": 8}
```"""

# 键是“拆分后的片段原文”（见 --debug）；值是中文，其中受保护内容要原样保留，
# CannedTranslator 会把它们替换回占位符。长句两段之间用 “，” 拼接。
DEMO_TM: dict[str, str] = {
    "The Transformer replaces recurrence with self-attention [1].":
        "Transformer 以自注意力取代循环结构 [1]。",
    "For a sequence of length $n$, the computational complexity of self-attention is $O(n^2 d)$, where $d$ is the model dimension.":
        "对于长度为 $n$ 的序列，自注意力的计算复杂度为 $O(n^2 d)$，其中 $d$ 为模型维度。",
    "In our implementation we set `d_model = 512` and use $h = 8$ attention heads.":
        "在我们的实现中，设置 `d_model = 512`，并使用 $h = 8$ 个注意力头。",

    # 长句：在 “, and” 处拆开
    "Although the model achieves strong results on the WMT 2014 English-to-German translation task, its quadratic memory footprint becomes prohibitive for long documents,":
        "尽管该模型在 WMT 2014 英德翻译任务上取得了优异结果，但其平方级的内存占用对于长文档而言已变得难以承受，",
    "and prior work has therefore explored sparse attention patterns [2, 3], which reduce the cost at the price of a lower effective capacity.":
        "而已有工作因此探索了稀疏注意力模式 [2, 3]，这些模式以较低的有效容量为代价降低了开销。",

    "The scaled dot-product attention is defined as\n$$\n\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V,\n$$\nwhere $d_k$ is the dimension of the keys [4–6].":
        "缩放点积注意力定义为\n$$\n\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V,\n$$\n其中 $d_k$ 为键的维度 [4–6]。",

    'We use the following configuration:\n```python\nconfig = {"d_model": 512, "num_heads": 8}\n```':
        '我们使用如下配置：\n```python\nconfig = {"d_model": 512, "num_heads": 8}\n```',
}


# ══════════════════════════════ 自检 ══════════════════════════════
class _StubTranslator:
    """只验证机制：返回“译”+ 本片段所有占位符，顺序与源一致。"""

    def __init__(self, drop: bool = False, fail_once: bool = False) -> None:
        self.drop = drop
        self.fail_once = fail_once
        self.calls: list[Unit] = []
        self.glossaries: list[dict[str, str]] = []
        self.repairs: list[PlaceholderIssue | None] = []

    def translate(self, unit, glossary=None, repair=None):
        self.calls.append(unit)
        self.glossaries.append(dict(glossary or {}))
        self.repairs.append(repair)
        if self.drop:
            return "译文丢掉了占位符"
        if self.fail_once and len(self.calls) == 1 and unit.placeholders:
            return "第一版丢掉了占位符"
        return "译" + "".join(_ph(pid) for pid in unit.placeholders)


def _selftest() -> int:
    passed = 0
    failed = 0

    def check(name: str, cond: bool, detail: Any = "") -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"[OK] {name}")
        else:
            failed += 1
            print(f"[XX] {name}  → {detail}")

    # 1. 分段：空行切，单换行不切；缩写不误切
    ps = split_paragraphs("A one.\n\nB two.\nC three.")
    check("空行切段、单换行不切", ps == ["A one.", "B two.\nC three."], ps)
    ss = split_sentences("We use Fig. 3 and et al. methods. Then we stop.")
    check("Fig./et al. 不误切句", len(ss) == 2, ss)

    # 2. 代码块内部的空行不切
    src = "Use:\n```python\na = 1\n\nb = 2\n```\nDone."
    ps = split_paragraphs(src)
    check("代码块内空行不切段", len(ps) == 1, len(ps))

    # 3. 公式/代码/变量/引用全部保护并可无损还原
    src = "We set $d_{model}=512$ and use x_i with `torch` [1]."
    pp, mp = protect(src)
    check("公式未被翻译", "$d_{model}" not in pp and "x_i" not in pp, pp)
    check("引用未被翻译", "[1]" not in pp, pp)
    check("保护后无损还原", restore(pp, mp) == src, restore(pp, mp))

    # 4. 多行块级公式原样保留
    src = "Formally,\n$$\na = b\n$$\nend."
    pp, mp = protect(src)
    check("块级公式含换行仍无损", restore(pp, mp) == src, repr(restore(pp, mp)))

    # 5. 围栏代码整体保护（内部 $x$ 不再拆成多个占位符）
    src = "Use:\n```python\nx = $y$\n```\nDone."
    pp, mp = protect(src)
    check("围栏代码整体保护", len(mp) == 1 and "```python" in mp[0], mp)

    # 6. 货币不被当成行内公式
    pp, mp = protect("It costs $5 and $10.")
    check("$5 不被当作公式", not mp, mp)

    # 7. 行内公式仍然保护
    pp, mp = protect("Let $x$ be a variable.")
    check("行内公式保护", any(v == "$x$" for v in mp.values()), mp)

    # 8. 多种文献标记
    pp, mp = protect("Prior work [1, 2–3], ［4］ and \\cite{vaswani2017}.")
    vals = list(mp.values())
    check("数字引用保护", "[1, 2–3]" in vals and "［4］" in vals, vals)
    check("LaTeX 引用保护", "\\cite{vaswani2017}" in vals, vals)
    pp, mp = protect("Prior work (Smith et al., 2020; 王小明等，2021) and (Figure 1, 2020).")
    vals = list(mp.values())
    check("作者-年份引用保护",
          any("Smith et al., 2020" in v for v in vals)
          and any("王小明等，2021" in v for v in vals), vals)
    check("普通括注不误保护", not any("Figure 1" in v for v in vals), vals)

    # 9. 变量与赋值/复杂度保护
    pp, mp = protect("We set d_model = 512 and p < 0.05, where O(n^2) matters.")
    vals = list(mp.values())
    check("赋值表达式保护", "d_model = 512" in vals and "p < 0.05" in vals, vals)
    check("大 O 复杂度保护", any(v.startswith("O(n^2)") for v in vals), vals)

    pp, mp = protect("设 x 为输入，n 个样本，Transformer 模型。")
    vals = list(mp.values())
    check("裸单字母变量保护", "x" in vals and "n" in vals, vals)
    check("英文单词不误保护", not any("Transformer" in v for v in vals), vals)
    pp, mp = protect("a, b and c.")
    check("英文冠词不误保护", mp == {}, mp)

    # 10. URL / DOI 保护
    pp, mp = protect("See https://example.org/x?y=1 or doi:10.1000/xyz123.")
    vals = list(mp.values())
    check("URL/DOI 保护", any(v.startswith("https://") for v in vals) and
          any(v.startswith("10.1000/") for v in vals), vals)

    # 11. 宽容还原
    out = restore("⟦ PH0 ⟧ [PH1] 【PH2】", {0: "$x$", 1: "[1]", 2: "`c`"})
    check("宽容还原 ⟦⟧/[]/【】", out == "$x$ [1] `c`", out)

    # 12. 占位符丢失/重复/臆造
    issue = check_placeholders("没有占位符", {0: "$x$"})
    check("丢失可检测", issue.missing == (0,), issue)
    issue = check_placeholders("⟦PH0⟧ ⟦PH0⟧", {0: "$x$"})
    check("重复可检测", issue.duplicated == (0,), issue)
    issue = check_placeholders("⟦PH9⟧", {0: "$x$"})
    check("臆造可检测", issue.unknown == (9,), issue)

    # 13. 端到端：段落一一对应，占位符不出现在译文
    stub = _StubTranslator()
    doc = "We use $x$ [1].\n\nSecond paragraph."
    out = translate_paragraphs(doc, stub, max_unit_chars=200)
    check("输出只有 pairs", set(out) == {"pairs"}, set(out))
    check("段落一一对应", [p["en"] for p in out["pairs"]] == ["We use $x$ [1].", "Second paragraph."],
          out["pairs"])
    check("译文中无占位符残留", all("PH" not in p["zh"] for p in out["pairs"]), out["pairs"])
    check("公式已还原", "$x$" in out["pairs"][0]["zh"], out["pairs"][0])

    # 14. 长句拆分 + 连接符
    stub = _StubTranslator()
    long = ("Although the model is fast, its memory footprint grows with the sequence length, "
            "and this becomes prohibitive for long documents, which limits deployment.")
    out = translate_paragraphs(long, stub, max_unit_chars=80)
    zh = out["pairs"][0]["zh"]
    check("长句被拆成多片", len(stub.calls) >= 2, len(stub.calls))
    check("片段以逗号/分号拼接", "，" in zh or "；" in zh, zh)
    check("拆句后仍有句末标点", _ends_terminal(zh), zh)

    # 15. 首次丢占位符 → 带修复提示重试成功
    stub = _StubTranslator(fail_once=True)
    out = translate_paragraphs("We use $x$ [1].", stub, max_unit_chars=200)
    check("丢占位符触发重试", len(stub.calls) == 2, len(stub.calls))
    check("重试携带问题描述", isinstance(stub.repairs[-1], PlaceholderIssue)
          and not stub.repairs[-1].ok, stub.repairs[-1])
    check("重试后公式还原", "$x$" in out["pairs"][0]["zh"], out["pairs"][0]["zh"])

    # 16. 始终丢失 → 明确失败，不静默输出
    try:
        translate_paragraphs("We use $x$ [1].", _StubTranslator(drop=True))
        raised = False
    except TranslationError:
        raised = True
    check("占位符始终丢失则报错", raised)

    # 17. 术语表随每次翻译下发
    stub = _StubTranslator()
    translate_paragraphs("Attention matters.", stub, glossary={"attention": "注意力"})
    check("术语表下发", stub.glossaries[0] == {"attention": "注意力"}, stub.glossaries[0])

    # 18. 空输入
    check("空输入返回空 pairs", translate_paragraphs("", _StubTranslator()) == {"pairs": []})

    # 19. 原文自带类占位符也不会碰撞
    src = "A literal ⟦PH0⟧ and [PH1] token."
    pp, mp = protect(src)
    check("原文占位符先被保护", "⟦PH0⟧" in mp.values() and "[PH1]" in mp.values(), mp)
    check("原文占位符可还原", restore(pp, mp) == src, restore(pp, mp))

    # 20. 演示材料完整可译
    demo = translate_paragraphs(DEMO_TEXT, CannedTranslator(DEMO_TM), max_unit_chars=240)
    check("演示 4 段", len(demo["pairs"]) == 4, len(demo["pairs"]))
    check("演示译文非空且无占位符",
          all(p["zh"] and "PH" not in p["zh"] for p in demo["pairs"]), demo["pairs"])

    print(f"\nselftest: {passed}/{passed + failed} 通过")
    return 0 if failed == 0 else 1


# ══════════════════════════════ CLI ══════════════════════════════
def _load_glossary(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(x["en"]): str(x["zh"]) for x in data}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    raise ValueError("术语表必须是 {en: zh} 对象或 [{en, zh}] 列表")


def _debug_units(text: str, max_unit_chars: int) -> None:
    for no, raw in enumerate(split_paragraphs(text), 1):
        _, protected, mapping = _prepare_paragraph(raw)
        units = _make_units(protected, mapping, max_unit_chars, raw)
        print(f"\n[段落 {no}] 保护后：\n{protected}", file=sys.stderr)
        for i, u in enumerate(units, 1):
            print(f"  [{i}] fragment={u.fragment} joiner={u.joiner!r} raw={u.raw!r}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="学术段落英译中（占位符保护 + 长句拆分 + 双语对齐）")
    ap.add_argument("--selftest", action="store_true", help="跑内置自检")
    ap.add_argument("--demo", action="store_true", help="用内置材料离线演示")
    ap.add_argument("--text", help="直接给原文")
    ap.add_argument("--file", help="从 UTF-8 文件读原文；- 表示 stdin")
    ap.add_argument("--glossary", help="术语表 JSON：{en: zh} 或 [{en, zh}]")
    ap.add_argument("--max-unit-chars", type=int, default=240, help="超过该长度的句子才拆（最小 80）")
    ap.add_argument("--out", help="把 JSON 写入文件")
    ap.add_argument("--debug", action="store_true", help="把保护结果与片段切分打到 stderr")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    if args.selftest:
        return _selftest()

    if args.demo:
        text = DEMO_TEXT
        translator: Translator = CannedTranslator(DEMO_TM)
        glossary: dict[str, str] = {}
    else:
        if args.file == "-":
            text = sys.stdin.read()
        elif args.file:
            text = Path(args.file).read_text(encoding="utf-8")
        elif args.text:
            text = args.text
        else:
            ap.print_help()
            return 2
        glossary = _load_glossary(args.glossary)
        try:
            translator = LLMTranslator()   # Key 未配置时这里会给出明确错误
        except RuntimeError as exc:
            print(f"无法启动翻译：{exc}", file=sys.stderr)
            return 1

    if not text.strip():
        print("原文为空", file=sys.stderr)
        return 2
    if args.debug:
        _debug_units(text, max(80, args.max_unit_chars))

    try:
        result = translate_paragraphs(text, translator, glossary=glossary,
                                      max_unit_chars=args.max_unit_chars)
    except (TranslationError, PlaceholderError) as exc:
        print(f"翻译失败：{exc}", file=sys.stderr)
        return 1

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"written: {args.out}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
