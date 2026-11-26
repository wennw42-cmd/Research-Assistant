"""文献检索与引用含金量审计。

输入一篇论文标题，或一段参考文献列表；输出抓取到的候选与逐条引用评分。

流水线
──────
1. 解析    标题 / 编号参考文献 / 带上下文的引用（`ref || context || section`）/ JSON
2. 检索    标题模式：Semantic Scholar 取参考文献边（含 contexts / intents），
           失败时退回 Crossref 的 reference 字段；
           列表模式：Crossref、arXiv、可选 Semantic Scholar 逐条检索候选。
3. 评分    先看引用上下文（方法性动词、对比、差异定位、罗列堆叠、S2 intent），
           再看文献元数据（被引量、venue、DOI）。配置了真实 LLM 时，
           由模型读上下文打分，再与启发式结果融合；无 Key 则完全离线启发式。
4. 入库    低于阈值（默认 0.55）的引用标记 kept=false，只保留高价值文献。

输出（stdout 纯 JSON）：
    {"fetched": [...], "citation_quality": [...]}
    fetched          每条引用一条记录，含来源、元数据、实际检索式 queries、kept
    citation_quality 每条引用一条 {ref, score, reason, keep, verdict, context_used}

用法
────
    python -m app.citation_audit --selftest
    python -m app.citation_audit --demo
    python -m app.citation_audit "Attention Is All You Need"
    python -m app.citation_audit --file refs.txt --out result.json --library library.json

环境变量（均可选）：
    SEMANTIC_SCHOLAR_KEY       S2 提高限额；无 Key 时公共端点常返回 429，
                               标题模式会自动退回 Crossref 的参考文献列表
    CROSSREF_MAILTO            Crossref polite pool

说明：引用上下文只有 Semantic Scholar 提供。取不到上下文时只能按元数据
（被引量 / venue / DOI）低置信度打分，理由里会明确标注。
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

from app.utils import get_logger, short

log = get_logger("ra.citations")

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

DEFAULT_MIN_SCORE = 0.55
DEFAULT_LIMIT = 50
DEFAULT_TIMEOUT = 15
USER_AGENT = "ResearchAssistant/0.1 (citation-audit; +https://example.invalid)"


# ════════════════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════════════════


@dataclass
class Candidate:
    """某个数据源返回的一条文献记录。"""

    source: str
    title: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    doi: str = ""
    url: str = ""
    citation_count: int = 0
    paper_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "year": self.year,
            "authors": self.authors[:8],
            "venue": self.venue,
            "doi": self.doi,
            "url": self.url,
            "citation_count": self.citation_count,
            "source": self.source,
        }


@dataclass
class RefItem:
    """输入里的一条引用。context/section 由用户或数据源提供，可为空。"""

    raw: str
    title_guess: str = ""
    year: int | None = None
    doi: str = ""
    context: str = ""
    section: str = ""
    index: int = 0


@dataclass
class RefRecord:
    """一条引用 + 检索结果 + 上下文证据。"""

    ref: RefItem
    candidates: list[Candidate] = field(default_factory=list)
    best: Candidate | None = None
    contexts: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    influential: bool | None = None
    queries: list[str] = field(default_factory=list)


@dataclass
class Quality:
    score: float
    kind: str
    reason: str
    context_used: bool


@dataclass
class ParsedInput:
    mode: str  # title | refs
    title: str = ""
    refs: list[RefItem] = field(default_factory=list)


class RetrievalError(RuntimeError):
    pass


# ════════════════════════════════════════════════════════════════════
# 文本工具
# ════════════════════════════════════════════════════════════════════

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_ARXIV_RE = re.compile(r"\barXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_INDEX_RE = re.compile(r"^\s*(?:\[\s*\d+\s*\]|\(\s*\d+\s*\)|\d+\s*[.)、])\s*")
_SPACE_RE = re.compile(r"\s+")

_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with", "by",
    "from", "at", "as", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "we", "our", "their", "its", "it", "they", "them",
    "using", "based", "via", "towards", "toward", "into", "than", "then", "also",
    "can", "could", "may", "might", "will", "would", "should", "not", "no", "all",
    "more", "most", "some", "such", "which", "who", "what", "when", "where", "how",
    "paper", "study", "work", "approach", "method", "methods", "results", "result",
    "new", "novel", "efficient", "effective", "learning", "model", "models",
}


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def norm_title(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return _SPACE_RE.sub(" ", s).strip()


def jaccard(a: str, b: str) -> float:
    ta, tb = set(norm_title(a).split()), set(norm_title(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def title_ratio(a: str, b: str) -> float:
    """词序敏感：区分 "Attention Is All You Need" 与 "Is Attention All You Need?"。"""
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def keywords(text: str, k: int = 6) -> list[str]:
    """抽取检索关键词：去停用词、保序、去重。"""
    seen: dict[str, int] = {}
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text or ""):
        low = tok.lower().strip("-")
        if low in _STOPWORDS or len(low) < 3:
            continue
        seen[low] = seen.get(low, 0) + 1
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:k]]


_VENUE_LIKE_RE = re.compile(
    r"\b(?:advances in|proceedings|journal|transactions|conference|workshop|"
    r"symposium|findings of|lecture notes|technical report|arxiv|preprint|"
    r"press|springer|elsevier|ieee|acm|vol\.|pp\.)\b",
    re.I,
)
_AUTHOR_TAIL_RE = re.compile(r"(?:[A-Z][a-z\-']+\s+[A-Z]\.?|[A-Z]\.\s*[A-Z][a-z\-']+)$")


def _author_like(seg: str) -> bool:
    s = seg.strip().rstrip(".")
    if re.search(r"\bet al\b", s, re.I):
        return True
    chunks = [x.strip() for x in s.split(",") if x.strip()]
    if len(chunks) >= 2 and all(
        re.match(r"^[A-Z]", x) and 1 <= len(x.split()) <= 4 for x in chunks
    ):
        return True
    return bool(_AUTHOR_TAIL_RE.fullmatch(s))


def _title_candidate(seg: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", seg)
    return (
        2 <= len(words) <= 40
        and not _VENUE_LIKE_RE.search(seg)
        and not re.fullmatch(r"[\s\d.,;:()\-]+", seg)
    )


def guess_title(ref: str) -> str:
    """从参考文献串里猜标题：优先引号；其次取作者段之后的标题段；最后按字数打分。"""
    ref = (ref or "").strip()
    m = re.search(r"[“\"]([^”\"]{8,200})[”\"]", ref)
    if m:
        return m.group(1).strip(" .")

    body = _DOI_RE.sub(" ", ref)
    body = re.sub(r"https?://\S+", " ", body)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", body) if p.strip()]

    # 标准格式 "作者. 标题. venue. 年份."：作者段后第一段通常是标题
    for i, part in enumerate(parts):
        if not _author_like(part):
            continue
        for nxt in parts[i + 1:i + 3]:
            if _title_candidate(nxt):
                return nxt.rstrip(" .")[:220]
        break

    scored: list[tuple[float, str]] = []
    for p in parts:
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", p)
        if not (3 <= len(words) <= 40):
            continue
        penalty = 0.0
        if p.count(",") >= 2:
            penalty += 1.5 * p.count(",")
        if re.search(r"\bet al\b", p, re.I):
            penalty += 3.0
        if _VENUE_LIKE_RE.search(p):
            penalty += 4.0
        scored.append((min(len(words), 18) - penalty, p))
    if not scored:
        return ""
    scored.sort(key=lambda x: -x[0])
    return scored[0][1].rstrip(" .")[:220]


def _extract_year(text: str) -> int | None:
    cleaned = _DOI_RE.sub(" ", text or "")
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    years = [int(m.group(0)) for m in _YEAR_RE.finditer(cleaned)]
    return years[-1] if years else None


def _looks_like_ref(line: str) -> bool:
    if _INDEX_RE.match(line):
        return True
    if _DOI_RE.search(line) or _ARXIV_RE.search(line):
        return True
    if _YEAR_RE.search(line) and (
        line.count(",") >= 2 or re.search(r"\bet al\b", line, re.I)
    ):
        return True
    return False


def make_ref(raw: str, *, context: str = "", section: str = "", index: int = 0) -> RefItem:
    raw = _SPACE_RE.sub(" ", (raw or "").strip())
    raw = _INDEX_RE.sub("", raw).strip()
    return RefItem(
        raw=raw,
        title_guess=guess_title(raw),
        year=_extract_year(raw),
        doi=(_DOI_RE.search(raw).group(0) if _DOI_RE.search(raw) else ""),
        context=_SPACE_RE.sub(" ", context.strip()),
        section=section.strip(),
        index=index,
    )


def parse_input(text: str, *, mode: str = "auto") -> ParsedInput:
    """解析标题、纯文本参考文献列表或 JSON。"""
    if not text or not text.strip():
        raise ValueError("输入为空：请提供论文标题或参考文献列表")

    raw = text.strip()
    # 只把真正的 JSON 当 JSON；"[1] ..." 这种编号列表不能误判
    if raw.startswith("{") or re.match(r'\[\s*[\[{"]', raw):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败：{exc}") from exc
        return _parse_json_input(data)

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    is_title = mode == "title" or (
        mode == "auto"
        and len(lines) == 1
        and not _looks_like_ref(lines[0])
    )
    if is_title:
        return ParsedInput(mode="title", title=_SPACE_RE.sub(" ", raw))

    refs: list[RefItem] = []
    for line in lines:
        if re.fullmatch(r"(?i)\s*(references|bibliography|参考文献)\s*:?\s*", line):
            continue
        # 支持 "引用 || 上下文 || 章节"
        parts = [p.strip() for p in line.split("||")]
        ref = make_ref(parts[0], context=parts[1] if len(parts) > 1 else "",
                       section=parts[2] if len(parts) > 2 else "", index=len(refs))
        if len(ref.raw) < 8:
            continue
        refs.append(ref)
    refs = dedupe_refs(refs)
    if not refs:
        raise ValueError("没有解析出任何参考文献")
    return ParsedInput(mode="refs", refs=refs)


def _parse_json_input(data: Any) -> ParsedInput:
    if isinstance(data, list):
        return ParsedInput(mode="refs", refs=_refs_from_items(data))
    if not isinstance(data, dict):
        raise ValueError("JSON 必须是对象或数组")
    if data.get("refs"):
        return ParsedInput(mode="refs", refs=_refs_from_items(data["refs"]))
    if data.get("ref"):
        return ParsedInput(mode="refs", refs=_refs_from_items([data]))
    title = str(data.get("title") or "").strip()
    if title:
        return ParsedInput(mode="title", title=title)
    raise ValueError("JSON 里没有 title 或 refs")


def _refs_from_items(items: Iterable[Any]) -> list[RefItem]:
    out: list[RefItem] = []
    for item in items:
        if isinstance(item, str):
            out.append(make_ref(item, index=len(out)))
        elif isinstance(item, dict):
            raw = str(item.get("ref") or item.get("title") or "").strip()
            if not raw:
                continue
            out.append(make_ref(
                raw,
                context=str(item.get("context") or ""),
                section=str(item.get("section") or ""),
                index=len(out),
            ))
        else:
            raise ValueError("refs 数组元素必须是字符串或对象")
    out = dedupe_refs(out)
    if not out:
        raise ValueError("没有解析出任何参考文献")
    return out


def dedupe_refs(refs: Sequence[RefItem]) -> list[RefItem]:
    seen: set[str] = set()
    out: list[RefItem] = []
    for ref in refs:
        key = ref.doi.lower() or norm_title(ref.title_guess or ref.raw)
        if not key or key in seen:
            continue
        seen.add(key)
        ref.index = len(out)
        out.append(ref)
    return out


# ════════════════════════════════════════════════════════════════════
# 检索式
# ════════════════════════════════════════════════════════════════════


def ref_queries(ref: RefItem) -> list[str]:
    """一条引用生成 1–3 条检索式：DOI → 猜出的标题 → 关键词。"""
    out: list[str] = []
    if ref.doi:
        out.append(ref.doi)
    title = ref.title_guess
    if title:
        out.append(title)
    kw = keywords(title or ref.raw, 6)
    if kw:
        out.append(" ".join(kw))
    seen: set[str] = set()
    return [q for q in out if q and not (q.lower() in seen or seen.add(q.lower()))][:3]


def arxiv_query(text: str) -> str:
    terms = keywords(text, 6)
    if not terms:
        return ""
    return " AND ".join(f"all:{t}" for t in terms)


# ════════════════════════════════════════════════════════════════════
# API 解析（纯函数，便于离线测试）
# ════════════════════════════════════════════════════════════════════


def parse_arxiv(xml_text: str) -> list[Candidate]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out: list[Candidate] = []
    for entry in root.findall(f"{ATOM}entry"):
        title = (entry.findtext(f"{ATOM}title") or "").strip()
        if not title:
            continue
        published = entry.findtext(f"{ATOM}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            (a.findtext(f"{ATOM}name") or "").strip()
            for a in entry.findall(f"{ATOM}author")
            if (a.findtext(f"{ATOM}name") or "").strip()
        ]
        url = (entry.findtext(f"{ATOM}id") or "").strip()
        doi = (entry.findtext(f"{ARXIV}doi") or "").strip()
        venue = (entry.findtext(f"{ARXIV}journal_ref") or "").strip()
        out.append(Candidate(
            source="arxiv",
            title=_SPACE_RE.sub(" ", title),
            year=year,
            authors=authors,
            venue=venue or "arXiv",
            doi=doi,
            url=url,
            citation_count=0,
            paper_id=url.rsplit("/", 1)[-1] if url else "",
        ))
    return out


def _crossref_year(item: dict[str, Any]) -> int | None:
    parts = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
    if parts and isinstance(parts[0], int):
        return parts[0]
    return None


def _crossref_authors(item: dict[str, Any]) -> list[str]:
    out = []
    for a in item.get("author") or []:
        if isinstance(a, dict):
            name = " ".join(x for x in (a.get("given"), a.get("family")) if x)
            if name:
                out.append(name.strip())
        elif isinstance(a, str):
            out.append(a)
    return out


def parse_crossref(payload: Any) -> list[Candidate]:
    items = ((payload or {}).get("message") or {}).get("items") or []
    out: list[Candidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = _first(item.get("title"))
        if not title:
            continue
        out.append(Candidate(
            source="crossref",
            title=_SPACE_RE.sub(" ", title),
            year=_crossref_year(item),
            authors=_crossref_authors(item),
            venue=_first(item.get("container-title")),
            doi=str(item.get("DOI") or ""),
            url=str(item.get("URL") or ""),
            citation_count=int(item.get("is-referenced-by-count") or 0),
        ))
    return out


def parse_s2_search(payload: Any) -> list[Candidate]:
    out: list[Candidate] = []
    for item in (payload or {}).get("data") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        ext = item.get("externalIds") or {}
        out.append(Candidate(
            source="semantic_scholar",
            title=_SPACE_RE.sub(" ", title),
            year=item.get("year"),
            authors=[str(a.get("name")) for a in (item.get("authors") or []) if a.get("name")],
            venue=str(item.get("venue") or ""),
            doi=str(ext.get("DOI") or ""),
            url=str(item.get("url") or ""),
            citation_count=int(item.get("citationCount") or 0),
            paper_id=str(item.get("paperId") or ""),
        ))
    return out


def parse_s2_references(payload: Any) -> list[RefRecord]:
    """Semantic Scholar 参考文献边：上下文与 intent 都挂在这条边上。"""
    out: list[RefRecord] = []
    for item in (payload or {}).get("data") or []:
        if not isinstance(item, dict):
            continue
        paper = item.get("citedPaper") or {}
        title = str(paper.get("title") or "").strip()
        if not title:
            continue
        ext = paper.get("externalIds") or {}
        cand = Candidate(
            source="semantic_scholar",
            title=_SPACE_RE.sub(" ", title),
            year=paper.get("year"),
            authors=[str(a.get("name")) for a in (paper.get("authors") or []) if a.get("name")],
            venue=str(paper.get("venue") or ""),
            doi=str(ext.get("DOI") or ""),
            url=str(paper.get("url") or ""),
            citation_count=int(paper.get("citationCount") or 0),
            paper_id=str(paper.get("paperId") or ""),
        )
        contexts = [
            _SPACE_RE.sub(" ", str(c).strip())
            for c in (item.get("contexts") or [])
            if c and str(c).strip()
        ]
        intents = [str(i) for i in (item.get("intents") or []) if i]
        influential = item.get("isInfluential")
        ref = make_ref(_compose_ref(cand), index=len(out))
        out.append(RefRecord(
            ref=ref,
            candidates=[cand],
            best=cand,
            contexts=contexts,
            intents=intents,
            influential=influential if isinstance(influential, bool) else None,
        ))
    return out


def parse_crossref_references(work: dict[str, Any]) -> list[RefRecord]:
    """Crossref reference 列表兜底：只有书目信息，没有上下文。"""
    refs = ((work or {}).get("message") or {}).get("reference") or []
    out: list[RefRecord] = []
    for item in refs:
        if not isinstance(item, dict):
            continue
        unstructured = str(item.get("unstructured") or "").strip()
        title = str(item.get("article-title") or item.get("volume-title") or "").strip()
        raw = unstructured or " ".join(
            x for x in (str(item.get("author") or ""), title,
                        str(item.get("journal-title") or ""), str(item.get("year") or ""))
            if x
        )
        if len(raw) < 8:
            continue
        cand = Candidate(
            source="crossref",
            title=title or guess_title(raw),
            year=int(item["year"]) if str(item.get("year") or "").isdigit() else _extract_year(raw),
            authors=[str(item["author"])] if item.get("author") else [],
            venue=str(item.get("journal-title") or ""),
            doi=str(item.get("DOI") or ""),
            citation_count=0,
        )
        ref = make_ref(raw, index=len(out))
        out.append(RefRecord(ref=ref, candidates=[cand] if cand.title else [], best=cand if cand.title else None))
    return out


def _first(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _compose_ref(c: Candidate) -> str:
    first = c.authors[0] if c.authors else ""
    authors = f"{first} et al." if len(c.authors) > 2 and first else first
    bits = [authors, c.title, c.venue, str(c.year or "")]
    return ". ".join(x for x in bits if x).strip(". ")


def pick_best(cands: Sequence[Candidate], ref: RefItem) -> Candidate | None:
    """只接受可信匹配：DOI 相同、标题几乎一致，或相似度够高且年份相容。"""
    probe = ref.title_guess or ref.raw
    best: Candidate | None = None
    best_score = 0.0
    for c in cands:
        sim = jaccard(c.title, probe)
        doi_ok = bool(c.doi and ref.doi and c.doi.lower() == ref.doi.lower())
        year_ok = not ref.year or not c.year or abs(c.year - ref.year) <= 1
        if not (doi_ok or title_ratio(c.title, probe) >= 0.9 or (sim >= 0.55 and year_ok)):
            continue
        score = 0.7 * sim
        if c.year and ref.year and c.year == ref.year:
            score += 0.15
        if doi_ok:
            score += 0.5
        if score > best_score:
            best, best_score = c, score
    return best


def dedupe_candidates(cands: Sequence[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in cands:
        key = (c.doi.lower() if c.doi else "") or norm_title(c.title)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ════════════════════════════════════════════════════════════════════
# 引用含金量
# ════════════════════════════════════════════════════════════════════

_METHOD_RE = re.compile(
    r"\b(?:adopt\w*|employ\w*|extend\w*|build(?:s|ing)?\b|built\b|"
    r"rely(?:ing)? on|based on|leverag\w*|incorporat\w*|combin\w*|"
    r"modif\w*|replac\w*|follow\w*|fine-?tun\w*|implement\w*|"
    r"use|uses|used|using|train(?:ed|ing)? on)\b",
    re.I,
)
_METHOD_ZH = re.compile(r"(采用|沿用|借鉴|基于|扩展|改进|复现|作为基线|使用)")
_COMPARE_RE = re.compile(
    r"\b(?:compar(?:e|ed|ison|ing)|outperform\w*|better than|worse than|"
    r"baseline|versus|vs\.)\b",
    re.I,
)
_COMPARE_ZH = re.compile(r"(对比|优于|差于|基线|相比)")
_CONTRAST_RE = re.compile(
    r"\b(?:unlike|in contrast|by contrast|whereas|however|limitation\w*|"
    r"limited|fail(?:s|ed|ure)?|suffer\w*|cannot|does not)\b",
    re.I,
)
_DATA_RE = re.compile(
    r"\b(?:dataset|data set|benchmark|corpus|corpora|training data|"
    r"test set|evaluat\w*|experiment\w*)\b",
    re.I,
)
_DATA_ZH = re.compile(r"(数据集|基准|语料|评测|实验)")
_LISTING_RE = re.compile(
    r"\b(?:see also|for example|for instance|e\.g\.|among others|"
    r"and references therein|etc\.|a number of|a variety of|"
    r"several (?:works|studies|papers)|many (?:works|studies|papers)|"
    r"recent (?:works|studies|papers)|in the literature|a review of)\b",
    re.I,
)
_LISTING_ZH = re.compile(r"(等相关工作|等文献|参见|例如|综述)")
_CITE_BRACKET_RE = re.compile(r"\[([0-9][0-9\s,;–—-]*)\]")
_KIND_ZH = {
    "method": "方法性引用",
    "baseline": "基线/对比引用",
    "data": "数据/评测引用",
    "contrast": "差异定位引用",
    "background": "背景性引用",
    "listing": "罗列式引用",
    "metadata_only": "仅元数据判定",
    "unmatched": "未匹配到文献",
}
_TOP_VENUES = {
    "nature", "science", "cell", "neurips", "nips", "icml", "iclr", "acl",
    "emnlp", "naacl", "cvpr", "iccv", "eccv", "aaai", "ijcai", "kdd", "sigir",
    "tpami", "jmlr", "tacl", "osdi", "sosp", "nsdi", "sigcomm", "stoc", "focs",
    "usenix", "www", "wacv", "miccai", "icra", "iros", "acl", "coling",
}
_MID_VENUES = {
    "ieee", "acm", "springer", "elsevier", "plos", "frontiers", "journal",
    "transactions", "conference", "proceedings", "workshop",
}
_LOW_VENUES = {"arxiv", "biorxiv", "medrxiv", "preprint", "techrxiv", "ssrn", "researchgate"}


def cite_group_size(text: str) -> int:
    """同一方括号里并列了多少条文献，取最大值。"2–4" 展开为 3 条。"""
    biggest = 0
    for m in _CITE_BRACKET_RE.finditer(text or ""):
        total = 0
        for part in re.split(r"[,\s;]+", m.group(1).strip()):
            rng = re.fullmatch(r"(\d+)[–—-](\d+)", part)
            if rng:
                a, b = int(rng.group(1)), int(rng.group(2))
                total += b - a + 1 if b >= a else 1
            elif part.isdigit():
                total += 1
        biggest = max(biggest, total)
    return biggest


def venue_score(venue: str) -> float:
    v = (venue or "").lower()
    if not v:
        return 0.35
    if any(k in v for k in _LOW_VENUES):
        return 0.15
    if any(k in v for k in _TOP_VENUES):
        return 1.0
    if any(k in v for k in _MID_VENUES):
        return 0.55
    return 0.4


def meta_score(c: Candidate | None) -> float:
    if c is None:
        return 0.0
    cites = min(1.0, math.log10(1 + max(0, c.citation_count)) / 4.0)
    trace = 1.0 if (c.doi or c.url) else 0.0
    return clamp(0.6 * cites + 0.3 * venue_score(c.venue) + 0.1 * trace)


def _context_text(record: RefRecord) -> str:
    bits = list(record.contexts)
    if record.ref.context:
        bits.insert(0, record.ref.context)
    return " ".join(bits).strip()


def heuristic_quality(record: RefRecord) -> Quality:
    text = _context_text(record)
    has_ctx = bool(text)
    listing = bool(_LISTING_RE.search(text) or _LISTING_ZH.search(text))
    group = cite_group_size(text)
    method = bool(_METHOD_RE.search(text) or _METHOD_ZH.search(text))
    compare = bool(_COMPARE_RE.search(text) or _COMPARE_ZH.search(text))
    contrast = bool(_CONTRAST_RE.search(text))
    data = bool(_DATA_RE.search(text) or _DATA_ZH.search(text))
    meta = meta_score(record.best)
    bits: list[str] = []

    if has_ctx:
        score = 0.35
        n = len(record.contexts) + (1 if record.ref.context else 0)
        if n >= 2:
            score += 0.08
            bits.append(f"{n} 处提及")
        if len(text) >= 120:
            score += 0.05
        if method:
            score += 0.22
            bits.append("上下文表明被采用/扩展/作为实现基础")
        if data:
            score += 0.15
            bits.append("用于数据/评测设置")
        if compare:
            score += 0.15
            bits.append("作为基线或对比对象")
        if contrast:
            score += 0.08
            bits.append("用于差异定位")
        if listing:
            score -= 0.22
            bits.append("罗列式表述")
        if group >= 3:
            score -= 0.12
            bits.append(f"与 {group} 条文献并列堆叠")
        if record.intents:
            low = [i.lower() for i in record.intents]
            if "methodology" in low:
                score += 0.12
                bits.append("S2 intent=methodology")
            elif "result" in low:
                score += 0.08
                bits.append("S2 intent=result")
            elif "background" in low:
                score -= 0.06
                bits.append("S2 intent=background")
        if record.influential is True:
            score += 0.08
            bits.append("S2 isInfluential")
        if record.ref.section:
            sec = record.ref.section.lower()
            if any(k in sec for k in ("method", "approach", "experiment", "result", "evaluation", "方法", "实验", "结果")):
                score += 0.08
                bits.append(f"位于{record.ref.section}章节")
            elif any(k in sec for k in ("related", "background", "introduction", "相关工作", "背景", "引言")):
                score -= 0.06
                bits.append(f"位于{record.ref.section}章节")
        score += 0.18 * meta
        kind = (
            "listing" if listing
            else "method" if method
            else "data" if data
            else "baseline" if compare
            else "contrast" if contrast
            else "background"
        )
    else:
        score = 0.28
        bits.append("未提供引用上下文")
        score += 0.30 * meta
        if record.best and (record.best.doi or record.best.url):
            score += 0.04
        kind = "metadata_only"

    if record.best is None:
        score -= 0.12
        bits.append("未命中任何数据源")
        if not has_ctx:
            kind = "unmatched"

    if record.best is not None:
        b = record.best
        extra = [f"{b.source} 命中"]
        if b.citation_count:
            extra.append(f"被引 {b.citation_count}")
        if b.venue:
            extra.append(short(b.venue, 24))
        if b.doi:
            extra.append("有 DOI")
        bits.extend(extra)

    score = clamp(score)
    reason = f"{_KIND_ZH.get(kind, kind)}：" + "；".join(bits)
    if not has_ctx:
        reason += "（仅元数据，置信度低，建议人工复核）"
    return Quality(score=round(score, 3), kind=kind, reason=short(reason, 180), context_used=has_ctx)


def _judgeable(record: RefRecord) -> bool:
    return bool(_context_text(record))


def llm_judge(records: Sequence[RefRecord], llm: Any, *, batch: int = 8) -> dict[int, dict[str, Any]]:
    """批量让模型判引用质量。只接受合法 JSON；失败整体退回启发式。"""
    if llm is None or getattr(llm, "model_name", "") == "mock-llm":
        return {}
    indexed = [(i, r) for i, r in enumerate(records) if _judgeable(r)]
    if not indexed:
        return {}

    system = (
        "你是学术引用审计员。只依据给定的参考文献与引用上下文评分，不得使用外部知识，"
        "不得因为标题看起来重要而加分。\n"
        "0.8-1.0：方法/数据被实质采用、扩展或作为实验基线；0.5-0.79：明确讨论但非关键；"
        "0.25-0.49：背景铺垫或泛泛提及；0-0.24：仅罗列、堆叠。\n"
        "kind 只能取 method|baseline|data|contrast|background|listing。\n"
        '只输出 JSON：{"results":[{"index":0,"score":0.0,"kind":"method",'
        '"reason":"不超过60字，引用上下文中的具体证据"}]}'
    )

    found: dict[int, dict[str, Any]] = {}
    for start in range(0, len(indexed), batch):
        chunk = indexed[start:start + batch]
        payload = []
        for i, r in chunk:
            ctxs = ([r.ref.context] if r.ref.context else []) + list(r.contexts)
            payload.append({
                "index": i,
                "ref": short(r.ref.raw, 320),
                "contexts": [short(c, 600) for c in ctxs][:3],
                "section": r.ref.section,
            })
        user = "待判定引用：\n" + json.dumps(payload, ensure_ascii=False) + "\n只输出 JSON。"
        try:
            data = llm.chat_json([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM 引用判定失败，退回启发式：%s", exc)
            return found
        entries = data.get("results") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            try:
                idx = int(e.get("index"))
                score = clamp(float(e.get("score")))
            except (TypeError, ValueError):
                continue
            if idx not in {i for i, _ in chunk}:
                continue
            kind = str(e.get("kind") or "").lower()
            if kind not in _KIND_ZH:
                kind = ""
            found[idx] = {
                "score": score,
                "kind": kind,
                "reason": short(str(e.get("reason") or "").strip(), 120),
            }
    return found


def finalize_quality(record: RefRecord, base: Quality, judged: dict[str, Any] | None) -> Quality:
    if not judged or not _judgeable(record):
        return base
    blended = 0.7 * float(judged["score"]) + 0.3 * base.score
    reason = str(judged.get("reason") or "").strip()
    kind = judged.get("kind") or base.kind
    if base.kind == "listing" and blended > 0.5:
        blended = 0.5
        reason = (reason + "；检测到罗列式引用，已下调").strip("；")
    if not reason:
        reason = base.reason
    else:
        b = record.best
        suffix = []
        if b is not None:
            suffix.append(f"{b.source}")
            if b.citation_count:
                suffix.append(f"被引 {b.citation_count}")
            if b.venue:
                suffix.append(short(b.venue, 20))
        if suffix:
            reason += "（" + "，".join(suffix) + "）"
    return Quality(
        score=round(clamp(blended), 3),
        kind=kind if kind in _KIND_ZH else base.kind,
        reason=short(reason, 180),
        context_used=base.context_used,
    )


# ════════════════════════════════════════════════════════════════════
# HTTP + 数据源
# ════════════════════════════════════════════════════════════════════


class HttpClient:
    def __init__(self, *, timeout: int = DEFAULT_TIMEOUT, mailto: str = "", s2_key: str = "") -> None:
        self.timeout = timeout
        self.mailto = mailto
        self.s2_key = s2_key
        self._last_s2 = 0.0

    def _open(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, application/atom+xml, */*",
            **(headers or {}),
        })
        last: Exception | None = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code == 429 and attempt == 0:
                    time.sleep(2.0)
                    continue
                if 500 <= exc.code < 600 and attempt == 0:
                    time.sleep(1.0)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if attempt == 0:
                    time.sleep(0.6)
                    continue
                raise
        raise RuntimeError(str(last))

    def get_json(self, url: str, headers: dict[str, str] | None = None) -> Any:
        body = self._open(url, headers)
        return json.loads(body.decode("utf-8", "replace"))

    def get_text(self, url: str) -> str:
        return self._open(url).decode("utf-8", "replace")

    def s2_headers(self) -> dict[str, str]:
        return {"x-api-key": self.s2_key} if self.s2_key else {}

    def s2_wait(self) -> None:
        gap = 0.35 if self.s2_key else 1.05
        wait = gap - (time.monotonic() - self._last_s2)
        if wait > 0:
            time.sleep(wait)
        self._last_s2 = time.monotonic()


class Provider(Protocol):
    def title_references(self, title: str) -> tuple[Candidate | None, list[RefRecord]]: ...
    def search(self, ref: RefItem, sources: Sequence[str]) -> list[Candidate]: ...


class ApiProvider:
    """arXiv / Semantic Scholar / Crossref。所有失败都降级，不中断整体流程。"""

    def __init__(self, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient()
        self.errors: list[str] = []

    # ── 标题 → 参考文献（上下文只能从这里来）────────────────────────
    def title_references(self, title: str) -> tuple[Candidate | None, list[RefRecord]]:
        try:
            seed = self._s2_find(title)
            if seed and seed.paper_id:
                recs = parse_s2_references(self._s2_refs(seed.paper_id))
                if recs:
                    return seed, recs
        except Exception as exc:  # noqa: BLE001
            self._fail("semantic_scholar", exc)

        try:
            work = self._crossref_work(title)
            recs = parse_crossref_references(work)
            if recs:
                seed = parse_crossref({"message": {"items": [work.get("message") or {}]}})
                return (seed[0] if seed else None), recs
            raise RetrievalError("Crossref 未公开该文献的 reference 列表")
        except Exception as exc:  # noqa: BLE001
            self._fail("crossref", exc)

        raise RetrievalError(
            "无法获取该文献的参考文献：Semantic Scholar 与 Crossref 均失败。"
            "请改用手工参考文献列表（--file）。"
        )

    def _s2_find(self, title: str) -> Candidate | None:
        self.http.s2_wait()
        params = urllib.parse.urlencode({
            "query": title,
            "limit": 5,
            "fields": "paperId,title,year,venue,authors,externalIds,citationCount,url",
        })
        payload = self.http.get_json(
            f"https://api.semanticscholar.org/graph/v1/paper/search?{params}",
            self.http.s2_headers(),
        )
        hits = parse_s2_search(payload)
        if not hits:
            return None
        best = max(hits, key=lambda c: title_ratio(c.title, title))
        return best if title_ratio(best.title, title) >= 0.85 else None

    def _s2_refs(self, paper_id: str) -> Any:
        self.http.s2_wait()
        fields = (
            "title,year,venue,authors,externalIds,citationCount,url,"
            "contexts,intents,isInfluential"
        )
        params = urllib.parse.urlencode({"limit": 500, "fields": fields})
        return self.http.get_json(
            f"https://api.semanticscholar.org/graph/v1/paper/{quote_id(paper_id)}/references?{params}",
            self.http.s2_headers(),
        )

    def _crossref_work(self, title: str) -> dict[str, Any]:
        # query.title 对精确标题的排序远好于 query.bibliographic
        params = {"query.title": title, "rows": 8,
                  "select": "DOI,title,author,issued,container-title,is-referenced-by-count,URL"}
        if self.http.mailto:
            params["mailto"] = self.http.mailto
        payload = self.http.get_json(
            "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        )
        items = ((payload or {}).get("message") or {}).get("items") or []
        if not items:
            raise RetrievalError("Crossref 无结果")
        best = max(items, key=lambda it: title_ratio(_first(it.get("title")), title))
        if title_ratio(_first(best.get("title")), title) < 0.92:
            raise RetrievalError("Crossref 标题匹配度过低，拒绝用错误文献的参考文献")
        doi = str(best.get("DOI") or "")
        if not doi:
            return {"message": best}
        detail = self.http.get_json(
            "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
            + (("?mailto=" + urllib.parse.quote(self.http.mailto)) if self.http.mailto else "")
        )
        return detail if isinstance(detail, dict) else {"message": best}

    # ── 逐条引用检索 ────────────────────────────────────────────────
    def search(self, ref: RefItem, sources: Sequence[str]) -> list[Candidate]:
        out: list[Candidate] = []

        if "crossref" in sources:
            try:
                params = {"query.bibliographic": ref.raw, "rows": 5,
                          "select": "DOI,title,author,issued,container-title,is-referenced-by-count,URL"}
                if self.http.mailto:
                    params["mailto"] = self.http.mailto
                out += parse_crossref(self.http.get_json(
                    "https://api.crossref.org/works?" + urllib.parse.urlencode(params)))
            except Exception as exc:  # noqa: BLE001
                self._fail("crossref", exc)

        # Crossref 命中就不再打 arXiv；有 arXiv id 时例外（id 是权威匹配）
        arxiv_id = _ARXIV_RE.search(ref.raw)
        if "arxiv" in sources and (arxiv_id or pick_best(out, ref) is None):
            try:
                if arxiv_id:
                    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                        {"id_list": arxiv_id.group(1), "max_results": 3})
                else:
                    q = arxiv_query(ref.title_guess or ref.raw)
                    if q:
                        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                            {"search_query": q, "start": 0, "max_results": 5})
                    else:
                        url = ""
                if url:
                    out += parse_arxiv(self.http.get_text(url))
            except Exception as exc:  # noqa: BLE001
                self._fail("arxiv", exc)

        if "s2" in sources and (ref.title_guess or ref.raw) and pick_best(out, ref) is None:
            try:
                self.http.s2_wait()
                params = urllib.parse.urlencode({
                    "query": ref.title_guess or ref.raw,
                    "limit": 5,
                    "fields": "paperId,title,year,venue,authors,externalIds,citationCount,url",
                })
                out += parse_s2_search(self.http.get_json(
                    f"https://api.semanticscholar.org/graph/v1/paper/search?{params}",
                    self.http.s2_headers(),
                ))
            except Exception as exc:  # noqa: BLE001
                self._fail("semantic_scholar", exc)

        return dedupe_candidates(out)

    def _fail(self, source: str, exc: Exception) -> None:
        msg = f"{source}: {short(str(exc), 120)}"
        self.errors.append(msg)
        log.warning("数据源失败 %s", msg)


def quote_id(paper_id: str) -> str:
    return urllib.parse.quote(str(paper_id).strip(), safe=":")


# ════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════


def build_records(
    parsed: ParsedInput,
    provider: Provider,
    *,
    sources: Sequence[str] = ("crossref", "arxiv"),
    limit: int = DEFAULT_LIMIT,
) -> list[RefRecord]:
    if parsed.mode == "title":
        _, records = provider.title_references(parsed.title)
        for r in records:
            r.queries = ref_queries(r.ref)[:1]
        return records[:limit] if limit else records

    records: list[RefRecord] = []
    for ref in parsed.refs[:limit] if limit else parsed.refs:
        cands = provider.search(ref, sources)
        records.append(RefRecord(
            ref=ref,
            candidates=cands,
            best=pick_best(cands, ref),
            queries=ref_queries(ref),
        ))
    return records


def audit(
    text: str,
    *,
    mode: str = "auto",
    provider: Provider | None = None,
    llm: Any = None,
    sources: Sequence[str] = ("crossref", "arxiv"),
    min_score: float = DEFAULT_MIN_SCORE,
    limit: int = DEFAULT_LIMIT,
    use_llm: bool = True,
) -> dict[str, Any]:
    """给定标题或参考文献列表，返回 fetched + citation_quality。"""
    parsed = parse_input(text, mode=mode)
    prov = provider or ApiProvider()
    records = build_records(parsed, prov, sources=sources, limit=limit)

    judged: dict[int, dict[str, Any]] = {}
    if use_llm and llm is not None:
        judged = llm_judge(records, llm)

    qualities: list[Quality] = []
    for i, rec in enumerate(records):
        base = heuristic_quality(rec)
        qualities.append(finalize_quality(rec, base, judged.get(i)))

    fetched: list[dict[str, Any]] = []
    citation_quality: list[dict[str, Any]] = []
    for rec, q in zip(records, qualities):
        kept = q.score >= min_score and not (rec.best is None and not q.context_used)
        row = {
            "ref": rec.ref.raw,
            "title": rec.best.title if rec.best else (rec.ref.title_guess or short(rec.ref.raw, 120)),
            "year": rec.best.year if rec.best else rec.ref.year,
            "authors": rec.best.authors[:8] if rec.best else [],
            "venue": rec.best.venue if rec.best else "",
            "doi": rec.best.doi if rec.best else rec.ref.doi,
            "url": rec.best.url if rec.best else "",
            "source": rec.best.source if rec.best else "",
            "citation_count": rec.best.citation_count if rec.best else 0,
            "sources": sorted({c.source for c in rec.candidates}),
            "queries": rec.queries[:2],
            "matched": rec.best is not None,
            "kept": kept,
        }
        fetched.append(row)
        citation_quality.append({
            "ref": rec.ref.raw,
            "score": q.score,
            "reason": q.reason,
            "keep": kept,
            "verdict": q.kind,
            "context_used": q.context_used,
        })

    return {"fetched": fetched, "citation_quality": citation_quality}


# ════════════════════════════════════════════════════════════════════
# 离线 fixtures：--demo 与 --selftest 共用
# ════════════════════════════════════════════════════════════════════

_FIXTURE_S2_SEARCH = {
    "data": [{
        "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        "title": "Attention Is All You Need",
        "year": 2017,
        "venue": "Neural Information Processing Systems",
        "citationCount": 120000,
        "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
        "externalIds": {"DOI": "10.48550/arXiv.1706.03762", "ArXiv": "1706.03762"},
        "url": "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
    }]
}


def _s2_edge(title: str, year: int, venue: str, cites: int, contexts: list[str],
             intents: list[str], influential: bool | None = None,
             doi: str = "") -> dict[str, Any]:
    return {
        "citedPaper": {
            "paperId": "fixture-" + norm_title(title)[:18].replace(" ", "-"),
            "title": title,
            "year": year,
            "venue": venue,
            "citationCount": cites,
            "authors": [{"name": "Fixture Author"}],
            "externalIds": {"DOI": doi} if doi else {},
            "url": "https://www.semanticscholar.org/paper/" + norm_title(title)[:18].replace(" ", "-"),
        },
        "contexts": contexts,
        "intents": intents,
        "isInfluential": influential,
    }


_FIXTURE_S2_REFS = {
    "data": [
        _s2_edge(
            "Neural Machine Translation by Jointly Learning to Align and Translate",
            2015, "International Conference on Learning Representations", 42000,
            ["We build on the attention mechanism of Bahdanau et al. and extend it to multi-head self-attention.",
             "Our scaled dot-product attention follows the additive attention of Bahdanau et al."],
            ["methodology"], True, "10.48550/arXiv.1409.0473",
        ),
        _s2_edge(
            "Sequence to Sequence Learning with Neural Networks",
            2014, "Neural Information Processing Systems", 26000,
            ["Unlike recurrent sequence models such as the one of Sutskever et al., our model dispenses with recurrence entirely."],
            ["background"], True, "10.48550/arXiv.1409.3215",
        ),
        _s2_edge(
            "Long Short-Term Memory",
            1997, "Neural Computation", 90000,
            ["Recurrent networks, particularly LSTM, have long been the dominant approach for sequence modeling."],
            ["background"], False,
        ),
        _s2_edge(
            "Sparse Transformers for Long Documents",
            2019, "arXiv", 180,
            ["A number of works have explored sparse and local attention patterns [12, 13, 14]."],
            ["background"], False,
        ),
        _s2_edge(
            "Local Attention for Efficient Translation",
            2020, "arXiv", 95,
            ["A number of works have explored sparse and local attention patterns [12, 13, 14]."],
            ["background"], False,
        ),
        _s2_edge(
            "Efficient Attention with Fixed Patterns",
            2021, "Workshop on Efficient NLP", 40,
            ["A number of works have explored sparse and local attention patterns [12, 13, 14]."],
            ["background"], False,
        ),
        _s2_edge(
            "Findings of the 2014 Workshop on Statistical Machine Translation",
            2014, "Workshop on Statistical Machine Translation", 3100,
            ["We evaluate our model on the WMT 2014 English-to-German translation task and report BLEU scores."],
            ["result"], True,
        ),
        _s2_edge(
            "Adam: A Method for Stochastic Optimization",
            2015, "International Conference on Learning Representations", 180000,
            [], [], None, "10.48550/arXiv.1412.6980",
        ),
        _s2_edge(
            "A Preliminary Study of Tiny Attention Variants",
            2022, "Workshop on Tiny Models", 3,
            [], [], None,
        ),
    ]
}


class FixtureProvider:
    """离线演示：走与真实 API 完全相同的解析 → 评分 → 过滤路径。"""

    def __init__(self) -> None:
        self.seed = parse_s2_search(_FIXTURE_S2_SEARCH)[0]
        self.records = parse_s2_references(_FIXTURE_S2_REFS)
        self._by_title = {norm_title(r.ref.title_guess): r.best for r in self.records}

    def title_references(self, title: str) -> tuple[Candidate | None, list[RefRecord]]:
        return self.seed, [RefRecord(
            ref=r.ref, candidates=list(r.candidates), best=r.best,
            contexts=list(r.contexts), intents=list(r.intents),
            influential=r.influential, queries=ref_queries(r.ref)[:1],
        ) for r in self.records]

    def search(self, ref: RefItem, sources: Sequence[str]) -> list[Candidate]:
        hit = self._by_title.get(norm_title(ref.title_guess))
        return [hit] if hit else []


# ════════════════════════════════════════════════════════════════════
# 自检
# ════════════════════════════════════════════════════════════════════


class _StubJudge:
    model_name = "stub-judge"

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.calls = 0
        self.last_user = ""

    def chat_json(self, messages, **_: Any) -> Any:
        self.calls += 1
        self.last_user = messages[-1]["content"]
        return {"results": self.results}


def _selftest() -> int:
    checks = 0
    fails = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal checks, fails
        checks += 1
        if cond:
            print(f"[OK] {name}")
        else:
            fails += 1
            print(f"[XX] {name}" + (f"  → {detail}" if detail else ""))

    # 输入解析
    p = parse_input("Attention Is All You Need")
    check("单行标题识别", p.mode == "title" and p.title.startswith("Attention"))
    p = parse_input("References\n[1] Vaswani A, et al. Attention is all you need. NeurIPS 2017.\n"
                    "[2] Bahdanau D. Neural machine translation by jointly learning to align and translate. ICLR 2015.")
    check("多行参考文献识别", p.mode == "refs" and len(p.refs) == 2, str(p))
    check("去掉 [n] 编号", p.refs[0].raw.startswith("Vaswani"), p.refs[0].raw)
    check("年份提取", p.refs[1].year == 2015, str(p.refs[1].year))
    check("标题猜测", "align" in p.refs[1].title_guess.lower(), p.refs[1].title_guess)
    venue_ref = make_ref("Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. "
                         "Advances in Neural Information Processing Systems. 2017.")
    check("标题不被 venue 抢走", "Attention" in venue_ref.title_guess
          and "Advances" not in venue_ref.title_guess, venue_ref.title_guess)
    check("作者段识别", _author_like("Bahdanau D, Cho K, Bengio Y."))
    check("DOI 提取", make_ref("X. doi:10.1000/xyz123. 2020.").doi == "10.1000/xyz123")

    p = parse_input("[1] A. Paper one. 2019.\n[2] B. Paper two. 2020. || We adopt the method of [2] in our encoder.")
    check("|| 上下文切分", "adopt" in p.refs[1].context, p.refs[1].context)
    check("章节字段", make_ref("r", section="Method").section == "Method")

    pj = parse_input('{"title": "Attention Is All You Need"}')
    check("JSON 标题", pj.mode == "title")
    pj = parse_input('{"refs": [{"ref": "A. Paper. 2020.", "context": "We use it.", "section": "Method"}]}')
    check("JSON refs", pj.mode == "refs" and pj.refs[0].context == "We use it.")
    try:
        parse_input("   ")
        check("空输入报错", False)
    except ValueError:
        check("空输入报错", True)
    try:
        parse_input("{bad json")
        check("坏 JSON 报错", False)
    except ValueError:
        check("坏 JSON 报错", True)

    dup = parse_input("[1] A. Same Paper. 2020. doi:10.1000/abc\n[2] B. Same Paper. 2020. doi:10.1000/abc")
    check("DOI 去重", len(dup.refs) == 1, str(len(dup.refs)))

    check("关键词去停用词", "the" not in keywords("The model of the paper", 5))
    check("空关键词", keywords("", 3) == [])

    # API 解析
    arx = f"""<?xml version="1.0"?><feed xmlns="{ATOM[1:-1]}" xmlns:arxiv="{ARXIV[1:-1]}">
      <entry><id>http://arxiv.org/abs/1706.03762v7</id>
      <published>2017-06-12T00:00:00Z</published>
      <title>Attention Is All You Need</title>
      <author><name>Vaswani</name></author>
      <arxiv:doi>10.48550/arXiv.1706.03762</arxiv:doi>
      <arxiv:journal_ref>NeurIPS 2017</arxiv:journal_ref></entry></feed>"""
    a = parse_arxiv(arx)
    check("arXiv Atom 解析", len(a) == 1 and a[0].year == 2017 and a[0].doi.startswith("10.48550"),
          str(a))

    cr_payload = {"message": {"items": [{
        "title": ["Attention Is All You Need"], "DOI": "10.1/x",
        "author": [{"given": "Ashish", "family": "Vaswani"}],
        "issued": {"date-parts": [[2017, 6]]},
        "container-title": ["NeurIPS"], "is-referenced-by-count": 1000,
        "URL": "https://doi.org/10.1/x",
    }]}}
    c = parse_crossref(cr_payload)
    check("Crossref 解析", c[0].title.startswith("Attention") and c[0].year == 2017
          and c[0].citation_count == 1000, str(c))

    s2s = parse_s2_search(_FIXTURE_S2_SEARCH)
    check("S2 检索解析", s2s[0].paper_id and s2s[0].year == 2017)
    recs = parse_s2_references(_FIXTURE_S2_REFS)
    check("S2 参考文献解析", len(recs) == 9, str(len(recs)))
    check("S2 contexts 保留", len(recs[0].contexts) == 2, str(recs[0].contexts))
    check("S2 intents 保留", recs[0].intents == ["methodology"], str(recs[0].intents))
    check("S2 influential 保留", recs[0].influential is True)
    check("S2 空 contexts", recs[7].contexts == [])

    cr_refs = parse_crossref_references({"message": {"reference": [
        {"unstructured": "Vaswani A. Attention is all you need. NeurIPS 2017."},
        {"article-title": "BERT", "author": "Devlin", "year": "2019", "DOI": "10.1/bert"},
    ]}})
    check("Crossref 参考文献列表", len(cr_refs) == 2 and cr_refs[1].ref.year == 2019, str(cr_refs))

    # 匹配
    cands = [Candidate("crossref", "Attention Is All You Need", 2017),
             Candidate("arxiv", "Unrelated Topic", 2019)]
    ref = make_ref("Vaswani. Attention is all you need. NeurIPS 2017.")
    best = pick_best(cands, ref)
    check("标题相似匹配", best is not None and "Attention" in best.title, str(best))
    check("标题完全一致", title_ratio("Attention Is All You Need", "attention is all you need.") == 1.0)
    check("标题词序区分", title_ratio("Attention Is All You Need", "Is Attention All You Need?") < 0.92,
          str(title_ratio("Attention Is All You Need", "Is Attention All You Need?")))
    check("候选去重", len(dedupe_candidates(cands + cands)) == 2)
    wrong = [Candidate("crossref", "AligNART: Non-autoregressive Neural Machine Translation by "
                       "Jointly Learning to Estimate Alignment and Translate", 2021)]
    bahdanau = make_ref("Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly "
                        "learning to align and translate. ICLR 2015.")
    check("拒绝年份不符的弱匹配", pick_best(wrong, bahdanau) is None)
    check("接受同题异写", pick_best(
        [Candidate("crossref", "Attention Is All You Need", 2017)], ref) is not None)

    # 评分
    def q_for(record: RefRecord) -> Quality:
        return heuristic_quality(record)

    method_rec = RefRecord(
        ref=make_ref("X. Paper. 2020."),
        best=Candidate("crossref", "Paper", 2020, citation_count=500, venue="NeurIPS", doi="10.1/x"),
        contexts=["We adopt the encoder of this work and extend it to multi-head attention."],
        intents=["methodology"], influential=True,
    )
    list_rec = RefRecord(
        ref=make_ref("Y. Paper. 2020."),
        best=Candidate("crossref", "Paper", 2020, citation_count=50, venue="arXiv"),
        contexts=["A number of works have studied this problem [1, 2, 3, 4]."],
        intents=["background"],
    )
    no_ctx = RefRecord(
        ref=make_ref("Z. Paper. 2020."),
        best=Candidate("crossref", "Paper", 2020, citation_count=800, venue="ICML", doi="10.1/z"),
    )
    qm, ql, qn = q_for(method_rec), q_for(list_rec), q_for(no_ctx)
    check("方法性引用高分", qm.score >= 0.7, str(qm.score))
    check("罗列式引用低分", ql.score < 0.4, str(ql.score))
    check("方法性理由含证据", "采用" in qm.reason or "methodology" in qm.reason, qm.reason)
    check("罗列理由明确", "罗列" in ql.reason, ql.reason)
    check("无上下文分数封顶", qn.score <= 0.62, str(qn.score))
    check("无上下文标注置信度", "置信度低" in qn.reason, qn.reason)
    check("无上下文 kind", qn.kind == "metadata_only", qn.kind)
    check("并列引用计数", cite_group_size("see [1, 2–4] and [5]") == 4, str(cite_group_size("see [1, 2–4] and [5]")))
    check("venue 分层", venue_score("NeurIPS") > venue_score("") > venue_score("arXiv"))
    check("influential 加分", q_for(RefRecord(
        ref=method_rec.ref, best=method_rec.best, contexts=method_rec.contexts[:1],
        intents=["background"], influential=True,
    )).score > q_for(RefRecord(
        ref=method_rec.ref, best=method_rec.best, contexts=method_rec.contexts[:1],
        intents=["background"], influential=False,
    )).score)

    # LLM 融合与护栏
    judge = _StubJudge([{"index": 0, "score": 0.95, "kind": "method", "reason": "上下文明确采用其编码器"}])
    judged = llm_judge([method_rec], judge)
    q_llm = finalize_quality(method_rec, qm, judged.get(0))
    check("LLM 判定被采纳", q_llm.score > qm.score, f"{q_llm.score} vs {qm.score}")
    ref_ctx_rec = RefRecord(ref=make_ref("K. Paper. 2020.", context="We adopt its loss function."),
                            best=Candidate("crossref", "Paper", 2020))
    judge_ctx = _StubJudge([{"index": 0, "score": 0.8, "kind": "method", "reason": "采用其损失函数"}])
    llm_judge([ref_ctx_rec], judge_ctx)
    check("LLM 收到用户上下文", "loss function" in judge_ctx.last_user, judge_ctx.last_user[:80])
    judge2 = _StubJudge([{"index": 0, "score": 0.98, "kind": "background", "reason": "看起来很重要"}])
    q_guard = finalize_quality(list_rec, ql, llm_judge([list_rec], judge2).get(0))
    check("罗列护栏压分", q_guard.score <= 0.5, str(q_guard.score))
    check("MockLLM 不参与", llm_judge([method_rec], type("M", (), {"model_name": "mock-llm", "chat_json": lambda *a, **k: {}})()) == {})
    bad = _StubJudge([{"index": 0, "score": "oops", "kind": "method", "reason": "x"}])
    check("非法 LLM 分数被忽略", llm_judge([method_rec], bad) == {})

    # 端到端（离线 fixture）
    prov = FixtureProvider()
    out = audit("Attention Is All You Need", provider=prov, min_score=0.55, use_llm=False)
    check("输出顶层字段", set(out) == {"fetched", "citation_quality"}, str(set(out)))
    check("条目数一致", len(out["fetched"]) == len(out["citation_quality"]) == 9)
    check("评分字段齐全", all({"ref", "score", "reason", "keep"} <= set(x) for x in out["citation_quality"]))
    check("有保留有剔除",
          any(x["keep"] for x in out["citation_quality"])
          and any(not x["keep"] for x in out["citation_quality"]))
    check("标记与阈值一致",
          all((x["score"] >= 0.55) == x["keep"] for x in out["citation_quality"]))
    check("排序稳定", [x["ref"] for x in out["citation_quality"]]
          == [x["ref"] for x in out["fetched"]])
    check("方法性引用被保留", out["citation_quality"][0]["keep"] is True)
    check("罗列式引用被剔除", out["citation_quality"][3]["keep"] is False)
    check("无上下文低价值被剔除", out["citation_quality"][8]["keep"] is False)
    check("理由为中文", all(re.search(r"[一-鿿]", x["reason"]) for x in out["citation_quality"]))

    # 列表模式（fixture provider 按标题匹配）
    out2 = audit(
        "[1] Bahdanau. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015.\n"
        "[2] Tiny. A Preliminary Study of Tiny Attention Variants. 2022. || A number of works [1, 2].",
        provider=prov, mode="refs", min_score=0.55, use_llm=False,
    )
    check("列表模式可运行", len(out2["fetched"]) == 2)
    check("列表模式有查询式", bool(out2["fetched"][0]["queries"]))
    check("列表模式保留实质引用", out2["citation_quality"][0]["keep"] is True, str(out2["citation_quality"][0]))
    check("列表模式剔除低价值", out2["citation_quality"][1]["keep"] is False, str(out2["citation_quality"][1]))

    print(f"\nselftest: {checks - fails}/{checks} 通过")
    return 0 if fails == 0 else 1


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════


def _demo() -> dict[str, Any]:
    prov = FixtureProvider()
    out = audit("Attention Is All You Need", provider=prov, use_llm=False)
    return out


def _emit(out: dict[str, Any], args: argparse.Namespace) -> int:
    """打印 JSON，并按需写 --out / --library。"""
    if args.no_queries:
        for row in out["fetched"]:
            row.pop("queries", None)
    payload = json.dumps(out, ensure_ascii=False, indent=2)
    print(payload)
    try:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")
        if args.library:
            kept = [row for row in out["fetched"] if row.get("kept")]
            with open(args.library, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(kept, ensure_ascii=False, indent=2) + "\n")
    except OSError as exc:
        print(f"写文件失败：{exc}", file=sys.stderr)
        return 1
    if args.debug:
        kept_n = sum(1 for x in out["citation_quality"] if x["keep"])
        print(f"共 {len(out['citation_quality'])} 条，保留 {kept_n} 条，"
              f"剔除 {len(out['citation_quality']) - kept_n} 条", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(
        prog="python -m app.citation_audit",
        description="论文引用抓取与含金量审计",
    )
    ap.add_argument("text", nargs="?", help="论文标题或参考文献列表")
    ap.add_argument("--input", "-i", metavar="PATH", help="输入文件（- 表示 stdin）")
    ap.add_argument("--file", "-f", metavar="PATH", help="同 --input")
    ap.add_argument("--title", help="强制按论文标题处理")
    ap.add_argument("--mode", choices=["auto", "title", "refs"], default="auto")
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="最多处理多少条引用（0=全部）")
    ap.add_argument("--sources", default="crossref,arxiv",
                    help="列表模式的数据源：crossref,arxiv,s2（逗号分隔）")
    ap.add_argument("--out", metavar="PATH", help="完整结果写入文件")
    ap.add_argument("--library", metavar="PATH", help="只把 kept=true 的文献写入文件")
    ap.add_argument("--no-llm", action="store_true", help="只用启发式，不调用大模型")
    ap.add_argument("--no-queries", action="store_true", help="输出中省略 queries")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.demo:
        return _emit(_demo(), args)

    text = args.title or ""
    if not text:
        path = args.file or args.input
        if path == "-":
            text = sys.stdin.read()
        elif path:
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError as exc:
                print(f"读取文件失败：{exc}", file=sys.stderr)
                return 1
        elif args.text:
            text = args.text
        elif not sys.stdin.isatty():
            text = sys.stdin.read()

    if not text.strip():
        print("请提供论文标题或参考文献列表（位置参数 / --file / --title / stdin）", file=sys.stderr)
        return 1

    if not 0.0 <= args.min_score <= 1.0:
        print("--min-score 必须在 0..1", file=sys.stderr)
        return 1
    if args.limit < 0:
        print("--limit 不能为负", file=sys.stderr)
        return 1

    mode = "title" if args.title else args.mode
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    unknown = set(sources) - {"crossref", "arxiv", "s2"}
    if not sources or unknown:
        print(f"--sources 只能取 crossref,arxiv,s2：{','.join(sorted(unknown))}", file=sys.stderr)
        return 1

    mailto = s2_key = ""
    try:
        from app.config import get_settings
        st = get_settings()
        mailto = getattr(st, "CROSSREF_MAILTO", "") or ""
        s2_key = getattr(st, "SEMANTIC_SCHOLAR_KEY", "") or ""
    except Exception:  # noqa: BLE001
        pass
    provider = ApiProvider(HttpClient(timeout=args.timeout, mailto=mailto, s2_key=s2_key))

    llm = None
    if not args.no_llm:
        try:
            from app.providers.llm import get_llm
            llm = get_llm()
        except Exception as exc:  # noqa: BLE001
            print(f"LLM 初始化失败，使用启发式：{exc}", file=sys.stderr)

    try:
        out = audit(
            text,
            mode=mode,
            provider=provider,
            llm=llm,
            sources=sources,
            min_score=args.min_score,
            limit=args.limit,
            use_llm=not args.no_llm,
        )
    except (ValueError, RetrievalError) as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1

    if llm is not None and getattr(llm, "model_name", "") == "mock-llm" and args.debug:
        print("提示：未配置 LLM，引用判定使用启发式规则。", file=sys.stderr)
    if out["citation_quality"] and not any(q["context_used"] for q in out["citation_quality"]):
        print("提示：没有取到任何引用上下文，只能按元数据低置信度评分。"
              "配置 SEMANTIC_SCHOLAR_KEY，或用 `引用 || 上下文` 提供上下文。", file=sys.stderr)

    return _emit(out, args)


if __name__ == "__main__":
    raise SystemExit(main())
