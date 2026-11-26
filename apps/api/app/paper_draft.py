"""学术写作助手：根据 idea 与**给定的真实检索结果**生成论文框架草稿。

写作边界
────────
1. 只引用 `retrieved_refs` 中同时具备标题和可解析来源链接的条目；
   缺标题、缺链接、重复的条目会剔除，并在文末说明原因。
2. 不调用 LLM、不联网、不引入检索结果之外的文献；参考文献不编造。
3. 不生成任何具体实验数值。需要实验数据支撑的句子用
   `[需补充实验数据#A1：...]` 标记，文末汇总成清单。
4. 文献描述只依据题录与摘要；没有摘要时只按标题归类，不推断论文结论。

输入
────
::

    {
      "idea": "面向长文档问答的检索增强生成……",
      "retrieved_refs": [
        {"title": "Attention Is All You Need", "authors": ["Ashish Vaswani"],
         "year": 2017, "venue": "NeurIPS", "arxiv_id": "1706.03762",
         "url": "https://arxiv.org/abs/1706.03762",
         "abstract": "……", "keywords": ["transformer"]}
      ]
    }

`retrieved_refs` 也接受 `{"references":[...]}`、`{"fetched":[...]}`、
`{"papers":[...]}` 等常见检索/审计输出形状。每条至少要有标题，以及
`url` / `doi` / `arxiv_id` 之一；只有 DOI 或 arXiv 编号时会构造标准解析链接。

输出
────
Markdown：Abstract、Introduction、Related Work 的详细提纲与草稿，
后续章节建议，需补充实验数据清单，以及真实参考文献列表。

用法
────
::

    python -m app.paper_draft --selftest
    python -m app.paper_draft --demo
    python -m app.paper_draft -i input.json --out draft.md
    python -m app.paper_draft --idea "……" --refs refs.json
    python -m app.paper_draft -i input.json --json --strict
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

# ── 校验用正则 ──────────────────────────────────────────────────────────
CITE_RE = re.compile(r"\[@([^\[\]\s]+)\]")
REF_LINE_RE = re.compile(r"^- \*\*\[@([^\[\]]+)\]\*\*", re.M)
NEED_RE = re.compile(r"\[需补充实验数据#([A-Z]\d+)：([^\]]+)\]")
URL_RE = re.compile(r"https?://[^\s<>()]+", re.I)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
ARXIV_RE = re.compile(r"\barXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
SPACE_RE = re.compile(r"\s+")
SENT_RE = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)\s+|\n+")

# 只用于分组/取标题词，不参与任何结论生成。
STOP = {
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "with", "via",
    "using", "from", "by", "as", "at", "or", "is", "are", "be", "been",
    "being", "was", "were", "we", "our", "this", "that", "these", "those",
    "it", "its", "their", "not", "all", "more", "than", "into", "over",
    "under", "between", "through", "about", "across", "can", "could", "may",
    "might", "will", "would", "should", "must", "do", "does", "did", "has",
    "have", "had", "how", "what", "why", "when", "where", "new", "novel",
    "learning", "model", "models", "method", "methods", "approach",
    "approaches", "paper", "study", "analysis", "based", "large", "towards",
    "toward", "problem", "problems", "task", "tasks", "data",
    "result", "results", "propose", "proposed", "introduce", "introduced",
    "present", "presented", "show", "shows", "also", "however", "thus",
    "such", "leveraging", "one",
    "two", "three", "first", "second", "third", "using", "use", "used",
    "研究", "本文", "方法", "问题", "模型", "工作", "文献", "结果", "实验",
    "提出", "基于", "面向", "一种", "我们", "以及", "通过", "进行", "可以",
    "相关", "现有", "然而", "因此", "其中", "这个", "这些", "那些", "不同",
    "主要", "进一步", "使用", "采用", "一个", "两种", "能够", "需要", "具有",
    "针对", "对于", "由于", "如果", "但是", "并且", "或者", "已经", "仍然",
    "目前", "未来", "方向", "分析", "系统", "框架", "任务", "数据", "性能",
    "效果", "提升", "改进", "增强", "场景", "设定", "目标", "贡献", "结论",
    "背景", "挑战",
}

# 自检用：生成文本里不允许出现的夸大措辞（引文和标题会先剔除）。
HYPE = (
    "颠覆", "革命性", "划时代", "史无前例", "完美", "最强", "无敌",
    "无可匹敌", "遥遥领先", "碾压", "完胜", "突破性", "显著优于",
    "大幅超越", "state-of-the-art", "groundbreaking", "revolutionary",
)


# ════════════════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════════════════


@dataclass
class Ref:
    """一条通过白名单校验的真实文献。"""

    key: str
    title: str
    url: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "keywords": self.keywords,
            "source": self.source,
        }


@dataclass
class Idea:
    """用户 idea 的归一化结果。字典输入可携带更具体的字段。"""

    text: str
    title: str
    method: str = ""
    problem: str = ""
    dataset: str = ""
    baselines: str = ""
    metrics: str = ""
    contributions: list[str] = field(default_factory=list)

    @classmethod
    def from_any(cls, obj: Any) -> "Idea":
        if isinstance(obj, dict):
            text = _clean(
                obj.get("idea")
                or obj.get("description")
                or obj.get("text")
                or obj.get("summary")
                or obj.get("title")
                or ""
            )
            title = _clean_title(obj.get("title") or "") or _title_from_text(text)
            method = _clean(obj.get("method") or obj.get("approach") or "")
            problem = _clean(obj.get("problem") or obj.get("challenge") or "")
            dataset = _clean(
                obj.get("dataset") or obj.get("datasets") or obj.get("benchmark") or ""
            )
            baselines = _clean(obj.get("baselines") or obj.get("baseline") or "")
            metrics = _clean(obj.get("metrics") or obj.get("evaluation") or "")
            raw_contrib = obj.get("contributions") or []
            if isinstance(raw_contrib, str):
                raw_contrib = [raw_contrib]
            contributions = [_clean(x) for x in raw_contrib if _clean(x)]
        else:
            text = _clean(obj)
            title = _title_from_text(text)
            method = problem = dataset = baselines = metrics = ""
            contributions = []

        if not text:
            raise ValueError("idea 为空：请提供 --idea 或输入 JSON 中的 idea 字段")
        return cls(text=text, title=title or "未命名研究", method=method, problem=problem,
                   dataset=dataset, baselines=baselines, metrics=metrics,
                   contributions=contributions)


# ════════════════════════════════════════════════════════════════════
# 文本工具
# ════════════════════════════════════════════════════════════════════


def _clean(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).replace(" ", " ")
    s = re.sub(r"[\r\n\t]+", " ", s)
    return SPACE_RE.sub(" ", s).strip()


def _clip(text: str, limit: int) -> str:
    text = _clean(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;；，。") + "…"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text or "") if s and s.strip()]


def _title_from_text(text: str) -> str:
    s = _clean(text)
    if not s:
        return ""
    s = re.sub(r"^\s*#+\s*", "", s)
    s = re.sub(r"^(?:idea|topic|title|题目|标题|研究设想)\s*[:：]\s*", "", s, flags=re.I)
    first = _sentences(s)[0] if _sentences(s) else s
    first = first.strip(" .;；:：。！？!?")
    return _clip(first, 80) or "未命名研究"


def _sanitize_md(text: str) -> str:
    """防止题名/摘要里的方括号被误识别成引用或待补标记。"""
    s = _clean(text)
    s = s.replace("[@", "[ @")
    s = s.replace("[需补充实验数据#", "[需补充实验数据 #")
    return s


def _clean_title(value: Any) -> str:
    s = _sanitize_md(value)
    s = s.strip(" .;；:：")
    s = re.sub(r"^#+\s*", "", s)
    return s


def _norm_title(value: Any) -> str:
    return re.sub(r"[^0-9a-z一-鿿]+", " ", str(value or "").lower()).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    return [value]


def _year(value: Any) -> int | None:
    if isinstance(value, int) and 1900 <= value <= 2100:
        return value
    m = YEAR_RE.search(str(value or ""))
    if not m:
        return None
    year = int(m.group(0))
    return year if 1900 <= year <= 2100 else None


def _author_names(value: Any) -> list[str]:
    names: list[str] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            name = item.get("name") or item.get("full_name") or ""
            if not name:
                given = _clean(item.get("given") or item.get("first") or "")
                family = _clean(item.get("family") or item.get("last") or "")
                name = " ".join(x for x in (given, family) if x)
        else:
            name = _clean(item)
        name = re.sub(r"\s+et\s+al\.?$", "", name, flags=re.I).strip(" ,;；")
        if name and name not in names:
            names.append(name)
    return names


def _keywords(value: Any) -> list[str]:
    out: list[str] = []
    for item in _as_list(value):
        for part in re.split(r"[;,；、]", _clean(item)):
            part = part.strip()
            if part and part.lower() not in {x.lower() for x in out}:
                out.append(part)
    return out


def _valid_link(value: str) -> bool:
    if not value or any(c.isspace() or c in '<>"' for c in value):
        return False
    try:
        u = urlparse(value)
    except ValueError:
        return False
    return u.scheme in ("http", "https") and bool(u.netloc)


def _doi_link(value: str) -> str:
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I)
    doi = doi.strip().strip(" .;；")
    if DOI_RE.fullmatch(doi):
        return f"https://doi.org/{doi}"
    return ""


def _arxiv_link(value: str) -> str:
    s = re.sub(r"^(?:arxiv:\s*|https?://arxiv\.org/(?:abs|pdf)/)", "", value, flags=re.I)
    s = re.sub(r"\.pdf$", "", s.strip(), flags=re.I)
    m = re.match(r"(\d{4}\.\d{4,5})(?:v\d+)?", s)
    return f"https://arxiv.org/abs/{m.group(1)}" if m else ""


def _link_from_value(value: str) -> str:
    value = _clean(value).strip("<>")
    if not value:
        return ""
    if value.startswith("//"):
        value = "https:" + value
    if value.startswith("www."):
        value = "https://" + value
    if re.match(r"https?://", value, re.I) and _valid_link(value):
        return value
    return _doi_link(value) or _arxiv_link(value)


def _link_for(raw: dict[str, Any]) -> str:
    for key in ("url", "link", "source_url", "html_url", "pdf_url", "paper_url"):
        link = _link_from_value(_clean(raw.get(key)))
        if link:
            return link

    candidate = _clean(raw.get("doi"))
    if candidate:
        link = _doi_link(candidate)
        if link:
            return link

    candidate = _clean(raw.get("arxiv_id") or raw.get("arxiv"))
    if candidate:
        link = _arxiv_link(candidate)
        if link:
            return link

    # 字符串输入或 raw 字段里可能直接带链接。
    blob = _clean(raw.get("raw") or "")
    if blob:
        m = URL_RE.search(blob)
        if m and _valid_link(m.group(0)):
            return m.group(0)
        m = DOI_RE.search(blob)
        if m:
            return f"https://doi.org/{m.group(0)}"
        m = ARXIV_RE.search(blob)
        if m:
            return f"https://arxiv.org/abs/{m.group(1)}"
    return ""


def _guess_title(raw: str) -> str:
    """从字符串型检索结果里尽量取标题；失败时返回空，绝不拼造。"""
    s = URL_RE.sub(" ", raw or "")
    s = DOI_RE.sub(" ", s)
    s = re.sub(r"arXiv:\s*\S+", " ", s, flags=re.I)
    s = re.sub(r"^\s*(?:\[\s*\d+\s*\]|\(\s*\d+\s*\)|\d+\s*[.)、])\s*", "", s)
    s = _clean(s)
    if not s:
        return ""
    if "|" in s:
        s = s.split("|", 1)[0].strip()
    m = re.search(r"[“\"]([^”\"]{8,200})[”\"]", s)
    if m:
        return _clean_title(m.group(1))
    parts = [p.strip(" .;；") for p in re.split(r"\.\s+|。\s*", s) if p.strip()]
    candidates = [p for p in parts if 2 <= len(re.findall(r"[A-Za-z0-9一-鿿]+", p)) <= 40]
    return _clean_title(max(candidates, key=len) if candidates else s)


def _theme_for(ref: Ref) -> str:
    if ref.keywords:
        for kw in ref.keywords:
            if kw and kw.lower() not in STOP:
                return _clip(kw, 24)
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", ref.title)
             if w.lower() not in STOP]
    if words:
        return words[0][:1].upper() + words[0][1:]
    return _clip(ref.title, 18)


def _quote_sentence(abstract: str) -> str:
    """从摘要里挑一句贡献/方法句，原样引用；不做改写。"""
    sents = [s for s in _sentences(abstract) if 25 <= len(s) <= 260]
    if not sents:
        return ""
    cues = ("we propose", "we introduce", "we present", "we develop", "we design",
            "this paper", "in this work", "提出", "本文", "介绍", "设计", "构建")
    scored = []
    for idx, sent in enumerate(sents):
        low = sent.lower()
        score = sum(1 for cue in cues if cue in low)
        if abs(len(sent) - 140) < abs(len(sents[0]) - 140):
            score += 0.1
        scored.append((score, -idx, sent))
    best = max(scored, key=lambda x: (x[0], x[1]))[2]
    best = best.replace("“", "'").replace("”", "'").replace('"', "'")
    return _clip(best, 260)


def _display_authors(ref: Ref) -> str:
    if not ref.authors:
        return "作者未提供"
    first = ref.authors[0]
    return f"{first} 等" if len(ref.authors) > 1 else first


def _ref_line(ref: Ref) -> str:
    authors = [a.rstrip(".") for a in ref.authors[:6]] or ["作者未提供"]
    if len(ref.authors) > 6:
        authors.append("等")
    author_text = ", ".join(authors)
    title = ref.title.rstrip(".")
    meta = []
    if ref.venue:
        meta.append(ref.venue)
    if ref.year:
        meta.append(str(ref.year))
    tail = ", ".join(meta) if meta else "出版信息未提供"
    return f"- **[@%s]** %s. %s. %s. [来源](<%s>)" % (
        ref.key, author_text, title, tail, ref.url
    )


# ════════════════════════════════════════════════════════════════════
# 参考文献白名单
# ════════════════════════════════════════════════════════════════════


def extract_refs(data: Any) -> list[Any]:
    """兼容常见检索/审计输出；找不到返回空列表，由上层显式降级。"""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("retrieved_refs", "references", "refs", "papers", "fetched",
                    "items", "results", "candidates"):
            value = data.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return list(value.values())
        if any(k in data for k in ("title", "url", "doi", "arxiv_id")):
            return [data]
    return []


def _ref_from_item(item: Any, index: int) -> Ref:
    if isinstance(item, str):
        raw: dict[str, Any] = {"raw": item}
    elif isinstance(item, dict):
        raw = item
    else:
        raise ValueError(f"第 {index} 条不是对象或字符串")

    title = _clean_title(
        raw.get("title") or raw.get("name") or raw.get("paper_title") or ""
    )
    if not title:
        title = _guess_title(_clean(raw.get("raw") or ""))
    if not title or len(re.sub(r"[^0-9A-Za-z一-鿿]", "", title)) < 3:
        raise ValueError("缺少可用标题")

    url = _link_for(raw)
    if not url:
        raise ValueError("缺少可访问的来源链接（url / doi / arxiv_id）")

    abstract = _sanitize_md(raw.get("abstract") or raw.get("summary") or raw.get("tldr") or "")
    if len(abstract) > 4000:
        abstract = abstract[:4000].rstrip() + "…"
    return Ref(
        key="",
        title=title,
        url=url,
        authors=_author_names(raw.get("authors") or raw.get("author") or raw.get("creators")),
        year=_year(raw.get("year") or raw.get("publication_year") or raw.get("published")
                   or raw.get("date") or raw.get("published_date")),
        venue=_clean(raw.get("venue") or raw.get("journal") or raw.get("conference")
                     or raw.get("booktitle") or raw.get("container_title") or ""),
        doi=re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "",
                   _clean(raw.get("doi") or ""), flags=re.I).strip(" .;；"),
        arxiv_id=_clean(raw.get("arxiv_id") or raw.get("arxiv") or ""),
        abstract=abstract,
        keywords=_keywords(raw.get("keywords") or raw.get("tags") or raw.get("topics")),
        source=_clean(raw.get("source") or raw.get("provider") or raw.get("database") or ""),
    )


def _key_base(ref: Ref, index: int) -> str:
    if ref.authors:
        first = ref.authors[0]
        surname_src = first.split(",", 1)[0] if "," in first else first.split()[-1]
        surname = re.sub(r"[^A-Za-z0-9]", "", surname_src.lower())
        base = surname
    else:
        base = ""
    if not base:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", ref.title)
                 if w.lower() not in STOP]
        base = "".join(w.lower() for w in words[:2])
    if not base:
        base = f"ref{index}"
    if not base[0].isalpha():
        base = "r" + base
    if ref.year:
        base += str(ref.year)
    return base[:48]


def normalize_refs(items: Sequence[Any]) -> tuple[list[Ref], list[dict[str, Any]]]:
    """校验并去重。只有带标题 + 可访问链接的条目才进入引用白名单。"""
    refs: list[Ref] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    used_keys: set[str] = set()

    for index, item in enumerate(items, 1):
        try:
            ref = _ref_from_item(item, index)
        except ValueError as exc:
            raw_title = ""
            if isinstance(item, dict):
                raw_title = _clean(item.get("title") or item.get("name") or "")
            elif isinstance(item, str):
                raw_title = _guess_title(item)
            rejected.append({
                "index": index,
                "title": _clip(raw_title, 60) or "（无标题）",
                "reason": str(exc),
            })
            continue

        identities = []
        if ref.doi:
            identities.append("doi:" + ref.doi.lower())
        if ref.arxiv_id:
            identities.append("arxiv:" + ref.arxiv_id.lower())
        identities.append("title:" + _norm_title(ref.title))
        duplicate = next((k for k in identities if k in seen), "")
        if duplicate:
            rejected.append({
                "index": index,
                "title": _clip(ref.title, 60),
                "reason": f"与第 {seen[duplicate]} 条重复，已保留先出现的版本",
            })
            continue

        base = _key_base(ref, index)
        key = base
        n = 2
        while key in used_keys:
            key = f"{base}{n}"
            n += 1
        ref.key = key
        used_keys.add(key)
        for ident in identities:
            seen[ident] = index
        refs.append(ref)

    return refs, rejected


# ════════════════════════════════════════════════════════════════════
# 草稿生成
# ════════════════════════════════════════════════════════════════════


def _need(marker: str, text: str) -> str:
    return f"[需补充实验数据#{marker}：{_clean(text).replace(']', '）')}]"


def _todo(text: str) -> str:
    return f"[需补充：{_clean(text).replace(']', '）')}]"


def _cite(keys: Iterable[str]) -> str:
    return "".join(f"[@{k}]" for k in keys if k)


def _theme_groups(refs: Sequence[Ref], limit: int = 6) -> list[tuple[str, list[Ref]]]:
    groups: dict[str, list[Ref]] = {}
    for ref in refs:
        groups.setdefault(_theme_for(ref), []).append(ref)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if len(ordered) <= limit:
        return ordered
    main = ordered[:limit - 1]
    rest = [r for _, group in ordered[limit - 1:] for r in group]
    return main + [("其他相关工作", rest)]


def _ref_sentence(ref: Ref) -> str:
    quote = _quote_sentence(ref.abstract)
    who = _display_authors(ref)
    if quote:
        return f"{who}的《{ref.title}》在摘要中指出：“{quote}” [@{ref.key}]。"
    return (f"{who}的《{ref.title}》涉及该主题 [@{ref.key}]；"
            "该条未提供摘要，具体方法与结论需核对原文。")


def _related_draft(refs: Sequence[Ref]) -> list[str]:
    if not refs:
        return [
            "当前未提供可核验的真实检索结果。为避免编造文献，本节不生成具体文献讨论；"
            "请在补充 `retrieved_refs` 后重新生成。",
            "",
        ]
    lines = [
        f"本节仅依据输入中的 {len(refs)} 条可核验检索结果组织，按题名、摘要或关键词归类；"
        "未引入检索结果之外的文献。",
        "",
    ]
    groups = _theme_groups(refs)
    for theme, group in groups:
        body = " ".join(_ref_sentence(r) for r in group)
        lines.append(f"**{theme}。** {body}")
        lines.append("")
    themes = "、".join(theme for theme, _ in groups)
    lines.append(
        f"综上，上述检索结果覆盖了 {themes} 等主题。就可获得的题录与摘要而言，"
        "这些工作与本文设想存在主题关联；本文的具体差异点仍需结合原文与实验进一步确认。"
        "该判断仅基于所给材料，不代表对完整文献空间的穷尽检索。"
    )
    lines.append("")
    return lines


def _idea_sentence(idea: Idea) -> str:
    text = idea.text.strip()
    if not text.endswith(("。", "！", "？", ".", "!", "?")):
        text += "。"
    return _clip(text, 320)


def _method_sentence(idea: Idea) -> str:
    if idea.method:
        problem = f"针对{idea.problem}，" if idea.problem else ""
        return f"{problem}本文拟采用{idea.method}，围绕上述问题构建可检验的方法框架。"
    problem = f"针对{idea.problem}，" if idea.problem else ""
    return (f"{problem}本文拟设计一套可检验的方法框架，"
            f"{_todo('方法的核心机制、模块组成与命名')}。")


def _abstract_draft(idea: Idea) -> list[str]:
    dataset = idea.dataset or _need("A1", "数据集/基准名称、版本与划分方式")
    baselines = idea.baselines or _need("A2", "基线方法及其实现来源")
    metrics = idea.metrics or _need("A3", "评价指标及其统计口径")
    same = _norm_title(idea.text) == _norm_title(idea.title)
    opening = f"本文围绕“{idea.title}”展开研究。"
    if not same:
        opening += _idea_sentence(idea)
    return [
        f"{opening}{_method_sentence(idea)}",
        "",
        f"实验部分计划在 {dataset} 上，与 {baselines} 进行比较，并报告 {metrics}。"
        f"主要实验结果 {_need('A4', '主实验结果、对比数值与统计显著性')} "
        "将用于检验所提框架是否达到预期目标。",
        "",
        "最后，本文将总结方法在目标场景中的适用条件与局限，并讨论仍需进一步研究的问题。",
        "",
    ]


def _intro_draft(idea: Idea, refs: Sequence[Ref]) -> list[str]:
    same = _norm_title(idea.text) == _norm_title(idea.title)
    opening = f"本文研究“{idea.title}”。"
    if not same:
        opening += _idea_sentence(idea)
    lines = [opening, ""]

    if refs:
        groups = _theme_groups(refs)
        bits = []
        for theme, group in groups:
            bits.append(f"{theme}（{_cite(r.key for r in group)}）")
        lines.append(
            "围绕上述问题，本文检索到的相关研究涉及 " + "、".join(bits) + "。"
            "这些工作构成了本文的直接背景。本文的关注点在于 idea 中的具体设想，"
            "而不是重复上述研究；与这些工作的具体差异将在系统检索、原文核对"
            "和实验完成后进一步明确。"
        )
    else:
        lines.append(
            "当前未提供可核验的真实检索结果，因此本文不在此处列举具体文献；"
            "相关工作的系统梳理需在补充检索后完成。"
        )
    lines += [
        "",
        "本文将问题形式化为：给定 "
        f"{_todo('输入数据形态、可用信息与约束条件')}，输出 "
        f"{_todo('预测或生成目标')}；评价目标是 "
        f"{_todo('评价指标、统计口径与比较方式')}。",
        "",
        "为回答上述问题，本文拟采用如下研究路径："
        f"{_todo('方法总览与关键模块')}。方法设计将围绕 idea 中的核心设想展开，"
        "不预先假定其一定优于现有方案。",
        "",
        "本文的贡献初步概括为：",
        "",
    ]
    if idea.contributions:
        for i, contrib in enumerate(idea.contributions, 1):
            lines.append(f"{i}. {contrib}")
    else:
        lines += [
            "1. 问题层面：明确本文要检验的研究问题、输入输出与评价目标。",
            "2. 方法层面：提出或整理一套可复现的方法框架，并说明其关键设计。",
            "3. 实验层面：在真实数据上开展主实验、消融与对比分析。",
        ]
    lines += [
        "",
        f"其中，实验层面的贡献需要由真实结果支撑：{_need('I1', '主实验设置、对比结果与结论边界')}。"
        "在数据补齐前，本文不声称任何性能优势。",
        "",
        f"实验部分还将报告消融、效率或案例分析结果：{_need('I2', '消融/效率/案例分析的具体设置与结果')}。",
        "",
    ]
    return lines


def _abstract_outline(idea: Idea) -> list[str]:
    return [
        "- 背景与问题：用 1–2 句说明本文关注的场景与核心问题。",
        "- 研究缺口：只写作者准备检验的缺口，不把未核实的“已有工作都没解决”写成事实。",
        f"- 方法：概述{idea.method or '拟采用的方法'}的关键机制。",
        "- 实验：给出数据集、基线、指标与主要结果；结果未产生时保留数据标记。",
        "- 结论：在结果补齐后总结发现、适用条件与局限。",
    ]


def _intro_outline(idea: Idea, refs: Sequence[Ref]) -> list[str]:
    n = len(refs)
    return [
        "- 1.1 背景与动机：说明研究场景、为什么值得研究，以及本文的出发点。",
        "- 1.2 问题定义：写清输入、输出、约束与评价目标。",
        f"- 1.3 现有研究：仅引用提供的 {n} 条真实检索结果，按主题归纳，不补造文献。",
        "- 1.4 本文思路：说明方法框架与关键假设。",
        "- 1.5 贡献：区分“方法/问题贡献”与“需要实验支撑的贡献”。",
        "- 1.6 实验预览：列出主实验、消融与效率/案例分析，但不写具体数值。",
    ]


def _related_outline(refs: Sequence[Ref]) -> list[str]:
    if not refs:
        return ["- 当前没有可核验的真实文献；本节暂不展开，待补充 `retrieved_refs`。"]
    groups = _theme_groups(refs)
    lines = []
    for i, (theme, group) in enumerate(groups, 1):
        titles = "；".join(_clip(r.title, 42) for r in group[:3])
        if len(group) > 3:
            titles += " 等"
        lines.append(f"- 2.{i} {theme}：{titles}。")
    lines.append(f"- 2.{len(groups) + 1} 小结：归纳主题覆盖范围，说明本文定位与检索边界。")
    return lines


def _later_sections() -> list[str]:
    return [
        "## 3. 后续章节建议",
        "",
        "- **3 方法**：问题形式化、方法总览、关键模块、复杂度与实现细节。",
        "- **4 实验**：数据集与基线、主实验、消融、效率/案例分析、局限讨论。",
        "- **5 结论**：回答研究问题、总结贡献、说明适用条件与未来工作。",
        "",
    ]


def _collect_needs(markdown: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    section = ""
    for line in markdown.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        for match in NEED_RE.finditer(line):
            marker, need = match.group(1), match.group(2).strip()
            if marker in seen:
                continue
            seen.add(marker)
            start, end = match.span()
            left = max(line.rfind(p, 0, start) for p in "。！？.!?") + 1
            rights = [line.find(p, end) for p in "。！？.!?"]
            rights = [x for x in rights if x >= 0]
            right = min(rights) + 1 if rights else len(line)
            sentence = _clip(line[left:right].strip(), 220)
            items.append({
                "id": marker,
                "section": section or "正文",
                "need": need,
                "sentence": sentence,
            })
    return items


def _needs_section(markdown: str) -> list[str]:
    items = _collect_needs(markdown)
    lines = ["## 需补充实验数据", ""]
    if not items:
        lines += ["本稿未包含需要实验数据支撑的句子。", ""]
        return lines
    lines += [
        "以下标记必须由作者用真实实验替换；在补齐前请勿删除标记后直接投稿。",
        "",
    ]
    for item in items:
        lines.append(f"- **{item['id']}**（{item['section']}）：{item['need']}")
        lines.append(f"  - 原句：{item['sentence']}")
    lines.append("")
    return lines


def _notes_section(refs: Sequence[Ref], rejected: Sequence[dict[str, Any]],
                   cited: Sequence[str]) -> list[str]:
    lines = [
        "## 材料与校验说明",
        "",
        f"- 可用文献：{len(refs)} 条；正文引用：{len(cited)} 个键；剔除：{len(rejected)} 条。",
        "- 文献信息只来自输入；摘要缺失时只按标题归类，未推断论文结论。",
        "- 来源链接优先使用输入 URL；仅有 DOI 或 arXiv 编号时，构造对应标准解析链接。",
        "- 未调用语言模型、未联网，未引入 `retrieved_refs` 之外的文献。",
    ]
    if rejected:
        lines.append("- 剔除条目：")
        for row in rejected[:10]:
            lines.append(f"  - 第 {row['index']} 条《{row['title']}》：{row['reason']}")
        if len(rejected) > 10:
            lines.append(f"  - 其余 {len(rejected) - 10} 条略。")
    else:
        lines.append("- 未剔除此轮输入中的任何条目。")
    lines.append("")
    return lines


def _split_section(markdown: str, heading: str) -> str:
    body = markdown.split("## 参考文献")[0]
    parts = re.split(r"^## ", body, flags=re.M)
    for part in parts[1:]:
        head, _, text = part.partition("\n")
        if head.strip() == heading:
            return text
    return ""


def _ordered_cited(markdown: str) -> list[str]:
    seen: list[str] = []
    for key in CITE_RE.findall(markdown):
        if key not in seen:
            seen.append(key)
    return seen


def verify(markdown: str, refs: Sequence[Ref], rejected: Sequence[dict[str, Any]] = ()) -> list[str]:
    """引用白名单、链接、数据标记与章节完整性检查。"""
    issues: list[str] = []
    valid = {r.key for r in refs}
    body = markdown.split("## 参考文献")[0]
    cited = CITE_RE.findall(markdown)
    body_cited = set(CITE_RE.findall(body))

    undefined = sorted(set(cited) - valid)
    if undefined:
        issues.append(f"正文引用了白名单外的文献：{', '.join(undefined)}")
    if valid and body_cited != valid:
        missing = sorted(valid - body_cited)
        if missing:
            issues.append(f"有可用文献未在正文引用：{', '.join(missing)}")

    if "## 参考文献" not in markdown:
        issues.append("缺少参考文献章节")
    else:
        ref_marks = REF_LINE_RE.findall(markdown)
        if len(ref_marks) != len(set(ref_marks)):
            issues.append("参考文献存在重复键")
        missing_refs = sorted(set(cited) - set(ref_marks))
        extra_refs = sorted(set(ref_marks) - set(cited))
        if missing_refs:
            issues.append(f"正文引用未列入参考文献：{', '.join(missing_refs)}")
        if extra_refs:
            issues.append(f"参考文献中存在未被正文引用的条目：{', '.join(extra_refs)}")
        for line in markdown.splitlines():
            if REF_LINE_RE.match(line) and not re.search(r"\[来源\]\(<https?://[^>\s]+>\)", line):
                issues.append(f"参考文献缺少可访问链接：{line[:60]}")
                break

    for heading in ("## Abstract", "## 1. Introduction", "## 2. Related Work"):
        if heading not in markdown:
            issues.append(f"缺少章节：{heading}")
    if refs:
        if not CITE_RE.search(_split_section(markdown, "1. Introduction")):
            issues.append("Introduction 没有引用任何提供的真实文献")
        if not CITE_RE.search(_split_section(markdown, "2. Related Work")):
            issues.append("Related Work 没有引用任何提供的真实文献")
    elif "当前未提供可核验的真实检索结果" not in markdown:
        issues.append("没有可用文献，但正文未显式说明降级原因")

    if not refs:
        issues.append("未提供可核验的真实文献；References 为空，正文不生成引用")

    draft_only = body.split("## 需补充实验数据")[0]
    markers = [m[0] for m in NEED_RE.findall(draft_only)]
    if len(markers) != len(set(markers)):
        issues.append("实验数据标记编号重复")
    listed = set(re.findall(r"^- \*\*([A-Z]\d+)\*\*（", markdown, re.M))
    if set(markers) != listed:
        missing = sorted(set(markers) - listed)
        extra = sorted(listed - set(markers))
        if missing:
            issues.append(f"数据标记未进入清单：{', '.join(missing)}")
        if extra:
            issues.append(f"数据清单存在正文没有的标记：{', '.join(extra)}")

    if rejected:
        issues.append(f"有 {len(rejected)} 条检索结果因缺标题/链接或重复被剔除")

    # 夸大措辞检查：先去掉引文、标题和参考文献行。
    prose = re.sub(r"“[^”]*”", "", markdown)
    prose = re.sub(r"《[^》]*》", "", prose)
    prose = re.sub(r"^- \*\*\[@.*$", "", prose, flags=re.M)
    low = prose.lower()
    for word in HYPE:
        if word.lower() in low:
            issues.append(f"夸大措辞：{word}")
    return issues


def generate(idea_obj: Any, data: Any = None) -> dict[str, Any]:
    """生成论文框架。`data` 可以是完整输入对象，也可以直接是参考文献列表。"""
    if data is None and isinstance(idea_obj, dict):
        data = idea_obj
    idea = Idea.from_any(idea_obj)
    raw_refs = extract_refs(data)
    refs, rejected = normalize_refs(raw_refs)

    lines: list[str] = [
        f"# {idea.title}",
        "",
        f"> 论文框架草稿｜{len(refs)} 条带来源链接的检索结果｜引用白名单：`retrieved_refs`。",
        "> 工作标题为占位，可替换；本文件是写作框架，不是可直接投稿的定稿。",
        "",
        "## Abstract",
        "",
        "### 提纲",
        "",
    ]
    lines += _abstract_outline(idea)
    lines += ["", "### 草稿", ""]
    lines += _abstract_draft(idea)

    lines += ["## 1. Introduction", "", "### 提纲", ""]
    lines += _intro_outline(idea, refs)
    lines += ["", "### 草稿", ""]
    lines += _intro_draft(idea, refs)

    lines += ["## 2. Related Work", "", "### 提纲", ""]
    lines += _related_outline(refs)
    lines += ["", "### 草稿", ""]
    lines += _related_draft(refs)

    lines += _later_sections()

    # 先把正文拼好，再从正文里收集数据标记，避免把清单里的编号算进去。
    body = "\n".join(lines).rstrip() + "\n\n"
    lines += _needs_section(body)
    lines += _notes_section(refs, rejected, _ordered_cited(body))
    markdown_wo_refs = "\n".join(lines).rstrip() + "\n\n"

    cited = _ordered_cited(markdown_wo_refs)
    by_key = {r.key: r for r in refs}
    ref_lines = ["## 参考文献", ""]
    if cited:
        ref_lines.append("以下条目均来自输入 `retrieved_refs`，按正文首次引用顺序排列；"
                         "未使用任何外部或生成式文献。")
        ref_lines.append("")
        for key in cited:
            if key in by_key:
                ref_lines.append(_ref_line(by_key[key]))
    else:
        ref_lines.append("当前没有可核验的真实文献，故不生成参考文献列表。未编造任何条目。")
    ref_lines.append("")
    markdown = markdown_wo_refs + "\n".join(ref_lines)

    issues = verify(markdown, refs, rejected)
    needs = _collect_needs(markdown)
    return {
        "markdown": markdown,
        "idea": idea.text,
        "title": idea.title,
        "cited": cited,
        "references": [r.to_dict() for r in refs],
        "rejected": rejected,
        "needs_data": needs,
        "issues": issues,
        "stats": {
            "n_refs": len(refs),
            "n_cited": len(cited),
            "n_rejected": len(rejected),
            "n_needs_data": len(needs),
        },
    }


# ════════════════════════════════════════════════════════════════════
# 演示与自检
# ════════════════════════════════════════════════════════════════════


def demo_input() -> dict[str, Any]:
    """演示只使用真实论文题录；不填摘要，避免演示数据中出现编造引文。"""
    return {
        "idea": (
            "面向长文档问答的检索增强生成：通过自适应证据压缩与引用一致性约束，"
            "提高多跳问答的事实性与可追溯性。"
        ),
        "retrieved_refs": [
            {
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar",
                            "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez",
                            "Łukasz Kaiser", "Illia Polosukhin"],
                "year": 2017,
                "venue": "NeurIPS",
                "arxiv_id": "1706.03762",
                "url": "https://arxiv.org/abs/1706.03762",
            },
            {
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus",
                            "Fabio Petroni", "Vladimir Karpukhin", "Naman Goyal",
                            "Heinrich Küttler", "Mike Lewis", "Wen-tau Yih",
                            "Tim Rocktäschel", "Sebastian Riedel", "Douwe Kiela"],
                "year": 2020,
                "venue": "NeurIPS",
                "arxiv_id": "2005.11401",
                "url": "https://arxiv.org/abs/2005.11401",
            },
            {
                "title": "Longformer: The Long-Document Transformer",
                "authors": ["Iz Beltagy", "Matthew E. Peters", "Arman Cohan"],
                "year": 2020,
                "venue": "arXiv preprint",
                "arxiv_id": "2004.05150",
                "url": "https://arxiv.org/abs/2004.05150",
            },
            {
                "title": "Dense Passage Retrieval for Open-Domain Question Answering",
                "authors": ["Vladimir Karpukhin", "Barlas Oğuz", "Sewon Min",
                            "Patrick Lewis", "Ledell Wu", "Sergey Edunov",
                            "Danqi Chen", "Wen-tau Yih"],
                "year": 2020,
                "venue": "EMNLP",
                "arxiv_id": "2004.04906",
                "url": "https://arxiv.org/abs/2004.04906",
            },
            {
                "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
                "authors": ["Gautier Izacard", "Edouard Grave"],
                "year": 2021,
                "venue": "EACL",
                "arxiv_id": "2007.01282",
                "url": "https://arxiv.org/abs/2007.01282",
            },
            {
                "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
                "authors": ["Yunfan Gao", "Yun Xiong", "Xinyu Gao", "Kangxiang Jia",
                            "Jinliu Pan", "Yuxi Bi", "Yi Dai", "Jiawei Sun",
                            "Haofen Wang"],
                "year": 2023,
                "venue": "arXiv preprint",
                "arxiv_id": "2312.10997",
                "url": "https://arxiv.org/abs/2312.10997",
            },
        ],
    }


def _selftest() -> int:
    checks = 0
    failures = 0

    def ok(name: str, cond: bool, detail: str = "") -> None:
        nonlocal checks, failures
        checks += 1
        if cond:
            print(f"[OK] {name}")
        else:
            failures += 1
            print(f"[XX] {name}{'  → ' + detail if detail else ''}")

    # 白名单：URL / DOI / arXiv / 缺链接 / 重复
    refs, rejected = normalize_refs([
        {"title": "A Real Paper", "authors": ["Ada Lovelace"], "year": 2024,
         "url": "https://example.org/a"},
        {"title": "DOI Only Paper", "year": 2023, "doi": "10.1234/example.2023"},
        {"title": "Arxiv Only Paper", "year": 2022, "arxiv_id": "2201.00001"},
        {"title": "No Link Paper", "year": 2021},
        {"title": "A Real Paper", "year": 2024, "url": "https://example.org/a2"},
        {"title": "", "url": "https://example.org/empty"},
    ])
    ok("有效文献保留 3 条", len(refs) == 3, str(len(refs)))
    ok("缺链接条目被剔除", any("来源链接" in r["reason"] for r in rejected))
    ok("重复条目被剔除", any("重复" in r["reason"] for r in rejected))
    ok("空标题条目被剔除", any("标题" in r["reason"] for r in rejected))
    ok("DOI 构造标准链接",
       any(r.url == "https://doi.org/10.1234/example.2023" for r in refs))
    ok("arXiv 构造标准链接",
       any(r.url == "https://arxiv.org/abs/2201.00001" for r in refs))
    ok("引用键唯一", len({r.key for r in refs}) == len(refs))

    string_refs, string_rej = normalize_refs([
        "A Survey of Retrieval Augmented Generation | https://arxiv.org/abs/2312.10997",
        {"title": "Bad URL", "url": "https://example.org/a b"},
    ])
    ok("字符串条目可解析标题与链接",
       len(string_refs) == 1 and string_refs[0].url == "https://arxiv.org/abs/2312.10997")
    ok("含空格链接被剔除", len(string_rej) == 1, str(string_rej))

    # 摘要引用是原句，不是模型改写
    ref = Ref(key="demo", title="Demo", url="https://example.org/demo",
              abstract="We propose a method for long documents. It is evaluated on two tasks.")
    quote = _quote_sentence(ref.abstract)
    ok("摘要引句来自原文", quote in ref.abstract, quote)

    # 完整生成
    out = generate(demo_input())
    md = out["markdown"]
    ok("包含三个规定章节",
       all(h in md for h in ("## Abstract", "## 1. Introduction", "## 2. Related Work")))
    ok("参考文献来自输入", len(out["references"]) == 6, str(len(out["references"])))
    ok("正文有真实引用", "[@" in md and out["cited"])
    ok("每篇文献都被引用", set(out["cited"]) == {r["key"] for r in out["references"]})
    ok("参考文献含来源链接", md.count("[来源](<https://") == len(out["references"]))
    ok("存在需补充实验数据标记", "[需补充实验数据#A1" in md and "[需补充实验数据#I1" in md)
    ok("数据清单与正文一致", {x["id"] for x in out["needs_data"]}
       == {"A1", "A2", "A3", "A4", "I1", "I2"})
    ok("校验通过", not out["issues"], "; ".join(out["issues"][:3]))

    # 无文献时显式降级
    empty = generate("A study of something without any retrieved references.")
    ok("无文献时不编造引用", "[@" not in empty["markdown"].split("## 参考文献")[0])
    ok("无文献时 References 为空提示", "未编造任何条目" in empty["markdown"])
    ok("无文献时 strict 能发现问题", any("未提供可核验" in x for x in empty["issues"]))

    # 引用白名单校验能抓到伪造引用
    bad = md.replace("## 参考文献", "本文还引用了 [@ghost]。\n\n## 参考文献")
    ok("能抓出白名单外引用", any("白名单外" in x for x in verify(bad, refs)))

    # idea 字典字段能填入草稿
    dict_out = generate({
        "title": "A Test Idea",
        "description": "Study a problem.",
        "dataset": "Dataset X",
        "baselines": "Baseline Y",
        "metrics": "Accuracy",
    }, [{"title": "Ref", "url": "https://example.org/r"}])
    ok("idea 字典字段被使用",
       "Dataset X" in dict_out["markdown"] and "Baseline Y" in dict_out["markdown"])
    ok("idea 字典标题被使用", "# A Test Idea" in dict_out["markdown"])
    ok("未提供结果时仍标记 A4", "[需补充实验数据#A4" in dict_out["markdown"])

    print(f"\npaper_draft: {checks - failures}/{checks} 通过")
    return 0 if failures == 0 else 1


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════


def _read_json(path: str) -> Any:
    try:
        if path == "-":
            raw = sys.stdin.read()
        else:
            raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"读取失败：{path}：{exc}") from exc
    if not raw.strip():
        raise SystemExit(f"输入为空：{path}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON 解析失败（第 {exc.lineno} 行第 {exc.colno} 列）：{exc.msg}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(
        prog="python -m app.paper_draft",
        description="根据 idea 与真实检索结果生成论文框架（引用白名单，无 LLM）。",
    )
    ap.add_argument("file", nargs="?", help="输入 JSON；也可用 -i / --demo / stdin")
    ap.add_argument("-i", "--input", help="输入 JSON 路径")
    ap.add_argument("--idea", help="直接提供 idea（覆盖输入 JSON 中的 idea）")
    ap.add_argument("--refs", help="参考文献 JSON（覆盖输入 JSON 中的 retrieved_refs）")
    ap.add_argument("-o", "--out", help="Markdown 输出路径（默认打印到 stdout）")
    ap.add_argument("--json", action="store_true", help="输出结构化 JSON（含 markdown 字段）")
    ap.add_argument("--strict", action="store_true", help="校验不通过时返回非零退出码")
    ap.add_argument("--demo", action="store_true", help="用 6 条真实论文题录演示")
    ap.add_argument("--selftest", action="store_true", help="运行内置自检")
    ap.add_argument("--debug", action="store_true", help="打印统计与校验详情")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.demo:
        data: Any = demo_input()
    else:
        if args.input or args.file:
            data = _read_json(args.input or args.file)
        elif args.idea:
            data = {"idea": args.idea, "retrieved_refs": []}
        else:
            ap.print_help()
            return 2

        if not isinstance(data, dict):
            raise SystemExit("输入 JSON 必须是对象，包含 idea 与 retrieved_refs")
        if args.idea:
            data["idea"] = args.idea
            data.pop("title", None)  # 覆盖 idea 时，不复用输入里可能过时的标题
        if args.refs:
            data["retrieved_refs"] = extract_refs(_read_json(args.refs))

    try:
        result = generate(data, data)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.out:
        try:
            Path(args.out).write_text(result["markdown"], encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"写入失败：{args.out}：{exc}") from exc
        print(f"[paper_draft] 已写入 {args.out}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif not args.out:
        sys.stdout.write(result["markdown"])

    stats = result["stats"]
    print(
        f"[paper_draft] {stats['n_refs']} 条真实文献 / {stats['n_cited']} 个引用 / "
        f"{stats['n_needs_data']} 处待补数据 / {stats['n_rejected']} 条剔除",
        file=sys.stderr,
    )
    if args.debug:
        print(f"[paper_draft] stats={json.dumps(result['stats'], ensure_ascii=False)}",
              file=sys.stderr)
    if result["issues"]:
        for issue in result["issues"]:
            print(f"[paper_draft] 检查未通过：{issue}", file=sys.stderr)
        return 1 if args.strict else 0
    print("[paper_draft] 引用与数据标记检查通过", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
