"""从多篇论文的 Future Work 中挖掘仍开放的探索方向（提取式、可审计）。

流程：抽取未来工作点 → 按 IDF 加权词面相似度聚类去重 → 用后续论文判断是否已解决
→ 输出 `resolved` / `open_directions`。不调用 LLM，不检索材料外文献。

输入 `{{future_work_with_dates}}` 支持::

    {"papers": [
        {"id": "p1", "title": "...", "date": "2021-03",
         "abstract": "...", "future_work": "In future work, we plan to ..."},
        {"id": "p2", "title": "...", "date": "2024-05",
         "abstract": "We propose ... and address ..."}          # 只作为“后续进展”证据
     ],
     "progress": [                                              # 可选：人工标注的进展
        {"idea": "study privacy leakage", "paper_id": "p9",
         "date": "2024-09", "claim": "We solve ... with ...", "status": "solved"}
     ]}

`future_work` 可以是字符串、字符串列表、`{"text": ...}` 或段落列表；
也可用 `sections: [{"heading": "Future Work", "text": "..."}]`。

判定规则（全部可在输出里追溯）：

* **已解决**：后续论文的标题/摘要覆盖该方向的核心词，且出现“address/solve/
  we propose”等表述；或 `progress` 明确标注 `status=solved`。若该论文自己仍把
  同一方向列为未来工作，则不算解决。
* **仍开放**：材料中没有晚于该方向最后一次出现的解决性证据。若该方向在最新
  论文中仍被重复提出，会在 `why_open` 中给出具体出处。
* `angle` 是方法层面的切入建议，按方向关键词套用模板，不陈述任何文献事实。

用法::

    python -m app.future_work --demo
    python -m app.future_work -i materials.json --out directions.json
    python -m app.future_work -i materials.json --explicit-only
    python -m app.future_work --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.retrieval.bm25 import tokenize
from app.survey import _sentences
from app.utils import short

# ────────────────────────── 日期 ──────────────────────────


@dataclass(frozen=True)
class PubDate:
    """只精确到月的发表日期；月未知时按年末处理，避免把同年论文误判为更晚。"""

    year: int
    month: int = 0

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}" if self.month else f"{self.year:04d}"

    @property
    def key(self) -> tuple[int, int]:
        return (self.year, self.month or 12)


_MONTHS = {
    name.lower(): i
    for i, name in enumerate(
        ("", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    )
    if name
}


def parse_date(value: Any) -> PubDate | None:
    """支持 2023 / 2023-05 / 2023-05-01 / May 2023 / 2023年5月。"""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        year = int(value)
        return PubDate(year) if 1900 <= year <= 2100 else None
    text = str(value).strip()
    m = re.search(r"((?:19|20)\d{2})\s*[-/.年]\s*(\d{1,2})", text)
    if m:
        year, month = int(m.group(1)), min(12, max(1, int(m.group(2))))
        return PubDate(year, month)
    m = re.search(r"((?:19|20)\d{2})", text)
    if not m:
        return None
    year = int(m.group(1))
    low = text.lower()
    month = next((idx for name, idx in _MONTHS.items() if name in low), 0)
    return PubDate(year, month)


def _later(a: PubDate | None, b: PubDate | None) -> bool:
    """a 是否严格晚于 b；同年且任一方月份未知时判否，避免误报“已解决”。"""
    if a is None or b is None:
        return False
    if a.year != b.year:
        return a.year > b.year
    if not a.month or not b.month:
        return False
    return a.month > b.month


# ────────────────────────── 材料结构 ──────────────────────────


@dataclass
class Paper:
    id: str
    title: str = ""
    date: PubDate | None = None
    abstract: str = ""
    conclusion: str = ""
    future_texts: list[str] = field(default_factory=list)

    @property
    def date_label(self) -> str:
        return self.date.label if self.date else "时间未提供"

    @property
    def resolution_text(self) -> str:
        return " ".join(x for x in (self.title, self.abstract, self.conclusion) if x)


@dataclass
class Point:
    text: str
    paper: Paper
    order: int
    tokens: set[str] = field(default_factory=set)

    @property
    def date(self) -> PubDate | None:
        return self.paper.date


@dataclass
class Cluster:
    rep: Point
    points: list[Point] = field(default_factory=list)
    resolved_by: Paper | None = None
    evidence: str = ""

    @property
    def latest(self) -> Point:
        return max(self.points, key=lambda p: (p.date.key if p.date else (0, 0), p.order))

    @property
    def source_ids(self) -> set[str]:
        return {p.paper.id for p in self.points}


# ────────────────────────── 文本抽取 ──────────────────────────


_BULLET_RE = re.compile(r"^\s*(?:[-–—•·*]+|\d+[.)、]|[（(]\d+[)）])\s*")
_ADMIN_RE = re.compile(
    r"\b(?:release|publish|make\s+(?:the\s+)?(?:code|data)\s+public|"
    r"open[- ]source\s+(?:the\s+)?(?:code|data)|journal\s+version|extended\s+version)\b"
    r"|(?:开源|公开|发布)(?:代码|数据|模型)",
    re.I,
)
_META_RE = re.compile(
    r"^(?:future\s+work|future\s+research|limitations?|conclusion|discussion|"
    r"more\s+experiments|further\s+experiments|未来工作|后续工作)[.。]?$",
    re.I,
)

_FRAME_PATTERNS = [
    re.compile(r"^(?:in|for)\s+(?:the\s+)?future(?:\s+work|\s+research)?\s*[,;:\-–—]?\s*", re.I),
    re.compile(
        r"^(?:future|further)\s+(?:work|research|studies)\s*"
        r"(?:includes?|will|should|may|could|might|focus(?:es)?\s+on|is|are)?\s*[:;\-–—,]?\s*",
        re.I,
    ),
    re.compile(
        r"^we\s+(?:(?:also|further)\s+)?"
        r"(?:plan|intend|aim|hope|want|propose|would\s+like)\s+to\s+",
        re.I,
    ),
    re.compile(r"^we\s+(?:will|next)\s+", re.I),
    re.compile(r"^(?:plan|intend|aim|hope|want)\s+to\s+", re.I),
    re.compile(r"^(?:并且?|以及|同时|另外|此外|其次|还(?:要|将|会)?)\s*"),
    re.compile(
        r"^(?:additionally|moreover|furthermore|next|in\s+addition|also|besides)\s*,?\s*"
        r"(?:we\s+(?:plan\s+to\s+|will\s+|intend\s+to\s+))?",
        re.I,
    ),
    re.compile(
        r"^(?:one|a)\s+(?:promising|possible|natural|important)\s+(?:future\s+)?"
        r"(?:direction|extension|next\s+step)\s+(?:is|would\s+be)\s+",
        re.I,
    ),
    re.compile(r"^(?:a\s+)?promising\s+(?:future\s+)?(?:direction|avenue)\s+(?:is|would\s+be)\s+", re.I),
    re.compile(r"^未来工作(?:包括|将|是|方面)?\s*[：:，,]?\s*"),
    re.compile(r"^后续工作(?:包括|将|是)?\s*[：:，,]?\s*"),
    re.compile(r"^本文(?:的)?未来工作\s*[：:，,]?\s*"),
    re.compile(r"^(?:一个)?(?:有前景|值得|重要)的?(?:未来)?(?:方向|延伸)(?:是|包括)?\s*[：:，,]?\s*"),
    re.compile(r"^(?:to\s+)", re.I),
]

_LEAVE_EN_RE = re.compile(
    r"(?:we\s+leave|we\s+defer|we\s+postpone)\s+(.{8,260}?)\s+"
    r"(?:for|to)\s+(?:future(?:\s+work|\s+research)?)",
    re.I,
)
_LEAVE_ZH_RE = re.compile(r"(.{6,200}?)\s*(?:留待|作为)(?:未来工作|后续工作)")
_LEFT_PASSIVE_RE = re.compile(
    r"^(.{6,220}?)\s+(?:is|are|remains?)\s+(?:left\s+)?(?:for|as)\s+"
    r"future\s+(?:work|research)",
    re.I,
)


def _clean_point(raw: str) -> str:
    """去掉列表符号与“未来工作包括”这类框架话，返回一个方向原句。"""
    text = (raw or "").strip()
    text = _BULLET_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)

    for pattern in (_LEAVE_EN_RE, _LEAVE_ZH_RE, _LEFT_PASSIVE_RE):
        m = pattern.search(text)
        if m:
            text = m.group(1)
            break

    for _ in range(3):
        before = text
        for pattern in _FRAME_PATTERNS:
            text = pattern.sub("", text, count=1)
        if text == before:
            break

    text = re.sub(
        r"\s*(?:is|are|remains?)\s+(?:left\s+)?(?:for|as)\s+future\s+(?:work|research).*$",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"[，,;；\s]*?(?:留待|作为)(?:未来工作|后续工作).*$", "", text)
    text = text.strip(" \t\r\n-–—•·*;；,，")
    min_len = 4 if re.search(r"[一-鿿]", text) else 6
    if len(text) < min_len:
        return ""
    if _META_RE.match(text):
        return ""
    if _ADMIN_RE.search(text):
        return ""
    return text


def _flatten_future(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out += _flatten_future(item)
        return out
    if isinstance(value, dict):
        for key in ("text", "idea", "direction", "content", "future_work"):
            if key in value:
                return _flatten_future(value[key])
        out = []
        for item in value.values():
            out += _flatten_future(item)
        return out
    return []


_SPLIT_CUE_RE = re.compile(
    r"(?<=[.!?。！？])\s+(?=(?:we\s+(?:also\s+|further\s+)?"
    r"(?:plan|will|intend|aim|hope|propose|would|next)|"
    r"(?:another|additionally|moreover|furthermore|in\s+addition|next)\b|also\b|"
    r"此外|另外|其次|未来工作))",
    re.I,
)


def _cue_split(text: str) -> list[str]:
    """在同一段里按“We also plan / Another / 此外”等新方向提示切句。"""
    parts = [p.strip() for p in _SPLIT_CUE_RE.split(text) if p and p.strip()]
    return parts or [text]


_CHUNK_SPLIT_RE = re.compile(
    r"\n+|[;；]+|[，,]\s*(?=(?:并且?|以及|同时|另外|此外|其次|还(?:要|将|会)?))"
)


def _split_future_text(text: str) -> list[str]:
    """按行/分号/中文并列连词切成条目；段落内按新方向提示或句号继续切分。"""
    chunks = [c for c in _CHUNK_SPLIT_RE.split(text) if c and c.strip()]
    out: list[str] = []
    for chunk in chunks:
        cleaned = _clean_point(chunk)
        if not cleaned:
            continue
        pieces = _sentences(cleaned) if len(cleaned) > 320 else [cleaned]
        for piece in pieces:
            for part in _cue_split(piece):
                item = _clean_point(part)
                if item:
                    out.append(item)
    return out


def _paper_future_texts(raw: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in (
        "future_work", "future_works", "future", "future_directions",
        "future_research", "open_problems", "next_steps",
    ):
        texts += _flatten_future(raw.get(key))
    if not texts:
        for section in raw.get("sections") or []:
            if not isinstance(section, dict):
                continue
            heading = str(section.get("heading") or section.get("title") or "")
            if re.search(r"future|未来工作|展望|next\s+steps", heading, re.I):
                texts += _flatten_future(section.get("text") or section.get("content"))
    return texts


def _slug(text: str) -> str:
    slug = re.sub(r"[^0-9a-z一-鿿]+", "-", (text or "").lower()).strip("-")
    return slug[:40]


def _paper_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("papers", "items", "future_work_with_dates", "documents"):
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            return [value]
    if "title" in data or any(
        k in data for k in ("future_work", "future_works", "future", "future_directions")
    ):
        return [data]
    # 以 paper_id 为键的映射
    values = [v for v in data.values() if isinstance(v, dict)]
    return values


def load_papers(data: Any) -> list[Paper]:
    papers: list[Paper] = []
    seen: set[str] = set()
    for idx, item in enumerate(_paper_items(data)):
        nested = item.get("paper") if isinstance(item.get("paper"), dict) else {}
        raw = {**nested, **item}
        title = str(raw.get("title") or raw.get("name") or "").strip()
        raw_id = str(
            raw.get("id") or raw.get("paper_id") or raw.get("arxiv_id") or raw.get("doi") or ""
        ).strip()
        pid = raw_id or _slug(title) or f"p{idx + 1}"
        if pid in seen:
            suffix = 2
            while f"{pid}-{suffix}" in seen:
                suffix += 1
            pid = f"{pid}-{suffix}"
        seen.add(pid)
        papers.append(
            Paper(
                id=pid,
                title=title,
                date=parse_date(
                    raw.get("date") or raw.get("published") or raw.get("publication_date")
                    or raw.get("pub_date") or raw.get("year") or raw.get("time")
                ),
                abstract=str(raw.get("abstract") or raw.get("summary") or raw.get("tldr") or "").strip(),
                conclusion=str(raw.get("conclusion") or raw.get("conclusions") or "").strip(),
                future_texts=_paper_future_texts(raw),
            )
        )
    return papers


# ────────────────────────── 分词与聚类 ──────────────────────────

SYNONYMS = {
    "crosslingual": "multilingual", "cross-lingual": "multilingual",
    "multi-lingual": "multilingual", "multilingual": "multilingual",
    "latency": "efficiency", "efficient": "efficiency", "latencies": "efficiency",
    "speed": "efficiency", "runtime": "efficiency", "compute": "efficiency",
    "generalisation": "generalization", "generalize": "generalization",
    "generalise": "generalization", "robust": "robustness",
    "multi-modal": "multimodal", "multimodal": "multimodal",
    "long-context": "longcontext", "longcontext": "longcontext",
    "long-document": "longdocument", "longdocument": "longdocument",
    "long-sequence": "longsequence", "longsequence": "longsequence",
}

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "our", "are", "was",
    "were", "will", "can", "could", "should", "would", "may", "might", "have", "has",
    "had", "not", "but", "also", "such", "more", "most", "less", "very", "using", "use",
    "used", "based", "new", "different", "various", "several", "some", "many", "much",
    "other", "others", "these", "those", "their", "its", "it", "we", "they", "them",
    "you", "your", "paper", "work", "works", "future", "direction", "directions",
    "approach", "approaches", "method", "methods", "model", "models", "study", "studies",
    "research", "problem", "problems", "challenge", "challenges", "task", "tasks",
    "result", "results", "experiment", "experiments", "performance", "improve",
    "improving", "improvement", "extend", "extending", "extension", "explore",
    "exploring", "investigate", "investigating", "propose", "proposing", "consider",
    "considering", "plan", "plans", "focus", "further", "potential", "possible",
    "promising", "novel", "better", "good", "large", "small", "high", "low", "setting",
    "settings", "scenario", "scenarios", "setting", "enable", "enabling", "provide",
    "providing", "require", "requiring", "need", "needs", "important", "significant",
    "future", "recent", "current", "existing", "given", "make", "making", "show",
    "showing", "demonstrate", "demonstrating", "achieve", "achieving",
    "input", "inputs", "reduce", "reducing", "lower", "lowering", "under", "over",
    # 中文 bigram 常见虚词/泛词
    "未来", "工作", "方向", "方法", "模型", "数据", "研究", "问题", "挑战", "任务",
    "效果", "性能", "实验", "结果", "分析", "提出", "使用", "基于", "新的", "这种",
    "这些", "目前", "仍然", "需要", "可能", "不同", "相关", "进一步", "以及", "通过",
    "进行", "可以", "我们", "本文", "一个", "并且", "而且", "但是", "因为", "所以",
    "如果", "已经", "还有", "其他", "各种", "一些", "很多", "更加", "有效", "重要",
    "方面", "情况", "能力", "过程", "系统", "技术", "领域", "包括", "主要", "一般",
    "扩展", "降低", "推理", "评估", "场景", "提升", "增强", "优化",
}


def _stem(word: str) -> str:
    if len(word) <= 4:
        return word
    for suffix in ("izations", "ization", "ations", "ation", "ing", "ies", "ied", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            base = word[: -len(suffix)]
            if suffix in ("ies", "ied"):
                base += "y"
            if base.endswith("e") and len(base) > 4:
                base = base[:-1]
            return base
    return word


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in tokenize(text):
        canon = SYNONYMS.get(raw, raw)
        for part in re.split(r"[-_/]", canon):
            part = part.strip().strip(".")
            if not part or "." in part:
                continue
            if re.fullmatch(r"[a-z0-9]+", part):
                if len(part) < 3 or part.isdigit() or part in STOPWORDS:
                    continue
                part = _stem(part)
            elif len(part) < 2:
                continue
            if part in STOPWORDS:
                continue
            out.add(part)
    long_forms = {"longcontext", "longdocument", "longsequence"}
    if out & long_forms or ("long" in out and out & {"document", "context", "sequence"}):
        out -= long_forms | {"long", "document", "context", "sequence"}
        out.add("longinput")
    return out


def _build_idf(points: Sequence[Point]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for p in points:
        df.update(p.tokens)
    n = len(points)
    return {
        t: math.log(1 + (n - c + 0.5) / (c + 0.5))
        for t, c in df.items()
    }


def _sim_weights(points: Sequence[Point], idf: dict[str, float]) -> dict[str, float]:
    """聚类相似度只计“在材料中至少出现两次”的词。

    一次性修饰语（某篇独有的措辞）会把短句的相似度稀释掉；只保留跨论文复现的
    主题词，既能让同义复述聚到一起，又不会把无关方向硬拉在一起。
    """
    df: Counter[str] = Counter()
    for p in points:
        df.update(p.tokens)
    return {t: (idf.get(t, 1.0) if c >= 2 else 0.0) for t, c in df.items()}


def _sim(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    """IDF 加权重叠系数：共享词的权重 / 较短一方的权重。"""
    if not a or not b:
        return 0.0
    shared = a & b
    if not shared:
        return 0.0
    num = sum(idf.get(t, 1.0) for t in shared)
    den = min(
        sum(idf.get(t, 1.0) for t in a),
        sum(idf.get(t, 1.0) for t in b),
    )
    return num / den if den else 0.0


def _anchors(tokens: set[str], idf: dict[str, float], limit: int = 6) -> set[str]:
    ranked = sorted(tokens, key=lambda t: (-idf.get(t, 1.0), t))
    distinctive = [t for t in ranked if idf.get(t, 1.0) >= 1.0]
    return set((distinctive or ranked)[:limit])


def _coverage(anchors: set[str], tokens: set[str], idf: dict[str, float]) -> tuple[float, set[str]]:
    if not anchors:
        return 0.0, set()
    shared = anchors & tokens
    den = sum(idf.get(t, 1.0) for t in anchors)
    return (sum(idf.get(t, 1.0) for t in shared) / den if den else 0.0), shared


def _cluster_core_tokens(cluster: Cluster) -> set[str]:
    """聚类内各条共同出现的词；单条或没有交集时退回代表条目的词。"""
    if len(cluster.points) <= 1:
        return set(cluster.rep.tokens)
    shared = set(cluster.points[0].tokens)
    for point in cluster.points[1:]:
        shared &= point.tokens
    return shared or set(cluster.rep.tokens)


def extract_points(papers: Sequence[Paper]) -> list[Point]:
    points: list[Point] = []
    order = 0
    seen: set[tuple[str, str]] = set()
    for paper in sorted(papers, key=lambda p: (p.date.key if p.date else (9999, 12), p.id)):
        for text in paper.future_texts:
            for idea in _split_future_text(text):
                norm = re.sub(r"\s+", " ", idea).lower()
                if (paper.id, norm) in seen:
                    continue
                seen.add((paper.id, norm))
                order += 1
                points.append(Point(text=idea, paper=paper, order=order))
    return points


def cluster_points(
    points: Sequence[Point],
    weights: dict[str, float],
    threshold: float = 0.55,
) -> list[Cluster]:
    clusters: list[Cluster] = []
    ordered = sorted(points, key=lambda p: (p.date.key if p.date else (9999, 12), p.paper.id, p.order))
    for point in ordered:
        best: Cluster | None = None
        best_sim = 0.0
        for cluster in clusters:
            score = _sim(point.tokens, cluster.rep.tokens, weights)
            if score > best_sim:
                best, best_sim = cluster, score
        if best is not None and best_sim >= threshold:
            best.points.append(point)
        else:
            clusters.append(Cluster(rep=point, points=[point]))
    return clusters


# ────────────────────────── 解决判定 ──────────────────────────

_CLAIM_EN = re.compile(
    r"\b(?:we|this\s+(?:paper|work))\s+"
    r"(?:propos|introduc|present|develop|design|show|demonstrat|provid|build|construct|evaluat)\w*"
    r"|\b(?:proposed|introduced|presented)\b",
    re.I,
)
_CLAIM_ZH = ("提出", "介绍", "设计", "构建", "实现", "证明", "表明", "展示")

_RESOLVE_EN = re.compile(
    r"\b(?:solve[sd]?|solving|address(?:es|ed)?|addressing|overcome[sd]?|overcoming|"
    r"resolve[sd]?|resolving|tackl(?:e|es|ed|ing)|eliminat(?:e|es|ed|ing)|"
    r"fill(?:s|ed)?\s+the\s+gap|close[sd]?\s+the\s+gap)\b",
    re.I,
)
_RESOLVE_ZH = ("已解决", "解决了", "弥补", "填补", "克服", "消除", "解决了这一")

_SOLVED_STATUS = {"solved", "resolved", "closed", "done", "addressed", "已解决", "完成"}
_OPEN_STATUS = {"open", "ongoing", "unresolved", "partial", "planned", "in progress", "未解决", "进行中"}

_DISPLAY = {
    "longinput": "长输入",
    "efficiency": "效率/时延",
    "multilingual": "多语言",
    "evalu": "评测",
    "learn": "学习",
    "fairnes": "公平性",
    "federat": "联邦",
    "privaci": "隐私",
}


def _labels(tokens: Iterable[str], limit: int = 4) -> str:
    labels = sorted({_DISPLAY.get(t, t) for t in tokens})
    return "、".join(labels[:limit])


def _paper_restates(paper: Paper, cluster: Cluster, idf: dict[str, float]) -> bool:
    """该论文自己仍把同一方向列为未来工作 → 不能算作解决者。"""
    core = _cluster_core_tokens(cluster)
    for text in paper.future_texts:
        for idea in _split_future_text(text):
            if _sim(_tokens(idea), core, idf) >= 0.6:
                return True
    return False


def _resolve_evidence(paper: Paper, cluster: Cluster, idf: dict[str, float]) -> str | None:
    """返回证据说明；不足以判定解决时返回 None。"""
    anchors = _anchors(_cluster_core_tokens(cluster), idf)
    text = paper.resolution_text
    tokens = _tokens(text)
    coverage, shared = _coverage(anchors, tokens, idf)
    title_cov, _ = _coverage(anchors, _tokens(paper.title), idf)
    if not shared:
        return None

    low = text.lower()
    has_resolve = bool(_RESOLVE_EN.search(low)) or any(z in text for z in _RESOLVE_ZH)
    has_claim = bool(_CLAIM_EN.search(low)) or any(z in text for z in _CLAIM_ZH)
    strong = (
        len(shared) >= 2
        or max(idf.get(t, 1.0) for t in shared) >= 1.25
        or title_cov >= 0.6
    )

    if not strong:
        return None
    if coverage >= 0.5 and has_resolve:
        return (
            f"标题/摘要覆盖 {_labels(shared)}（{coverage:.0%}），并出现解决性表述"
        )
    if title_cov >= 0.65 and has_claim and coverage >= 0.45:
        return (
            f"标题直接覆盖 {_labels(shared)}（标题覆盖 {title_cov:.0%}），并提出对应方法"
        )
    return None


def _auto_resolve(
    cluster: Cluster,
    papers: Sequence[Paper],
    idf: dict[str, float],
) -> tuple[Paper, str] | None:
    latest_point = cluster.latest
    if latest_point.date is None:
        return None
    sources = cluster.source_ids
    candidates = [
        p for p in papers
        if p.date is not None
        and p.id not in sources
        and _later(p.date, latest_point.date)
    ]
    for paper in sorted(candidates, key=lambda p: (p.date.key, p.id)):
        if _paper_restates(paper, cluster, idf):
            continue
        evidence = _resolve_evidence(paper, cluster, idf)
        if evidence:
            return paper, evidence
    return None


def _related_paper(
    cluster: Cluster,
    papers: Sequence[Paper],
    idf: dict[str, float],
) -> tuple[Paper, float, set[str]] | None:
    """找一篇“主题相近但未明确解决”的后续论文，用于 why_open 的解释。"""
    latest_point = cluster.latest
    if latest_point.date is None:
        return None
    anchors = _anchors(_cluster_core_tokens(cluster), idf)
    best: tuple[Paper, float, set[str]] | None = None
    for paper in papers:
        if paper.date is None or paper.id in cluster.source_ids:
            continue
        if not _later(paper.date, latest_point.date):
            continue
        if _paper_restates(paper, cluster, idf):
            continue
        coverage, shared = _coverage(anchors, _tokens(paper.resolution_text), idf)
        if coverage >= 0.25 and (best is None or coverage > best[1]):
            best = (paper, coverage, shared)
    return best


def _progress_entries(data: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return entries
    for key in ("progress", "updates", "later_work", "solutions", "solved"):
        value = data.get(key)
        if isinstance(value, list):
            entries += [x for x in value if isinstance(x, dict)]
        elif isinstance(value, dict):
            entries.append(value)
    return entries


def _apply_progress(
    entries: Sequence[dict[str, Any]],
    clusters: Sequence[Cluster],
    papers: Sequence[Paper],
    idf: dict[str, float],
) -> None:
    by_id = {p.id: p for p in papers}
    for idx, entry in enumerate(entries):
        status = str(entry.get("status") or "").strip().lower()
        if status in _OPEN_STATUS:
            continue
        idea_text = str(entry.get("idea") or entry.get("direction") or entry.get("problem") or "")
        claim = str(
            entry.get("claim") or entry.get("evidence") or entry.get("description")
            or entry.get("result") or ""
        )
        title = str(entry.get("title") or "")
        match_text = " ".join(x for x in (idea_text, claim, title) if x)
        if not match_text:
            continue
        match_tokens = _tokens(match_text)

        best: Cluster | None = None
        best_sim = 0.0
        for cluster in clusters:
            score = _sim(match_tokens, cluster.rep.tokens, idf)
            if score > best_sim:
                best, best_sim = cluster, score
        if best is None or best_sim < 0.35:
            continue

        pid = str(entry.get("paper_id") or entry.get("id") or f"progress-{idx + 1}").strip()
        paper = by_id.get(pid)
        if paper is None:
            paper = Paper(
                id=pid,
                title=title or "进度条目",
                date=parse_date(entry.get("date") or entry.get("published")),
                abstract=claim,
            )
        solved = status in _SOLVED_STATUS
        marker = bool(_RESOLVE_EN.search((claim + " " + title).lower())) or any(
            z in (claim + " " + title) for z in _RESOLVE_ZH
        )
        if solved or marker:
            best.resolved_by = paper
            detail = claim or title or "材料标注为已解决"
            best.evidence = f"材料标注的进展：{short(detail, 140)}"


# ────────────────────────── 原因与角度 ──────────────────────────

_ANGLE_RULES: list[tuple[tuple[str, ...], str]] = [
    (
        ("multilingual", "cross-lingual", "low-resource", "多语言", "跨语言", "低资源"),
        "先在目标语言/低资源语言上建立小规模评测集，复用原文模型与指标作基线，"
        "再只替换语言相关组件，定位迁移损失来源。",
    ),
    (
        ("efficiency", "latency", "memory", "compute", "scalab", "效率", "时延", "显存", "算力", "规模"),
        "固定效果阈值，测量时延与显存随输入长度/模型规模的曲线，对瓶颈模块逐项消融，"
        "区分算法复杂度与实现开销。",
    ),
    (
        ("robust", "adversarial", "noise", "distribution shift", "domain shift",
         "鲁棒", "对抗", "噪声", "分布偏移", "泛化"),
        "构造同分布/异分布与对抗测试集，分层报告结果，定位性能下降来自表示、"
        "训练目标还是数据覆盖。",
    ),
    (
        ("data", "label", "annotat", "corpus", "dataset", "数据", "标注", "语料"),
        "并行评估人工标注与合成数据两条路线，比较单位标注成本带来的增益，"
        "再做跨域迁移检验。",
    ),
    (
        ("multimodal", "vision", "image", "audio", "多模态", "视觉", "图像", "语音"),
        "从单模态基线逐步加入另一模态，做模态缺失与对齐消融，隔离多模态增益。",
    ),
    (
        ("fair", "bias", "privacy", "security", "公平", "偏见", "隐私", "安全"),
        "先定义可操作的失败案例与度量，再在受控子集上做最小干预实验，"
        "同时报告收益与效用损失。",
    ),
    (
        ("longcontext", "longdocument", "longsequence", "长文本", "长文档", "长序列"),
        "按长度分层评测（如 4k/16k/64k），固定信息位置做检索-生成分解，"
        "判断瓶颈在召回还是推理。",
    ),
    (
        ("theory", "proof", "interpret", "explain", "理论", "证明", "解释", "可解释"),
        "先给出可证伪的假设与简化设定，构造反例测试边界，再逐步放宽假设。",
    ),
    (
        ("deploy", "real-world", "clinical", "production", "实际", "部署", "临床", "生产"),
        "先离线回溯验证，再做小规模前瞻性试点，预设失败回退与人工复核流程。",
    ),
]

_DEFAULT_ANGLE = (
    "以原始论文设置为基线，先复现其结论，再只改变该方向涉及的一个变量，"
    "做可归因的对照实验，并报告失败案例。"
)


def suggest_angle(idea: str) -> str:
    low = idea.lower()
    for keywords, angle in _ANGLE_RULES:
        if any(k in low for k in keywords):
            return angle
    return _DEFAULT_ANGLE


def _cite(paper_id: str) -> str:
    return f"[@{paper_id}]"


def _source(cluster: Cluster) -> str:
    seen: set[str] = set()
    ordered = sorted(
        cluster.points,
        key=lambda p: (p.date.key if p.date else (9999, 12), p.paper.id),
    )
    parts: list[str] = []
    for point in ordered:
        paper = point.paper
        if paper.id in seen:
            continue
        seen.add(paper.id)
        title = short(paper.title, 38) if paper.title else "标题未提供"
        parts.append(f"{_cite(paper.id)}（{paper.date_label}）《{title}》")
    if len(parts) > 3:
        return "；".join(parts[:3]) + f"；等 {len(parts)} 篇"
    return "；".join(parts)


def _why_open(
    cluster: Cluster,
    papers: Sequence[Paper],
    idf: dict[str, float],
    max_date: PubDate | None,
    explicit_only: bool = False,
) -> str:
    rep = cluster.rep
    latest = cluster.latest
    if len(cluster.source_ids) > 1:
        text = (
            f"该方向在 {_cite(rep.paper.id)}（{rep.paper.date_label}）与 "
            f"{_cite(latest.paper.id)}（{latest.paper.date_label}）中被重复提出，"
            f"最新一次为 {latest.paper.date_label}"
        )
        if latest.date and max_date and latest.date.key >= max_date.key:
            text += "（材料中的最新时间）"
        text += (
            "；explicit-only 模式未启用自动判定"
            if explicit_only
            else "，之后未达到自动判定条件"
        )
    else:
        text = f"该方向由 {_cite(rep.paper.id)}（{rep.paper.date_label}）提出"
        if explicit_only:
            text += "；explicit-only 模式未启用自动判定"
        elif max_date and rep.date and rep.date.key < max_date.key:
            text += "；材料覆盖到更晚时间，但未达到自动判定条件"
        else:
            text += "；材料未覆盖其后的进展"

    related = _related_paper(cluster, papers, idf)
    if related:
        paper, coverage, shared = related
        terms = _labels(shared, 3) or "部分术语"
        reason = "仅作参考" if explicit_only else "未达到自动判定阈值"
        text += (
            f"。后续 {_cite(paper.id)}（{paper.date_label}）主题相近"
            f"（{terms}，覆盖 {coverage:.0%}），但{reason}"
        )
    text += "。本判断仅基于所给材料，未检索材料外文献。"
    return text


# ────────────────────────── 主流程 ──────────────────────────


def analyze(
    data: Any,
    *,
    threshold: float = 0.55,
    explicit_only: bool = False,
) -> dict[str, Any]:
    """返回严格符合约定 schema 的 JSON 对象。"""
    papers = load_papers(data)
    points = extract_points(papers)
    for point in points:
        point.tokens = _tokens(point.text)
    idf = _build_idf(points)
    weights = _sim_weights(points, idf)
    clusters = cluster_points(points, weights, threshold=threshold)

    for cluster in clusters:
        if explicit_only:
            break
        found = _auto_resolve(cluster, papers, idf)
        if found:
            cluster.resolved_by, cluster.evidence = found

    _apply_progress(_progress_entries(data), clusters, papers, idf)

    max_date = max(
        (p.date for p in papers if p.date),
        key=lambda d: d.key,
        default=None,
    )

    resolved: list[dict[str, str]] = []
    opened: list[tuple[tuple[int, int], int, int, dict[str, str]]] = []
    for cluster in clusters:
        if cluster.resolved_by is not None:
            paper = cluster.resolved_by
            resolved.append(
                {
                    "idea": cluster.rep.text,
                    "solved_by": (
                        f"{_cite(paper.id)}（{paper.date_label}）"
                        f"《{short(paper.title or '进度条目', 50)}》—— {cluster.evidence}"
                    ),
                }
            )
            continue
        latest_key = cluster.latest.date.key if cluster.latest.date else (0, 0)
        earliest_key = cluster.rep.date.key if cluster.rep.date else (9999, 12)
        opened.append(
            (
                latest_key,
                len(cluster.points),
                earliest_key,
                {
                    "idea": cluster.rep.text,
                    "source": _source(cluster),
                    "why_open": _why_open(cluster, papers, idf, max_date, explicit_only),
                    "angle": suggest_angle(cluster.rep.text),
                },
            )
        )

    opened.sort(
        key=lambda row: (
            -row[0][0], -row[0][1], -row[1], -row[2][0], -row[2][1], row[3]["idea"]
        )
    )
    resolved.sort(key=lambda row: row["idea"])
    return {"resolved": resolved, "open_directions": [row[3] for row in opened]}


# ────────────────────────── 演示与自检 ──────────────────────────


def demo_data() -> dict[str, Any]:
    """合成材料：3 个方向被后续工作解决，1 个仍开放，1 条人工进展。"""
    return {
        "papers": [
            {
                "id": "p1", "title": "A Baseline for Document Understanding", "date": "2021-03",
                "future_work": (
                    "In future work, we plan to extend our method to multilingual settings. "
                    "We also plan to reduce inference latency on long documents."
                ),
            },
            {
                "id": "p2", "title": "Follow-up Study on Document Models", "date": "2022-06",
                "future_work": "Future work includes multilingual evaluation and low-resource transfer.",
            },
            {
                "id": "p3", "title": "Efficient Transformers", "date": "2023-02",
                "future_work": (
                    "- Improve inference efficiency for long-context inputs.\n"
                    "- We will release the code and data."
                ),
            },
            {
                "id": "p4", "title": "A Multilingual Evaluation Benchmark", "date": "2024-05",
                "abstract": (
                    "We propose a multilingual evaluation benchmark covering 20 languages. "
                    "The benchmark addresses the lack of multilingual evaluation in prior work."
                ),
            },
            {
                "id": "p5", "title": "Latency-Aware Attention for Long-Context Inputs", "date": "2023-11",
                "abstract": (
                    "We address the high inference latency of long-context inputs with a "
                    "latency-aware attention kernel. Experiments show lower memory and compute."
                ),
            },
            {
                "id": "p6", "title": "Privacy in Federated Learning", "date": "2020-04",
                "future_work": "A promising direction is to study privacy leakage in federated learning.",
            },
            {
                "id": "p7", "title": "Federated Learning Revisited", "date": "2024-08",
                "future_work": "Privacy leakage in federated learning remains an open problem.",
            },
            {
                "id": "p8", "title": "Differentially Private Federated Learning", "date": "2024-09",
                "abstract": "We solve privacy leakage in federated learning with differential privacy.",
            },
            {
                "id": "p9", "title": "Fairness under Distribution Shift", "date": "2025-01",
                "future_work": "We plan to evaluate fairness under distribution shift.",
            },
        ],
        "progress": [
            {
                "idea": "study privacy leakage in federated learning",
                "paper_id": "p8",
                "date": "2024-09",
                "claim": "We solve privacy leakage in federated learning with differential privacy.",
                "status": "solved",
            }
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

    # 文本抽取
    cleaned = _clean_point("In future work, we plan to extend our method to multilingual settings.")
    ok("去掉未来工作框架话", cleaned.lower().startswith("extend our method"), cleaned)
    ok("过滤代码发布类条目", _clean_point("We will release the code and data.") == "")
    ok("提取被动式留待句", "multilingual" in _clean_point("Multilingual evaluation is left for future work.").lower())
    ok(
        "拆分多条未来工作",
        len(_split_future_text(
            "Future work includes multilingual evaluation.\n"
            "- Study privacy leakage.\n"
            "- Improve inference efficiency."
        )) == 3,
    )
    ok("中文未来工作可抽取", "多语言" in _clean_point("未来工作包括：将方法扩展到多语言场景。"))

    # 日期
    ok("日期 年-月", parse_date("2023-05").label == "2023-05")
    ok("日期 仅年份", parse_date("2023").label == "2023")
    ok("日期 英文月份", parse_date("May 2023").label == "2023-05")
    ok("日期 中文", parse_date("2023年5月").label == "2023-05")
    ok("同年未知月份不判为更晚",
       not _later(parse_date("2024"), parse_date("2024-05"))
       and not _later(parse_date("2024-05"), parse_date("2024")))

    data = demo_data()
    out = analyze(data)
    resolved = out["resolved"]
    opened = out["open_directions"]

    ok("输出只有两个顶层键", set(out) == {"resolved", "open_directions"})
    ok("resolved 条目字段正确", all(set(r) == {"idea", "solved_by"} for r in resolved))
    ok("open 条目字段正确", all(set(o) == {"idea", "source", "why_open", "angle"} for o in opened))
    ok("多语言方向被判为已解决", any("multilingual" in r["idea"].lower() and "p4" in r["solved_by"] for r in resolved),
       str(resolved)[:300])
    ok("效率方向被判为已解决", any("efficiency" in r["idea"].lower() or "latency" in r["idea"].lower() for r in resolved),
       str(resolved)[:300])
    ok("隐私方向由人工进展解决", any("privacy" in r["idea"].lower() and "p8" in r["solved_by"] for r in resolved),
       str(resolved)[:300])
    ok("已解决方向不重复出现在开放列表",
       all(r["idea"] not in {o["idea"] for o in opened} for r in resolved))
    ok("公平性方向仍开放", any("fairness" in o["idea"].lower() for o in opened), str(opened)[:300])
    ok("开放项 source 含原始出处", all("[@" in o["source"] for o in opened))
    ok("开放项 why_open 有解释", all(len(o["why_open"]) > 20 for o in opened))
    ok("开放项 angle 非空", all(len(o["angle"]) > 20 for o in opened))
    ordered_out = analyze(
        {
            "papers": [
                {"id": "old", "title": "Old", "date": "2020-01",
                 "future_work": "study privacy leakage in federated learning"},
                {"id": "new", "title": "New", "date": "2024-01",
                 "future_work": "evaluate fairness under distribution shift"},
            ]
        }
    )["open_directions"]
    ok("开放项按最新时间降序",
       len(ordered_out) == 2 and "[@new]" in ordered_out[0]["source"],
       str([o["source"] for o in ordered_out]))
    ok("重复分析结果一致", analyze(data) == out)
    ok("JSON 可序列化", isinstance(json.dumps(out, ensure_ascii=False), str))

    # explicit-only：关闭自动判定，保留人工进展
    strict_out = analyze(data, explicit_only=True)
    ok("explicit-only 不自动排除方向",
       not any("p4" in r["solved_by"] for r in strict_out["resolved"]))
    ok("explicit-only 仍采用人工进展",
       any("p8" in r["solved_by"] for r in strict_out["resolved"]))

    # 后续论文自己仍列同一方向 → 不能算解决
    restate = {
        "papers": [
            {"id": "a", "title": "A", "date": "2020-01",
             "future_work": "extend to multilingual settings"},
            {"id": "b", "title": "A Multilingual Benchmark", "date": "2024-01",
             "abstract": "We propose a multilingual benchmark.",
             "future_work": "Future work includes multilingual evaluation."},
        ]
    }
    restate_out = analyze(restate)
    ok("复述方向不被误判为已解决", not restate_out["resolved"], str(restate_out["resolved"]))
    ok("复述后 why_open 指向最新论文",
       any("b" in o["why_open"] for o in restate_out["open_directions"]))

    # 无关方向不应合并
    unrelated = {
        "papers": [
            {"id": "x", "title": "X", "date": "2021-01",
             "future_work": "study privacy leakage in federated learning"},
            {"id": "y", "title": "Y", "date": "2021-02",
             "future_work": "reduce inference latency on long documents"},
        ]
    }
    un_out = analyze(unrelated)
    ok("无关方向不合并", len(un_out["open_directions"]) == 2, str(un_out["open_directions"]))

    # 空输入 / 无 future work
    empty = analyze({"papers": []})
    ok("空材料返回空列表", empty == {"resolved": [], "open_directions": []})
    no_future = analyze({"papers": [{"id": "z", "title": "Z", "date": "2024"}]})
    ok("无未来工作不崩溃", no_future == {"resolved": [], "open_directions": []})

    # 同一篇论文的两条相近表述，不应写成“在 [@s] 与 [@s] 中重复提出”
    same_paper = analyze(
        {
            "papers": [
                {"id": "s", "title": "S", "date": "2021-01",
                 "future_work": "We plan to evaluate multilingual robustness. "
                                 "We also plan to evaluate multilingual fairness."},
            ]
        }
    )["open_directions"]
    ok("同篇重复表述不写成两篇", bool(same_paper) and "被重复提出" not in same_paper[0]["why_open"],
       str(same_paper)[:300])

    # 单条方向没有“复发词”时，人工进展仍应能匹配
    singleton_progress = analyze(
        {
            "papers": [
                {"id": "a", "title": "A", "date": "2020",
                 "future_work": "study quantum error mitigation"},
            ],
            "progress": [
                {"idea": "quantum error mitigation", "paper_id": "b", "date": "2024",
                 "claim": "We solve quantum error mitigation.", "status": "solved"},
            ],
        }
    )
    ok("单条方向也能被人工进展解决", bool(singleton_progress["resolved"]),
       str(singleton_progress)[:300])

    # status=open 的进展条目不能算解决
    open_progress = {
        "papers": [{"id": "a", "title": "A", "date": "2020",
                    "future_work": "study privacy leakage in federated learning"}],
        "progress": [{"idea": "privacy leakage", "paper_id": "b", "status": "open",
                      "claim": "Privacy leakage remains unsolved."}],
    }
    ok("status=open 不计为解决", not analyze(open_progress)["resolved"])

    # 短材料 / 无日期
    no_date = analyze({"papers": [{"id": "n", "title": "N",
                                   "future_work": "study multilingual robustness"}]})
    ok("无日期不崩溃", len(no_date["open_directions"]) == 1)
    ok("无日期 why_open 说明时间缺失",
       "时间未提供" in no_date["open_directions"][0]["why_open"])

    print(f"\nfuture_work: {checks - failures}/{checks} 通过")
    return 0 if failures == 0 else 1


# ────────────────────────── CLI ──────────────────────────


def _read_input(path: str | None) -> Any:
    try:
        if path and path != "-":
            raw = Path(path).read_text(encoding="utf-8")
        else:
            raw = sys.stdin.read()
    except OSError as e:
        raise SystemExit(f"读取材料失败：{e}") from e
    if not raw.strip():
        raise SystemExit("没有输入：请传 FILE / -i FILE / --data JSON / --demo，或从 stdin 管道传入。")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"JSON 解析失败（第 {e.lineno} 行第 {e.colno} 列）：{e.msg}") from e


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(
        prog="python -m app.future_work",
        description="抽取并判定 Future Work 中仍开放的探索方向（提取式，无 LLM）。",
    )
    ap.add_argument("file", nargs="?", help="材料 JSON；也可用 -i / --data / stdin")
    ap.add_argument("-i", "--input", help="材料 JSON 路径")
    ap.add_argument("--data", help="材料 JSON 字符串")
    ap.add_argument("-o", "--out", help="输出 JSON 路径（默认 stdout）")
    ap.add_argument("--threshold", type=float, default=0.55, help="聚类相似度阈值（默认 0.55）")
    ap.add_argument(
        "--explicit-only",
        action="store_true",
        help="只采用 progress 中明确标注的解决证据，不自动判定后续论文",
    )
    ap.add_argument("--demo", action="store_true", help="用合成材料演示")
    ap.add_argument("--selftest", action="store_true", help="运行内置自检")
    ap.add_argument("--debug", action="store_true", help="打印聚类与判定统计")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.demo:
        data = demo_data()
    elif args.data:
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--data JSON 解析失败：{e.msg}") from e
    else:
        data = _read_input(args.input or args.file)

    result = analyze(data, threshold=max(0.1, min(0.95, args.threshold)),
                     explicit_only=args.explicit_only)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        try:
            Path(args.out).write_text(text, encoding="utf-8")
        except OSError as e:
            raise SystemExit(f"写入输出失败：{e}") from e
        print(f"[future_work] 已写入 {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)

    print(
        f"[future_work] 已解决 {len(result['resolved'])} 条 / "
        f"开放 {len(result['open_directions'])} 条",
        file=sys.stderr,
    )
    if args.debug:
        papers = load_papers(data)
        points = extract_points(papers)
        print(
            f"[future_work] 论文 {len(papers)} 篇 / 未来工作点 {len(points)} 条 / "
            f"阈值 {args.threshold}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
