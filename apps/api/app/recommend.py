"""个性化 arXiv 论文推荐 Agent（零密钥可跑，联网可选）。

数据流
──────
    arXiv Atom API ──feedparser / 标准库兜底──┐
    本地文献库 ──TF-IDF 余弦──┐               ├─> 相关性 + 分类 + 新鲜度 + 代码 + 反馈 ─> 排序
    点赞/踩 JSONL ──权重回流──┘               │
    GitHub 代码链接（摘要/评论；可选 API 检索）┘

为什么不是 Sickle：Sickle 走 OAI-PMH，适合全量收割；这里要的是“最新论文”，
arXiv Atom API + feedparser 更轻。feedparser 未安装时自动用 urllib + ElementTree，
所以本文件在零第三方依赖下也能跑（只有 networkx 那个图模块才需要额外依赖）。

反馈回流
────────
    feedback.jsonl 只追加 {"arxiv_id","action","features"}；
    每次运行重放全部事件，得到关键词/分类/作者/库文献/代码偏好权重，
    再以 ±0.20 的幅度调节基础分；history.json 记录已推荐项，默认不再重复推。
    这样没有“在线学习状态漂移”，权重随时可审计，删掉 JSONL 即冷启动。

用法
────
    python -m app.recommend --demo
    python -m app.recommend --library lib.json --candidates new.json
    python -m app.recommend --library lib.json --refresh --days 7
    python -m app.recommend --library lib.json --refresh --watch 60 --out rec.json
    python -m app.recommend --state-dir data/recommend --like 2601.10001
    python -m app.recommend --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.retrieval.bm25 import tokenize
from app.utils import get_logger, setup_logging, short

log = get_logger("ra.recommend")

APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STATE = APP_DIR / "data" / "recommend"
ATOM = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
_UA = "research-assistant/0.1 (recommend agent; mailto:research@example.invalid)"

# 只影响“推荐理由/关键词”的可读性，不影响召回；停用词表刻意小，避免误删领域词。
_STOP = {
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "with", "via",
    "using", "from", "by", "as", "at", "or", "is", "are", "be", "we", "our",
    "this", "that", "these", "those", "it", "its", "their", "not", "all",
    "more", "than", "into", "over", "under", "between", "through", "about",
    "across", "can", "how", "what", "why", "when", "where", "new", "novel",
    "learning", "model", "models", "method", "methods", "approach",
    "approaches", "paper", "study", "analysis", "based", "large", "towards",
    "toward", "results", "show", "we", "our", "they", "them", "also", "such",
    "two", "one", "use", "used", "using", "propose", "proposed",
}

_GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
    re.I,
)
_GITHUB_RESERVED = {
    "features", "topics", "about", "pricing", "login", "join", "sponsors",
    "marketplace", "orgs", "users", "settings", "notifications", "explore",
    "trending", "collections", "events", "security", "site", "readme", "apps",
    "enterprise", "team", "solutions", "resources", "customer-stories",
    "open-source", "github", "pulls", "issues", "discussions",
}
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", re.I)


# ────────────────────────── 基础工具 ──────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _parse_dt(value: Any) -> datetime | None:
    s = _clean(value)
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
        if not m:
            return None
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)


def _age_days(value: Any, now: datetime) -> int | None:
    dt = _parse_dt(value)
    if not dt:
        return None
    return max(0, (now - dt).days)


def _terms(text: Any) -> list[str]:
    out: list[str] = []
    for t in tokenize(str(text or "")):
        if t in _STOP or t.isdigit() or len(t) < 2:
            continue
        out.append(t)
    return out


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in re.split(r"[,;，；]", value) if x.strip()]
    if isinstance(value, (list, tuple, set)):
        out = []
        for x in value:
            if isinstance(x, dict):
                out.append(_clean(x.get("name") or x.get("term") or x.get("id")))
            else:
                out.append(_clean(x))
        return [x for x in out if x]
    return [_clean(value)]


def _stable_id(title: str) -> str:
    return "manual-" + hashlib.sha1(_clean(title).encode("utf-8")).hexdigest()[:10]


def _normalize(vec: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {k: v / norm for k, v in vec.items()} if norm > 0 else {}


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _read_json_any(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


# ────────────────────────── 数据模型 ──────────────────────────


@dataclass
class LibraryDoc:
    doc_id: str
    title: str
    abstract: str = ""
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    arxiv_id: str = ""
    url: str = ""

    def text(self) -> str:
        return " ".join([self.title, self.abstract, " ".join(self.tags)])


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    published: str = ""
    updated: str = ""
    url: str = ""
    comment: str = ""
    code_url: str = ""
    code_source: str = ""
    code_stars: int = 0
    code_confidence: str = ""
    keywords: list[str] = field(default_factory=list)
    source: str = "arxiv"


@dataclass
class State:
    dir: Path

    def __post_init__(self) -> None:
        self.dir = Path(self.dir)

    @property
    def feedback_path(self) -> Path:
        return self.dir / "feedback.jsonl"

    @property
    def history_path(self) -> Path:
        return self.dir / "history.json"

    @property
    def cache_path(self) -> Path:
        return self.dir / "code_cache.json"


# ────────────────────────── 输入归一化 ──────────────────────────


def load_library_data(data: Any) -> list[LibraryDoc]:
    """接受 {"docs":[...]} / {"library":[...]} / {"fetched":[...]} / 列表 / 字符串。"""
    if isinstance(data, dict):
        for key in ("docs", "library", "papers", "fetched", "items"):
            if isinstance(data.get(key), (list, dict)):
                data = data[key]
                break
        else:
            data = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
    if isinstance(data, str):
        data = [data]
    if not isinstance(data, (list, tuple)):
        data = [data]

    docs: list[LibraryDoc] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            title = _clean(item)
            docs.append(LibraryDoc(doc_id=f"doc_{i + 1}", title=title or f"doc_{i + 1}"))
            continue
        if not isinstance(item, dict):
            continue
        doc_id = _clean(item.get("doc_id") or item.get("id") or item.get("doi") or f"doc_{i + 1}")
        title = _clean(item.get("title") or item.get("name") or doc_id)
        year = item.get("year")
        try:
            year = int(year) if year not in (None, "") else None
        except (TypeError, ValueError):
            year = None
        docs.append(
            LibraryDoc(
                doc_id=doc_id,
                title=title or doc_id,
                abstract=_clean(item.get("abstract") or item.get("summary") or item.get("text")),
                tags=_as_str_list(item.get("tags") or item.get("keywords") or item.get("topics")),
                categories=_as_str_list(item.get("categories") or item.get("category") or item.get("fields")),
                authors=_as_str_list(item.get("authors") or item.get("author")),
                year=year,
                arxiv_id=_clean(item.get("arxiv_id") or ""),
                url=_clean(item.get("url") or ""),
            )
        )
    return [d for d in docs if d.title]


def load_candidate_data(data: Any) -> list[Paper]:
    """接受 {"candidates":[...]} / {"papers":[...]} / {"arxiv_candidates":[...]} / 列表。"""
    if isinstance(data, dict):
        for key in ("arxiv_candidates", "candidates", "papers", "items", "results"):
            if isinstance(data.get(key), (list, dict)):
                data = data[key]
                break
        else:
            data = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
    if isinstance(data, str):
        data = [data]
    if not isinstance(data, (list, tuple)):
        data = [data]

    papers: list[Paper] = []
    seen: set[str] = set()
    for item in data:
        if isinstance(item, str):
            title = _clean(item)
            if not title:
                continue
            pid = _stable_id(title)
            paper = Paper(arxiv_id=pid, title=title, source="file")
        elif isinstance(item, dict):
            title = _clean(item.get("title") or item.get("name"))
            if not title:
                continue
            raw_id = _clean(item.get("arxiv_id") or item.get("id") or item.get("entry_id") or item.get("url"))
            m = _ARXIV_ID_RE.search(raw_id) if raw_id else None
            if m:
                pid = m.group(1)
            elif raw_id and not raw_id.lower().startswith(("http://", "https://")):
                pid = raw_id
            else:
                pid = _stable_id(title)
            published = _clean(item.get("published") or item.get("published_at") or item.get("date"))
            if published:
                published = published[:10]
            paper = Paper(
                arxiv_id=pid,
                title=title,
                abstract=_clean(item.get("abstract") or item.get("summary")),
                authors=_as_str_list(item.get("authors") or item.get("author")),
                categories=_as_str_list(
                    item.get("categories") or item.get("category") or item.get("primary_category")
                ),
                published=published,
                updated=_clean(item.get("updated"))[:10],
                url=_clean(item.get("url") or item.get("link") or (f"https://arxiv.org/abs/{pid}" if m else "")),
                comment=_clean(item.get("comment")),
                code_url=_clean(item.get("code_url") or item.get("github") or item.get("repo")),
                code_source="input" if (item.get("code_url") or item.get("github") or item.get("repo")) else "",
                code_stars=int(item.get("stars") or item.get("code_stars") or 0),
                keywords=_as_str_list(item.get("keywords") or item.get("tags")),
                source=_clean(item.get("source") or "file"),
            )
            if not paper.code_url:
                code_url, code_source = extract_code(
                    (paper.abstract, "abstract"), (paper.comment, "comment")
                )
                if code_url:
                    paper.code_url, paper.code_source = code_url, code_source
        else:
            continue
        if paper.arxiv_id in seen:
            continue
        seen.add(paper.arxiv_id)
        papers.append(paper)
    return papers


# ────────────────────────── arXiv 刷新 ──────────────────────────


class FetchError(RuntimeError):
    pass


def _ssl_context() -> ssl.SSLContext | None:
    """Windows 上 Python 可能找不到系统根证书，优先用 certifi 的 CA 包。"""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except (ImportError, OSError, ssl.SSLError):
        return None


def _http_get(url: str, timeout: int = 30, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, **(headers or {})})
    context = _ssl_context()
    last: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504) and attempt == 0:
                time.sleep(1.5)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if attempt == 0:
                time.sleep(1.0)
                continue
    raise FetchError(str(last))


def _arxiv_id_from(raw: str) -> str:
    raw = _clean(raw)
    m = _ARXIV_URL_RE.search(raw)
    if m:
        raw = m.group(1)
    raw = re.sub(r"\.pdf$", "", raw, flags=re.I)
    m = _ARXIV_ID_RE.search(raw)
    return m.group(1) if m else raw


def build_arxiv_query(docs: list[LibraryDoc], days: int = 14) -> str:
    cats = Counter()
    terms = Counter()
    for d in docs:
        for c in d.categories:
            if c:
                cats[c] += 1
        for t in _terms(d.title):
            terms[t] += 3
        for t in _terms(d.abstract):
            terms[t] += 1
    cat_part = "(" + " OR ".join(f"cat:{c}" for c, _ in cats.most_common(2)) + ")" if cats else ""
    kw = [t for t, _ in terms.most_common(4)]
    kw_part = "(" + " OR ".join(f'all:"{t}"' for t in kw) + ")" if kw else ""
    query = " AND ".join(x for x in (cat_part, kw_part) if x) or "cat:cs.CL"
    if days and days > 0:
        end = _now()
        start = end - timedelta(days=days)
        query += f" AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    return query


def _arxiv_url(query: str, max_results: int) -> str:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max(1, min(100, max_results)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
        quote_via=urllib.parse.quote_plus,
    )
    return "https://export.arxiv.org/api/query?" + params


def _make_arxiv_paper(
    entry_id: str,
    title: str,
    abstract: str,
    authors: list[str],
    categories: list[str],
    published: str,
    updated: str,
    link: str,
    comment: str,
) -> Paper:
    aid = _arxiv_id_from(entry_id)
    code_url, code_source = extract_code((abstract, "abstract"), (comment, "comment"))
    return Paper(
        arxiv_id=aid,
        title=_clean(title),
        abstract=_clean(abstract),
        authors=authors,
        categories=categories,
        published=_clean(published)[:10],
        updated=_clean(updated)[:10],
        url=link or (f"https://arxiv.org/abs/{aid}" if aid else ""),
        comment=_clean(comment),
        code_url=code_url,
        code_source=code_source,
        source="arxiv",
    )


def _parse_atom_stdlib(raw: bytes) -> list[Paper]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise FetchError(f"arXiv 返回的不是合法 Atom XML: {e}") from e
    ns = {"a": ATOM, "arxiv": ARXIV_NS}
    out: list[Paper] = []
    for entry in root.findall("a:entry", ns):
        eid = entry.findtext("a:id", default="", namespaces=ns)
        title = entry.findtext("a:title", default="", namespaces=ns)
        summary = entry.findtext("a:summary", default="", namespaces=ns)
        published = entry.findtext("a:published", default="", namespaces=ns)
        updated = entry.findtext("a:updated", default="", namespaces=ns)
        comment = entry.findtext("arxiv:comment", default="", namespaces=ns)
        authors = [
            _clean(x.findtext("a:name", default="", namespaces=ns))
            for x in entry.findall("a:author", ns)
        ]
        categories = [c.get("term", "") for c in entry.findall("a:category", ns)]
        link = ""
        for ln in entry.findall("a:link", ns):
            if ln.get("rel") in (None, "alternate"):
                link = ln.get("href", "")
                break
        paper = _make_arxiv_paper(
            eid, title, summary, [a for a in authors if a], [c for c in categories if c],
            published, updated, link, comment,
        )
        if paper.title:
            out.append(paper)
    return out


def parse_arxiv_feed(raw: bytes) -> list[Paper]:
    """feedparser 优先（装了就用），否则标准库解析；两条路径产出同一数据结构。"""
    try:
        import feedparser  # type: ignore
    except Exception:  # noqa: BLE001
        feedparser = None
    if feedparser is not None:
        try:
            feed = feedparser.parse(raw)
            entries = getattr(feed, "entries", [])
            out = []
            for e in entries:
                authors = [_clean(a.get("name")) for a in e.get("authors", [])]
                cats = [_clean(t.get("term")) for t in e.get("tags", [])]
                paper = _make_arxiv_paper(
                    str(e.get("id", "")),
                    str(e.get("title", "")),
                    str(e.get("summary", "")),
                    [a for a in authors if a],
                    [c for c in cats if c],
                    str(e.get("published", "")),
                    str(e.get("updated", "")),
                    str(e.get("link", "")),
                    str(e.get("arxiv_comment") or e.get("comment") or ""),
                )
                if paper.title:
                    out.append(paper)
            if out:
                return out
        except Exception:  # noqa: BLE001
            log.debug("feedparser 解析失败，回退标准库", exc_info=True)
    return _parse_atom_stdlib(raw)


def fetch_arxiv(query: str, max_results: int = 20, days: int = 14, timeout: int = 30) -> list[Paper]:
    try:
        raw = _http_get(_arxiv_url(query, max_results), timeout=timeout)
    except Exception as e:  # noqa: BLE001
        raise FetchError(
            f"arXiv 拉取失败：{e}。可离线用 --candidates papers.json；"
            "若公司网络限制 export.arxiv.org，请配置代理后重试。"
        ) from e
    papers = parse_arxiv_feed(raw)
    if days and days > 0:
        cutoff = _now() - timedelta(days=days)
        papers = [p for p in papers if (_parse_dt(p.published) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
    papers.sort(key=lambda p: (p.published, p.arxiv_id), reverse=True)
    return papers


# ────────────────────────── 代码链接 ──────────────────────────


def extract_code(*texts: tuple[str, str]) -> tuple[str, str]:
    """从摘要/评论提取 GitHub 仓库；返回 (url, source)，找不到返回 ("","")."""
    for text, source in texts:
        for m in _GITHUB_RE.finditer(text or ""):
            owner = m.group(1).strip(".")
            repo = m.group(2).strip(".").rstrip(".,;:)]}'\"")
            repo = re.sub(r"\.git$", "", repo, flags=re.I)
            if not owner or not repo:
                continue
            if owner.lower() in _GITHUB_RESERVED:
                continue
            if repo.lower() in {"issues", "pulls", "discussions", "actions", "releases", "wiki"}:
                continue
            return f"https://github.com/{owner}/{repo}", source
    return "", ""


class GitHubClient:
    """可选：GitHub 仓库检索 + README 自动核验。默认关闭，避免无谓的网络/限流。"""

    def __init__(self, token: str = "", cache_path: Path | None = None, timeout: int = 20) -> None:
        self.token = token
        self.cache_path = cache_path
        self.timeout = timeout
        self.cache: dict[str, dict[str, Any]] = {}
        self.disabled = False
        self._last_search = 0.0
        if cache_path and cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.cache = {}

    def save(self) -> None:
        if self.cache_path:
            _write_json(self.cache_path, self.cache)

    def _cached(self, key: str) -> dict[str, Any] | None:
        item = self.cache.get(key)
        if not item:
            return None
        try:
            checked = datetime.fromisoformat(item.get("checked_at", ""))
        except ValueError:
            return None
        if (_now() - checked).days > 14:
            return None
        return item

    def _headers(self, raw: bool = False) -> dict[str, str]:
        h = {"Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _search(self, query: str) -> list[dict[str, Any]]:
        if self.disabled:
            return []
        gap = 2.2 if self.token else 6.5
        wait = gap - (time.monotonic() - self._last_search)
        if wait > 0:
            time.sleep(wait)
        self._last_search = time.monotonic()
        url = (
            "https://api.github.com/search/repositories?"
            + urllib.parse.urlencode({"q": query, "per_page": 5, "sort": "stars", "order": "desc"})
        )
        try:
            data = json.loads(_http_get(url, timeout=self.timeout, headers=self._headers()))
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                self.disabled = True
                raise FetchError("GitHub 检索限流（未认证 10 次/分钟），本轮停止代码检索") from e
            raise
        return list(data.get("items") or [])

    def _readme(self, full_name: str) -> str:
        url = f"https://api.github.com/repos/{full_name}/readme"
        try:
            raw = _http_get(url, timeout=self.timeout, headers=self._headers(raw=True))
            return raw[:200_000].decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ""
            if e.code in (403, 429):
                self.disabled = True
            return ""

    @staticmethod
    def _verify(paper: Paper, readme: str) -> tuple[bool, str]:
        low = readme.lower()
        if _ARXIV_ID_RE.fullmatch(paper.arxiv_id or "") and paper.arxiv_id.lower() in low:
            return True, "high"
        words = [t for t in _terms(paper.title) if len(t) >= 3]
        if not words:
            return False, ""
        hit = sum(1 for w in words if w in low) / len(words)
        return (hit >= 0.6), ("medium" if hit >= 0.6 else "")

    def find(self, paper: Paper) -> dict[str, Any]:
        cached = self._cached(paper.arxiv_id)
        if cached is not None:
            return cached
        item: dict[str, Any] = {"url": "", "checked_at": _iso(_now())}
        try:
            if _ARXIV_ID_RE.fullmatch(paper.arxiv_id or ""):
                items = self._search(f'"{paper.arxiv_id}" in:readme')
            else:
                items = self._search(f'"{paper.title}" in:readme')
            for repo in items[:3]:
                full = repo.get("full_name") or ""
                if not full:
                    continue
                ok, conf = self._verify(paper, self._readme(full))
                if ok:
                    item = {
                        "url": repo.get("html_url") or f"https://github.com/{full}",
                        "stars": int(repo.get("stargazers_count") or 0),
                        "confidence": conf,
                        "full_name": full,
                        "checked_at": _iso(_now()),
                    }
                    break
        except FetchError:
            raise
        except Exception:  # noqa: BLE001
            log.debug("GitHub 检索异常", exc_info=True)
        self.cache[paper.arxiv_id] = item
        return item


def enrich_with_github(
    scored: list[tuple[Paper, float]],
    state: State,
    top: int = 3,
    timeout: int = 20,
) -> list[str]:
    """只对预排序前 top 篇做检索；返回警告列表。"""
    token = os.environ.get("GITHUB_TOKEN", "")
    try:
        from app.config import get_settings

        settings = get_settings()
        token = token or getattr(settings, "GITHUB_TOKEN", "")
    except Exception:  # noqa: BLE001
        pass
    client = GitHubClient(token=token, cache_path=state.cache_path, timeout=timeout)
    warnings: list[str] = []
    try:
        for paper, _ in scored[:max(0, top)]:
            if paper.code_url:
                continue
            hit = client.find(paper)
            if hit.get("url"):
                paper.code_url = hit["url"]
                paper.code_source = "github_search"
                paper.code_stars = int(hit.get("stars") or 0)
                paper.code_confidence = hit.get("confidence") or "medium"
    except FetchError as e:
        warnings.append(str(e))
    finally:
        client.save()
    if client.disabled:
        warnings.append("GitHub 检索已触发限流并停止；可设置 GITHUB_TOKEN 提高配额。")
    return warnings


# ────────────────────────── 相关性索引 ──────────────────────────


class LibraryIndex:
    """文献库 TF-IDF 余弦索引：候选同时对齐“最佳单篇”和“整体画像”。"""

    def __init__(self, docs: list[LibraryDoc]) -> None:
        self.docs = docs
        token_lists = [_terms(d.text()) for d in docs]
        self.doc_terms = [set(t) for t in token_lists]
        self.doc_counts = [Counter(t) for t in token_lists]
        n = len(docs)
        df = Counter(t for toks in token_lists for t in set(toks))
        self.idf = {t: math.log((n + 1) / (df[t] + 1)) + 1.0 for t in df}
        self.max_idf = math.log(n + 1) + 1.0 if n else 1.0
        self.vecs = [self._vec(c) for c in self.doc_counts]
        profile: dict[str, float] = {}
        for v in self.vecs:
            for k, x in v.items():
                profile[k] = profile.get(k, 0.0) + x
        self.profile = _normalize(profile)
        self.categories = Counter(c.lower() for d in docs for c in d.categories)

    def _vec(self, counts: Counter) -> dict[str, float]:
        return _normalize(
            {t: (1.0 + math.log(n)) * self.idf.get(t, self.max_idf) for t, n in counts.items()}
        )

    def score(self, text: str) -> tuple[float, int, float, list[float]]:
        q = self._vec(Counter(_terms(text)))
        sims = [_dot(q, v) for v in self.vecs]
        best = max(range(len(sims)), key=lambda i: sims[i]) if sims else -1
        return (sims[best] if best >= 0 else 0.0), best, _dot(q, self.profile), sims

    def paper_keywords(self, paper: Paper, limit: int = 8) -> list[str]:
        counts = Counter(_terms(paper.title + " " + paper.abstract))
        title_counts = Counter(_terms(paper.title))
        scored = {
            t: (n + 2.0 * title_counts.get(t, 0)) * self.idf.get(t, self.max_idf)
            for t, n in counts.items()
        }
        return [t for t, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]

    def shared_categories(self, paper: Paper) -> list[str]:
        lib = set(self.categories)
        return [c for c in paper.categories if c.lower() in lib]


# ────────────────────────── 反馈模型 ──────────────────────────


@dataclass
class FeedbackModel:
    keywords: dict[str, float] = field(default_factory=dict)
    categories: dict[str, float] = field(default_factory=dict)
    authors: dict[str, float] = field(default_factory=dict)
    docs: dict[str, float] = field(default_factory=dict)
    code: float = 0.0
    events: int = 0
    likes: int = 0
    dislikes: int = 0
    invalid: int = 0


_LR = 0.20
_CLAMP = 0.8


def _bump(d: dict[str, float], key: str, delta: float) -> None:
    if not key:
        return
    d[key] = _clamp(d.get(key, 0.0) + delta, -_CLAMP, _CLAMP)


def build_model(events: list[dict[str, Any]]) -> FeedbackModel:
    m = FeedbackModel()
    for e in events:
        action = _clean(e.get("action")).lower()
        if action in ("like", "up", "赞", "点赞"):
            label = 1
            m.likes += 1
        elif action in ("dislike", "down", "踩", "点踩"):
            label = -1
            m.dislikes += 1
        else:
            m.invalid += 1
            continue
        m.events += 1
        f = e.get("features") or {}
        for t in list(f.get("keywords") or [])[:6]:
            _bump(m.keywords, _clean(t), _LR * label)
        for c in list(f.get("categories") or [])[:2]:
            _bump(m.categories, _clean(c), _LR * label * 0.8)
        for a in list(f.get("authors") or [])[:2]:
            _bump(m.authors, _clean(a), _LR * label * 0.5)
        if f.get("match_doc"):
            _bump(m.docs, _clean(f["match_doc"]), _LR * label * 0.7)
        if f.get("has_code"):
            m.code = _clamp(m.code + _LR * label * 0.6, -_CLAMP, _CLAMP)
    return m


def feedback_adjust(model: FeedbackModel, paper: Paper, best_doc_id: str) -> float:
    def avg(weights: dict[str, float], keys: list[str]) -> float:
        vals = [weights.get(k, 0.0) for k in keys if k]
        return sum(vals) / len(vals) if vals else 0.0

    kw = avg(model.keywords, paper.keywords[:6])
    cat = avg(model.categories, paper.categories[:2])
    auth = avg(model.authors, paper.authors[:3])
    doc = model.docs.get(best_doc_id, 0.0) if best_doc_id else 0.0
    code = model.code if paper.code_url else 0.0
    adj = 0.18 * kw + 0.08 * cat + 0.06 * auth + 0.08 * doc + 0.04 * code
    return _clamp(adj, -0.20, 0.20)


def _top_weights(d: dict[str, float], n: int = 5) -> list[dict[str, Any]]:
    rows = [
        {"key": k, "weight": round(v, 4)}
        for k, v in sorted(d.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
        if abs(v) > 1e-9
    ]
    return rows[:n]


def feedback_summary(model: FeedbackModel) -> dict[str, Any]:
    return {
        "events": model.events,
        "likes": model.likes,
        "dislikes": model.dislikes,
        "invalid": model.invalid,
        "top_keywords": _top_weights(model.keywords),
        "top_categories": _top_weights(model.categories),
        "top_docs": _top_weights(model.docs),
        "code_bias": round(model.code, 4),
    }


def load_events(state: State) -> list[dict[str, Any]]:
    path = state.feedback_path
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            if isinstance(e, dict):
                out.append(e)
        except json.JSONDecodeError:
            continue
    return out


def record_event(
    state: State,
    arxiv_id: str,
    action: str,
    features: dict[str, Any] | None = None,
    note: str = "",
) -> dict[str, Any]:
    state.dir.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": _iso(_now()),
        "arxiv_id": _clean(arxiv_id),
        "action": _clean(action).lower(),
        "note": _clean(note),
        "features": features or {},
    }
    with state.feedback_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def load_history(state: State) -> dict[str, dict[str, Any]]:
    if not state.history_path.exists():
        return {}
    try:
        data = json.loads(state.history_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_history(state: State, history: dict[str, dict[str, Any]]) -> None:
    _write_json(state.history_path, history)


# ────────────────────────── 打分与解释 ──────────────────────────


def _related(index: LibraryIndex, paper: Paper, sims: list[float]) -> list[dict[str, Any]]:
    if not sims:
        return []
    cand_terms = set(_terms(paper.title + " " + paper.abstract))
    rows = []
    for i in sorted(range(len(sims)), key=lambda j: (-sims[j], j))[:2]:
        sim = sims[i]
        if sim <= 0.005:
            continue
        doc = index.docs[i]
        shared = [t for t in paper.keywords if t in index.doc_terms[i]]
        if not shared:
            shared = sorted(cand_terms & index.doc_terms[i], key=lambda t: (-index.idf.get(t, 0), t))
        rows.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "similarity": round(sim, 4),
                "shared_terms": shared[:5],
            }
        )
    return rows


def _reason(
    paper: Paper,
    best_doc: LibraryDoc | None,
    best_sim: float,
    shared_terms: list[str],
    shared_cats: list[str],
    fb: float,
    now: datetime,
    has_library: bool,
) -> str:
    parts = []
    if best_doc and best_sim > 0.02:
        shared = "、".join(shared_terms[:3]) or "相近主题"
        parts.append(f"与库中《{short(best_doc.title, 46)}》最相关（相似度 {best_sim:.2f}，共享 {shared}）")
    elif has_library:
        parts.append("与库中文献的主题关联较弱")
    else:
        parts.append("本地文献库为空，暂按新鲜度与可复现性推荐")
    if shared_cats:
        parts.append(f"同属 {', '.join(shared_cats[:2])}")
    if paper.code_url:
        if paper.code_source == "github_search":
            star = f"，stars={paper.code_stars}" if paper.code_stars else ""
            parts.append(f"GitHub 检索到含本文 arXiv 编号的仓库 {paper.code_url}（已自动核验 README{star}）")
        else:
            where = {"abstract": "摘要", "comment": "arXiv 评论", "input": "输入数据"}.get(paper.code_source, "摘要/评论")
            parts.append(f"{where}中给出开源实现 {paper.code_url}（可复现性高）")
    else:
        parts.append("未在摘要、评论中发现公开代码链接，复现需自行实现或等待作者开源")
    age = _age_days(paper.published, now)
    if age is not None:
        parts.append(f"{age} 天前发布")
    if abs(fb) >= 0.02:
        parts.append(f"反馈回流：{'点赞过的相近主题加权' if fb > 0 else '踩过相近主题降权'} {fb:+.3f}")
    return "；".join(parts) + "。"


def score_paper(
    paper: Paper,
    index: LibraryIndex,
    model: FeedbackModel,
    now: datetime,
) -> dict[str, Any]:
    text = f"{paper.title}\n{paper.abstract}\n{paper.comment}"
    best_sim, best_i, profile_sim, sims = index.score(text)

    input_terms: list[str] = []
    for k in paper.keywords:
        input_terms.extend(_terms(k) or [_clean(k).lower()])
    paper.keywords = list(
        dict.fromkeys(input_terms + index.paper_keywords(paper))
    )[:8]

    best_doc = index.docs[best_i] if best_i >= 0 and best_sim > 0 else None
    related = _related(index, paper, sims)
    shared_terms = [t for t in paper.keywords if best_doc and t in index.doc_terms[best_i]]
    if not shared_terms and best_doc:
        shared_terms = [r for row in related for r in row["shared_terms"]][:4]
    shared_cats = index.shared_categories(paper)
    fb = feedback_adjust(model, paper, best_doc.doc_id if best_doc else "")

    best_norm = min(1.0, best_sim / 0.22)
    prof_norm = min(1.0, profile_sim / 0.18)
    rel = 0.72 * (0.70 * best_norm + 0.30 * prof_norm)
    cat_bonus = min(0.08, 0.04 * len(shared_cats))
    age = _age_days(paper.published, now)
    fresh = 0.07 * (max(0.0, 1.0 - age / 30.0) if age is not None else 0.5)
    code_bonus = 0.13 if paper.code_url else 0.0
    total = _clamp(rel + cat_bonus + fresh + code_bonus + fb, 0.0, 1.0)

    if paper.code_url:
        repro = {
            "has_code": True,
            "url": paper.code_url,
            "source": paper.code_source or "input",
            "stars": paper.code_stars or None,
            "confidence": paper.code_confidence or ("high" if paper.code_source in ("abstract", "comment", "input") else "medium"),
            "note": "代码链接来自论文元数据；GitHub 检索结果已用 README 自动核验，使用前仍建议人工确认。",
        }
    else:
        repro = {
            "has_code": False,
            "url": "",
            "source": "",
            "stars": None,
            "confidence": "none",
            "note": "摘要、评论中未发现代码链接；可加 --github-search 进一步检索，不代表作者一定未开源。",
        }

    why = (
        f"与《{short(best_doc.title, 46)}》主题相近（相似度 {best_sim:.2f}）"
        if best_doc and best_sim > 0.02
        else ("库中暂无高相似文献" if index.docs else "文献库为空")
    )
    if shared_terms:
        why += f"，共享关键词 {'、'.join(shared_terms[:3])}"
    if paper.code_url:
        why += "；有公开代码"
    rec = {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "published": paper.published,
        "updated": paper.updated,
        "url": paper.url,
        "authors": paper.authors,
        "categories": paper.categories,
        "score": round(total, 4),
        "reason": _reason(paper, best_doc, best_sim, shared_terms, shared_cats, fb, now, bool(index.docs)),
        "why_relevant": why + "。",
        "reproducibility": repro,
        "related_library": related,
        "score_breakdown": {
            "relevance": round(rel, 4),
            "category": round(cat_bonus, 4),
            "freshness": round(fresh, 4),
            "code": round(code_bonus, 4),
            "feedback": round(fb, 4),
            "total": round(total, 4),
        },
        "feedback_signal": round(fb, 4),
        "source": paper.source,
        "_paper": paper,
        "_score_raw": total,
        "_features": {
            "keywords": paper.keywords[:6],
            "categories": paper.categories[:3],
            "authors": paper.authors[:3],
            "match_doc": best_doc.doc_id if best_doc else "",
            "has_code": bool(paper.code_url),
        },
    }
    return rec


def _public(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


# ────────────────────────── 主流程 ──────────────────────────


def run_cycle(
    docs: list[LibraryDoc],
    papers: list[Paper],
    state: State,
    *,
    top: int = 5,
    min_score: float = 0.0,
    include_seen: bool = False,
    use_github: bool = False,
    code_top: int = 3,
    source_label: str = "file",
    query_label: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or _now()
    warnings: list[str] = []
    index = LibraryIndex(docs)
    events = load_events(state)
    model = build_model(events)
    history = load_history(state)

    if not docs:
        warnings.append("文献库为空：相关性分量恒为 0，推荐主要由新鲜度与代码可用性决定。")
    if not papers:
        warnings.append("没有候选论文。")

    seen = set(history)
    candidates = papers if include_seen else [p for p in papers if p.arxiv_id not in seen]
    suppressed = len(papers) - len(candidates)
    if suppressed:
        warnings.append(f"已过滤 {suppressed} 篇历史推荐（--include-seen 可重新展示）。")

    prelim = [score_paper(p, index, model, now) for p in candidates]
    if use_github and prelim:
        ranked_pre = sorted(prelim, key=lambda r: (-r["_score_raw"], r["arxiv_id"]))
        warnings.extend(enrich_with_github([(r["_paper"], r["_score_raw"]) for r in ranked_pre], state, top=code_top))
        prelim = [score_paper(p, index, model, now) for p in candidates]

    picked = sorted(
        [r for r in prelim if r["score"] >= min_score],
        key=lambda r: (-r["_score_raw"], r["arxiv_id"]),
    )[: max(0, top)]

    if not picked and candidates:
        warnings.append(f"所有候选均低于 --min-score {min_score:.3f}。")
    if not candidates and papers:
        warnings.append("所有候选都已推荐过；如需重复展示请加 --include-seen。")

    for rank, rec in enumerate(picked, 1):
        rec["rank"] = rank
        history[rec["_paper"].arxiv_id] = {
            "title": rec["title"],
            "ts": _iso(now),
            "score": rec["score"],
            "features": rec["_features"],
        }
    save_history(state, history)

    public = [_public(r) for r in picked]
    return {
        "recommendations": public,
        "meta": {
            "generated_at": _iso(now),
            "source": source_label,
            "query": query_label,
            "library": {
                "docs": len(docs),
                "categories": sorted({c for d in docs for c in d.categories}),
            },
            "candidates": len(papers),
            "after_seen_filter": len(candidates),
            "suppressed_seen": suppressed,
            "returned": len(public),
            "github_search": bool(use_github),
            "state_dir": str(state.dir),
            "feedback": feedback_summary(model),
            "warnings": warnings,
        },
    }


def record_feedback_from_history(
    state: State,
    history: dict[str, dict[str, Any]],
    items: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    recorded = []
    for arxiv_id, action in items:
        entry = history.get(arxiv_id) or {}
        features = entry.get("features") or {}
        recorded.append(record_event(state, arxiv_id, action, features))
    return recorded


# ────────────────────────── 演示数据 ──────────────────────────


def _days_ago(n: int) -> str:
    return (_now() - timedelta(days=n)).strftime("%Y-%m-%d")


def demo_library() -> list[LibraryDoc]:
    return [
        LibraryDoc(
            doc_id="doc_survey",
            title="Efficient Transformers: A Survey",
            abstract="Efficient transformers reduce the quadratic cost of self-attention for long sequences "
            "using sparse attention, low-rank approximations, linear attention and memory-efficient kernels.",
            tags=["efficient transformer", "sparse attention", "long sequence"],
            categories=["cs.CL", "cs.LG"],
            year=2020,
        ),
        LibraryDoc(
            doc_id="doc_rag",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            abstract="Retrieval-augmented generation combines dense passage retrieval with a sequence-to-sequence "
            "language model, improving factual accuracy on knowledge-intensive question answering.",
            tags=["retrieval augmented generation", "dense retrieval", "question answering"],
            categories=["cs.CL"],
            year=2020,
        ),
        LibraryDoc(
            doc_id="doc_attn",
            title="Attention Is All You Need",
            abstract="The Transformer architecture relies entirely on self-attention and is the foundation of "
            "modern language models.",
            tags=["transformer", "self-attention", "sequence transduction"],
            categories=["cs.CL", "cs.LG"],
            year=2017,
        ),
    ]


def demo_candidates() -> list[Paper]:
    return load_candidate_data(
        {
            "candidates": [
                {
                    "arxiv_id": "2610.10001",
                    "title": "Sparse Attention for Long-Context Language Models",
                    "abstract": "We propose a sparse attention mechanism that scales transformers to long documents "
                    "while preserving accuracy on language modeling and retrieval tasks. "
                    "Code: https://github.com/demo-lab/sparse-attention",
                    "authors": ["A. Researcher", "B. Builder"],
                    "categories": ["cs.CL"],
                    "published": _days_ago(2),
                    "url": "https://arxiv.org/abs/2610.10001",
                    "source": "demo",
                },
                {
                    "arxiv_id": "2610.10002",
                    "title": "Efficient Attention with Learned Sparsity",
                    "abstract": "A learned sparsity pattern reduces the memory cost of self-attention for long sequences "
                    "and improves inference throughput.",
                    "authors": ["C. Scholar"],
                    "categories": ["cs.LG"],
                    "published": _days_ago(3),
                    "url": "https://arxiv.org/abs/2610.10002",
                    "source": "demo",
                },
                {
                    "arxiv_id": "2610.10003",
                    "title": "Retrieval-Augmented Generation with Adaptive Evidence Selection",
                    "abstract": "We improve retrieval-augmented generation by adaptively selecting evidence passages "
                    "for knowledge-intensive question answering.",
                    "authors": ["D. Retriever"],
                    "categories": ["cs.CL"],
                    "published": _days_ago(1),
                    "url": "https://arxiv.org/abs/2610.10003",
                    "comment": "Code: https://github.com/demo-lab/adaptive-rag",
                    "source": "demo",
                },
                {
                    "arxiv_id": "2610.10004",
                    "title": "A Survey of Efficient Attention Mechanisms",
                    "abstract": "This survey reviews sparse, linear and low-rank attention, with an emphasis on long-context "
                    "language modeling.",
                    "authors": ["E. Surveyor"],
                    "categories": ["cs.CL"],
                    "published": _days_ago(5),
                    "url": "https://arxiv.org/abs/2610.10004",
                    "source": "demo",
                },
                {
                    "arxiv_id": "2610.10005",
                    "title": "Vision-Language Pretraining for Remote Sensing",
                    "abstract": "A multimodal pretraining recipe for satellite imagery and geospatial question answering.",
                    "authors": ["F. Vision"],
                    "categories": ["cs.CV"],
                    "published": _days_ago(2),
                    "url": "https://arxiv.org/abs/2610.10005",
                    "source": "demo",
                },
                {
                    "arxiv_id": "2610.10006",
                    "title": "Long-Context Retrieval for Scientific Question Answering",
                    "abstract": "We combine dense retrieval with long-context language models for scientific literature "
                    "question answering.",
                    "authors": ["G. Scientist"],
                    "categories": ["cs.CL"],
                    "published": _days_ago(10),
                    "url": "https://arxiv.org/abs/2610.10006",
                    "source": "demo",
                },
            ]
        }
    )


# ────────────────────────── 自测 ──────────────────────────


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
            print(f"[XX] {name}" + (f"  -> {detail}" if detail else ""))

    # 1. 输入形状
    lib = load_library_data(
        {"docs": [{"doc_id": "d1", "title": "Sparse Attention", "categories": "cs.CL"}]}
    )
    ok("文献库 docs 形状", len(lib) == 1 and lib[0].categories == ["cs.CL"])
    ok("文献库字符串形状", len(load_library_data(["Paper A", "Paper B"])) == 2)
    cands = load_candidate_data(
        {"arxiv_candidates": [{"id": "http://arxiv.org/abs/2601.01234v2", "title": "T", "published": "2026-10-20"}]}
    )
    ok("候选 arXiv URL 提取 id", cands and cands[0].arxiv_id == "2601.01234", cands[0].arxiv_id if cands else "none")
    ok("候选日期截断", cands and cands[0].published == "2026-10-20")
    ok("候选无 id 稳定哈希", load_candidate_data(["Some Title"])[0].arxiv_id == _stable_id("Some Title"))

    # 2. Atom 解析
    atom = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2601.12345v1</id>
        <title>Sparse Attention Test</title>
        <summary>We study sparse attention. Code: https://github.com/demo/sparse.</summary>
        <published>2026-10-20T10:00:00Z</published>
        <updated>2026-10-21T10:00:00Z</updated>
        <author><name>Alice</name></author>
        <category term="cs.CL"/>
        <arxiv:comment>12 pages</arxiv:comment>
        <link rel="alternate" href="http://arxiv.org/abs/2601.12345v1"/>
      </entry>
    </feed>"""
    feed = parse_arxiv_feed(atom)
    ok("Atom 解析条数", len(feed) == 1)
    ok("Atom 去版本号", feed and feed[0].arxiv_id == "2601.12345")
    ok("Atom 作者/分类", feed and feed[0].authors == ["Alice"] and feed[0].categories == ["cs.CL"])
    ok("Atom 摘要代码识别", feed and feed[0].code_url == "https://github.com/demo/sparse")
    ok("Atom 评论保留", feed and feed[0].comment == "12 pages")

    # 3. 代码链接
    ok("评论区代码识别", extract_code(("", "abstract"), ("Code: https://github.com/a/b.git", "comment")) == ("https://github.com/a/b", "comment"))
    ok("非仓库 GitHub 链接忽略", extract_code(("https://github.com/features/actions", "abstract"))[0] == "")
    ok("无代码返回空", extract_code(("no code here", "abstract")) == ("", ""))

    # 4. 查询
    q = build_arxiv_query(lib, days=7)
    ok("查询含分类", "cat:cs.CL" in q, q)
    ok("查询含日期", "submittedDate:" in q, q)

    # 5. 相关性排序
    docs = demo_library()
    index = LibraryIndex(docs)
    model = FeedbackModel()
    now = _now()
    papers = demo_candidates()
    scored = {p.arxiv_id: score_paper(p, index, model, now) for p in papers}
    ok("稀疏注意力候选最相关库文献", scored["2610.10001"]["related_library"][0]["doc_id"] == "doc_survey",
       str(scored["2610.10001"]["related_library"]))
    ok("RAG 候选最相关库文献", scored["2610.10003"]["related_library"][0]["doc_id"] == "doc_rag",
       str(scored["2610.10003"]["related_library"]))
    ok("无关 CV 候选分数最低", scored["2610.10005"]["score"] < scored["2610.10001"]["score"])

    # 6. 代码加分
    no_code = load_candidate_data(
        {"candidates": [{"arxiv_id": "2610.10001", "title": "Sparse Attention for Long-Context Language Models",
                         "abstract": "We propose a sparse attention mechanism for long documents.",
                         "categories": ["cs.CL"], "published": _days_ago(2)}]}
    )[0]
    with_code = load_candidate_data(
        {"candidates": [{"arxiv_id": "2610.10001", "title": "Sparse Attention for Long-Context Language Models",
                         "abstract": "We propose a sparse attention mechanism for long documents. "
                                     "https://github.com/demo-lab/sparse-attention",
                         "categories": ["cs.CL"], "published": _days_ago(2)}]}
    )[0]
    s_no = score_paper(no_code, index, model, now)["score"]
    s_yes = score_paper(with_code, index, model, now)["score"]
    ok("代码链接提高分数", s_yes > s_no, f"{s_yes} vs {s_no}")

    # 7. 反馈回流
    rec1 = scored["2610.10001"]
    like_model = build_model([{"action": "like", "features": rec1["_features"]}])
    sib = load_candidate_data(
        {"candidates": [{"arxiv_id": "2610.10002", "title": "Efficient Attention with Learned Sparsity",
                         "abstract": "A learned sparsity pattern reduces the memory cost of self-attention for long sequences.",
                         "categories": ["cs.LG"], "published": _days_ago(3)}]}
    )[0]
    base = score_paper(sib, index, FeedbackModel(), now)["score"]
    boosted = score_paper(sib, index, like_model, now)["score"]
    ok("点赞提升相近论文分数", boosted > base, f"{boosted} vs {base}")
    ok("点赞写入关键词权重", like_model.keywords.get("sparse", 0) > 0, str(like_model.keywords))
    dislike_model = build_model([{"action": "dislike", "features": rec1["_features"]}])
    dropped = score_paper(sib, index, dislike_model, now)["score"]
    ok("点踩降低相近论文分数", dropped < base, f"{dropped} vs {base}")

    # 8. 理由字段
    rec = scored["2610.10001"]
    ok("理由含库中论文", "Efficient Transformers: A Survey" in rec["reason"] or "主题" in rec["reason"], rec["reason"])
    ok("理由含代码链接", "github.com" in rec["reason"], rec["reason"])
    ok("无代码理由明确说明", "未在摘要" in scored["2610.10004"]["reason"], scored["2610.10004"]["reason"])
    ok("可复现性字段", rec["reproducibility"]["has_code"] and rec["reproducibility"]["url"].startswith("https://github.com/"))

    # 9. 历史与已读过滤
    with tempfile.TemporaryDirectory() as td:
        state = State(Path(td))
        first = run_cycle(docs, papers, state, top=10, now=now)
        second = run_cycle(docs, papers, state, top=10, now=now)
        third = run_cycle(docs, papers, state, top=10, now=now, include_seen=True)
        ok("首次推荐全部候选", first["meta"]["returned"] == 6, str(first["meta"]["returned"]))
        ok("二次推荐过滤已读", second["meta"]["returned"] == 0 and second["meta"]["suppressed_seen"] == 6,
           str(second["meta"]))
        ok("include-seen 恢复", third["meta"]["returned"] == 6)
        ok("历史文件落盘", (Path(td) / "history.json").exists())
        ok("反馈事件追加", (record_event(state, "2610.10001", "like") is not None)
           and (Path(td) / "feedback.jsonl").exists())
        ok("反馈事件可读回", len(load_events(state)) == 1)
        ok("反馈特征可回查", load_history(state)["2610.10001"]["features"]["keywords"])

    # 10. 空库 / 无候选 / 中文
    empty = run_cycle([], papers[:1], State(Path(tempfile.mkdtemp())), top=5, now=now)
    ok("空库不崩溃且有警告", empty["recommendations"] and any("文献库为空" in w for w in empty["meta"]["warnings"]))
    no_c = run_cycle(docs, [], State(Path(tempfile.mkdtemp())), top=5, now=now)
    ok("无候选不崩溃", no_c["recommendations"] == [] and any("没有候选" in w for w in no_c["meta"]["warnings"]))
    zh_docs = load_library_data([{"doc_id": "zh1", "title": "稀疏注意力与长文本建模", "abstract": "面向长文本的稀疏注意力方法"}])
    zh_paper = load_candidate_data([{"id": "zh-paper", "title": "稀疏注意力长文本检索", "abstract": "长文本稀疏注意力"}])[0]
    zh = score_paper(zh_paper, LibraryIndex(zh_docs), FeedbackModel(), now)
    ok("中文相关性可计算", zh["related_library"] and zh["related_library"][0]["similarity"] > 0,
       str(zh["related_library"]))

    # 11. 输出契约与确定性
    a = run_cycle(docs, papers, State(Path(tempfile.mkdtemp())), top=4, now=now)
    b = run_cycle(docs, papers, State(Path(tempfile.mkdtemp())), top=4, now=now)
    ok("输出顶层字段", {"recommendations", "meta"} <= set(a))
    ok("推荐字段齐全", all({"arxiv_id", "title", "score", "reason", "reproducibility", "related_library",
                            "score_breakdown", "feedback_signal"} <= set(r) for r in a["recommendations"]))
    ok("JSON 可序列化", bool(json.dumps(a, ensure_ascii=False)))
    ok("排序确定", [r["arxiv_id"] for r in a["recommendations"]] == [r["arxiv_id"] for r in b["recommendations"]])
    ok("分数分项加总", all(abs(sum(r["score_breakdown"][k] for k in ("relevance", "category", "freshness", "code", "feedback"))
                              - r["score_breakdown"]["total"]) < 1e-3 for r in a["recommendations"]))
    ok("代码论文优先于同等相关无代码论文",
       next(r for r in a["recommendations"] if r["arxiv_id"] == "2610.10003")["score_breakdown"]["code"] == 0.13)

    # 12. 查询/标题核验
    ok("标题核验命中", GitHubClient._verify(cands[0], "This repository implements T from arXiv 2601.01234.")[0])
    ok("标题核验拒绝无关 README", not GitHubClient._verify(cands[0], "A cooking recipe.")[0])

    print(f"\nrecommend: {checks - failures}/{checks} 通过")
    return 1 if failures else 0


# ────────────────────────── CLI ──────────────────────────


def _load_feedback_items(args: argparse.Namespace) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for x in args.like or []:
        items.append((x, "like"))
    for x in args.dislike or []:
        items.append((x, "dislike"))
    if args.feedback:
        try:
            data = json.loads(args.feedback)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--feedback 不是合法 JSON：{e}") from e
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if isinstance(row, dict) and row.get("arxiv_id") and row.get("action"):
                items.append((str(row["arxiv_id"]), str(row["action"])))
            else:
                raise SystemExit("--feedback 需要 {\"arxiv_id\":..., \"action\":\"like|dislike\"}")
    return items


def _default_library() -> list[LibraryDoc]:
    try:
        from app.memory.store import memory

        return load_library_data(memory.get_library("demo"))
    except Exception:  # noqa: BLE001
        return []


def _emit(result: dict[str, Any], out: str | None) -> None:
    if out:
        path = _write_json(out, result)
        print(f"[OK] 写入 {path}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(
        prog="python -m app.recommend",
        description="个性化 arXiv 论文推荐：文献库相关性 + 代码优先 + 点赞/踩回流",
    )
    p.add_argument("--library", help="用户文献库 JSON（缺省用 MemoryStore 演示库）")
    p.add_argument("--candidates", help="候选论文 JSON，离线运行")
    p.add_argument("--refresh", action="store_true", help="从 arXiv Atom API 拉取最新论文")
    p.add_argument("--query", default="", help="覆盖自动生成的 arXiv 查询")
    p.add_argument("--max-results", type=int, default=20)
    p.add_argument("--days", type=int, default=14, help="只保留最近 N 天；0 表示不过滤")
    p.add_argument("--top", type=int, default=5, help="推荐条数")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--state-dir", default=str(DEFAULT_STATE), help="反馈/历史/缓存目录")
    p.add_argument("--out", "-o", help="把结果 JSON 写到文件（同时打印到 stdout）")
    p.add_argument("--include-seen", action="store_true", help="不过滤历史上已推荐的论文")
    p.add_argument("--github-search", action="store_true", help="对预排序前 N 篇检索 GitHub 并核验 README（较慢，易限流）")
    p.add_argument("--code-top", type=int, default=3, help="--github-search 检查的候选数")
    p.add_argument("--like", nargs="+", default=[], metavar="ARXIV_ID")
    p.add_argument("--dislike", nargs="+", default=[], metavar="ARXIV_ID")
    p.add_argument("--feedback", help='内联 JSON：{"arxiv_id":"...","action":"like"}')
    p.add_argument("--show-profile", action="store_true", help="只输出反馈学习到的权重")
    p.add_argument("--reset-feedback", action="store_true", help="删除 feedback.jsonl（冷启动）")
    p.add_argument("--watch", type=int, default=0, metavar="MIN", help="每 N 分钟刷新一次（需 --refresh）")
    p.add_argument("--demo", action="store_true", help="内置 3 篇库 + 6 篇候选，离线演示")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    setup_logging("DEBUG" if args.debug else "WARNING")
    if args.selftest:
        return _selftest()

    state = State(Path(args.state_dir))
    if args.demo and args.state_dir == str(DEFAULT_STATE):
        state = State(Path(tempfile.mkdtemp(prefix="ra-recommend-demo-")))
        print(f"[demo] 临时状态目录 {state.dir}（用 --state-dir 可持久化）", file=sys.stderr)

    if args.reset_feedback and state.feedback_path.exists():
        state.feedback_path.unlink()
        print(f"[OK] 已删除 {state.feedback_path}", file=sys.stderr)

    feedback_items = _load_feedback_items(args)
    history = load_history(state)
    if feedback_items:
        recorded = record_feedback_from_history(state, history, feedback_items)
        unknown = [r["arxiv_id"] for r in recorded if not r.get("features")]
        model = build_model(load_events(state))
        profile = feedback_summary(model)
        if unknown:
            print(f"[WARN] 以下条目没有历史特征，只计入次数、不影响权重：{', '.join(unknown)}", file=sys.stderr)
        has_run = args.demo or args.refresh or bool(args.candidates)
        if not has_run:
            print(json.dumps({"recorded": recorded, "feedback_model": profile}, ensure_ascii=False, indent=2))
            return 0
        history = load_history(state)

    if args.show_profile and not (args.demo or args.refresh or args.candidates):
        print(json.dumps({"feedback_model": feedback_summary(build_model(load_events(state)))}, ensure_ascii=False, indent=2))
        return 0

    if args.watch and not args.refresh:
        p.error("--watch 需要配合 --refresh")

    docs: list[LibraryDoc]
    if args.library:
        try:
            docs = load_library_data(_read_json_any(args.library))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] 读取文献库失败：{e}", file=sys.stderr)
            return 1
    elif args.demo:
        docs = demo_library()
    else:
        docs = _default_library()

    def cycle() -> dict[str, Any]:
        if args.demo:
            papers = demo_candidates()
            source = "demo"
            query = ""
        elif args.candidates:
            try:
                papers = load_candidate_data(_read_json_any(args.candidates))
            except Exception as e:  # noqa: BLE001
                raise FetchError(f"读取候选失败：{e}") from e
            source = "file"
            query = ""
        else:
            query = args.query or build_arxiv_query(docs, days=args.days)
            papers = fetch_arxiv(query, max_results=args.max_results, days=args.days)
            source = "arxiv"
        return run_cycle(
            docs,
            papers,
            state,
            top=args.top,
            min_score=args.min_score,
            include_seen=args.include_seen,
            use_github=args.github_search,
            code_top=args.code_top,
            source_label=source,
            query_label=query,
        )

    if args.watch:
        out = args.out or str(state.dir / "latest.json")
        while True:
            try:
                result = cycle()
                _write_json(out, result)
                print(
                    f"[watch] {result['meta']['generated_at']} 推荐 {result['meta']['returned']} 条 -> {out}",
                    file=sys.stderr,
                )
            except KeyboardInterrupt:
                return 0
            except Exception as e:  # noqa: BLE001
                print(f"[watch] 本轮失败：{e}", file=sys.stderr)
            try:
                time.sleep(max(1, args.watch) * 60)
            except KeyboardInterrupt:
                return 0

    try:
        result = cycle()
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] {e}", file=sys.stderr)
        if args.debug:
            raise
        return 1
    _emit(result, args.out)
    if args.demo:
        print(
            f"[OK] {result['meta']['returned']} 条推荐 / "
            f"{result['meta']['feedback']['events']} 条反馈事件",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
