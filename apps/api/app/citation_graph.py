"""引用图挖掘：PageRank / 中心性 / Louvain 社区 / 基石与边缘衍生论文。

图约定
──────
    边 u → v 表示 **u 引用 v**（citing → cited）。
    因此 PageRank 沿“引用”方向流向被引论文，基石论文得分高；
    in_degree = 被引次数，out_degree = 参考文献数。
    如果输入给的是“被引 → 引用”，结果会整体反向，请先反转边。

输入（自动识别）
────────────────
    {"papers":[{"id","title","year","authors","venue","keywords","doi",
                "arxiv_id","topic","references":[{"id","weight"}]}]}
    {"nodes":[...],"edges":[{"source","target","weight"}]}
    {"A":["B"],"B":[]}                  # 邻接表
    [["A","B"],["B","C"]]                # 边对
    [{"id":"A","references":["B"]}, ...]  # 论文列表
    ["A","B"]                            # 只有节点

    references 可为 id / 标题 / DOI / arXiv 编号；只解析到图谱内节点，
    解析不到的引用默认丢弃（--keep-external 保留为 ext: 占位节点）。
    nodes 边表里出现但未声明的端点会自动补成占位节点。

基石论文
────────
    对 5 个指标做 min-max 归一化后加权：
      0.38*PageRank + 0.22*介数 + 0.15*in-接近中心性
      + 0.15*度中心性 + 0.10*入度中心性
    取综合分前 K（默认 3）。分数是图内相对值，不能跨图比较。

边缘衍生论文
────────────
    in_degree <= 1（低被引）且 out_degree >= 2
    且跨社区邻居占比 >= 0.35
    且 innovation = 0.45*跨社区占比 + 0.25*新近度
                    + 0.30*min(1, 引用基石数/2) >= 0.55
    基石节点不参与；规则是“连接不同子领域、引用新基石”的启发式，
    不是论文质量的因果判断。

用法
────
    python -m app.citation_graph --demo
    python -m app.citation_graph citations.json --out graph.json
    python -m app.citation_graph --data '{"papers":[...]}'
    python -m app.citation_graph --selftest

依赖：networkx>=3.2（已加入 requirements.txt）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities, modularity
except ImportError as e:  # pragma: no cover - 环境缺依赖时给可执行的提示
    raise SystemExit("需要 networkx>=3.2：pip install 'networkx>=3.2'") from e

from app.retrieval.bm25 import tokenize
from app.utils import short

_STOP = {
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "with", "via",
    "using", "from", "by", "as", "at", "or", "is", "are", "be", "we", "our",
    "this", "that", "these", "those", "it", "its", "their", "not", "all",
    "more", "than", "into", "over", "under", "between", "through", "about",
    "across", "can", "how", "what", "why", "when", "where", "new", "novel",
    "learning", "model", "models", "method", "methods", "approach",
    "approaches", "paper", "study", "analysis", "based", "large", "towards",
    "toward", "results", "show", "two", "one", "use", "used", "propose",
    "proposed", "via", "per",
}

_PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295",
]

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s,;]+)", re.I)


# ────────────────────────── 基础工具 ──────────────────────────


def _clean(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def _terms(text: Any) -> list[str]:
    out = []
    for t in tokenize(str(text or "")):
        if t in _STOP or t.isdigit() or len(t) < 2:
            continue
        out.append(t)
    return out


def _norm_title(s: Any) -> str:
    return re.sub(r"[^0-9a-z一-鿿]+", " ", str(s or "").lower()).strip()


def _slug(text: str) -> str:
    return "p-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _year(v: Any) -> int | None:
    if v is None or v == "":
        return None
    m = re.match(r"\s*(\d{4})", str(v))
    if not m:
        return None
    y = int(m.group(1))
    return y if 1900 <= y <= 2100 else None


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in re.split(r"[,;，；]", v) if x.strip()]
    if isinstance(v, (list, tuple, set)):
        out = []
        for x in v:
            if isinstance(x, dict):
                out.append(_clean(x.get("name")))
            else:
                out.append(_clean(x))
        return [x for x in out if x]
    return [_clean(v)]


def _weight(v: Any) -> float:
    try:
        w = float(v)
    except (TypeError, ValueError):
        return 1.0
    return w if w > 0 else 1.0


def _minimal_meta(pid: str, title: str = "", placeholder: bool = True) -> dict[str, Any]:
    return {
        "id": pid,
        "title": title or pid,
        "year": None,
        "authors": [],
        "venue": "",
        "keywords": [],
        "doi": "",
        "arxiv_id": "",
        "topic": "",
        "label": "",
        "url": "",
        "placeholder": placeholder,
    }


def _meta_from_dict(d: dict[str, Any], pid: str) -> dict[str, Any]:
    meta = _minimal_meta(pid, _clean(d.get("title") or d.get("name") or pid), bool(d.get("placeholder", False)))
    meta.update(
        {
            "year": _year(d.get("year")),
            "authors": _as_list(d.get("authors") or d.get("author")),
            "venue": _clean(d.get("venue") or d.get("journal") or d.get("conference") or d.get("booktitle")),
            "keywords": _as_list(d.get("keywords") or d.get("tags")),
            "doi": _clean(d.get("doi")),
            "arxiv_id": _clean(d.get("arxiv_id")),
            "topic": _clean(d.get("topic") or d.get("field")),
            "label": _clean(d.get("label") or d.get("short")),
            "url": _clean(d.get("url")),
        }
    )
    return meta


def _is_edge_item(x: Any) -> bool:
    return isinstance(x, dict) and (
        ("source" in x or "from" in x or "citing" in x or "u" in x)
        and ("target" in x or "to" in x or "cited" in x or "v" in x)
    )


def _edge_tuple(e: Any) -> tuple[str, str, float] | None:
    if isinstance(e, dict):
        u = e.get("source") or e.get("from") or e.get("citing") or e.get("u")
        v = e.get("target") or e.get("to") or e.get("cited") or e.get("v")
        w = e.get("weight", e.get("w", 1.0))
    elif isinstance(e, (list, tuple)) and len(e) >= 2:
        u, v = e[0], e[1]
        w = e[2] if len(e) > 2 else 1.0
    else:
        return None
    if u is None or v is None:
        return None
    u, v = _clean(u), _clean(v)
    if not u or not v:
        return None
    return u, v, _weight(w)


# ────────────────────────── 输入解析 ──────────────────────────


@dataclass
class ParsedGraph:
    papers: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[tuple[str, str], float] = field(default_factory=dict)
    shape: str = ""
    warnings: list[str] = field(default_factory=list)
    unknown_refs: int = 0
    self_loops: int = 0
    duplicate_edges: int = 0
    placeholders: int = 0


def _looks_like_single_paper(d: dict[str, Any]) -> bool:
    return any(k in d for k in ("id", "paper_id", "title", "references", "refs", "doi", "arxiv_id"))


def _is_adjacency(d: dict[str, Any]) -> bool:
    if not d or _looks_like_single_paper(d):
        return False
    if any(k in d for k in ("papers", "nodes", "edges", "source", "target", "from", "to", "citing", "cited")):
        return False
    return all(isinstance(v, (list, tuple, set, dict, str)) for v in d.values())


def parse_citation_data(data: Any, keep_external: bool = False) -> ParsedGraph:
    """把多种输入形状归一成 papers + 去重后的边（权重相加）。"""
    out = ParsedGraph()
    papers = out.papers
    edges = out.edges
    refs: list[tuple[str, Any, float]] = []

    def add_paper(meta: dict[str, Any]) -> None:
        pid = meta["id"]
        if pid not in papers:
            papers[pid] = meta

    def add_edge(u: str, v: str, w: float) -> None:
        if u == v:
            out.self_loops += 1
            return
        key = (u, v)
        if key in edges:
            out.duplicate_edges += 1
            edges[key] += w
        else:
            edges[key] = w

    def ensure_node(pid: str) -> None:
        if pid not in papers:
            papers[pid] = _minimal_meta(pid, placeholder=True)
            out.placeholders += 1

    def collect_paper_item(item: Any, fallback_id: str) -> None:
        if isinstance(item, str):
            pid = _clean(item)
            if pid:
                add_paper(_minimal_meta(pid, placeholder=False))
            return
        if not isinstance(item, dict):
            return
        pid = _clean(
            item.get("id")
            or item.get("paper_id")
            or item.get("key")
            or item.get("doi")
            or item.get("arxiv_id")
            or item.get("title")
            or fallback_id
        )
        title = _clean(item.get("title") or item.get("name") or pid)
        if not pid:
            pid = _slug(title or fallback_id)
        meta = _meta_from_dict(item, pid)
        add_paper(meta)
        raw_refs = item.get("references", item.get("refs", item.get("cites", [])))
        if raw_refs is None:
            raw_refs = []
        if isinstance(raw_refs, (str, dict)):
            raw_refs = [raw_refs]
        for r in raw_refs:
            w = _weight(r.get("weight", 1.0)) if isinstance(r, dict) else 1.0
            refs.append((pid, r, w))

    # 1) 显式 papers（可同时带 edges）
    if isinstance(data, dict) and "papers" in data:
        out.shape = "papers"
        raw = data["papers"]
        if isinstance(raw, dict):
            items = []
            for k, v in raw.items():
                if isinstance(v, dict):
                    vv = dict(v)
                    vv.setdefault("id", k)
                    items.append(vv)
                elif isinstance(v, str):
                    items.append({"id": k, "title": v})
                else:
                    items.append({"id": k})
        elif isinstance(raw, (list, tuple)):
            items = list(raw)
        else:
            items = [raw]
        for i, item in enumerate(items):
            collect_paper_item(item, f"p{i + 1}")
        for e in data.get("edges") or []:
            t = _edge_tuple(e)
            if not t:
                continue
            u, v, w = t
            ensure_node(u)
            ensure_node(v)
            add_edge(u, v, w)

    # 2) 单篇论文 dict
    elif isinstance(data, dict) and _looks_like_single_paper(data):
        out.shape = "paper"
        collect_paper_item(data, "p1")

    # 3) nodes / edges
    elif isinstance(data, dict) and ("nodes" in data or "edges" in data):
        out.shape = "nodes_edges"
        raw_nodes = data.get("nodes") or []
        if isinstance(raw_nodes, dict):
            node_items = []
            for k, v in raw_nodes.items():
                if isinstance(v, dict):
                    vv = dict(v)
                    vv.setdefault("id", k)
                    node_items.append(vv)
                else:
                    node_items.append({"id": k, "title": str(v)})
        elif isinstance(raw_nodes, (list, tuple)):
            node_items = list(raw_nodes)
        else:
            node_items = [raw_nodes]
        for i, item in enumerate(node_items):
            if isinstance(item, str):
                pid = _clean(item)
                if pid:
                    add_paper(_minimal_meta(pid, placeholder=False))
            elif isinstance(item, dict):
                pid = _clean(item.get("id") or item.get("paper_id") or item.get("title") or f"n{i + 1}")
                add_paper(_meta_from_dict(item, pid))
        for e in data.get("edges") or []:
            t = _edge_tuple(e)
            if not t:
                continue
            u, v, w = t
            ensure_node(u)
            ensure_node(v)
            add_edge(u, v, w)

    # 4) 邻接表
    elif isinstance(data, dict) and _is_adjacency(data):
        out.shape = "adjacency"
        for src_raw, targets in data.items():
            src = _clean(src_raw)
            if not src:
                continue
            add_paper(_minimal_meta(src, placeholder=False))
            if isinstance(targets, dict):
                for dst_raw, w in targets.items():
                    dst = _clean(dst_raw)
                    if dst:
                        ensure_node(dst)
                        add_edge(src, dst, _weight(w))
            elif isinstance(targets, (list, tuple, set)):
                for dst_raw in targets:
                    dst = _clean(dst_raw)
                    if dst:
                        ensure_node(dst)
                        add_edge(src, dst, 1.0)
            elif isinstance(targets, str) and targets.strip():
                dst = _clean(targets)
                ensure_node(dst)
                add_edge(src, dst, 1.0)

    # 5) 列表：论文 / 边 / 节点混合
    elif isinstance(data, list):
        if not data:
            out.shape = "empty"
        elif all(isinstance(x, str) for x in data):
            out.shape = "nodes"
            for x in data:
                pid = _clean(x)
                if pid:
                    add_paper(_minimal_meta(pid, placeholder=False))
        elif all(isinstance(x, (list, tuple)) for x in data):
            out.shape = "edges"
            for e in data:
                t = _edge_tuple(e)
                if not t:
                    continue
                u, v, w = t
                ensure_node(u)
                ensure_node(v)
                add_edge(u, v, w)
        else:
            out.shape = "mixed"
            for i, item in enumerate(data):
                if isinstance(item, str):
                    pid = _clean(item)
                    if pid:
                        add_paper(_minimal_meta(pid, placeholder=False))
                elif _is_edge_item(item):
                    t = _edge_tuple(item)
                    if t:
                        u, v, w = t
                        ensure_node(u)
                        ensure_node(v)
                        add_edge(u, v, w)
                elif isinstance(item, (list, tuple)):
                    t = _edge_tuple(item)
                    if t:
                        u, v, w = t
                        ensure_node(u)
                        ensure_node(v)
                        add_edge(u, v, w)
                elif isinstance(item, dict):
                    collect_paper_item(item, f"p{i + 1}")

    # 6) 单条边
    elif isinstance(data, dict):
        t = _edge_tuple(data)
        if t:
            out.shape = "edge"
            u, v, w = t
            ensure_node(u)
            ensure_node(v)
            add_edge(u, v, w)

    if not papers and not edges:
        if out.shape:
            out.warnings.append("输入没有任何可用节点或边。")
            return out
        raise ValueError("输入为空或无法识别：需要 papers/nodes/edges/邻接表/边对之一")

    # 引用解析：id / 标题 / DOI / arXiv 编号 → 图内节点
    alias_low: dict[str, str] = {}
    alias_norm: dict[str, str] = {}
    for pid, m in papers.items():
        for key in (pid, m.get("doi", ""), m.get("arxiv_id", "")):
            if key:
                alias_low.setdefault(_clean(key).lower(), pid)
        alias_norm.setdefault(_norm_title(m.get("title") or pid), pid)
        alias_norm.setdefault(_norm_title(pid), pid)

    def resolve(ref: Any) -> str | None:
        if isinstance(ref, dict):
            for key in ("id", "paper_id", "doi", "arxiv_id", "title"):
                if ref.get(key):
                    got = resolve(ref[key])
                    if got:
                        return got
            return None
        s = _clean(ref)
        if not s:
            return None
        if s.lower() in alias_low:
            return alias_low[s.lower()]
        nt = _norm_title(s)
        if nt in alias_norm:
            return alias_norm[nt]
        m = _ARXIV_ID_RE.search(s)
        if m and m.group(1).lower() in alias_low:
            return alias_low[m.group(1).lower()]
        m = _DOI_RE.search(s)
        if m and m.group(1).lower().rstrip(".,;") in alias_low:
            return alias_low[m.group(1).lower().rstrip(".,;")]
        return None

    for src, ref, w in refs:
        dst = resolve(ref)
        if dst is None:
            out.unknown_refs += 1
            if not keep_external:
                continue
            if isinstance(ref, dict):
                raw_title = _clean(ref.get("title") or ref.get("id") or ref.get("doi"))
            else:
                raw_title = _clean(ref)
            if not raw_title:
                continue
            dst = "ext:" + (_norm_title(raw_title) or _slug(raw_title))
            if dst not in papers:
                papers[dst] = _minimal_meta(dst, title=raw_title, placeholder=True)
                out.placeholders += 1
        add_edge(src, dst, w)

    if out.unknown_refs:
        if keep_external:
            out.warnings.append(
                f"{out.unknown_refs} 条引用指向集合外论文，已保留为 ext: 占位节点（{out.placeholders} 个）。"
            )
        else:
            out.warnings.append(
                f"{out.unknown_refs} 条引用指向集合外论文，已丢弃；--keep-external 可保留为占位节点。"
            )
    if out.self_loops:
        out.warnings.append(f"忽略 {out.self_loops} 条自引用。")
    if out.duplicate_edges:
        out.warnings.append(f"{out.duplicate_edges} 组重复边已合并（权重相加）。")
    return out


def build_graph(parsed: ParsedGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    for pid in sorted(parsed.papers):
        meta = {k: v for k, v in parsed.papers[pid].items() if k != "id"}
        G.add_node(pid, **meta)
    for (u, v), w in sorted(parsed.edges.items()):
        G.add_edge(u, v, weight=float(w))
    return G


# ────────────────────────── 指标与社区 ──────────────────────────


def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def _rank_desc(d: dict[str, float]) -> dict[str, int]:
    order = sorted(d, key=lambda k: (-d[k], k))
    return {k: i + 1 for i, k in enumerate(order)}


def _pagerank(G: nx.DiGraph, alpha: float = 0.85, tol: float = 1e-10, max_iter: int = 500) -> dict[str, float]:
    """有向加权 PageRank 的幂迭代实现。

    不用 nx.pagerank：它在 networkx 3.7 里默认走 scipy，而本仓库未装 scipy；
    20 个节点的幂迭代比多装一个科学计算栈更划算。公式与 nx.pagerank 一致。
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: 1.0}
    idx = {u: i for i, u in enumerate(nodes)}
    out_w = [0.0] * n
    for u in nodes:
        out_w[idx[u]] = sum(
            max(0.0, float(d.get("weight", 1.0))) for _, _, d in G.out_edges(u, data=True)
        )
    pr = [1.0 / n] * n
    dangling = [i for i, w in enumerate(out_w) if w <= 0]
    base = (1.0 - alpha) / n
    for _ in range(max_iter):
        new = [base] * n
        dsum = alpha * sum(pr[i] for i in dangling) / n
        if dsum:
            for i in range(n):
                new[i] += dsum
        for u, v, d in G.edges(data=True):
            i, j = idx[u], idx[v]
            if out_w[i] > 0:
                new[j] += alpha * pr[i] * max(0.0, float(d.get("weight", 1.0))) / out_w[i]
        delta = sum(abs(new[i] - pr[i]) for i in range(n))
        pr = new
        if delta < tol:
            break
    total = sum(pr)
    return {u: (pr[i] / total if total else 1.0 / n) for i, u in enumerate(nodes)}


def _community_label(
    members: list[str],
    G: nx.DiGraph,
    df: Counter,
    n_communities: int,
    cid: int,
) -> str:
    topics = Counter(
        _clean(G.nodes[m].get("topic"))
        for m in members
        if _clean(G.nodes[m].get("topic"))
    )
    if topics:
        topic, count = topics.most_common(1)[0]
        if count / max(1, len(members)) >= 0.6:
            return topic

    tf: Counter = Counter()
    for m in members:
        node = G.nodes[m]
        title_terms = _terms(node.get("title", ""))
        kw_terms = _terms(" ".join(node.get("keywords") or []))
        for t in title_terms:
            tf[t] += 3
        tf.update(kw_terms)
    if not tf:
        return f"社区 {cid + 1}"

    scored = {
        t: tf[t] * math.log((n_communities + 1) / (df.get(t, 1) + 0.5))
        for t in tf
    }
    top = sorted(scored, key=lambda t: (-scored[t], t))[:2]
    return " · ".join(top) if top else f"社区 {cid + 1}"


def analyze(
    G: nx.DiGraph,
    *,
    top_keystone: int = 3,
    peripheral_max_indeg: int = 1,
    peripheral_min_outdeg: int = 2,
    peripheral_min_cross: float = 0.35,
    peripheral_min_innovation: float = 0.55,
    resolution: float = 1.0,
    seed: int = 42,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    warnings = list(warnings or [])
    nodes = sorted(G.nodes())
    n = len(nodes)
    m = G.number_of_edges()
    empty = {
        "nodes": [],
        "edges": [],
        "keystone": [],
        "peripheral": [],
        "meta": {
            "n_nodes": 0,
            "n_edges": 0,
            "n_communities": 0,
            "modularity": 0.0,
            "warnings": warnings + ["图为空：没有节点可分析。"],
            "thresholds": {
                "peripheral_max_indeg": peripheral_max_indeg,
                "peripheral_min_outdeg": peripheral_min_outdeg,
                "peripheral_min_cross": peripheral_min_cross,
                "peripheral_min_innovation": peripheral_min_innovation,
            },
        },
    }
    if n == 0:
        return empty

    # ── 中心性 ──
    pr = _pagerank(G)
    bt = dict(nx.betweenness_centrality(G, normalized=True, weight=None))
    # networkx 对 DiGraph 的 closeness 内部先反转：closeness(G) 是“别人到我”，
    # closeness(G.reverse()) 才是“我到别人”，别把 in/out 写反。
    cl_in = dict(nx.closeness_centrality(G))
    cl_out = dict(nx.closeness_centrality(G.reverse()))

    in_deg = {u: int(d) for u, d in G.in_degree()}
    out_deg = {u: int(d) for u, d in G.out_degree()}
    deg = {u: in_deg[u] + out_deg[u] for u in nodes}
    win = {u: float(d) for u, d in G.in_degree(weight="weight")}
    wout = {u: float(d) for u, d in G.out_degree(weight="weight")}
    deg_c = {u: (deg[u] / (2 * (n - 1)) if n > 1 else 0.0) for u in nodes}
    in_c = {u: (in_deg[u] / (n - 1) if n > 1 else 0.0) for u in nodes}

    npr, nbt, nci = _minmax({u: float(pr[u]) for u in nodes}), _minmax({u: float(bt[u]) for u in nodes}), _minmax({u: float(cl_in[u]) for u in nodes})
    ndeg, nin = _minmax({u: float(deg_c[u]) for u in nodes}), _minmax({u: float(in_c[u]) for u in nodes})
    kscore = {
        u: 0.38 * npr[u] + 0.22 * nbt[u] + 0.15 * nci[u] + 0.15 * ndeg[u] + 0.10 * nin[u]
        for u in nodes
    }
    if n == 1:
        kscore[nodes[0]] = 1.0

    # ── Louvain（无向投影，双向边权重相加）──
    # ponytail: 精确介数 O(nm)，20 篇够用；上千节点改 betweenness_centrality(k=200)。
    if m == 0:
        parts = [set(nodes)]
        mod = 0.0
        warnings.append("图没有任何边：社区发现退化为单组，中心性只反映孤立节点。")
    else:
        U = nx.Graph()
        U.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 1.0))
            if U.has_edge(u, v):
                U[u][v]["weight"] += w
            else:
                U.add_edge(u, v, weight=w)
        parts = [set(p) for p in louvain_communities(U, weight="weight", resolution=resolution, seed=seed)]
        parts.sort(key=lambda s: (-len(s), min(s) if s else ""))
        if not parts:
            parts = [set(nodes)]
        covered = set().union(*parts)
        for u in nodes:
            if u not in covered:
                parts.append({u})
        mod = float(modularity(U, parts, weight="weight", resolution=resolution))

    parts = [set(p) for p in parts]
    comm: dict[str, int] = {}
    for cid, members in enumerate(parts):
        for u in members:
            comm[u] = cid
    for u in nodes:
        if u not in comm:
            comm[u] = len(parts)
            parts.append({u})

    n_comm = len(parts)
    comm_terms = []
    for members in parts:
        s: set[str] = set()
        for u in members:
            nd = G.nodes[u]
            s.update(_terms(nd.get("title", "")))
            s.update(_terms(" ".join(nd.get("keywords") or [])))
        comm_terms.append(s)
    df = Counter(t for s in comm_terms for t in s)
    labels = {
        cid: _community_label(sorted(members), G, df, n_comm, cid)
        for cid, members in enumerate(parts)
    }
    colors = {cid: _PALETTE[cid % len(_PALETTE)] for cid in range(n_comm)}

    if n_comm == 1 and n > 1 and m > 0:
        warnings.append("Louvain 只得到 1 个社区：图可能过稠密或 resolution 偏低（可试 --resolution 1.5）。")

    # ── 邻居跨社区信息 ──
    nbr_comms: dict[str, list[int]] = {}
    cross_ratio: dict[str, float] = {}
    for u in nodes:
        neighbors = set(G.predecessors(u)) | set(G.successors(u))
        others = sorted({comm[x] for x in neighbors} - {comm[u]})
        nbr_comms[u] = others
        cross_ratio[u] = (
            sum(1 for x in neighbors if comm[x] != comm[u]) / len(neighbors) if neighbors else 0.0
        )

    # ── 基石论文 ──
    ks_order = sorted(nodes, key=lambda u: (-kscore[u], -float(pr[u]), u))
    ks_list = [u for u in ks_order if kscore[u] > 1e-12][: max(0, top_keystone)]
    ks_set = set(ks_list)
    if not ks_list:
        warnings.append("没有节点的基石分 >0：图无边或所有节点完全孤立。")

    ranks = {
        "pagerank": _rank_desc({u: float(pr[u]) for u in nodes}),
        "betweenness": _rank_desc({u: float(bt[u]) for u in nodes}),
        "closeness_in": _rank_desc({u: float(cl_in[u]) for u in nodes}),
        "degree_centrality": _rank_desc({u: float(deg_c[u]) for u in nodes}),
        "in_degree_centrality": _rank_desc({u: float(in_c[u]) for u in nodes}),
        "in_degree": _rank_desc({u: float(in_deg[u]) for u in nodes}),
        "degree": _rank_desc({u: float(deg[u]) for u in nodes}),
    }

    def keystone_why(u: str) -> str:
        nbr = nbr_comms[u]
        bits = [
            f"PageRank {pr[u]:.3f}（第{ranks['pagerank'][u]}）",
            f"介数中心性 {bt[u]:.3f}（第{ranks['betweenness'][u]}）",
            f"被引 {in_deg[u]} 次（第{ranks['in_degree'][u]}）/ 引用他人 {out_deg[u]} 次",
            f"in-接近中心性 {cl_in[u]:.3f}（第{ranks['closeness_in'][u]}）",
        ]
        if nbr:
            names = "、".join(labels.get(c, f"社区 {c + 1}") for c in nbr[:3])
            bits.append(f"邻接 {len(nbr)} 个其他社区（{names}）")
        return "；".join(bits) + "。"

    keystone = []
    for r, u in enumerate(ks_list, 1):
        nd = G.nodes[u]
        keystone.append(
            {
                "id": u,
                "rank": r,
                "score": round(float(kscore[u]), 6),
                "title": nd.get("title") or u,
                "year": nd.get("year"),
                "community": comm[u],
                "group": labels[comm[u]],
                "metrics": {
                    "pagerank": round(float(pr[u]), 6),
                    "betweenness": round(float(bt[u]), 6),
                    "closeness_in": round(float(cl_in[u]), 6),
                    "closeness_out": round(float(cl_out[u]), 6),
                    "degree": deg[u],
                    "in_degree": in_deg[u],
                    "out_degree": out_deg[u],
                    "degree_centrality": round(float(deg_c[u]), 6),
                    "in_degree_centrality": round(float(in_c[u]), 6),
                },
                "ranks": {
                    k: ranks[k][u]
                    for k in ("pagerank", "betweenness", "closeness_in", "degree_centrality", "in_degree_centrality")
                },
                "neighbor_communities": nbr_comms[u],
                "bridges": len(nbr_comms[u]),
                "why": keystone_why(u),
            }
        )

    # ── 边缘衍生论文 ──
    years = [y for y in (G.nodes[u].get("year") for u in nodes) if y]
    max_year = max(years) if years else None
    peripheral_rows = []
    for u in nodes:
        if u in ks_set:
            continue
        year = G.nodes[u].get("year")
        if year is not None and max_year is not None:
            recency = max(0.0, min(1.0, 1.0 - (max_year - year) / 5.0))
        else:
            recency = 0.5
        cites_ks = sum(1 for s in G.successors(u) if s in ks_set)
        innov = (
            0.45 * cross_ratio[u]
            + 0.25 * recency
            + 0.30 * min(1.0, cites_ks / 2.0)
        )
        if (
            in_deg[u] <= peripheral_max_indeg
            and out_deg[u] >= peripheral_min_outdeg
            and cross_ratio[u] >= peripheral_min_cross
            and innov >= peripheral_min_innovation
        ):
            peripheral_rows.append(
                {
                    "id": u,
                    "score": round(innov, 6),
                    "title": G.nodes[u].get("title") or u,
                    "year": year,
                    "community": comm[u],
                    "group": labels[comm[u]],
                    "in_degree": in_deg[u],
                    "out_degree": out_deg[u],
                    "cross_community_ratio": round(cross_ratio[u], 4),
                    "neighbor_communities": nbr_comms[u],
                    "cites_keystone": cites_ks,
                    "recency": round(recency, 4),
                }
            )
    peripheral_rows.sort(key=lambda r: (-r["score"], r["in_degree"], r["id"]))
    for r, row in enumerate(peripheral_rows, 1):
        row["rank"] = r
        nbr = row["neighbor_communities"]
        names = "、".join(labels.get(c, f"社区 {c + 1}") for c in nbr[:3]) or "无"
        row["why"] = (
            f"入度 {row['in_degree']}（≤{peripheral_max_indeg}，低被引）；"
            f"出度 {row['out_degree']}；跨社区邻居占比 {row['cross_community_ratio']:.2f}"
            f"（连接 {len(nbr)} 个社区：{names}）；引用 {row['cites_keystone']} 篇基石论文；"
            f"{row['year'] if row['year'] else '年份未知'} 年发表（新近度 {row['recency']:.2f}）；"
            f"创新关联分 {row['score']:.2f} ≥ {peripheral_min_innovation:.2f}，判为边缘衍生论文。"
        )
    if not peripheral_rows and n >= 5:
        warnings.append("没有节点满足边缘衍生规则（可放宽 --peripheral-* 阈值）。")

    roles = {u: "normal" for u in nodes}
    for u in ks_set:
        roles[u] = "keystone"
    for row in peripheral_rows:
        roles[row["id"]] = "peripheral"

    # ── 可视化节点/边 ──
    node_rows = []
    for u in nodes:
        nd = G.nodes[u]
        node_rows.append(
            {
                "id": u,
                "label": nd.get("label") or short(nd.get("title") or u, 42),
                "title": nd.get("title") or u,
                "year": nd.get("year"),
                "authors": nd.get("authors") or [],
                "venue": nd.get("venue") or "",
                "keywords": nd.get("keywords") or [],
                "doi": nd.get("doi") or "",
                "arxiv_id": nd.get("arxiv_id") or "",
                "placeholder": bool(nd.get("placeholder")),
                "community": comm[u],
                "group": labels[comm[u]],
                "group_id": comm[u],
                "color": colors[comm[u]],
                "size": round(14.0 + 3.0 * deg[u] + 30.0 * float(pr[u]), 1),
                "degree": deg[u],
                "in_degree": in_deg[u],
                "out_degree": out_deg[u],
                "weighted_in_degree": round(win[u], 3),
                "weighted_out_degree": round(wout[u], 3),
                "pagerank": round(float(pr[u]), 6),
                "betweenness": round(float(bt[u]), 6),
                "closeness_in": round(float(cl_in[u]), 6),
                "closeness_out": round(float(cl_out[u]), 6),
                "degree_centrality": round(float(deg_c[u]), 6),
                "in_degree_centrality": round(float(in_c[u]), 6),
                "keystone_score": round(float(kscore[u]), 6),
                "role": roles[u],
            }
        )

    edge_rows = []
    for i, (u, v, d) in enumerate(sorted(G.edges(data=True))):
        edge_rows.append(
            {
                "id": f"e{i}",
                "source": u,
                "target": v,
                "weight": round(float(d.get("weight", 1.0)), 3),
                "label": "cites",
                "type": "citation",
                "cross_community": comm[u] != comm[v],
                "source_community": comm[u],
                "target_community": comm[v],
            }
        )

    return {
        "nodes": node_rows,
        "edges": edge_rows,
        "keystone": keystone,
        "peripheral": peripheral_rows,
        "meta": {
            "n_nodes": n,
            "n_edges": m,
            "n_communities": n_comm,
            "community_labels": {str(cid): labels[cid] for cid in range(n_comm)},
            "community_sizes": {str(cid): len(parts[cid]) for cid in range(n_comm)},
            "communities": [
                {
                    "id": cid,
                    "label": labels[cid],
                    "color": colors[cid],
                    "size": len(parts[cid]),
                    "members": sorted(parts[cid]),
                }
                for cid in range(n_comm)
            ],
            "modularity": round(mod, 4),
            "directed": True,
            "weighted": True,
            "algorithms": {
                "pagerank": "power iteration, alpha=0.85, weight='weight', tol=1e-10 (equivalent to nx.pagerank)",
                "betweenness": "networkx.betweenness_centrality(normalized=True, weight=None)",
                "closeness_in": "networkx.closeness_centrality(G)  # DiGraph 内部反转，衡量可达性",
                "closeness_out": "networkx.closeness_centrality(G.reverse())",
                "community": "networkx.algorithms.community.louvain_communities(weight='weight', seed=...)",
                "modularity": "networkx.algorithms.community.modularity(weight='weight', resolution=...)",
            },
            "edge_direction": "source cites target",
            "keystone_formula": "0.38*pagerank_norm + 0.22*betweenness_norm + 0.15*closeness_in_norm + 0.15*degree_centrality_norm + 0.10*in_degree_centrality_norm",
            "keystone_rule": f"综合分 >0 的前 {top_keystone} 名；分数为图内相对值",
            "peripheral_rule": (
                f"in_degree <= {peripheral_max_indeg} AND out_degree >= {peripheral_min_outdeg} "
                f"AND cross_community_ratio >= {peripheral_min_cross:.2f} "
                f"AND innovation >= {peripheral_min_innovation:.2f}; "
                "innovation = 0.45*cross_ratio + 0.25*recency + 0.30*min(1, cites_keystone/2); "
                "keystone nodes excluded"
            ),
            "thresholds": {
                "top_keystone": top_keystone,
                "peripheral_max_indeg": peripheral_max_indeg,
                "peripheral_min_outdeg": peripheral_min_outdeg,
                "peripheral_min_cross": peripheral_min_cross,
                "peripheral_min_innovation": peripheral_min_innovation,
            },
            "seed": seed,
            "resolution": resolution,
            "warnings": warnings,
        },
    }


def build_result(data: Any, *, keep_external: bool = False, **kwargs: Any) -> dict[str, Any]:
    parsed = parse_citation_data(data, keep_external=keep_external)
    G = build_graph(parsed)
    result = analyze(G, warnings=list(parsed.warnings), **kwargs)
    result["meta"]["input"] = {
        "shape": parsed.shape,
        "unknown_refs": parsed.unknown_refs,
        "self_loops_skipped": parsed.self_loops,
        "duplicate_edges_merged": parsed.duplicate_edges,
        "placeholder_nodes": parsed.placeholders,
    }
    result["meta"]["n_nodes"] = G.number_of_nodes()
    result["meta"]["n_edges"] = G.number_of_edges()
    return result


# ────────────────────────── 20 篇演示数据 ──────────────────────────


def _r(pid: str, w: int = 1) -> dict[str, Any]:
    return {"id": pid, "weight": w}


def demo_data() -> dict[str, Any]:
    """3 个子领域 + 1 篇跨领域边缘论文；数据为合成示例，不代表真实引用。"""
    papers = [
        # ── Transformer 主干（T）──
        {"id": "att", "title": "Attention Is All You Need", "year": 2017,
         "authors": ["Vaswani et al."], "venue": "NeurIPS",
         "keywords": ["transformer", "self-attention", "sequence transduction"],
         "references": []},
        {"id": "bert", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "year": 2018,
         "authors": ["Devlin et al."], "venue": "NAACL",
         "keywords": ["pretraining", "transformer", "language representation"],
         "references": [_r("att", 3)]},
        {"id": "gpt", "title": "Improving Language Understanding by Generative Pre-Training", "year": 2018,
         "authors": ["Radford et al."], "venue": "OpenAI",
         "keywords": ["pretraining", "transformer", "language model"],
         "references": [_r("att", 2)]},
        {"id": "roberta", "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "year": 2019,
         "authors": ["Liu et al."], "venue": "arXiv",
         "keywords": ["pretraining", "transformer", "robustness"],
         "references": [_r("bert", 3)]},
        {"id": "t5", "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
         "year": 2019, "authors": ["Raffel et al."], "venue": "JMLR",
         "keywords": ["text-to-text", "transformer", "transfer learning"],
         "references": [_r("att", 2), _r("bert", 2)]},
        {"id": "distilbert", "title": "DistilBERT: A Distilled Version of BERT", "year": 2019,
         "authors": ["Sanh et al."], "venue": "arXiv",
         "keywords": ["distillation", "transformer", "efficiency"],
         "references": [_r("bert", 3)]},
        {"id": "albert", "title": "ALBERT: A Lite BERT for Self-supervised Learning", "year": 2019,
         "authors": ["Lan et al."], "venue": "ICLR",
         "keywords": ["pretraining", "transformer", "parameter sharing"],
         "references": [_r("bert", 2)]},
        {"id": "llama", "title": "LLaMA: Open and Efficient Foundation Language Models", "year": 2023,
         "authors": ["Touvron et al."], "venue": "arXiv",
         "keywords": ["large language model", "transformer", "scaling"],
         "references": [_r("att", 2), _r("gpt", 2)]},
        # ── 高效注意力（E）──
        {"id": "longformer", "title": "Longformer: The Long-Document Transformer", "year": 2020,
         "authors": ["Beltagy et al."], "venue": "arXiv",
         "keywords": ["sparse attention", "long document", "efficient transformer"],
         "references": [_r("att", 2), _r("bert", 1)]},
        {"id": "bigbird", "title": "Big Bird: Transformers for Longer Sequences", "year": 2020,
         "authors": ["Zaheer et al."], "venue": "NeurIPS",
         "keywords": ["sparse attention", "long sequence", "efficient transformer"],
         "references": [_r("longformer", 2), _r("bert", 1)]},
        {"id": "informer", "title": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series",
         "year": 2020, "authors": ["Zhou et al."], "venue": "AAAI",
         "keywords": ["efficient attention", "long sequence", "time series"],
         "references": [_r("att", 1)]},
        {"id": "performer", "title": "Rethinking Attention with Performers", "year": 2020,
         "authors": ["Choromanski et al."], "venue": "ICLR",
         "keywords": ["linear attention", "kernel", "efficient transformer"],
         "references": [_r("att", 1)]},
        {"id": "flash", "title": "FlashAttention: Fast and Memory-Efficient Exact Attention", "year": 2022,
         "authors": ["Dao et al."], "venue": "NeurIPS",
         "keywords": ["io-aware", "attention kernel", "efficient transformer"],
         "references": [_r("longformer", 1), _r("bigbird", 1)]},
        {"id": "mamba", "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", "year": 2023,
         "authors": ["Gu & Dao"], "venue": "arXiv",
         "keywords": ["state space model", "long sequence", "linear time"],
         "references": [_r("att", 1), _r("flash", 1)]},
        # ── 检索增强（R）──
        {"id": "dpr", "title": "Dense Passage Retrieval for Open-Domain Question Answering", "year": 2020,
         "authors": ["Karpukhin et al."], "venue": "EMNLP",
         "keywords": ["dense retrieval", "question answering", "bert"],
         "references": [_r("bert", 1)]},
        {"id": "rag", "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP", "year": 2020,
         "authors": ["Lewis et al."], "venue": "NeurIPS",
         "keywords": ["retrieval-augmented generation", "dense retrieval", "generation"],
         "references": [_r("dpr", 3), _r("bert", 1)]},
        {"id": "realm", "title": "REALM: Retrieval-Augmented Language Model Pre-Training", "year": 2020,
         "authors": ["Guu et al."], "venue": "ICML",
         "keywords": ["retrieval", "pretraining", "knowledge"],
         "references": [_r("dpr", 2)]},
        {"id": "atlas", "title": "Atlas: Few-shot Learning with Retrieval Augmented Language Models", "year": 2022,
         "authors": ["Izacard et al."], "venue": "JMLR",
         "keywords": ["retrieval-augmented", "few-shot", "question answering"],
         "references": [_r("rag", 2), _r("t5", 1)]},
        {"id": "selfrag", "title": "Self-RAG: Learning to Retrieve, Generate, and Critique", "year": 2023,
         "authors": ["Asai et al."], "venue": "ICLR",
         "keywords": ["self-reflection", "retrieval", "generation"],
         "references": [_r("rag", 2)]},
        # ── 边缘衍生（P）──
        {"id": "pbridge", "title": "Bridging Sparse Attention and Retrieval for Efficient Long-Context QA",
         "year": 2024, "authors": ["Example et al."], "venue": "arXiv",
         "keywords": ["sparse attention", "retrieval", "long-context question answering"],
         "references": [_r("longformer", 2), _r("flash", 1), _r("bigbird", 1),
                        _r("rag", 2), _r("selfrag", 1), _r("att", 1)]},
    ]
    return {"papers": papers}


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

    def by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {n["id"]: n for n in result["nodes"]}

    # ── 1. 解析与方向 ──
    path = {"papers": [
        {"id": "A", "title": "Paper A", "year": 2020, "references": ["B"]},
        {"id": "B", "title": "Paper B", "year": 2021, "references": ["C"]},
        {"id": "C", "title": "Paper C", "year": 2022},
    ]}
    parsed = parse_citation_data(path)
    ok("论文形状节点数", len(parsed.papers) == 3)
    ok("引用方向 A→B", ("A", "B") in parsed.edges and ("B", "A") not in parsed.edges)
    res = build_result(path, top_keystone=3)
    nodes = by_id(res)
    ok("PageRank 沿引用流向被引论文", nodes["C"]["pagerank"] > nodes["B"]["pagerank"] > nodes["A"]["pagerank"],
       f"{nodes['A']['pagerank']},{nodes['B']['pagerank']},{nodes['C']['pagerank']}")
    ok("PageRank 归一", abs(sum(n["pagerank"] for n in res["nodes"]) - 1.0) < 1e-4)
    ok("入度/出度", nodes["B"]["in_degree"] == 1 and nodes["B"]["out_degree"] == 1)
    ok("介数中心 B 最高", max(res["nodes"], key=lambda n: n["betweenness"])["id"] == "B")
    ok("接近中心性方向", nodes["A"]["closeness_out"] > nodes["C"]["closeness_out"]
       and nodes["C"]["closeness_in"] > nodes["A"]["closeness_in"])

    # ── 2. 标题 / DOI / arXiv 解析 ──
    alias = {"papers": [
        {"id": "b", "title": "Paper B Title", "doi": "10.1000/b", "arxiv_id": "2401.00002"},
        {"id": "a", "title": "Paper A", "references": ["Paper B Title"]},
        {"id": "c", "title": "Paper C", "references": ["https://doi.org/10.1000/b"]},
        {"id": "d", "title": "Paper D", "references": ["arXiv:2401.00002v1"]},
    ]}
    pa = parse_citation_data(alias)
    ok("标题解析引用", ("a", "b") in pa.edges)
    ok("DOI URL 解析引用", ("c", "b") in pa.edges)
    ok("arXiv 编号解析引用", ("d", "b") in pa.edges)
    ok("所有引用均解析", pa.unknown_refs == 0, str(pa.unknown_refs))

    # ── 3. 未知引用 / 自引用 / 重复边 ──
    messy = {"papers": [
        {"id": "A", "references": [{"id": "B", "weight": 2}, {"id": "B", "weight": 3},
                                  {"id": "A"}, "External Paper"]},
        {"id": "B"},
    ]}
    pm = parse_citation_data(messy)
    ok("重复边权重相加", pm.edges.get(("A", "B")) == 5.0, str(pm.edges))
    ok("自引用被跳过", pm.self_loops == 1 and ("A", "A") not in pm.edges)
    ok("未知引用被丢弃", pm.unknown_refs == 1 and "External Paper" not in pm.papers)
    ok("重复边计数", pm.duplicate_edges == 1)
    pm_ext = parse_citation_data(messy, keep_external=True)
    ok("--keep-external 保留占位节点", any(k.startswith("ext:") for k in pm_ext.papers))
    ok("保留外部节点后边仍存在", ("A", "ext:external paper") in pm_ext.edges, str(pm_ext.edges))

    # ── 4. nodes/edges、邻接表、边对、单边 ──
    ne = {"nodes": [{"id": "a", "title": "A", "year": 2020}, "b"],
          "edges": [{"source": "a", "target": "b", "weight": 2}]}
    rne = build_result(ne)
    ok("nodes/edges 形状", rne["meta"]["n_nodes"] == 2 and rne["meta"]["n_edges"] == 1)
    ok("节点元数据保留", by_id(rne)["a"]["title"] == "A")
    adj = parse_citation_data({"A": ["B"], "B": []})
    ok("邻接表解析", len(adj.papers) == 2 and adj.edges.get(("A", "B")) == 1.0)
    pairs = parse_citation_data([["A", "B"], ["B", "C"]])
    ok("边对解析 + 自动补节点", len(pairs.papers) == 3 and len(pairs.edges) == 2)
    single_edge = parse_citation_data({"source": "A", "target": "B"})
    ok("单边解析", len(single_edge.papers) == 2 and ("A", "B") in single_edge.edges)
    names_only = parse_citation_data(["A", "B", "C"])
    ok("纯节点列表", len(names_only.papers) == 3 and not names_only.edges)

    # ── 5. Louvain 两个团 ──
    clique_edges = []
    for group in (["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]):
        for u in group:
            for v in group:
                if u != v:
                    clique_edges.append({"source": u, "target": v, "weight": 2})
    clique_edges.append({"source": "a1", "target": "b1", "weight": 1})
    rc = build_result({"nodes": [f"a{i}" for i in range(1, 5)] + [f"b{i}" for i in range(1, 5)],
                       "edges": clique_edges})
    ok("Louvain 两个团", rc["meta"]["n_communities"] == 2, str(rc["meta"]["n_communities"]))
    ok("模块度 >0", rc["meta"]["modularity"] > 0, str(rc["meta"]["modularity"]))
    ok("同一团同社区", by_id(rc)["a1"]["community"] == by_id(rc)["a2"]["community"])
    ok("不同团不同社区", by_id(rc)["a1"]["community"] != by_id(rc)["b1"]["community"])

    # ── 6. 演示图：20 篇、三社区、基石、边缘 ──
    demo = build_result(demo_data())
    dn = by_id(demo)
    ok("演示 20 个节点", demo["meta"]["n_nodes"] == 20, str(demo["meta"]["n_nodes"]))
    ok("演示边数 >25", demo["meta"]["n_edges"] > 25, str(demo["meta"]["n_edges"]))
    ok("演示识别出多个子领域", demo["meta"]["n_communities"] >= 3, str(demo["meta"]["n_communities"]))
    ok("演示模块度 >0.3", demo["meta"]["modularity"] > 0.3, str(demo["meta"]["modularity"]))
    ok("主干/高效/检索代表节点分属不同社区",
       len({dn["att"]["community"], dn["longformer"]["community"], dn["rag"]["community"]}) == 3)
    ok("检索子领域聚合",
       len({dn[x]["community"] for x in ("dpr", "rag", "realm", "atlas", "selfrag")}) == 1)
    ok("ATT 与 BERT 均为基石", {"att", "bert"} <= {k["id"] for k in demo["keystone"]},
       str([k["id"] for k in demo["keystone"]]))
    ok("基石按分数降序", [k["score"] for k in demo["keystone"]] ==
       sorted([k["score"] for k in demo["keystone"]], reverse=True))
    ok("基石理由含 PageRank", "PageRank" in demo["keystone"][0]["why"])
    ok("pbridge 判为边缘", any(p["id"] == "pbridge" for p in demo["peripheral"]),
       str([p["id"] for p in demo["peripheral"]]))
    ok("边缘理由含低入度与跨社区", all(k in next(p for p in demo["peripheral"] if p["id"] == "pbridge")["why"]
                                      for k in ("入度", "跨社区")))
    ok("边缘论文不在基石列表", "pbridge" not in {k["id"] for k in demo["keystone"]})
    ok("pbridge 指向 att", ("pbridge", "att") in {(e["source"], e["target"]) for e in demo["edges"]})
    ok("跨社区边被标记", any(e["cross_community"] for e in demo["edges"]))
    ok("边缺省 label=cites", all(e["label"] == "cites" and e["type"] == "citation" for e in demo["edges"]))
    ok("节点角色字段", dn["att"]["role"] == "keystone" and dn["pbridge"]["role"] == "peripheral")

    # ── 7. 输出契约 ──
    required_node = {"id", "label", "title", "community", "group", "group_id", "color", "size",
                     "degree", "in_degree", "out_degree", "pagerank", "betweenness",
                     "closeness_in", "closeness_out", "degree_centrality", "keystone_score", "role"}
    required_edge = {"id", "source", "target", "weight", "label", "type", "cross_community",
                     "source_community", "target_community"}
    ok("节点字段齐全", all(required_node <= set(n) for n in demo["nodes"]))
    ok("边字段齐全", all(required_edge <= set(e) for e in demo["edges"]))
    ok("社区标签非空", all(v for v in demo["meta"]["community_labels"].values()))
    ok("社区颜色非空", all(c["color"].startswith("#") for c in demo["meta"]["communities"]))
    ok("节点颜色与社区一致", all(n["color"] == demo["meta"]["communities"][n["community"]]["color"]
                              for n in demo["nodes"]))
    ok("JSON 可序列化", bool(json.dumps(demo, ensure_ascii=False)))
    ok("同种子输出确定",
       json.dumps(build_result(demo_data()), sort_keys=True, ensure_ascii=False) ==
       json.dumps(build_result(demo_data()), sort_keys=True, ensure_ascii=False))

    # ── 8. 边界情况 ──
    no_edge = build_result({"nodes": ["A", "B"]})
    ok("无边图不崩溃", no_edge["meta"]["n_edges"] == 0 and no_edge["meta"]["n_communities"] == 1)
    ok("无边图模块度为 0", no_edge["meta"]["modularity"] == 0.0)
    ok("无边图无基石", no_edge["keystone"] == [])
    ok("无边图有警告", any("没有任何边" in w for w in no_edge["meta"]["warnings"]))
    one = build_result({"papers": [{"id": "solo", "title": "Solo Paper", "year": 2024}]})
    ok("单节点视为基石", len(one["keystone"]) == 1 and one["keystone"][0]["id"] == "solo")
    ok("单节点社区", one["meta"]["n_communities"] == 1 and one["meta"]["n_nodes"] == 1)
    empty = build_result({"papers": []})
    ok("空图不崩溃", empty["meta"]["n_nodes"] == 0 and empty["keystone"] == [])
    try:
        parse_citation_data({})
        ok("空输入报错", False)
    except ValueError:
        ok("空输入报错", True)

    # ── 9. 中文 ──
    zh = build_result({"papers": [
        {"id": "z1", "title": "稀疏注意力与长文本建模", "keywords": ["稀疏注意力", "长文本"], "year": 2024},
        {"id": "z2", "title": "稀疏注意力在检索增强中的应用", "keywords": ["稀疏注意力", "检索增强"],
         "year": 2024, "references": ["z1"]},
        {"id": "z3", "title": "稠密检索与生成", "keywords": ["稠密检索", "生成"], "year": 2023,
         "references": ["z1"]},
    ]})
    ok("中文社区标签非空", all(v for v in zh["meta"]["community_labels"].values()), str(zh["meta"]["community_labels"]))
    ok("中文节点标签", any("稀疏" in n["label"] or "注意力" in n["label"] for n in zh["nodes"]))

    print(f"\ncitation_graph: {checks - failures}/{checks} 通过")
    return 1 if failures else 0


# ────────────────────────── CLI ──────────────────────────


def _read_input(args: argparse.Namespace) -> Any:
    if args.data:
        return json.loads(args.data)
    path = args.input_file or args.file
    if path == "-":
        text = sys.stdin.read()
    elif path:
        text = Path(path).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise SystemExit("需要输入：位置参数 FILE、-i FILE、--data JSON，或从 stdin 管道传入；--demo 可离线演示。")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"JSON 解析失败（第 {e.lineno} 行第 {e.colno} 列）：{e.msg}") from e


def _write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(
        prog="python -m app.citation_graph",
        description="引用图：PageRank/中心性找基石、Louvain 找子领域、规则识别边缘衍生论文",
    )
    p.add_argument("file", nargs="?", help="引用数据 JSON；- 读 stdin")
    p.add_argument("-i", "--input", dest="input_file", help="输入文件（与位置参数等价）")
    p.add_argument("--data", help="内联 JSON")
    p.add_argument("--out", "-o", help="写出结果 JSON（同时打印到 stdout）")
    p.add_argument("--top-keystone", type=int, default=3)
    p.add_argument("--peripheral-max-indeg", type=int, default=1)
    p.add_argument("--peripheral-min-outdeg", type=int, default=2)
    p.add_argument("--peripheral-min-cross", type=float, default=0.35)
    p.add_argument("--peripheral-min-innovation", type=float, default=0.55)
    p.add_argument("--keep-external", action="store_true", help="把集合外引用保留为 ext: 占位节点")
    p.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution，越大社区越多")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo", action="store_true", help="20 篇合成引用图")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()

    data = demo_data() if args.demo else _read_input(args)
    try:
        result = build_result(
            data,
            keep_external=args.keep_external,
            top_keystone=args.top_keystone,
            peripheral_max_indeg=args.peripheral_max_indeg,
            peripheral_min_outdeg=args.peripheral_min_outdeg,
            peripheral_min_cross=args.peripheral_min_cross,
            peripheral_min_innovation=args.peripheral_min_innovation,
            resolution=args.resolution,
            seed=args.seed,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] {e}", file=sys.stderr)
        if args.debug:
            raise
        return 1

    if args.out:
        path = _write_json(args.out, result)
        print(f"[OK] 写入 {path}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
