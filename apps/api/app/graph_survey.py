"""基于引用图谱撰写领域简短综述（提取式：图谱统计 + 原文摘句，不调用 LLM）。

材料 `{{graph_and_papers}}` 支持两类输入，可混合在同一个 JSON 里：

1. `app.citation_graph` 的输出
   `{"nodes": [...], "edges": [...], "keystone": [...], "peripheral": [...], "meta": {...}}`
2. 原始论文数据（带摘要）
   `{"papers": [{"id": "...", "title": "...", "year": 2020,
                 "abstract": "...", "keywords": [...], "references": [...]}]}`

正文固定四节：背景 → 发展脉络 → 代表工作 → 当前挑战。
每个论点后用 `[@论文ID]` 标注依据，文末按首次出现顺序列出参考文献。
摘要引句逐字取自材料；材料未提供摘要、年份或出版信息时明确写出，
不用外部知识补全，也不对论文水平作主观评价。

用法::

    python -m app.graph_survey --demo
    python -m app.graph_survey -i materials.json --topic "检索增强生成" --out survey.md
    python -m app.graph_survey --demo --strict
    python -m app.graph_survey --selftest

`--strict` 在引用检查或措辞检查不通过时返回非零退出码；默认只告警。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.citation_graph import _norm_title, build_result, demo_data
from app.survey import _norm, _sentences
from app.utils import short

# ── 章节与检查 ───────────────────────────────────────────────────────────
SECTIONS = ("一、背景", "二、发展脉络", "三、代表工作", "四、当前挑战")
REF_HEADING = "## 参考文献"

# 引用标记：[@节点ID]。ID 来自 citation_graph 输出，文末必须能查到。
CITE_RE = re.compile(r"\[@([^\[\]\s]+)\]")
REF_LINE_RE = re.compile(r"^- \*\*\[@([^\[\]]+)\]\*\*", re.M)

# 生成文本中禁止出现的夸大措辞（引号内的原文引用豁免）。
HYPE_WORDS = (
    "颠覆", "革命性", "划时代", "史无前例", "完美", "最强", "无敌",
    "无可匹敌", "遥遥领先", "碾压", "完胜", "突破性", "显著优于",
    "大幅超越", "首次", "state-of-the-art", "groundbreaking", "revolutionary",
)

# 摘要中「贡献句」的触发词。只用于挑选原句，不参与任何结论生成。
CONTRIB_WORDS: dict[str, float] = {
    "we propose": 4.0, "we introduce": 3.5, "we present": 3.0,
    "we design": 3.0, "we show": 2.0, "we demonstrate": 2.5,
    "this paper presents": 3.0, "in this work": 2.5, "this work": 2.0,
    "提出": 3.0, "介绍": 2.0, "设计": 2.0, "构建": 2.0, "实现": 1.0,
    "表明": 1.5, "证明": 2.0, "展示": 1.0, "贡献": 2.0, "方法": 0.8,
    "framework": 1.0, "approach": 0.8, "novel": 1.5, "survey": 3.0,
    "review": 2.5, "综述": 3.0, "回顾": 2.0,
}

# 摘要句入选长度区间（字符）；太短像标题，太长不适合引用。
QUOTE_MIN, QUOTE_MAX = 30, 200


@dataclass
class Materials:
    """已归一化、可直接用于写作的材料。"""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    keystone: list[dict[str, Any]]
    peripheral: list[dict[str, Any]]
    meta: dict[str, Any]
    content: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ids(self) -> set[str]:
        return {n["id"] for n in self.nodes}

    @property
    def by_id(self) -> dict[str, dict[str, Any]]:
        return {n["id"]: n for n in self.nodes}


# ────────────────────────── 材料加载 ──────────────────────────


def _paper_keys(d: dict[str, Any]) -> list[str]:
    """一个论文对象可能被引用的所有键：id / 标题 / DOI / arXiv。"""
    keys: list[str] = []
    for k in ("id", "paper_id", "key", "doi", "arxiv_id"):
        v = d.get(k)
        if v:
            keys.append(str(v).strip())
    if d.get("title"):
        keys.append(_norm_title(d["title"]))
    return [k for k in keys if k]


def _looks_like_paper(d: dict[str, Any]) -> bool:
    if not d.get("title"):
        return False
    return any(
        d.get(k) not in (None, "", [], {})
        for k in ("abstract", "summary", "tldr", "keywords", "authors", "year")
    )


def collect_contents(data: Any) -> dict[str, dict[str, Any]]:
    """递归收集材料中所有论文对象的正文信息（摘要/关键词等）。

    citation_graph 的节点只保留元数据，摘要会在构图时丢失；
    这里从原始输入里单独捞出来，按 id / 标题 / DOI / arXiv 建索引。
    """
    index: dict[str, dict[str, Any]] = {}

    def merge(key: str, item: dict[str, Any]) -> None:
        cur = index.setdefault(key, {})
        for k in ("title", "abstract", "keywords", "authors", "venue", "year", "doi", "arxiv_id"):
            v = item.get(k)
            if v in (None, "", [], {}):
                continue
            if k == "abstract" and cur.get("abstract") and len(str(cur["abstract"])) >= len(str(v)):
                continue
            cur[k] = v

    def visit(obj: Any, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(obj, dict):
            if _looks_like_paper(obj):
                abstract = obj.get("abstract") or obj.get("summary") or obj.get("tldr") or ""
                item = {
                    "title": obj.get("title"),
                    "abstract": abstract,
                    "keywords": obj.get("keywords") or obj.get("tags") or [],
                    "authors": obj.get("authors") or obj.get("author") or [],
                    "venue": obj.get("venue") or obj.get("journal") or obj.get("conference"),
                    "year": obj.get("year"),
                    "doi": obj.get("doi"),
                    "arxiv_id": obj.get("arxiv_id"),
                }
                for key in _paper_keys(obj):
                    merge(key, item)
            for v in obj.values():
                visit(v, depth + 1)
        elif isinstance(obj, list):
            for v in obj:
                visit(v, depth + 1)

    visit(data)
    return index


def _content_for(node: dict[str, Any], index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for key in (node.get("id"), _norm_title(node.get("title")), node.get("doi"), node.get("arxiv_id")):
        if key and key in index:
            return index[key]
    return {}


def build_materials(data: Any, *, keep_external: bool = False) -> Materials:
    """构图 + 合并内容。只保留非占位节点，保证正文引用都能在参考文献里查到。"""
    result = build_result(data, keep_external=keep_external)
    contents = collect_contents(data)

    all_nodes = result.get("nodes") or []
    nodes: list[dict[str, Any]] = []
    for n in all_nodes:
        if n.get("placeholder"):
            continue
        c = _content_for(n, contents)
        n = dict(n)
        n["_abstract"] = str(c.get("abstract") or "")
        if not n.get("keywords") and c.get("keywords"):
            n["keywords"] = c["keywords"]
        if not n.get("authors") and c.get("authors"):
            n["authors"] = c["authors"]
        if not n.get("venue") and c.get("venue"):
            n["venue"] = c["venue"]
        if not n.get("year") and c.get("year"):
            n["year"] = c["year"]
        nodes.append(n)

    if not nodes:
        raise ValueError("材料中没有可用论文（除外部占位节点外为空）")

    active = {n["id"] for n in nodes}
    edges = [
        e for e in (result.get("edges") or [])
        if e.get("source") in active and e.get("target") in active
    ]
    return Materials(
        nodes=nodes,
        edges=edges,
        keystone=[k for k in (result.get("keystone") or []) if k.get("id") in active],
        peripheral=[p for p in (result.get("peripheral") or []) if p.get("id") in active],
        meta=result.get("meta") or {},
        content=contents,
    )


# ────────────────────────── 文本小工具 ──────────────────────────


def _cite(ids: Iterable[str]) -> str:
    return "".join(f"[@{i}]" for i in ids if i)


def _name(node: dict[str, Any], n: int = 44) -> str:
    return f"{_cite([node['id']])}《{short(node.get('title') or node['id'], n)}》"


def _meta_text(node: dict[str, Any]) -> str:
    bits = []
    if node.get("year"):
        bits.append(str(node["year"]))
    if node.get("venue"):
        bits.append(str(node["venue"]))
    return f"（{'，'.join(bits)}）" if bits else "（出版信息未提供）"


def _clip(sentence: str, limit: int = 200) -> str:
    if len(sentence) <= limit:
        return sentence
    cut = max(
        sentence.rfind("，", 0, limit),
        sentence.rfind("；", 0, limit),
        sentence.rfind("。", 0, limit),
        sentence.rfind(" ", 0, limit),
        sentence.rfind(",", 0, limit),
    )
    if cut < 60:
        cut = limit
    return sentence[:cut].rstrip("，；、, ;") + "…"


def pick_quote(abstract: str) -> str:
    """从摘要中挑一句最像「贡献陈述」的原句；挑不到返回空串。"""
    abstract = (abstract or "").strip()
    if len(abstract) < QUOTE_MIN:
        return ""
    best, best_score = "", -1.0
    for sent in _sentences(abstract):
        if not (QUOTE_MIN <= len(sent) <= 420):
            continue
        if any(ch in sent for ch in "“”\""):  # 避免引号嵌套，保持逐字引用
            continue
        low = sent.lower()
        score = sum(w for word, w in CONTRIB_WORDS.items() if word in low)
        if score > best_score:
            best, best_score = sent, score
    if not best:
        return ""
    return _clip(best, QUOTE_MAX)


def _evidence_quote(node: dict[str, Any]) -> str:
    q = pick_quote(node.get("_abstract") or "")
    if not q:
        return ""
    return f"其摘要原文表述为：“{q}”{_cite([node['id']])}。"


def _keyword_df(nodes: Sequence[dict[str, Any]]) -> list[tuple[str, set[str]]]:
    df: dict[str, set[str]] = defaultdict(set)
    for n in nodes:
        for kw in n.get("keywords") or []:
            k = str(kw).strip().lower()
            if len(k) >= 2:
                df[k].add(n["id"])
    rows = [(k, v) for k, v in df.items() if len(v) >= 2]
    if not rows:
        rows = list(df.items())
    return sorted(rows, key=lambda kv: (-len(kv[1]), kv[0]))


def _communities(mat: Materials) -> list[dict[str, Any]]:
    by_id = mat.by_id
    out = []
    for c in mat.meta.get("communities") or []:
        members = [by_id[m] for m in (c.get("members") or []) if m in by_id]
        if not members:
            continue
        out.append(
            {
                "id": c.get("id"),
                "label": c.get("label") or f"社区 {int(c.get('id', 0)) + 1}",
                "members": members,
                "size": len(members),
            }
        )
    out.sort(key=lambda c: (-c["size"], int(c["id"])))
    return out


def _years(mat: Materials) -> list[int]:
    return sorted({int(n["year"]) for n in mat.nodes if n.get("year")})


def _comm_of(mat: Materials) -> dict[str, int]:
    return {n["id"]: int(n.get("community", 0)) for n in mat.nodes}


def _neighbor_communities(node_id: str, edges: Sequence[dict[str, Any]], comm: dict[str, int]) -> set[int]:
    """邻接的“其他”社区（不含自身所在社区），与 citation_graph 的 bridges 口径一致。"""
    own = comm.get(node_id)
    out: set[int] = set()
    for e in edges:
        if e["source"] == node_id:
            out.add(comm.get(e["target"], -1))
        elif e["target"] == node_id:
            out.add(comm.get(e["source"], -1))
    return {c for c in out if c >= 0 and c != own}


# ────────────────────────── 四个章节 ──────────────────────────


def _background(mat: Materials, *, include_quotes: bool = True) -> list[str]:
    nodes = mat.nodes
    years = _years(mat)
    comms = _communities(mat)
    paras: list[str] = []

    span = f"{years[0]}–{years[-1]} 年" if years else "年份信息不完整"
    mod = mat.meta.get("modularity")
    mod_txt = f"，模块度 Q={float(mod):.2f}" if mod is not None else ""
    top = sorted(nodes, key=lambda n: (-float(n.get("pagerank") or 0.0), n["id"]))[:3]
    if len(comms) <= 1:
        structure = "社区发现未形成明显的子领域划分"
    else:
        structure = f"Louvain 社区发现得到 {len(comms)} 个子领域"
    paras.append(
        f"本综述以 {len(nodes)} 篇论文及其 {len(mat.edges)} 条有向引用关系为材料"
        f"（边由引用方指向被引方），文献年份跨度为 {span}；"
        f"{structure}{mod_txt}。纳入分析的节点包括 {_cite([n['id'] for n in top])} 等。"
    )

    if len(comms) > 1:
        described = []
        for c in comms[:3]:
            members = sorted(c["members"], key=lambda n: -float(n.get("pagerank") or 0.0))[:2]
            described.append(f"「{c['label']}」（{c['size']} 篇，如 {_cite([m['id'] for m in members])}）")
        paras.append("从社区结构看，材料覆盖的主要方向为 " + "、".join(described) + "。")

    kws = _keyword_df(nodes)
    if kws:
        items = []
        for term, ids in kws[:4]:
            ordered = sorted(
                (mat.by_id[i] for i in ids),
                key=lambda n: (n.get("year") or 9999, n["id"]),
            )[:2]
            items.append(f"{term}（{len(ids)} 篇，如 {_cite([n['id'] for n in ordered])}）")
        paras.append(
            "关键词分布上，" + "、".join(items) + "的出现频率较高；"
            "该分布只反映材料内部的主题构成，不代表领域全貌。"
        )
    else:
        paras.append(
            "材料未提供关键词字段，主题构成只能依据社区划分与标题判断，"
            f"相关节点包括 {_cite([n['id'] for n in top])}。"
        )

    surveys = [
        n for n in nodes
        if re.search(r"\b(survey|review|overview)\b|综述|回顾", n.get("title") or "", re.I)
    ]
    if surveys:
        s = max(surveys, key=lambda n: float(n.get("pagerank") or 0.0))
        if include_quotes:
            q = pick_quote(s.get("_abstract") or "")
            if q:
                paras.append(f"材料中的综述性文献 {_name(s)} 在摘要中自述：“{q}”{_cite([s['id']])}。")
    return paras


def _year_groups(mat: Materials) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    """按年份分组；年份跨度过大时合并为三段。"""
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    unknown: list[dict[str, Any]] = []
    for n in sorted(mat.nodes, key=lambda n: (n.get("year") or 9999, n["id"])):
        if n.get("year"):
            by_year[int(n["year"])].append(n)
        else:
            unknown.append(n)
    years = sorted(by_year)
    if not years:
        return [], unknown
    if len(years) <= 8:
        return [(f"{y} 年", by_year[y]) for y in years], unknown

    size = max(1, (len(years) + 2) // 3)
    groups = []
    for i in range(0, len(years), size):
        chunk = years[i:i + size]
        label = f"{chunk[0]} 年" if len(chunk) == 1 else f"{chunk[0]}–{chunk[-1]} 年"
        items = [n for y in chunk for n in by_year[y]]
        groups.append((label, items))
    return groups, unknown


def _development(mat: Materials) -> list[str]:
    paras: list[str] = []
    groups, unknown = _year_groups(mat)
    if groups:
        for label, items in groups:
            shown = sorted(items, key=lambda n: -float(n.get("pagerank") or 0.0))[:4]
            names = "、".join(f"{_name(n, 30)}" for n in shown)
            tail = " 等" if len(items) > len(shown) else ""
            paras.append(f"- **{label}**（{len(items)} 篇）：{names}{tail}。")
    else:
        paras.append(
            "- 材料未提供任何论文年份，无法按时间排序；"
            f"节点包括 {_cite([n['id'] for n in mat.nodes[:4]])}。"
        )

    if unknown:
        paras.append(
            f"另有 {len(unknown)} 篇未提供年份，未纳入时间排序"
            f"（{_cite([n['id'] for n in unknown[:3]])} 等）。"
        )

    # 引用方向：后发表 → 更早发表
    older = []
    for e in mat.edges:
        sy = mat.by_id.get(e["source"], {}).get("year")
        ty = mat.by_id.get(e["target"], {}).get("year")
        if sy and ty and int(sy) > int(ty):
            older.append(e)
    if older:
        cnt = Counter(e["target"] for e in older)
        top_id, top_n = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        paras.append(
            f"引用方向上，后发表论文引用更早工作的边共 {len(older)} 条，"
            f"其中被引最多的是 {_cite([top_id])}（{top_n} 条）。"
        )
    elif len(mat.edges) > 0:
        paras.append("图谱中的引用边未形成跨年份的后向引用，材料内部的继承关系不明显。")

    # 关键词迁移（只描述材料内部，不推断领域趋势）
    years = _years(mat)
    if len(years) >= 3:
        mid = years[len(years) // 2]
        early = {
            str(k).strip().lower()
            for n in mat.nodes if n.get("year") and int(n["year"]) < mid
            for k in (n.get("keywords") or [])
        }
        late = {
            str(k).strip().lower()
            for n in mat.nodes if n.get("year") and int(n["year"]) >= mid
            for k in (n.get("keywords") or [])
        }
        new_terms = sorted(late - early)
        if new_terms:
            shown_terms = []
            for term in new_terms[:3]:
                ids = [
                    n["id"] for n in mat.nodes
                    if n.get("year") and int(n["year"]) >= mid
                    and term in {str(k).strip().lower() for k in (n.get("keywords") or [])}
                ]
                shown_terms.append(f"{term}（{_cite(ids[:2])}）")
            paras.append(
                f"从材料内部看，{mid} 年及之后出现、此前未出现的主题词包括 "
                + "、".join(shown_terms)
                + "；词表变化只说明材料构成的变化，不等同于领域趋势。"
            )

    if mat.keystone:
        items = []
        for k in mat.keystone[:3]:
            node = mat.by_id.get(k["id"])
            if not node:
                continue
            metrics = k.get("metrics") or {}
            items.append(
                f"{_cite([k['id']])}《{short(node.get('title') or k['id'], 28)}》"
                f"（PageRank {float(metrics.get('pagerank') or 0):.3f}，"
                f"被引 {metrics.get('in_degree', 0)} 次，"
                f"介数 {float(metrics.get('betweenness') or 0):.3f}）"
            )
        if items:
            paras.append("结构指标上，按综合基石分排序的前几位为 " + "；".join(items) + "。")
    return paras


def _works(mat: Materials, *, max_communities: int = 5, include_quotes: bool = True) -> list[str]:
    paras: list[str] = []
    comms = _communities(mat)
    pr_order = {
        n["id"]: i + 1
        for i, n in enumerate(sorted(mat.nodes, key=lambda n: (-float(n.get("pagerank") or 0.0), n["id"])))
    }
    comm = _comm_of(mat)

    if mat.keystone:
        head = "、".join(
            f"{_cite([k['id']])}（{float(k.get('score') or 0):.2f}）" for k in mat.keystone[:3]
        )
        paras.append(f"综合各项中心性指标，图谱中的基石论文依次为 {head}。")

    for c in comms[:max_communities]:
        members = sorted(
            c["members"],
            key=lambda n: (-float(n.get("keystone_score") or 0.0), -float(n.get("pagerank") or 0.0), n["id"]),
        )
        primary = members[0]
        nbr = _neighbor_communities(primary["id"], mat.edges, comm)
        text = (
            f"**{c['label']}**（{c['size']} 篇）。代表工作 {_name(primary)}{_meta_text(primary)}："
            f"图谱中被 {primary.get('in_degree', 0)} 篇引用、引用 {primary.get('out_degree', 0)} 篇"
        )
        ks_row = next((k for k in mat.keystone if k.get("id") == primary["id"]), None)
        if ks_row:
            text += (
                f"；综合基石分 {float(ks_row.get('score') or 0):.2f}，"
                f"邻接 {len(nbr)} 个其他社区"
            )
        else:
            text += f"；PageRank 全图排名第 {pr_order.get(primary['id'], '—')}"
        text += "。"
        if include_quotes:
            quote = _evidence_quote(primary)
            if quote:
                text += quote
            else:
                text += "材料未提供该文摘要，此处仅依据图谱结构，不对其方法细节作内容判断。"
        else:
            text += "（按参数要求未引用摘要原文。）"
        paras.append(text)

        second = next((n for n in members[1:] if n["id"] != primary["id"]), None)
        if second:
            kws = "、".join(str(k) for k in (second.get("keywords") or [])[:3]) or "未提供关键词"
            paras.append(
                f"同社区内，{_name(second, 32)}{_meta_text(second)}的 PageRank 为 "
                f"{float(second.get('pagerank') or 0):.3f}，关键词为 {kws} {_cite([second['id']])}。"
            )

    if len(comms) > max_communities:
        rest = comms[max_communities:]
        rest_nodes = sorted(
            (n for c in rest for n in c["members"]),
            key=lambda n: -float(n.get("pagerank") or 0.0),
        )
        paras.append(
            f"其余 {len(rest)} 个社区共 {len(rest_nodes)} 篇未逐一展开，"
            f"包括 {_cite([n['id'] for n in rest_nodes[:5]])} 等。"
        )
    return paras


def _challenges(mat: Materials) -> list[str]:
    bullets: list[str] = []
    comm = _comm_of(mat)
    nodes = mat.nodes
    m = len(mat.edges)
    years = _years(mat)
    comms = _communities(mat)

    # 1) 跨社区整合
    cross = [e for e in mat.edges if comm.get(e["source"]) != comm.get(e["target"])]
    if m and cross:
        share = 100.0 * len(cross) / m
        bridge_count: Counter[str] = Counter()
        for e in cross:
            bridge_count[e["source"]] += 1
            bridge_count[e["target"]] += 1
        top_bridge, top_edges = sorted(bridge_count.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        nbr = _neighbor_communities(top_bridge, mat.edges, comm)
        if share < 15.0:
            bullets.append(
                f"**跨社区整合依赖少数节点。** {m} 条引用中跨社区边仅 {len(cross)} 条"
                f"（{share:.0f}%），{_cite([top_bridge])} 参与的跨社区边最多"
                f"（{top_edges} 条，连接 {len(nbr)} 个社区）。社区间的知识流动因此较为集中，"
                "对桥接节点的完整性较敏感。"
            )
        else:
            bullets.append(
                f"**子领域边界存在交叉。** 跨社区引用 {len(cross)} 条（占 {share:.0f}%），"
                f"其中 {_cite([top_bridge])} 参与 {top_edges} 条、连接 {len(nbr)} 个社区；"
                "按社区归纳时需注意主题重叠，不宜把社区标签当作互斥分类。"
            )
    elif m:
        bullets.append(
            "**子领域之间缺少引用关联。** 图谱中没有跨社区引用，各社区在引用层面相互独立；"
            "这可能与材料选取范围有关，不能据此推断领域整体的割裂程度。"
            + (f"相关节点示例：{_cite([n['id'] for n in nodes[:3]])}。" if nodes else "")
        )

    # 2) 最新工作的引用验证
    if years and len(years) >= 2:
        latest = years[-1]
        recent = [n for n in mat.nodes if n.get("year") and int(n["year"]) == latest]
        zero = [n for n in recent if int(n.get("in_degree") or 0) == 0]
        if zero:
            bullets.append(
                f"**最新工作尚未进入引用链。** 材料中最新年份为 {latest} 年，该年 {len(recent)} 篇"
                f"论文中有 {len(zero)} 篇尚无入链引用"
                f"（{_cite([n['id'] for n in zero[:3]])}{' 等' if len(zero) > 3 else ''}）。"
                "这些工作的影响尚待后续研究检验，本综述不对其水平作评价。"
            )
        elif recent:
            bullets.append(
                f"**新近工作的影响仍待观察。** {latest} 年的 {len(recent)} 篇论文均已获得至少 1 条入链引用，"
                "但引用量整体偏低，长期影响仍需跟踪。"
                + f"代表节点：{_cite([n['id'] for n in recent[:3]])}。"
            )

    # 3) 边缘衍生论文
    if mat.peripheral:
        shown = mat.peripheral[:3]
        items = [
            f"{_cite([p['id']])}（跨 {len(p.get('neighbor_communities') or [])} 个社区，"
            f"引用 {p.get('cites_keystone', 0)} 篇基石论文）"
            for p in shown
        ]
        bullets.append(
            f"**边缘衍生方向尚未稳定。** 按低入度（≤1）、高出度（≥2）与跨社区关联规则，"
            f"识别出 {len(mat.peripheral)} 篇边缘衍生论文，如 " + "、".join(items) +
            (" 等" if len(mat.peripheral) > len(shown) else "") +
            "。这类节点可能对应新兴交叉方向，但当前引用链尚不稳定，需结合全文进一步判断。"
        )

    # 4) 结构集中度
    total_pr = sum(float(n.get("pagerank") or 0.0) for n in nodes)
    if total_pr > 0:
        top3 = sorted(nodes, key=lambda n: -float(n.get("pagerank") or 0.0))[:3]
        share = 100.0 * sum(float(n.get("pagerank") or 0.0) for n in top3) / total_pr
        if share >= 50.0:
            bullets.append(
                f"**引用重要性高度集中。** PageRank 前 3 的节点合计占 {share:.0f}%"
                f"（{_cite([n['id'] for n in top3])}）。以被引量作为质量代理时，"
                "新近工作与小众方向容易被低估；这是指标解释层面的限制，不构成对具体论文质量的判断。"
            )
        else:
            bullets.append(
                f"**引用分布相对分散。** PageRank 前 3 的节点合计占 {share:.0f}%"
                f"（{_cite([n['id'] for n in top3])}），未出现单一工作主导引用流向的情况。"
            )

    # 规模过小时明确说明结论的稳健性边界
    if len(nodes) < 5 or m < 4:
        bullets.append(
            f"**材料规模有限。** 当前图谱仅含 {len(nodes)} 篇论文、{m} 条引用边"
            f"（{_cite([n['id'] for n in nodes[:3]])} 等），上述结构判断的稳健性有限，"
            "扩充材料后应重新核算。"
        )

    if not bullets:
        bullets.append(
            "材料中的节点与边不足以支持结构性挑战的归纳；"
            f"现有节点为 {_cite([n['id'] for n in nodes[:3]])}，需补充引用关系后再分析。"
        )
    return bullets


# ────────────────────────── 渲染与校验 ──────────────────────────


def _reference_line(node: dict[str, Any]) -> str:
    authors = "、".join(str(a) for a in (node.get("authors") or [])[:6]) or "作者未提供"
    title = node.get("title") or node["id"]
    tail = []
    if node.get("venue") and node.get("year"):
        tail.append(f"{node['venue']}, {node['year']}")
    elif node.get("year"):
        tail.append(str(node["year"]))
    elif node.get("venue"):
        tail.append(str(node["venue"]))
    if node.get("arxiv_id"):
        tail.append(f"arXiv:{node['arxiv_id']}")
    if node.get("doi"):
        tail.append(f"DOI:{node['doi']}")
    suffix = "；".join(tail) if tail else "出版信息未提供"
    authors = authors.rstrip(".")
    title = title.rstrip(".")
    return f"- **[@%s]** %s. %s. %s." % (node["id"], authors, title, suffix)


def _ordered_cited(markdown: str) -> list[str]:
    seen: list[str] = []
    for key in CITE_RE.findall(markdown):
        if key not in seen:
            seen.append(key)
    return seen


def lint_language(markdown: str) -> list[str]:
    """检查夸大措辞。引号内的原文引用与《标题》豁免。"""
    text = re.sub(r"“[^”]*”", "", markdown)
    text = re.sub(r"《[^》]*》", "", text)
    low = text.lower()
    return [w for w in HYPE_WORDS if w.lower() in low]


def _split_sections(markdown: str) -> dict[str, str]:
    body = markdown.split(REF_HEADING)[0]
    parts = re.split(r"^## ", body, flags=re.M)
    out = {}
    for part in parts[1:]:
        head, _, text = part.partition("\n")
        out[head.strip()] = text
    return out


def verify(markdown: str, valid_ids: Iterable[str]) -> list[str]:
    """校验：引用都有定义、每节都有引用、参考文献与正文一致。"""
    issues: list[str] = []
    valid = set(valid_ids)
    cited = CITE_RE.findall(markdown)
    undefined = sorted({k for k in cited if k not in valid})
    if undefined:
        issues.append(f"引用了图谱中不存在的节点：{', '.join(undefined)}")

    sections = _split_sections(markdown)
    for name in SECTIONS:
        if name not in sections:
            issues.append(f"缺少章节：{name}")
        elif not CITE_RE.search(sections[name]):
            issues.append(f"章节没有引用标注：{name}")

    if REF_HEADING not in markdown:
        issues.append("缺少参考文献章节")
        return issues
    ref_keys = set(REF_LINE_RE.findall(markdown))
    missing = sorted(set(cited) - ref_keys)
    extra = sorted(ref_keys - set(cited))
    if missing:
        issues.append(f"正文引用未列入参考文献：{', '.join(missing)}")
    if extra:
        issues.append(f"参考文献中存在未被正文引用的条目：{', '.join(extra)}")
    return issues


def generate(
    data: Any,
    topic: str | None = None,
    *,
    max_communities: int = 5,
    include_quotes: bool = True,
    keep_external: bool = False,
    source_note: str = "",
) -> dict[str, Any]:
    """生成综述。返回 markdown、引用集合、校验问题与材料统计。"""
    mat = build_materials(data, keep_external=keep_external)
    years = _years(mat)
    comms = _communities(mat)
    kws = _keyword_df(mat.nodes)
    if not topic:
        if kws:
            term = kws[0][0]
            if term[:1].isascii() and term[:1].isalpha():
                term = term[0].upper() + term[1:]
            topic = f"{term} 研究综述：基于引用图谱的结构与内容分析"
        else:
            topic = "基于引用图谱的领域研究综述"

    body: list[str] = [f"# {topic}", ""]
    meta_line = (
        f"> 材料：{len(mat.nodes)} 篇论文 / {len(mat.edges)} 条引用 / {len(comms)} 个社区"
        f" ｜ 生成方式：图谱统计与摘要原文摘句，未调用语言模型，未引入材料外知识。"
    )
    if source_note:
        meta_line += f" {source_note}"
    body += [meta_line, ""]

    def add_section(title: str, paras: Sequence[str]) -> None:
        body.append(f"## {title}")
        body.append("")
        body.extend(paras)
        body.append("")

    add_section("一、背景", _background(mat, include_quotes=include_quotes))
    add_section("二、发展脉络", _development(mat))
    add_section("三、代表工作", _works(mat, max_communities=max_communities, include_quotes=include_quotes))
    add_section("四、当前挑战", _challenges(mat))

    # 材料与范围限制放在挑战之后，属于元信息，不作为领域论点。
    limitations = []
    input_meta = (mat.meta.get("input") or {})
    unknown = int(input_meta.get("unknown_refs") or 0)
    placeholders = int(input_meta.get("placeholder_nodes") or 0)
    if unknown:
        limitations.append(f"{unknown} 条引用指向集合外论文，已在构图时舍弃")
    if placeholders:
        limitations.append(f"保留 {placeholders} 个外部占位节点，未参与正文引用")
    missing_abs = [n for n in mat.nodes if not n.get("_abstract")]
    if missing_abs:
        limitations.append(
            f"{len(missing_abs)} 篇未提供摘要（{_cite([n['id'] for n in missing_abs[:3]])} 等），"
            "其内容层判断仅依据标题与图谱结构"
        )
    if not limitations:
        limitations.append("材料字段完整，正文结论限于图谱统计与摘要原文")
    body += [f"*材料与范围限制：{'；'.join(limitations)}。*", ""]

    markdown_wo_refs = "\n".join(body).rstrip() + "\n\n"
    cited = _ordered_cited(markdown_wo_refs)
    refs = [f"## 参考文献", ""]
    for key in cited:
        node = mat.by_id.get(key)
        if node:
            refs.append(_reference_line(node))
    refs.append("")
    markdown = markdown_wo_refs + "\n".join(refs)

    issues = verify(markdown, mat.ids)
    issues += [f"夸大措辞：{w}" for w in lint_language(markdown)]
    return {
        "markdown": markdown,
        "cited": cited,
        "issues": issues,
        "stats": {
            "n_nodes": len(mat.nodes),
            "n_edges": len(mat.edges),
            "n_communities": len(comms),
            "years": years,
            "modularity": mat.meta.get("modularity"),
            "cited_count": len(cited),
        },
    }


# ────────────────────────── 演示与自检 ──────────────────────────

_ABSTRACT: dict[str, str] = {
    "att": (
        "We propose a new simple network architecture, the Transformer, based solely on attention "
        "mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine "
        "translation tasks show that the model achieves better quality while being more parallelizable."
    ),
    "bert": (
        "We introduce a new language representation model called BERT, designed to pre-train deep "
        "bidirectional representations from unlabeled text. The pre-trained model can be fine-tuned "
        "with just one additional output layer to create state-of-the-art models for a wide range of tasks."
    ),
    "longformer": (
        "We propose the Longformer, a modified Transformer architecture with an attention pattern that "
        "scales linearly with sequence length. The model enables processing of documents with thousands "
        "of tokens on current hardware."
    ),
    "rag": (
        "We introduce RAG, a general-purpose fine-tuning approach that combines parametric and "
        "non-parametric memory for language generation. The model accesses an external document index "
        "and can be updated without re-training the generator."
    ),
    "pbridge": (
        "We present a bridging framework that connects sparse attention and retrieval-augmented "
        "generation for long-document question answering. The framework is evaluated on synthetic "
        "benchmarks and leaves open the question of stable citation impact."
    ),
}


def demo_materials() -> dict[str, Any]:
    """复用 citation_graph 的 20 篇合成数据，并给其中 5 篇补上摘要。"""
    data = demo_data()
    for p in data.get("papers") or []:
        text = _ABSTRACT.get(str(p.get("id")))
        if text:
            p["abstract"] = text
    return data


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

    data = demo_materials()
    out = generate(data, topic="Transformer 与检索增强生成")
    md = out["markdown"]

    ok("四节齐全", all(f"## {s}" in md for s in SECTIONS))
    ok("章节顺序正确", [md.index(f"## {s}") for s in SECTIONS] == sorted(md.index(f"## {s}") for s in SECTIONS))
    ok("标题使用指定主题", md.startswith("# Transformer 与检索增强生成"))
    ok("引用检查通过", not out["issues"], str(out["issues"])[:300])
    ok("无夸大措辞", not lint_language(md), str(lint_language(md)))
    ok("重复生成结果一致", generate(data, topic="Transformer 与检索增强生成")["markdown"] == md)
    ok("引用均来自图谱", set(out["cited"]) <= {n["id"] for n in data["papers"]})
    ok("每节都有引用", all(CITE_RE.search(_split_sections(md)[s]) for s in SECTIONS))
    ok("参考文献覆盖正文引用", set(out["cited"]) == set(REF_LINE_RE.findall(md)))
    ok("基石论文进入正文", "[@bert]" in md and "[@att]" in md)
    ok("边缘论文进入挑战节", "[@pbridge]" in _split_sections(md)["四、当前挑战"])

    quotes = re.findall(r"“([^”]{20,})”", _split_sections(md)["三、代表工作"])
    abstracts = " ".join(_ABSTRACT.values())
    ok("至少引用一句摘要原文", bool(quotes))
    ok(
        "引句逐字可回查",
        bool(quotes) and all(
            _norm(q.rstrip("…")) in _norm(abstracts) for q in quotes
        ),
        str(quotes[:1]),
    )
    ok("最新年份未引挑战出现", "尚无入链引用" in md)
    ok("跨社区挑战包含具体论文", "跨社区" in _split_sections(md)["四、当前挑战"])

    # 直接喂 citation_graph 的输出（nodes/edges/meta）也应可用
    graph_only = generate(build_result(demo_data()), topic="图谱直读")
    ok("直接读取图谱输出", not graph_only["issues"] and graph_only["stats"]["n_nodes"] == 20,
       str(graph_only["issues"])[:200])

    # 缺摘要时不许编内容
    no_abs = generate({"papers": [{"id": "noabs", "title": "A Work Without Abstract", "year": 2020}]})
    ok("缺摘要时明确标注", "未提供该文摘要" in no_abs["markdown"])
    ok("缺摘要时仍有引用", "[@noabs]" in no_abs["markdown"])
    ok("单篇材料不崩溃", "材料规模有限" in no_abs["markdown"])

    # 未知引用应写入限制说明
    data2 = demo_materials()
    data2["papers"][0].setdefault("references", []).append({"id": "ghost-paper"})
    out2 = generate(data2, topic="未知引用测试")
    ok("未知引用被记录", "条引用指向集合外论文" in out2["markdown"])
    ok("未知引用不进入正文", "ghost-paper" not in "".join(out2["cited"]))

    # 占位节点不能出现在正文引用里
    out3 = generate(data2, topic="占位节点测试", keep_external=True)
    ok("占位节点被排除", not any(k.startswith("ext:") for k in out3["cited"]))
    ok("保留占位节点时仍有正文", len(out3["cited"]) > 0)

    # 空材料
    try:
        generate({"papers": []})
        ok("空材料报错", False)
    except ValueError:
        ok("空材料报错", True)

    # 校验器本身
    ok("未定义引用被捕获", bool(verify("## 一、背景\n[@ghost]", {"a"})))
    ok("缺章节被捕获", bool(verify("# x\n[@a]", {"a"})))
    ok("夸大措辞被识别", bool(lint_language("这是一项颠覆性的工作。")))
    ok("引号内措辞豁免", not lint_language("摘要称：“state-of-the-art 结果”。"))
    ok("标题内措辞豁免", not lint_language("参见《A State-of-the-Art Survey》。"))

    # 无年份材料
    no_year = generate({"papers": [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}]})
    ok("无年份材料不崩溃", "未提供任何论文年份" in no_year["markdown"])

    # 摘要过长时应截断且保留省略号
    clipped = _clip("we propose " + "a detailed method " * 30, 200)
    ok("长引句被截断", len(clipped) <= 202 and clipped.endswith("…"), clipped[-20:])

    print(f"\ngraph_survey: {checks - failures}/{checks} 通过")
    return 0 if failures == 0 else 1


# ────────────────────────── CLI ──────────────────────────


def _read_input(path: str | None) -> Any:
    if path:
        if path == "-":
            raw = sys.stdin.read()
        else:
            raw = Path(path).read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()
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
        prog="python -m app.graph_survey",
        description="基于引用图谱生成四节式领域综述（提取式，无 LLM）。",
    )
    ap.add_argument("file", nargs="?", help="材料 JSON；也可用 -i / --data / stdin")
    ap.add_argument("-i", "--input", help="材料 JSON 路径")
    ap.add_argument("--data", help="材料 JSON 字符串")
    ap.add_argument("--topic", help="综述标题；缺省时按高频关键词生成")
    ap.add_argument("-o", "--out", help="Markdown 输出路径（默认打印到 stdout）")
    ap.add_argument("--max-communities", type=int, default=5, help="最多展开几个社区（默认 5）")
    ap.add_argument("--no-quotes", action="store_true", help="不引用摘要原文，只写图谱结构")
    ap.add_argument("--keep-external", action="store_true", help="保留集合外引用为 ext: 占位节点（不参与正文）")
    ap.add_argument("--strict", action="store_true", help="引用/措辞检查不通过时返回非零退出码")
    ap.add_argument("--demo", action="store_true", help="用 20 篇合成数据演示")
    ap.add_argument("--selftest", action="store_true", help="运行内置自检")
    ap.add_argument("--debug", action="store_true", help="打印材料统计")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.demo:
        data = demo_materials()
        note = "演示数据为合成样本，不代表真实文献结论。"
    elif args.data:
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--data JSON 解析失败：{e.msg}") from e
        note = ""
    else:
        data = _read_input(args.input or args.file)
        note = ""

    try:
        result = generate(
            data,
            topic=args.topic,
            max_communities=max(1, args.max_communities),
            include_quotes=not args.no_quotes,
            keep_external=args.keep_external,
            source_note=note,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    md = result["markdown"]
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"[graph_survey] 已写入 {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(md)

    stats = result["stats"]
    print(
        f"[graph_survey] {stats['n_nodes']} 篇 / {stats['n_edges']} 条边 / "
        f"{stats['n_communities']} 个社区 / {stats['cited_count']} 个引用",
        file=sys.stderr,
    )
    if args.debug:
        print(f"[graph_survey] stats={json.dumps(stats, ensure_ascii=False)}", file=sys.stderr)
    if result["issues"]:
        for issue in result["issues"]:
            print(f"[graph_survey] 检查未通过：{issue}", file=sys.stderr)
        if args.strict:
            return 1
    else:
        print("[graph_survey] 引用与措辞检查通过", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
