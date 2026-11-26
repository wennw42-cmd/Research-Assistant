"""根据系统架构描述生成可直接运行的绘图脚本（TikZ / Graphviz / Mermaid / Matplotlib）。

输入 `{{arch_desc}}` 支持三种写法::

    # 纯文本：箭头或“A 调用 B”（动词两侧留空格），[分组] 切换分组
    用户 -> Web 客户端: 访问
    Web 客户端 调用 API 网关
    [数据层]
    向量数据库

    # JSON 结构
    {"title": "...",
     "nodes": [{"id": "u", "label": "用户", "kind": "client", "group": "接入层"}],
     "edges": [{"from": "u", "to": "g", "label": "请求"}]}

    # JSON 邻接表
    {"用户": ["API 网关"], "API 网关": ["检索服务", "向量数据库"]}

输出 `{{format}}` 为 tikz / graphviz / mermaid / matplotlib 之一。四种输出都包含
图例；节点统一编号为 n1..nN，原标签保留。本模块自身只用标准库，生成物所需的
外部工具/库会在脚本头部声明。

用法::

    python -m app.diagram --demo --format mermaid
    python -m app.diagram -i arch.json --format tikz --out architecture.tex
    python -m app.diagram --arch "用户 -> API 网关 -> 检索服务" -f graphviz
    python -m app.diagram --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

FORMATS = ("tikz", "graphviz", "mermaid", "matplotlib")
KIND_ORDER = ("client", "service", "store", "external", "model")
KIND_STYLE: dict[str, dict[str, str]] = {
    "client": {"fill": "#E8F1FF", "draw": "#3B6FB6", "label": "客户端 / 用户"},
    "service": {"fill": "#EEF3FB", "draw": "#4A6FA5", "label": "服务"},
    "store": {"fill": "#EAF6EF", "draw": "#2F7D5B", "label": "存储"},
    "external": {"fill": "#FDF1E3", "draw": "#B26B1E", "label": "外部系统"},
    "model": {"fill": "#F3ECFA", "draw": "#6B4FA0", "label": "模型 / 推理"},
}
META_KEYS = {"title", "name", "description", "desc", "meta", "version"}

EDGE_RE = re.compile(r"^(.*?)\s*(?:-->|->|→|⇒|=>)\s*(.*)$")
REL_RE = re.compile(
    r"^(.{1,28}?)\s+(?:调用|请求|访问|查询|读取|写入|连接到|连接|发送到|发送给|"
    r"发送|转发到|转发|依赖|calls?|requests?|accesses|queries|reads?|writes?|"
    r"connects?\s+to|sends?\s+to|depends?\s+on)\s+(.{1,28})$",
    re.I,
)
GROUP_RE = re.compile(r"^\s*(?:\[(.+?)\]|【(.+?)】)\s*$")
MD_HEAD_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
NODE_DESC_RE = re.compile(r"^(.{1,40}?)\s*[:：]\s*(.+)$")
BULLET_RE = re.compile(r"^\s*(?:[-*+•·]|\d+[.)、]|[（(]\d+[)）])\s*")


# ────────────────────────── 数据模型 ──────────────────────────


@dataclass
class Node:
    id: str
    label: str
    kind: str = "service"
    group: str = ""
    description: str = ""

    def to_dict(self, x: float | None = None, y: float | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "group": self.group,
            "description": self.description,
        }
        if x is not None and y is not None:
            d["x"], d["y"] = round(x, 2), round(y, 2)
        return d


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"from": self.src, "to": self.dst, "label": self.label}


@dataclass
class Diagram:
    title: str
    nodes: list[Node]
    edges: list[Edge]
    issues: list[str] = field(default_factory=list)


# ────────────────────────── 输入解析 ──────────────────────────


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).strip().lower()


def _clean_label(text: str) -> str:
    s = BULLET_RE.sub("", str(text)).strip()
    s = s.strip("\"'“”‘’")
    s = re.sub(r"[。；;，,]\s*$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _infer_kind(label: str, description: str = "") -> str:
    text = f"{label} {description}".lower()
    if re.search(r"用户|客户端|浏览器|前端|界面|移动端|手机|user|client|browser|frontend|\bui\b", text):
        return "client"
    if re.search(r"数据库|存储|缓存|队列|索引库|数据仓库|文件系统|向量库|"
                 r"database|\bdb\b|cache|redis|kafka|storage|vector\s*(db|store)", text):
        return "store"
    if re.search(r"第三方|外部|上游|供应商|支付|短信|oauth|sso|third[- ]?party|external|vendor", text):
        return "external"
    if re.search(r"模型|大模型|推理|嵌入|向量化|重排|llm|embedding|rerank|classifier|"
                 r"neural|transformer|model\s+serv", text):
        return "model"
    return "service"


class _Builder:
    """按首次出现顺序建节点，统一编号 n1..nN。"""

    def __init__(self, title: str = ""):
        self.title = title or "系统架构图"
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.by_key: dict[str, Node] = {}
        self.by_alias: dict[str, Node] = {}
        self.edge_keys: set[tuple[str, str, str]] = set()
        self.issues: list[str] = []
        self.group = ""

    def add_node(self, label: str, group: str | None = None, kind: str = "",
                 description: str = "", alias: str = "", quiet: bool = False) -> Node | None:
        label = _clean_label(label)
        if not label:
            if not quiet:
                self.issues.append("忽略空节点标签")
            return None
        key = _norm(label)
        if key in self.by_key:
            node = self.by_key[key]
            if description and not node.description:
                node.description = description
            if group and not node.group:
                node.group = group
            if alias:
                self.by_alias[_norm(alias)] = node
            return node

        kind = (kind or "").strip().lower()
        if kind not in KIND_STYLE:
            kind = _infer_kind(label, description)
        node = Node(
            id=f"n{len(self.nodes) + 1}",
            label=label,
            kind=kind,
            group=(self.group if group is None else group),
            description=description.strip(),
        )
        self.nodes.append(node)
        self.by_key[key] = node
        self.by_alias[key] = node
        if alias:
            self.by_alias[_norm(alias)] = node
        return node

    def resolve(self, token: str, quiet: bool = False, warn: bool = False) -> Node | None:
        cleaned = _clean_label(token)
        if not cleaned:
            return None
        key = _norm(cleaned)
        node = self.by_alias.get(key) or self.by_key.get(key)
        if node:
            return node
        node = self.add_node(cleaned, quiet=quiet)
        if node and warn:
            self.issues.append(f"连线端点“{cleaned}”未在节点列表中，已自动补为节点")
        return node

    def add_edge(self, src: str, dst: str, label: str = "",
                 quiet: bool = False, warn: bool = False) -> None:
        a = self.resolve(src, quiet=quiet, warn=warn)
        b = self.resolve(dst, quiet=quiet, warn=warn)
        if not a or not b:
            return
        label = _clean_label(label)
        key = (a.id, b.id, label.lower())
        if key in self.edge_keys:
            return
        self.edge_keys.add(key)
        self.edges.append(Edge(a.id, b.id, label))

    def build(self) -> Diagram:
        if not self.nodes:
            raise ValueError("没有识别到任何节点：请用 A -> B、A 调用 B 或 JSON 的 nodes/edges 描述架构")
        if not self.edges:
            self.issues.append("未识别到连线，仅生成节点图；可用 A -> B 或 A 调用 B 补充关系")
        return Diagram(self.title, self.nodes, self.edges, self.issues)


def _split_edge_label(text: str) -> tuple[str, str]:
    s = text.strip()
    m = re.match(r"^(.+?)\s*[:：]\s*(.+)$", s)
    if m:
        return m.group(1), m.group(2)
    return s, ""


def _clauses(line: str) -> list[str]:
    if EDGE_RE.search(line) or len(line) <= 40:
        return [line]
    parts = [p.strip() for p in re.split(r"[。；;]", line) if p.strip()]
    out: list[str] = []
    for part in parts:
        if len(part) <= 40:
            out.append(part)
        else:
            out.extend(p.strip() for p in re.split(r"[，,]", part) if p.strip())
    return out or [line]


def _consume(b: _Builder, clause: str) -> None:
    s = BULLET_RE.sub("", clause).strip()
    if not s:
        return

    m = EDGE_RE.match(s)
    if m:
        src, rest = m.group(1).strip(), m.group(2).strip()
        dst, label = _split_edge_label(rest)
        if src and dst:
            b.add_edge(src, dst, label)
            return

    m = REL_RE.match(s)
    if m:
        b.add_edge(m.group(1), m.group(2))
        return

    m = NODE_DESC_RE.match(s)
    if m:
        b.add_node(m.group(1), description=m.group(2))
        return

    s = re.sub(r"^(?:系统|平台|架构)?\s*(?:包括|包含|由|分为|主要有|有)\s*", "", s)
    for token in re.split(r"[、,]", s):
        token = _clean_label(token)
        if token:
            b.add_node(token)


def parse_text(desc: str, title: str | None = None) -> Diagram:
    b = _Builder(title or "系统架构图")
    first = True
    for raw in desc.splitlines():
        line = raw.strip()
        if not line:
            continue
        if first:
            first = False
            m = re.match(r"^#\s+(.+)$", line)
            if m and not title:
                b.title = _clean_label(m.group(1))
                continue
        m = GROUP_RE.match(line) or MD_HEAD_RE.match(line)
        if m:
            b.group = _clean_label(next(g for g in m.groups() if g))
            continue
        for clause in _clauses(line):
            _consume(b, clause)

    if not b.nodes:  # 兜底：整段按标点切成节点，保证脚本仍可生成
        for part in re.split(r"[。；;，,\n]", desc):
            part = _clean_label(part)
            if part and not GROUP_RE.match(part):
                b.add_node(part)
    return b.build()


def _as_list(value: Any, name: str, issues: list[str]) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [value]
    issues.append(f"{name} 应为数组，已忽略：{type(value).__name__}")
    return []


def parse_structured(data: Any, title: str | None = None) -> Diagram:
    b = _Builder(title or "系统架构图")
    nodes_raw: list[Any] = []
    edges_raw: list[Any] = []
    quiet_resolve = False

    if isinstance(data, list):
        if data and isinstance(data[0], dict) and any(
            k in data[0] for k in ("from", "source", "src")
        ):
            edges_raw = data
            quiet_resolve = True
        else:
            nodes_raw = data
    elif isinstance(data, dict):
        b.title = title or str(data.get("title") or data.get("name") or "系统架构图")
        structured = any(
            k in data for k in ("nodes", "components", "edges", "links", "connections", "relations")
        )
        if structured:
            nodes_value = next(
                (data[k] for k in ("nodes", "components") if k in data), None)
            edges_value = next(
                (data[k] for k in ("edges", "links", "connections", "relations") if k in data),
                None,
            )
            nodes_raw = _as_list(nodes_value, "nodes", b.issues)
            edges_raw = _as_list(edges_value, "edges", b.issues)
            quiet_resolve = not nodes_raw
        else:
            # 邻接表：{"用户": ["API 网关"], ...}
            for src, targets in data.items():
                if src in META_KEYS:
                    continue
                if isinstance(targets, str):
                    targets = [targets]
                if not isinstance(targets, (list, tuple)):
                    b.issues.append(f"忽略无法解释的字段：{src}")
                    continue
                b.add_node(src)
                for tgt in targets:
                    if isinstance(tgt, dict):
                        dst = tgt.get("to") or tgt.get("target") or tgt.get("name") or ""
                        label = tgt.get("label") or tgt.get("description") or ""
                    else:
                        dst, label = str(tgt), ""
                    if dst:
                        b.add_edge(src, dst, label, quiet=True)
            return b.build()
    else:
        raise ValueError("架构描述必须是文本、JSON 对象或 JSON 数组")

    for item in nodes_raw:
        if isinstance(item, str):
            b.add_node(item)
        elif isinstance(item, dict):
            label = item.get("label") or item.get("title") or item.get("name") or item.get("text") or ""
            if not label:
                b.issues.append("忽略没有 label/name 的节点")
                continue
            b.add_node(
                str(label),
                group=str(item.get("group") or item.get("cluster") or item.get("layer") or ""),
                kind=str(item.get("kind") or item.get("type") or ""),
                description=str(item.get("description") or item.get("desc") or ""),
                alias=str(item.get("id") or item.get("key") or ""),
            )
        else:
            b.issues.append(f"忽略未知节点类型：{type(item).__name__}")

    for item in edges_raw:
        if isinstance(item, str):
            m = EDGE_RE.match(item)
            if not m:
                b.issues.append(f"忽略无法解析的连线：{item}")
                continue
            dst, label = _split_edge_label(m.group(2))
            b.add_edge(m.group(1), dst, label, quiet=quiet_resolve, warn=bool(nodes_raw))
        elif isinstance(item, dict):
            a = (item.get("from") or item.get("source") or item.get("src")
                 or item.get("start") or "")
            z = (item.get("to") or item.get("target") or item.get("dst")
                 or item.get("end") or "")
            label = item.get("label") or item.get("name") or item.get("description") or ""
            if not a or not z:
                b.issues.append("忽略缺少 from/to 的连线")
                continue
            b.add_edge(str(a), str(z), str(label), quiet=quiet_resolve, warn=bool(nodes_raw))
        else:
            b.issues.append(f"忽略未知连线类型：{type(item).__name__}")

    return b.build()


def parse_arch(desc: Any, title: str | None = None) -> Diagram:
    if isinstance(desc, (dict, list)):
        return parse_structured(desc, title)
    text = str(desc or "").strip()
    if not text:
        raise ValueError("架构描述为空：请传 arch.json / --arch / stdin")
    if text.startswith("{"):
        try:
            return parse_structured(json.loads(text), title)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"架构描述以 {{ 开头但不是合法 JSON（第 {exc.lineno} 行第 {exc.colno} 列）：{exc.msg}"
            ) from exc
    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(data, list):
                return parse_structured(data, title)
    return parse_text(text, title)


# ────────────────────────── 布局与文本工具 ──────────────────────────


def layout(d: Diagram) -> tuple[dict[str, tuple[float, float]], list[str]]:
    """最长路径分层；只用于 TikZ / Matplotlib，图引擎自行布局的格式不受影响。"""
    ids = [n.id for n in d.nodes]
    adj: dict[str, list[str]] = {i: [] for i in ids}
    indeg: dict[str, int] = {i: 0 for i in ids}
    for e in d.edges:
        if e.src == e.dst or e.src not in adj or e.dst not in adj:
            continue
        adj[e.src].append(e.dst)
        indeg[e.dst] += 1

    layers = {i: 0 for i in ids}
    issues: list[str] = []
    has_cycle = bool(ids) and not any(v == 0 for v in indeg.values())
    q: deque[str] = deque(i for i in ids if indeg[i] == 0)
    if not q and ids:
        q.append(ids[0])
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in adj[u]:
            layers[v] = max(layers[v], layers[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if has_cycle or seen < len(ids):
        issues.append("引用关系存在环，分层布局可能不理想；Graphviz/Mermaid 会自动处理")

    by_id = {n.id: n for n in d.nodes}
    buckets: dict[int, list[Node]] = {}
    for n in d.nodes:
        buckets.setdefault(layers[n.id], []).append(n)

    coords: dict[str, tuple[float, float]] = {}
    x_gap, y_gap = 4.8, 1.7
    for level in sorted(buckets):
        group = buckets[level]
        for k, node in enumerate(group):
            coords[node.id] = (level * x_gap, ((len(group) - 1) / 2 - k) * y_gap)
    return coords, issues


def _width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def _wrap(text: str, limit: int = 18) -> list[str]:
    lines: list[str] = []
    cur, w = "", 0
    for ch in str(text):
        cw = _width(ch)
        if cur and w + cw > limit:
            lines.append(cur)
            cur, w = "", 0
            if ch == " ":
                continue
        cur += ch
        w += cw
    if cur:
        lines.append(cur)
    return lines or [""]


def _has_non_ascii(*texts: str) -> bool:
    return any(ord(ch) > 127 for text in texts for ch in text)


def _num(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


# ────────────────────────── TikZ ──────────────────────────


_TEX_MAP = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _tex(text: str) -> str:
    return "".join(_TEX_MAP.get(ch, ch) for ch in str(text))


def _tex_label(text: str, limit: int = 18) -> str:
    return r"\\".join(_tex(line) for line in _wrap(text, limit))


def emit_tikz(d: Diagram, coords: dict[str, tuple[float, float]]) -> str:
    non_ascii = _has_non_ascii(d.title, *[n.label for n in d.nodes],
                               *[n.description for n in d.nodes],
                               *[e.label for e in d.edges])
    run = "xelatex architecture.tex" if non_ascii else "pdflatex architecture.tex"
    xs = [p[0] for p in coords.values()] or [0.0]
    ys = [p[1] for p in coords.values()] or [0.0]
    cx = (min(xs) + max(xs)) / 2
    legend_y = min(ys) - 2.2

    lines = [
        "% 由 app.diagram 生成；完整可编译的 TikZ 源码。",
        f"% 运行：{run}（生成 architecture.pdf）",
        "% 依赖：TeX Live / MiKTeX 的 tikz、arrows.meta、shapes.geometric；"
        "含中文时额外需要 ctex（随 TeX 发行版安装）。",
        "\\documentclass[tikz,border=12pt]{standalone}",
        "\\usepackage{tikz}",
        "\\usetikzlibrary{positioning,arrows.meta,shapes.geometric,calc}",
    ]
    if non_ascii:
        lines.append("\\usepackage[UTF8]{ctex}")
    for kind, style in KIND_STYLE.items():
        lines.append(f"\\definecolor{{{kind}fill}}{{HTML}}{{{style['fill'].lstrip('#')}}}")
        lines.append(f"\\definecolor{{{kind}draw}}{{HTML}}{{{style['draw'].lstrip('#')}}}")

    lines += [
        "",
        "\\begin{document}",
        "\\begin{tikzpicture}[",
        "  kindclient/.style={draw=clientdraw, fill=clientfill, rounded corners=6pt, "
        "line width=0.7pt, inner sep=5pt, align=center, font=\\small},",
        "  kindservice/.style={draw=servicedraw, fill=servicefill, rounded corners=2pt, "
        "line width=0.7pt, inner sep=5pt, align=center, font=\\small},",
        "  kindstore/.style={draw=storedraw, fill=storefill, shape=cylinder, aspect=0.25, "
        "inner sep=4pt, align=center, font=\\small, minimum height=1.1cm},",
        "  kindexternal/.style={draw=externaldraw, fill=externalfill, dashed, "
        "rounded corners=4pt, line width=0.8pt, inner sep=5pt, align=center, font=\\small},",
        "  kindmodel/.style={draw=modeldraw, fill=modelfill, shape=hexagon, "
        "inner sep=4pt, align=center, font=\\small, minimum width=2.4cm},",
        "  arrow/.style={-{Latex[length=2.2mm]}, draw=black!55, line width=0.7pt}",
        "]",
        "",
        f"\\node[font=\\Large\\bfseries, anchor=south] at ({_num(cx)},{_num(max(ys) + 1.3)}) "
        f"{{{_tex(d.title)}}};",
        "",
    ]

    ordered = sorted(d.nodes, key=lambda n: (coords[n.id][0], -coords[n.id][1]))
    for n in ordered:
        x, y = coords[n.id]
        desc = f"% {_tex(n.description)}" if n.description else ""
        lines.append(
            f"\\node[kind{n.kind}, text width=3.2cm] ({n.id}) at ({_num(x)},{_num(y)}) "
            f"{{{_tex_label(n.label)}}};{desc}"
        )

    lines += ["", "% 连线"]
    for e in d.edges:
        if e.label:
            lines.append(
                f"\\draw[arrow] ({e.src}) -- node[above, font=\\scriptsize, text=black!75, "
                f"fill=white, inner sep=1.5pt, align=center] {{{_tex_label(e.label, 12)}}} ({e.dst});"
            )
        else:
            lines.append(f"\\draw[arrow] ({e.src}) -- ({e.dst});")

    lines += [
        "",
        "% 图例",
        f"\\node[anchor=west, font=\\bfseries\\small] at (0,{_num(legend_y + 0.9)}) {{图例}};",
    ]
    for i, kind in enumerate(KIND_ORDER):
        x = i * 3.6
        lines.append(
            f"\\node[kind{kind}, minimum width=1.1cm, minimum height=0.5cm] "
            f"(lg{i}) at ({_num(x)},{_num(legend_y)}) {{}};"
        )
        lines.append(
            f"\\node[anchor=west, font=\\small] at ({_num(x + 0.8)},{_num(legend_y)}) "
            f"{{{KIND_STYLE[kind]['label']}}};"
        )

    lines += ["", "\\end{tikzpicture}", "\\end{document}", ""]
    return "\n".join(lines)


# ────────────────────────── Graphviz ──────────────────────────


def _gv(text: str) -> str:
    s = str(text).replace("\\", "\\\\").replace('"', '\\"')
    return s.replace("\n", "\\n")


def _gv_label(text: str, limit: int = 18) -> str:
    return _gv("\n".join(_wrap(text, limit)))


def _gv_node_attrs(n: Node, label: str | None = None) -> str:
    shape = {"store": "cylinder", "model": "hexagon"}.get(n.kind, "box")
    style = {
        "external": "rounded,dashed,filled",
        "store": "filled",
        "model": "filled",
    }.get(n.kind, "rounded,filled")
    parts = [
        f'label="{_gv_label(label if label is not None else n.label)}"',
        f"shape={shape}",
        f'style="{style}"',
        f'fillcolor="{KIND_STYLE[n.kind]["fill"]}"',
        f'color="{KIND_STYLE[n.kind]["draw"]}"',
        "penwidth=1.2",
        'margin="0.16,0.10"',
    ]
    if n.description:
        parts.append(f'tooltip="{_gv(n.description)}"')
    return ", ".join(parts)


def emit_graphviz(d: Diagram) -> str:
    lines = [
        "// 由 app.diagram 生成；完整可运行的 Graphviz DOT 源码。",
        "// 运行：dot -Tpng architecture.dot -o architecture.png",
        "// 依赖：Graphviz（dot）。如中文显示为方框，把 fontname 改成系统可用中文字体。",
        "digraph architecture {",
        '  graph [rankdir=LR, fontname="Helvetica", fontsize=14, labelloc=t, '
        f'label="{_gv_label(d.title, 40)}", bgcolor="white", pad="0.35", '
        'nodesep="0.45", ranksep="0.9", splines="spline"];',
        '  node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled", '
        'fillcolor="#EEF3FB", color="#4A6FA5", penwidth=1.2, margin="0.16,0.10"];',
        '  edge [fontname="Helvetica", fontsize=9, color="#6B7280", '
        'fontcolor="#4B5563", arrowsize=0.7];',
        "",
    ]

    groups: dict[str, list[Node]] = {}
    for n in d.nodes:
        if n.group:
            groups.setdefault(n.group, []).append(n)
    grouped_ids = {n.id for group in groups.values() for n in group}

    for i, (name, nodes) in enumerate(groups.items()):
        lines.append(f"  subgraph cluster_{i} {{")
        lines.append(f'    label="{_gv(name)}"; fontsize=11; '
                     'style="rounded,dashed"; color="#CBD5E1";')
        for n in nodes:
            lines.append(f"    {n.id} [{_gv_node_attrs(n)}];")
        lines.append("  }")
        lines.append("")

    ungrouped = [n for n in d.nodes if n.id not in grouped_ids]
    for n in ungrouped:
        lines.append(f"  {n.id} [{_gv_node_attrs(n)}];")
    if ungrouped:
        lines.append("")

    for e in d.edges:
        label = f' [label="{_gv_label(e.label, 12)}"]' if e.label else ""
        lines.append(f"  {e.src} -> {e.dst}{label};")

    lines += ["", "  // 图例", "  subgraph cluster_legend {"]
    lines.append('    label="图例"; fontsize=11; style="rounded,dashed"; color="#CBD5E1";')
    for i, kind in enumerate(KIND_ORDER):
        stub = Node(f"lg{i}", KIND_STYLE[kind]["label"], kind=kind)
        lines.append(f"    {stub.id} [{_gv_node_attrs(stub)}];")
    lines += ["  }", "}", ""]
    return "\n".join(lines)


# ────────────────────────── Mermaid ──────────────────────────


def _mmd(text: str) -> str:
    return str(text).replace('"', "'").replace("<", "&lt;").replace(">", "&gt;")


def _mmd_label(text: str, limit: int = 18) -> str:
    return "<br/>".join(_mmd(line) for line in _wrap(text, limit))


def _mmd_node(n: Node, label: str) -> str:
    if n.kind == "client":
        body = f'(["{label}"])'
    elif n.kind == "store":
        body = f'[("{label}")]'
    elif n.kind == "external":
        body = f'[/"{label}"/]'
    elif n.kind == "model":
        body = '{{"' + label + '"}}'
    else:
        body = f'["{label}"]'
    return f"    {n.id}{body}:::{n.kind}"


def emit_mermaid(d: Diagram) -> str:
    lines = [
        "---",
        f'title: "{_mmd(d.title)}"',
        "---",
        "flowchart TB",
        "  %% 由 app.diagram 生成；保存为 architecture.mmd 后运行：",
        "  %% npx -y @mermaid-js/mermaid-cli -i architecture.mmd -o architecture.png",
        "  %% 也可粘贴到 https://mermaid.live 预览。依赖：Node.js（mermaid-cli）。",
        "",
    ]

    groups: dict[str, list[Node]] = {}
    for n in d.nodes:
        if n.group:
            groups.setdefault(n.group, []).append(n)
    grouped_ids = {n.id for group in groups.values() for n in group}

    for i, (name, nodes) in enumerate(groups.items()):
        lines.append(f'  subgraph g{i}["{_mmd(name)}"]')
        lines.append("    direction TB")
        for n in nodes:
            lines.append(_mmd_node(n, _mmd_label(n.label)))
        lines.append("  end")
        lines.append("")

    for n in d.nodes:
        if n.id not in grouped_ids:
            lines.append(_mmd_node(n, _mmd_label(n.label)))
    if any(n.id not in grouped_ids for n in d.nodes):
        lines.append("")

    for e in d.edges:
        if e.label:
            lines.append(f'  {e.src} -->|"{_mmd_label(e.label, 12)}"| {e.dst}')
        else:
            lines.append(f"  {e.src} --> {e.dst}")
    lines.append("")

    lines += ['  subgraph legend_box["图例"]', "    direction LR"]
    for i, kind in enumerate(KIND_ORDER):
        stub = Node(f"lg{i}", KIND_STYLE[kind]["label"], kind=kind)
        lines.append(_mmd_node(stub, _mmd(stub.label)))
    lines += ["  end", ""]

    for kind, style in KIND_STYLE.items():
        lines.append(
            f"  classDef {kind} fill:{style['fill']},stroke:{style['draw']},"
            f"color:#1F2937,stroke-width:1.2px"
        )
    lines.append("")
    return "\n".join(lines)


# ────────────────────────── Matplotlib ──────────────────────────


_MPL_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 由 app.diagram 生成；依赖：matplotlib>=3.5（pip install matplotlib）
# 运行：python architecture_plot.py architecture.png

import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch
except ImportError:
    raise SystemExit("需要 matplotlib：pip install matplotlib")

NODES = __NODES__
EDGES = __EDGES__
TITLE = __TITLE__
COLORS = __COLORS__
KIND_NAMES = __KIND_NAMES__
BOX_W, BOX_H = 3.0, 0.95
ORDER = ["client", "service", "store", "external", "model"]


def draw(out_path: str) -> None:
    # 中文优先使用系统字体；缺字体时回退 DejaVu Sans（中文可能显示为方框）。
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(__FIG_W__, __FIG_H__))
    pos = {n["id"]: (n["x"], n["y"]) for n in NODES}

    # 先画箭头，再用节点框盖住中心段，箭线自然停在框边。
    for e in EDGES:
        x1, y1 = pos[e["src"]]
        x2, y2 = pos[e["dst"]]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#6B7280", lw=1.1,
                            shrinkA=0.0, shrinkB=0.0, mutation_scale=12),
            zorder=1,
        )

    for n in NODES:
        fill, edge = COLORS[n["kind"]]
        ax.add_patch(FancyBboxPatch(
            (n["x"] - BOX_W / 2, n["y"] - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fill, edgecolor=edge, linewidth=1.3,
            linestyle="--" if n["kind"] == "external" else "-",
            zorder=2,
        ))
        ax.text(n["x"], n["y"], n["label"], ha="center", va="center",
                fontsize=9, color="#1F2937", zorder=3, linespacing=1.25)

    for e in EDGES:
        if not e["label"]:
            continue
        x1, y1 = pos[e["src"]]
        x2, y2 = pos[e["dst"]]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if abs(x2 - x1) >= abs(y2 - y1):
            my += 0.16
        else:
            mx += 0.16
        ax.text(mx, my, e["label"], ha="center", va="center", fontsize=7.5,
                color="#4B5563", zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))

    handles = [
        Patch(facecolor=COLORS[k][0], edgecolor=COLORS[k][1],
              linestyle="--" if k == "external" else "-", label=KIND_NAMES[k])
        for k in ORDER if any(n["kind"] == k for n in NODES)
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=max(1, len(handles)), frameon=True, fontsize=9)

    ax.set_title(TITLE, fontsize=14, fontweight="bold", pad=14)
    ax.set_aspect("equal")
    ax.axis("off")
    xs = [n["x"] for n in NODES]
    ys = [n["y"] for n in NODES]
    ax.set_xlim(min(xs) - BOX_W * 0.8, max(xs) + BOX_W * 0.8)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已生成：{out_path}")


if __name__ == "__main__":
    draw(sys.argv[1] if len(sys.argv) > 1 else "architecture.png")
'''


def emit_matplotlib(d: Diagram, coords: dict[str, tuple[float, float]]) -> str:
    nodes = []
    for n in d.nodes:
        x, y = coords[n.id]
        nodes.append({
            "id": n.id,
            "label": "\n".join(_wrap(n.label, 18)),
            "kind": n.kind,
            "group": n.group,
            "x": round(x, 2),
            "y": round(y, 2),
        })
    edges = [e.to_dict() for e in d.edges]
    colors = {k: (v["fill"], v["draw"]) for k, v in KIND_STYLE.items()}
    names = {k: v["label"] for k, v in KIND_STYLE.items()}

    xs = [p[0] for p in coords.values()] or [0.0]
    ys = [p[1] for p in coords.values()] or [0.0]
    fig_w = max(7.0, min(22.0, (max(xs) - min(xs)) * 0.42 + 4.0))
    fig_h = max(5.0, min(16.0, (max(ys) - min(ys)) * 0.55 + 3.5))

    script = _MPL_TEMPLATE
    for token, value in (
        ("__NODES__", repr(nodes)),
        ("__EDGES__", repr(edges)),
        ("__TITLE__", repr(d.title)),
        ("__COLORS__", repr(colors)),
        ("__KIND_NAMES__", repr(names)),
        ("__FIG_W__", repr(round(fig_w, 2))),
        ("__FIG_H__", repr(round(fig_h, 2))),
    ):
        script = script.replace(token, value)
    return script


# ────────────────────────── 生成入口 ──────────────────────────


FILENAMES = {
    "tikz": "architecture.tex",
    "graphviz": "architecture.dot",
    "mermaid": "architecture.mmd",
    "matplotlib": "architecture_plot.py",
}
RUNS = {
    "graphviz": "dot -Tpng architecture.dot -o architecture.png",
    "mermaid": "npx -y @mermaid-js/mermaid-cli -i architecture.mmd -o architecture.png",
    "matplotlib": "python architecture_plot.py architecture.png",
}


def generate(arch_desc: Any, fmt: str, title: str | None = None) -> dict[str, Any]:
    """解析架构描述并返回选定格式的完整脚本、文件名与运行命令。"""
    fmt = str(fmt or "").strip().lower()
    if fmt not in FORMATS:
        raise ValueError(f"不支持的目标格式：{fmt!r}；可选 {', '.join(FORMATS)}")

    diagram = parse_arch(arch_desc, title)
    coords, layout_issues = layout(diagram)
    for issue in layout_issues:
        if issue not in diagram.issues:
            diagram.issues.append(issue)

    if fmt == "tikz":
        script = emit_tikz(diagram, coords)
        non_ascii = _has_non_ascii(
            diagram.title,
            *[n.label for n in diagram.nodes],
            *[e.label for e in diagram.edges],
        )
        run = "xelatex architecture.tex"
        if not non_ascii:
            run = "pdflatex architecture.tex"
    elif fmt == "graphviz":
        script = emit_graphviz(diagram)
        run = RUNS["graphviz"]
    elif fmt == "mermaid":
        script = emit_mermaid(diagram)
        run = RUNS["mermaid"]
    else:
        script = emit_matplotlib(diagram, coords)
        run = RUNS["matplotlib"]

    issues = list(diagram.issues)
    for n in diagram.nodes:
        if n.id not in script:
            issues.append(f"节点 {n.id} 未出现在生成的脚本中")
    issues = list(dict.fromkeys(issues))

    return {
        "format": fmt,
        "title": diagram.title,
        "filename": FILENAMES[fmt],
        "run": run,
        "script": script,
        "nodes": [n.to_dict(*coords[n.id]) for n in diagram.nodes],
        "edges": [e.to_dict() for e in diagram.edges],
        "issues": issues,
        "stats": {
            "n_nodes": len(diagram.nodes),
            "n_edges": len(diagram.edges),
            "n_groups": len({n.group for n in diagram.nodes if n.group}),
        },
    }


# ────────────────────────── 演示与自检 ──────────────────────────


def demo_input() -> dict[str, Any]:
    return {
        "title": "检索增强问答系统架构",
        "nodes": [
            {"id": "user", "label": "用户", "kind": "client", "group": "接入层"},
            {"id": "web", "label": "Web / App 客户端", "kind": "client", "group": "接入层"},
            {"id": "gateway", "label": "API 网关", "kind": "service", "group": "接入层",
             "description": "鉴权、限流、路由"},
            {"id": "oauth", "label": "统一认证（第三方）", "kind": "external", "group": "外部服务"},
            {"id": "orchestrator", "label": "对话编排服务", "kind": "service", "group": "服务层"},
            {"id": "retriever", "label": "检索服务", "kind": "service", "group": "服务层"},
            {"id": "reranker", "label": "重排序模型", "kind": "model", "group": "服务层"},
            {"id": "llm", "label": "大语言模型服务", "kind": "model", "group": "服务层"},
            {"id": "vector", "label": "向量数据库", "kind": "store", "group": "数据层"},
            {"id": "docs", "label": "文档库 / 对象存储", "kind": "store", "group": "数据层"},
            {"id": "cache", "label": "会话缓存", "kind": "store", "group": "数据层"},
        ],
        "edges": [
            {"from": "user", "to": "web", "label": "访问"},
            {"from": "web", "to": "gateway", "label": "HTTPS 请求"},
            {"from": "gateway", "to": "oauth", "label": "校验令牌"},
            {"from": "gateway", "to": "orchestrator", "label": "转发对话"},
            {"from": "orchestrator", "to": "retriever", "label": "发起检索"},
            {"from": "retriever", "to": "vector", "label": "向量检索"},
            {"from": "retriever", "to": "docs", "label": "读取全文"},
            {"from": "retriever", "to": "reranker", "label": "候选证据"},
            {"from": "reranker", "to": "llm", "label": "重排上下文"},
            {"from": "orchestrator", "to": "llm", "label": "组装提示词"},
            {"from": "orchestrator", "to": "cache", "label": "读写会话"},
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

    # 文本解析：箭头 + 中文关系动词 + 分组
    d = parse_arch("用户 -> API 网关: 发起请求\nAPI 网关 调用 检索服务\n检索服务 -> 向量数据库")
    ok("文本箭头生成 4 个节点", len(d.nodes) == 4, str([n.label for n in d.nodes]))
    ok("文本箭头生成 3 条连线", len(d.edges) == 3, str(len(d.edges)))
    ok("边标签被解析", any(e.label == "发起请求" for e in d.edges))
    ok("中文关系动词被解析",
       any(e.src == "n2" and e.dst == "n3" for e in d.edges),
       str([(e.src, e.dst, e.label) for e in d.edges]))

    grouped = parse_arch("[数据层]\n向量数据库\n文档库\n[服务层]\n检索服务")
    ok("分组标题被识别",
       [n.group for n in grouped.nodes] == ["数据层", "数据层", "服务层"],
       str([n.group for n in grouped.nodes]))

    # JSON：别名、类型、标题
    js = parse_structured({
        "title": "T",
        "nodes": [
            {"id": "u", "label": "用户", "kind": "client", "group": "接入层"},
            {"id": "s", "label": "检索服务"},
        ],
        "edges": [{"from": "u", "to": "s", "label": "请求"}],
    })
    ok("JSON 标题被采用", js.title == "T")
    ok("JSON 节点统一编号为 n1..nN",
       [n.id for n in js.nodes] == ["n1", "n2"])
    ok("JSON 类型被保留", js.nodes[0].kind == "client")
    ok("JSON 边按别名解析", js.edges == [Edge("n1", "n2", "请求")], str(js.edges))
    ok("未声明 kind 时自动推断", js.nodes[1].kind == "service")

    adj = parse_structured({"用户": ["API 网关"], "API 网关": ["检索服务", "向量数据库"]})
    ok("邻接表生成 4 个节点", len(adj.nodes) == 4, str(len(adj.nodes)))
    ok("邻接表生成 3 条边", len(adj.edges) == 3, str(len(adj.edges)))

    unknown = parse_structured({
        "nodes": [{"id": "a", "label": "A"}],
        "edges": [{"from": "a", "to": "幽灵"}],
    })
    ok("未知端点自动补为节点", len(unknown.nodes) == 2)
    ok("未知端点给出提示", any("未在节点列表" in x for x in unknown.issues), str(unknown.issues))

    # 四种格式
    demo = demo_input()
    scripts = {fmt: generate(demo, fmt) for fmt in FORMATS}
    ok("Mermaid 生成 flowchart",
       scripts["mermaid"]["script"].startswith("---") and "flowchart TB" in scripts["mermaid"]["script"])
    ok("Graphviz 生成 digraph",
       "digraph architecture" in scripts["graphviz"]["script"])
    ok("TikZ 生成 standalone 文档",
       "\\documentclass" in scripts["tikz"]["script"]
       and "\\begin{document}" in scripts["tikz"]["script"])
    ok("Matplotlib 生成独立脚本",
       "import matplotlib" in scripts["matplotlib"]["script"]
       and "def draw" in scripts["matplotlib"]["script"])

    ok("Mermaid 含图例", "legend_box" in scripts["mermaid"]["script"])
    ok("Graphviz 含图例", "cluster_legend" in scripts["graphviz"]["script"])
    ok("TikZ 含图例", "图例" in scripts["tikz"]["script"])
    ok("Matplotlib 含图例", "ax.legend" in scripts["matplotlib"]["script"])

    ok("Mermaid 含全部节点",
       all(n["id"] in scripts["mermaid"]["script"] for n in scripts["mermaid"]["nodes"]))
    ok("Graphviz 含全部节点",
       all(n["id"] in scripts["graphviz"]["script"] for n in scripts["graphviz"]["nodes"]))
    ok("TikZ 含全部节点",
       all(n["id"] in scripts["tikz"]["script"] for n in scripts["tikz"]["nodes"]))
    ok("每个格式都有节点与边",
       all(s["stats"]["n_nodes"] == 11 and s["stats"]["n_edges"] == 11
           for s in scripts.values()))

    # Matplotlib 生成物必须是合法 Python
    try:
        compile(scripts["matplotlib"]["script"], "architecture_plot.py", "exec")
        compiled = True
    except SyntaxError as exc:
        compiled = False
        err = str(exc)
    else:
        err = ""
    ok("Matplotlib 脚本语法合法", compiled, err)
    ok("Matplotlib 声明依赖与缺库提示",
       "matplotlib>=3.5" in scripts["matplotlib"]["script"]
       and "pip install matplotlib" in scripts["matplotlib"]["script"])

    # 含中文的 TikZ 需要 ctex + xelatex
    ok("中文 TikZ 引入 ctex", "\\usepackage[UTF8]{ctex}" in scripts["tikz"]["script"])
    ok("中文 TikZ 使用 xelatex", scripts["tikz"]["run"].startswith("xelatex"))

    # 布局：链式结构逐层递增
    chain = generate({"nodes": ["A", "B", "C"],
                      "edges": [{"from": "A", "to": "B"}, {"from": "B", "to": "C"}]},
                     "tikz")
    xs = {n["label"]: n["x"] for n in chain["nodes"]}
    ok("分层布局逐层推进", xs["A"] < xs["B"] < xs["C"], str(xs))

    # 环不会崩溃
    cyc = generate({"nodes": ["A", "B"],
                    "edges": [{"from": "A", "to": "B"}, {"from": "B", "to": "A"}]},
                   "mermaid")
    ok("存在环时仍能生成", "flowchart" in cyc["script"])
    ok("存在环时给出提示", any("环" in x for x in cyc["issues"]), str(cyc["issues"]))

    # 自环不会破坏布局
    loop = generate({"nodes": ["A"], "edges": [{"from": "A", "to": "A"}]}, "graphviz")
    ok("自环不阻塞生成", "A" not in loop["script"] or "n1 -> n1" in loop["script"])

    # 转义：引号、&、括号
    tricky = generate({
        "title": 'T & "U"',
        "nodes": [{"label": 'A "B" [C]'}, {"label": "检索 & 重排"}],
        "edges": [{"from": 'A "B" [C]', "to": "检索 & 重排", "label": 'say "hi"'}],
    }, "graphviz")
    ok("Graphviz 引号被转义", '\\"' in tricky["script"])

    # 长标签换行
    long_d = parse_arch({"nodes": [{"label": "这是一个非常长的服务节点名称用于测试自动换行"}],
                         "edges": []})
    mmd = emit_mermaid(long_d)
    ok("Mermaid 长标签自动换行", "<br/>" in mmd)

    # 标题覆盖
    over = generate(demo, "mermaid", title="自定义标题")
    ok("--title 覆盖标题", over["title"] == "自定义标题")

    # 文件名与运行命令
    ok("文件名与格式对应",
       scripts["tikz"]["filename"] == "architecture.tex"
       and scripts["graphviz"]["filename"] == "architecture.dot"
       and scripts["mermaid"]["filename"] == "architecture.mmd"
       and scripts["matplotlib"]["filename"] == "architecture_plot.py")
    ok("运行说明含渲染命令",
       "dot -Tpng" in scripts["graphviz"]["run"]
       and "mermaid-cli" in scripts["mermaid"]["run"]
       and "architecture_plot.py" in scripts["matplotlib"]["run"])

    # 确定性
    again = generate(demo, "mermaid")
    ok("同一输入输出稳定", again["script"] == scripts["mermaid"]["script"])

    # 空输入报错
    try:
        generate("", "mermaid")
        empty_ok = False
    except ValueError:
        empty_ok = True
    ok("空架构描述被拒绝", empty_ok)

    try:
        generate(demo, "svg")
        fmt_ok = False
    except ValueError:
        fmt_ok = True
    ok("非法格式被拒绝", fmt_ok)

    print(f"\ndiagram: {checks - failures}/{checks} 通过")
    return 0 if failures == 0 else 1


# ────────────────────────── CLI ──────────────────────────


def _read_input(path: str | None) -> Any:
    if not path:
        if sys.stdin.isatty():
            raise SystemExit(
                "没有架构描述：请传 FILE / -i FILE / --arch TEXT，或从 stdin 管道传入。"
            )
        raw = sys.stdin.read()
        if not raw.strip():
            raise SystemExit("stdin 为空。")
        return raw
    try:
        raw = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SystemExit(f"读取失败：{path}：{exc}") from exc
    if not raw.strip():
        raise SystemExit(f"输入为空：{path}")
    if Path(path).suffix.lower() == ".json":
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"JSON 解析失败（第 {exc.lineno} 行第 {exc.colno} 列）：{exc.msg}"
            ) from exc
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(
        prog="python -m app.diagram",
        description="根据系统架构描述生成 TikZ / Graphviz / Mermaid / Matplotlib 绘图脚本。",
    )
    ap.add_argument("file", nargs="?", help="架构描述文件（.json 或文本）")
    ap.add_argument("-i", "--input", help="架构描述文件路径")
    ap.add_argument("--arch", help="架构描述文本或 JSON 字符串")
    ap.add_argument("-f", "--format", default="mermaid", choices=FORMATS,
                    help="目标格式（默认 mermaid）")
    ap.add_argument("--title", help="覆盖图标题")
    ap.add_argument("-o", "--out", help="脚本输出路径")
    ap.add_argument("--json", action="store_true", help="输出含脚本、运行命令与统计的 JSON")
    ap.add_argument("--demo", action="store_true", help="用内置 RAG 架构演示")
    ap.add_argument("--selftest", action="store_true", help="运行内置自检")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.demo:
        desc: Any = demo_input()
    elif args.arch is not None:
        desc = args.arch
    else:
        desc = _read_input(args.input or args.file)

    try:
        result = generate(desc, args.format, title=args.title)
    except ValueError as exc:
        raise SystemExit(f"[diagram] {exc}") from exc

    if args.json:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    elif args.out:
        try:
            Path(args.out).write_text(result["script"], encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"写入失败：{args.out}：{exc}") from exc
    else:
        sys.stdout.write(result["script"])

    if not args.json:
        print(
            f"[diagram] {result['format']} / {result['stats']['n_nodes']} 节点 / "
            f"{result['stats']['n_edges']} 连线 / 文件 {result['filename']}",
            file=sys.stderr,
        )
        print(f"[diagram] 运行：{result['run']}", file=sys.stderr)
        for issue in result["issues"]:
            print(f"[diagram] 注意：{issue}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
