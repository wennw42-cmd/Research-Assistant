"""多论文综述分析：时间线 / 对比表 / 继承关系与关键转折 / 引用列表。

**提取式（extractive）设计**：每个结论都是材料里的原句 + 出处，不调用 LLM，
结果可复现、可审计。不会用外部知识补全；材料没写的格子标 `missing`。

输入（`--papers papers.json`，顶层可以是列表或 `{"papers": [...]}`）::

    {"paper_id": "transformer",              # 必填，唯一
     "title": "Attention Is All You Need",
     "year": 2017,                            # 必填，时间线排序
     "venue": "NeurIPS",
     "aliases": ["Transformer"],              # 关键：后续论文正文里怎么称呼它
     "abstract": "……",
     "sections": [
        {"chunk_id": "transformer:method", "heading": "3 Model", "page": 3, "text": "……"}
     ]}

`aliases` 决定"继承关系"能否识别：一句话同时出现**较早论文的别名**和关系词
（沿用/基于/改为/替换…）才会连边。不提供别名时退回标题与 paper_id。

用法::

    python -m app.survey --selftest
    python -m app.survey --demo                 # Markdown
    python -m app.survey --demo --json          # 结构化结果
    python -m app.survey --papers papers.json --topic "Transformer 系架构演进"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# ── 对比维度与触发词（权重只影响"选哪一句作证据"，不参与结论生成）──────────────
DIMENSIONS: dict[str, dict[str, float]] = {
    "架构": {
        "架构": 1.5, "结构": 1.0, "编码器": 1.5, "解码器": 1.5, "注意力": 1.0,
        "自注意力": 1.0, "层": 0.4, "transformer": 0.5, "encoder": 1.5,
        "decoder": 1.5, "architecture": 1.5, "layer": 0.4, "双向": 0.5, "卷积": 0.3,
    },
    "训练目标": {
        "训练目标": 2.0, "预训练": 1.5, "目标": 1.0, "掩码": 2.0, "mlm": 2.0,
        "nsp": 2.0, "自回归": 1.5, "语言模型": 1.5, "微调": 1.0, "损失": 1.0,
        "最大似然": 1.5, "objective": 1.5, "autoregressive": 1.5, "token": 0.4,
    },
    "数据规模": {
        "数据": 2.0, "语料": 2.0, "corpus": 2.0, "dataset": 2.0, "参数": 1.5,
        "parameters": 1.5, "token": 1.0, "句对": 1.0, "词": 0.6, "亿": 0.5,
        "万": 0.3, "gb": 1.0, "本": 0.3, "网页": 0.5,
    },
    "创新点": {
        "提出": 2.0, "首次": 2.0, "创新": 2.0, "关键转折": 2.0, "贡献": 1.5,
        "改进": 1.5, "范式": 1.0, "证明": 1.0, "最优": 1.0, "核心": 0.6,
        "propose": 2.0, "introduce": 1.5, "contribution": 1.5, "novel": 1.5,
    },
}
DIM_ORDER = list(DIMENSIONS)
MIN_SCORE = 1.5

# 关系词按优先级排列：一句话同时含"沿用"和"改为"时，判为关键转折
_RELATIONS: list[tuple[str, tuple[str, ...]]] = [
    ("关键转折", ("改为", "替换", "不再", "不同于", "instead", "replace", "unlike", "contrast")),
    ("继承/扩展", ("沿用", "基于", "继承", "采用", "扩展", "延续",
                   "build on", "based on", "extend", "follow")),
    ("对比", ("相比", "对比", "比较", "compare", "outperform", "优于")),
]
_REL_PRIORITY = {name: len(_RELATIONS) - i for i, (name, _) in enumerate(_RELATIONS)}
_TURN_WORDS = ("关键转折", "转折点", "paradigm shift", "范式转变")

# 只在句末标点处切；中文分号常连接同一句的并列成分，切开会丢信息
_SENT_RE = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)\s+|\n+")


# ══════════════════════════════ 数据结构 ══════════════════════════════
@dataclass(frozen=True)
class Passage:
    chunk_id: str
    text: str
    heading: str = ""
    page: int = 0


@dataclass
class Paper:
    paper_id: str
    title: str
    year: int
    venue: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    passages: list[Passage] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    order: int = 0

    @classmethod
    def from_any(cls, obj: dict[str, Any], order: int = 0) -> "Paper":
        if not isinstance(obj, dict):
            raise ValueError(f"papers[{order}] 必须是对象，实际是 {type(obj).__name__}")
        pid = str(obj.get("paper_id") or obj.get("id") or "").strip()
        if not pid:
            raise ValueError(f"papers[{order}] 缺少 paper_id")
        title = str(obj.get("title") or pid).strip()
        try:
            year = int(obj.get("year"))
        except (TypeError, ValueError) as e:
            raise ValueError(f"{pid}: year 必须是整数") from e

        sections = obj.get("sections") or obj.get("passages") or []
        passages: list[Passage] = []
        for i, sec in enumerate(sections):
            if not isinstance(sec, dict):
                raise ValueError(f"{pid}.sections[{i}] 必须是对象")
            text = str(sec.get("text") or "").strip()
            if not text:
                raise ValueError(f"{pid}.sections[{i}] 的 text 为空")
            cid = str(sec.get("chunk_id") or f"{pid}:s{i}").strip()
            passages.append(Passage(cid, text, str(sec.get("heading") or ""),
                                    int(sec.get("page") or 0)))

        abstract = str(obj.get("abstract") or "").strip()
        if abstract and not any(p.text == abstract for p in passages):
            passages.insert(0, Passage(f"{pid}:abstract", abstract, "Abstract", 1))
        if not passages:
            raise ValueError(f"{pid}: 至少需要一个 abstract 或 sections 段落")

        seen: set[str] = set()
        for p in passages:
            if p.chunk_id in seen:
                raise ValueError(f"{pid}: chunk_id 重复：{p.chunk_id}")
            seen.add(p.chunk_id)

        aliases: list[str] = []
        for a in [*(obj.get("aliases") or []), title, pid]:
            a = str(a).strip()
            if a and a.lower() not in {x.lower() for x in aliases}:
                aliases.append(a)

        authors = obj.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        return cls(pid, title, year, str(obj.get("venue") or ""),
                   [str(a) for a in authors], abstract, passages, aliases, order)

    def all_text(self) -> Iterable[tuple[Passage, str]]:
        for p in self.passages:
            for s in _sentences(p.text):
                yield p, s


# ══════════════════════════════ 引用登记 ══════════════════════════════
class _Citations:
    """同一 (论文, chunk, 原句) 只登记一次，编号按首次出现顺序。"""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self._index: dict[tuple[str, str, str], int] = {}

    def add(self, paper: Paper, passage: Passage, quote: str) -> int:
        quote = quote.strip()
        if not _quote_in(quote, passage.text):
            raise ValueError(f"引用必须是原文子串：{paper.paper_id}/{passage.chunk_id}")
        key = (paper.paper_id, passage.chunk_id, quote)
        if key in self._index:
            return self._index[key]
        num = len(self.items) + 1
        self.items.append({"num": num, "paper_id": paper.paper_id, "file": paper.title,
                           "page": passage.page, "chunk_id": passage.chunk_id,
                           "quote": quote})
        self._index[key] = num
        return num


# ══════════════════════════════ 基础工具 ══════════════════════════════
def _sentences(text: str) -> list[str]:
    out = [s.strip() for s in _SENT_RE.split(text) if s and s.strip()]
    return [s for s in out if len(s) >= 6]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _quote_in(quote: str, text: str) -> bool:
    return _norm(quote) in _norm(text)


def _mentions(text: str, aliases: Sequence[str]) -> bool:
    for alias in aliases:
        if re.fullmatch(r"[A-Za-z0-9 .\-]+", alias):
            # 英文别名加边界：GPT 不能命中 GPT-2 / GPT-3
            pat = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9\-])"
            if re.search(pat, text, re.I):
                return True
        elif alias.lower() in text.lower():
            return True
    return False


def _relation(sentence: str) -> str | None:
    low = sentence.lower()
    for name, words in _RELATIONS:
        if any(w in low for w in words):
            return name
    return None


def _arch_class(text: str) -> str:
    """把架构描述归成 encoder-decoder / encoder-only / decoder-only。"""
    t = text.lower()
    if any(k in t for k in ("不使用解码器", "无解码器", "no decoder", "encoder-only",
                            "仅使用编码器", "双向编码器")):
        return "encoder-only"
    has_enc = ("编码器" in t) or ("encoder" in t)
    has_dec = ("解码器" in t) or ("decoder" in t)
    if ("decoder-only" in t or "仅使用解码器" in t or "只保留解码器" in t
            or (has_dec and not has_enc)):
        return "decoder-only"
    if has_enc and has_dec:
        return "encoder-decoder"
    if has_enc:
        return "encoder-only"
    if has_dec:
        return "decoder-only"
    return ""


def _score(sentence: str, keywords: dict[str, float], dim: str) -> float:
    t = sentence.lower()
    s = sum(w for k, w in keywords.items() if k.lower() in t)
    if dim == "数据规模" and re.search(r"\d", sentence):
        s += 0.5
    if any(k in t for k in ("本文", "我们", "we ", "our ")):
        s += 0.3
    return s


# ══════════════════════════════ 分析主体 ══════════════════════════════
def _contribution(paper: Paper) -> tuple[Passage, str]:
    """摘要里最能概括贡献的一句；没有摘要就退回第一段第一句。"""
    passages = [p for p in paper.passages if p.heading.lower() == "abstract"] or paper.passages
    best: tuple[float, Passage, str] | None = None
    for passage in passages:
        for sent in _sentences(passage.text):
            s = 0.0
            if re.search(r"提出|propose|introduce|present", sent, re.I):
                s += 2.0
            if re.search(r"本文|we |our ", sent, re.I):
                s += 0.5
            if best is None or s > best[0]:
                best = (s, passage, sent)
    if best is None:                                  # 理论上 from_any 已保证有文本
        raise ValueError(f"{paper.paper_id}: 没有可用句子")
    return best[1], best[2]


def _pick_cell(paper: Paper, dim: str, reg: _Citations,
               used: set[tuple[str, str]]) -> dict[str, Any] | None:
    best: tuple[tuple[float, int, int], Passage, str] | None = None
    for pi, passage in enumerate(paper.passages):
        for si, sent in enumerate(_sentences(passage.text)):
            s = _score(sent, DIMENSIONS[dim], dim)
            if (passage.chunk_id, sent) in used:
                s -= 0.8                                 # 尽量避免一格塞同一句
            if s < MIN_SCORE:
                continue
            key = (s, -pi, -si)                          # 同分取更靠前的段落/句子
            if best is None or key > best[0]:
                best = (key, passage, sent)
    if best is None:
        return None
    _, passage, sent = best
    used.add((passage.chunk_id, sent))
    return {"text": sent, "cite_ids": [reg.add(paper, passage, sent)],
            "status": "supported"}


def _edges(papers: Sequence[Paper], reg: _Citations) -> list[dict[str, Any]]:
    """较早论文 → 较晚论文：后者句子里同时出现前者的别名和关系词。"""
    out: list[dict[str, Any]] = []
    for i, src in enumerate(papers):
        for dst in papers[i + 1:]:
            best: tuple[int, int, int, str, Passage, str] | None = None
            for pi, passage in enumerate(dst.passages):
                for si, sent in enumerate(_sentences(passage.text)):
                    if not _mentions(sent, src.aliases):
                        continue
                    rel = _relation(sent)
                    if rel is None:
                        continue
                    cand = (_REL_PRIORITY[rel], -pi, -si, rel, passage, sent)
                    if best is None or cand[:3] > best[:3]:
                        best = cand
            if best is not None:
                _, _, _, rel, passage, sent = best
                out.append({"from_paper": src.paper_id, "to_paper": dst.paper_id,
                            "relation": rel, "statement": sent,
                            "cite_ids": [reg.add(dst, passage, sent)], "kind": "stated"})
    return out


def _turning_points(papers: Sequence[Paper], edges: Sequence[dict[str, Any]],
                    rows: dict[str, dict[str, Any]],
                    reg: _Citations) -> list[dict[str, Any]]:
    tps: list[dict[str, Any]] = []
    covered: set[tuple[str, str]] = set()
    seen_stmt: set[tuple[str, str]] = set()

    for e in edges:
        if e["relation"] != "关键转折":
            continue
        tps.append({"kind": "stated", "from_paper": e["from_paper"],
                    "to_paper": e["to_paper"], "statement": e["statement"],
                    "cite_ids": list(e["cite_ids"])})
        covered.add((e["from_paper"], e["to_paper"]))
        seen_stmt.add((e["to_paper"], e["statement"]))

    for paper in papers:                              # 论文自述的"关键转折"
        for passage, sent in paper.all_text():
            if not any(w in sent for w in _TURN_WORDS):
                continue
            if (paper.paper_id, sent) in seen_stmt:
                continue
            tps.append({"kind": "stated", "from_paper": "", "to_paper": paper.paper_id,
                        "statement": sent, "cite_ids": [reg.add(paper, passage, sent)]})

    for prev, cur in zip(papers, papers[1:]):         # 相邻两篇的架构类型变化 = 推断
        a = _arch_class(rows[prev.paper_id]["架构"]["text"]) if rows[prev.paper_id]["架构"] else ""
        b = _arch_class(rows[cur.paper_id]["架构"]["text"]) if rows[cur.paper_id]["架构"] else ""
        if not a or not b or a == b or (prev.paper_id, cur.paper_id) in covered:
            continue
        cites = [*rows[prev.paper_id]["架构"]["cite_ids"],
                 *rows[cur.paper_id]["架构"]["cite_ids"]]
        tps.append({"kind": "inferred", "from_paper": prev.paper_id,
                    "to_paper": cur.paper_id,
                    "statement": f"架构类型由 {a} 转为 {b}", "cite_ids": cites})

    rank = {p.paper_id: i for i, p in enumerate(papers)}
    tps.sort(key=lambda t: (rank.get(t["to_paper"], 0), t["kind"] != "stated"))
    return tps


def analyze(papers: Sequence[dict[str, Any]], *, topic: str = "学术文献演进") -> dict[str, Any]:
    ps = [Paper.from_any(p, i) for i, p in enumerate(papers)]
    if not ps:
        raise ValueError("papers 不能为空")
    ids = [p.paper_id for p in ps]
    dup = sorted({x for x in ids if ids.count(x) > 1})
    if dup:
        raise ValueError(f"paper_id 重复：{', '.join(dup)}")
    ps.sort(key=lambda p: (p.year, p.order))

    reg = _Citations()
    timeline: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for paper in ps:
        passage, sent = _contribution(paper)
        timeline.append({"paper_id": paper.paper_id, "title": paper.title,
                         "year": paper.year, "venue": paper.venue,
                         "contribution": sent,
                         "cite_ids": [reg.add(paper, passage, sent)], "kind": "stated"})

        used: set[tuple[str, str]] = set()
        cells: dict[str, dict[str, Any]] = {}
        for dim in DIM_ORDER:
            cell = _pick_cell(paper, dim, reg, used)
            if cell is None:
                cell = {"text": "材料未提供", "cite_ids": [], "status": "missing"}
                missing.append({"paper_id": paper.paper_id, "field": dim,
                                "reason": "关键段落中没有找到可引用的相关表述"})
            cells[dim] = cell
        rows.append({"paper_id": paper.paper_id, "title": paper.title,
                     "year": paper.year, "venue": paper.venue, "cells": cells})

    edges = _edges(ps, reg)
    tps = _turning_points(ps, edges, {r["paper_id"]: r["cells"] for r in rows}, reg)
    return {"topic": topic, "paper_count": len(ps),
            "timeline": timeline,
            "comparison": {"columns": DIM_ORDER, "rows": rows},
            "lineage": {"edges": edges},
            "turning_points": tps,
            "missing": missing,
            "citations": reg.items}


# ══════════════════════════════ Markdown ══════════════════════════════
def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _mark(ids: Sequence[int]) -> str:
    return " ".join(f"[{n}]" for n in ids)


def _short(title: str) -> str:
    return title.split(":")[0].strip()


def render_markdown(report: dict[str, Any]) -> str:
    titles = {t["paper_id"]: _short(t["title"]) for t in report["timeline"]}
    lines = [f"# 文献演进分析：{report['topic']}", "",
             "> 结论均取自所给材料；“推断”项是分析器基于引用证据的归纳，原文未直接陈述。", ""]

    lines += ["## 一、时间线", ""]
    for i, t in enumerate(report["timeline"], 1):
        venue = f"（{t['venue']}）" if t["venue"] else ""
        lines.append(f"{i}. **{t['year']} · {_md(t['title'])}**{venue} — "
                     f"{_md(t['contribution'])} {_mark(t['cite_ids'])}")

    lines += ["", "## 二、对比表", ""]
    cols = report["comparison"]["columns"]
    lines.append("| 年份 | 论文 | " + " | ".join(cols) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in cols) + " |")
    for row in report["comparison"]["rows"]:
        cells = [_md(row["cells"][d]["text"]) + " " + _mark(row["cells"][d]["cite_ids"])
                 for d in cols]
        lines.append(f"| {row['year']} | {_md(row['title'])} | " + " | ".join(cells) + " |")
    if report["missing"]:
        lines += ["", "**材料缺口**：" + "；".join(
            f"{m['paper_id']} / {m['field']}" for m in report["missing"])]

    lines += ["", "## 三、继承关系与关键转折", "", "**论文间关系**", ""]
    edges = report["lineage"]["edges"]
    if edges:
        for e in edges:
            lines.append(f"- {titles.get(e['from_paper'], e['from_paper'])} → "
                         f"{titles.get(e['to_paper'], e['to_paper'])}：{e['relation']} —— "
                         f"“{_md(e['statement'])}” {_mark(e['cite_ids'])}")
    else:
        lines.append("- （材料中未出现明确的继承关系表述）")

    lines += ["", "**关键转折**", ""]
    tps = report["turning_points"]
    if tps:
        for tp in tps:
            head = (f"{titles.get(tp['from_paper'], tp['from_paper'])} → "
                    f"{titles.get(tp['to_paper'], tp['to_paper'])}"
                    if tp["from_paper"] else titles.get(tp["to_paper"], tp["to_paper"]))
            if tp["kind"] == "inferred":
                lines.append(f"- （推断）{head} —— {tp['statement']} {_mark(tp['cite_ids'])}")
            else:
                label = "原文陈述" if tp["from_paper"] else "原文自述"
                lines.append(f"- （{label}）{head} —— “{_md(tp['statement'])}” "
                             f"{_mark(tp['cite_ids'])}")
    else:
        lines.append("- （材料中未出现明确的转折表述）")

    lines += ["", "## 四、引用列表", ""]
    for c in report["citations"]:
        lines.append(f"[{c['num']}] 《{_md(c['file'])}》 p.{c['page']} · "
                     f"{c['chunk_id']} — “{_md(c['quote'])}”")
    return "\n".join(lines) + "\n"


# ══════════════════════════════ 自检 ══════════════════════════════
def _selftest() -> int:
    from app.survey_samples import build_sample_papers

    papers = build_sample_papers()
    rep = analyze(papers, topic="Transformer 系架构演进")

    ids = [t["paper_id"] for t in rep["timeline"]]
    assert ids == ["transformer", "bert", "gpt1", "gpt2", "gpt3"], ids
    assert [t["year"] for t in rep["timeline"]] == sorted(t["year"] for t in rep["timeline"])

    # 每条引用都能回查到原文，且逐字存在
    source: dict[tuple[str, str], str] = {}
    for p in papers:
        for s in p["sections"]:
            source[(p["paper_id"], s["chunk_id"])] = s["text"]
        source[(p["paper_id"], f"{p['paper_id']}:abstract")] = p["abstract"]
    nums = [c["num"] for c in rep["citations"]]
    assert nums == list(range(1, len(nums) + 1)), nums
    for c in rep["citations"]:
        assert _quote_in(c["quote"], source[(c["paper_id"], c["chunk_id"])]), c

    # 所有结论都有引用
    for t in rep["timeline"]:
        assert t["cite_ids"]
    for row in rep["comparison"]["rows"]:
        for dim in rep["comparison"]["columns"]:
            cell = row["cells"][dim]
            assert cell["status"] in ("supported", "missing")
            if cell["status"] == "supported":
                assert cell["cite_ids"]
    for e in rep["lineage"]["edges"]:
        assert e["cite_ids"]
    for tp in rep["turning_points"]:
        assert tp["cite_ids"]

    # 继承/转折确实从材料里连了出来
    edges = {(e["from_paper"], e["to_paper"], e["relation"])
             for e in rep["lineage"]["edges"]}
    assert ("transformer", "bert", "关键转折") in edges, edges
    assert ("transformer", "gpt1", "关键转折") in edges, edges
    assert ("gpt1", "gpt2", "继承/扩展") in edges, edges
    assert ("gpt2", "gpt3", "继承/扩展") in edges, edges
    assert any(tp["kind"] == "inferred" for tp in rep["turning_points"])

    rows = {r["paper_id"]: r["cells"] for r in rep["comparison"]["rows"]}
    assert _arch_class(rows["bert"]["架构"]["text"]) == "encoder-only"
    assert _arch_class(rows["gpt1"]["架构"]["text"]) == "decoder-only"
    assert all(rows[p]["数据规模"]["status"] == "supported" for p in ids)

    # 材料不足时不编造
    thin = [{"paper_id": "x", "title": "X", "year": 2024,
             "abstract": "本文提出一个新模型。"}]
    rep2 = analyze(thin)
    assert rep2["comparison"]["rows"][0]["cells"]["数据规模"]["status"] == "missing"
    assert rep2["missing"]

    try:
        analyze([thin[0], dict(thin[0])])
    except ValueError:
        pass
    else:
        raise AssertionError("重复 paper_id 应报错")

    md = render_markdown(rep)
    for h in ("## 一、时间线", "## 二、对比表",
              "## 三、继承关系与关键转折", "## 四、引用列表"):
        assert h in md, h
    assert "[1]" in md

    print("selftest: 12/12 通过")
    return 0


# ══════════════════════════════ CLI ══════════════════════════════
def _load_papers(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("papers")
    if not isinstance(data, list):
        raise ValueError("papers JSON 必须是列表，或含 papers 列表的对象")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="多论文综述分析（提取式，全部结论可溯源）")
    ap.add_argument("--selftest", action="store_true", help="跑内置自检")
    ap.add_argument("--demo", action="store_true", help="用内置 Transformer/BERT/GPT 材料演示")
    ap.add_argument("--papers", help="论文 JSON 文件")
    ap.add_argument("--topic", default="学术文献演进")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而不是 Markdown")
    ap.add_argument("--out", help="写入文件（UTF-8）")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")      # Windows 控制台
    except Exception:                                  # noqa: BLE001
        pass

    if args.selftest:
        return _selftest()
    if args.papers:
        papers = _load_papers(args.papers)
    elif args.demo:
        from app.survey_samples import build_sample_papers
        papers = build_sample_papers()
    else:
        ap.print_help()
        return 2

    report = analyze(papers, topic=args.topic)
    text = (json.dumps(report, ensure_ascii=False, indent=2) if args.json
            else render_markdown(report))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"written: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
