"""文献版本管理：语义去重 + 版本先后判定 + 逐段修订差异。

设计要点（对应"不得臆造 / 不评价"的约束）：

1. **判定依据必须可追溯**：`similarity` 只由 title / abstract / 正文段落三组向量的余弦相似度
   加权得到，并把**命中的原文字段**放进 `evidence`，而不是只丢一个数字出来。
2. **元数据规则优先于语义**：同一 arXiv 主干（`1706.03762v1` / `v7`）直接判 duplicate，
   避免"同一篇论文因修订量大被判成两篇"（重复计数），这是两类错误里更贵的那一类。
3. **差异只陈述事实**：LLM 提示词显式禁用价值判断词；无 Key 时退回 `difflib` + 数值正则的
   规则 diff，同样只输出"新增了什么/删除了什么/数值从 X 变 Y"。

输出（四个主键固定，其余为可追溯细节）：
    {"verdict": "duplicate|related|distinct",
     "similarity": 0.93,
     "version_advice": "...",
     "diff_summary": "...",
     "evidence": [...], "diff_detail": [...], "versions": {...}, "engine": {...}}
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path as _Path
from typing import Any, Sequence

# 允许 `python app/agents/dedup.py` 直接运行：把 apps/api（含 app 包）加进模块搜索路径
if __package__ in (None, ""):
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

# ────────────────────────── 常量 ──────────────────────────
DUPLICATE = "duplicate"
RELATED = "related"
DISTINCT = "distinct"

DEFAULT_DUP_TH = 0.82      # ≥ 判为同一篇（不同版本）
DEFAULT_REL_TH = 0.55      # ≥ 判为相关（同主题不同论文）
WEIGHTS = {"title": 0.25, "abstract": 0.35, "body": 0.40}
ALIGN_TH = 0.45            # 段落对齐门槛：低于此值视为"新增/删除"而非"修订"
UNCHANGED_TH = 0.97        # ≥ 视为未修改

_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
# 前置字符允许"非字母"：arXiv ID 是 1706.03762v7（数字后紧跟 vN），
# 若只放行空白/分隔符，这类最常见的学术版本号会被判成"不可比较"。
_VER_RE = re.compile(r"(?:^|[^A-Za-z])v(?:ersion)?\s*(\d+(?:\.\d+){0,2})", re.I)
_PLAIN_NUM_RE = re.compile(r"^v?(\d+(?:\.\d+){0,2})$", re.I)
_DATE_RE = re.compile(r"(20\d{2})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?")


def _q(text: str, n: int = 90) -> str:
    """引用片段：截断但不改写（证据必须字面取自原文）。"""
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= n else t[: n - 1] + "…"


# ────────────────────────── 数据模型 ──────────────────────────
@dataclass
class Section:
    sid: str = ""
    heading: str = ""
    text: str = ""


@dataclass
class PaperDoc:
    """一篇论文的**内容**视图（与文件名/路径无关，避免"同名即重复"的假阳性）。"""

    doc_id: str
    title: str = ""
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    version: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # ── 构造 ──
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperDoc":
        raw_sections = data.get("sections") or data.get("blocks") or data.get("paragraphs") or []
        sections: list[Section] = []
        for i, s in enumerate(raw_sections):
            if isinstance(s, str):
                sections.append(Section(sid=s_id(s, i), heading="", text=s))
            elif isinstance(s, dict):
                sections.append(Section(
                    sid=str(s.get("id") or s.get("block_id") or s_id(s, i)),
                    heading=str(s.get("heading") or s.get("section") or ""),
                    text=str(s.get("text") or s.get("content") or ""),
                ))
        abstract = str(data.get("abstract") or "")
        if not abstract and sections:
            first = sections[0]
            if re.search(r"abstract|摘要", first.heading, re.I):
                abstract = first.text
                sections = sections[1:]
        return cls(
            doc_id=str(data.get("doc_id") or data.get("id") or data.get("paper_id") or ""),
            title=str(data.get("title") or ""),
            abstract=abstract,
            sections=[s for s in sections if s.text.strip()],
            version=str(data.get("version") or ""),
            meta={k: v for k, v in data.items()
                  if k not in {"doc_id", "id", "paper_id", "title", "abstract",
                               "sections", "blocks", "paragraphs", "version"}},
        )

    @classmethod
    def from_corpus(cls, doc_id: str) -> "PaperDoc":
        """从内置语料构建（离线演示用）。"""
        from app.agents.tools import corpus

        doc = corpus.DOCS.get(doc_id, {})
        blocks = [b for b in corpus.BLOCKS if b["doc_id"] == doc_id]
        abstract = next((b["text"] for b in blocks
                         if re.search(r"abstract|摘要", b.get("section", ""), re.I)), "")
        by_section: dict[str, Section] = {}
        for b in blocks:
            if abstract and b["text"] == abstract:
                continue
            sec = b.get("section") or "Body"
            s = by_section.setdefault(sec, Section(sid=f"{doc_id}#{sec}", heading=sec, text=""))
            s.text = (s.text + " " + b["text"]).strip()
        return cls(doc_id=doc_id, title=str(doc.get("title", "")), abstract=abstract,
                   sections=list(by_section.values()), version=str(doc_id),
                   meta={k: v for k, v in doc.items() if k != "title"})

    @classmethod
    def from_version_chain(cls, vid: str) -> "PaperDoc":
        """从版本链语料构建（arxiv:1706.03762v1 / v7）。"""
        from app.agents.tools import corpus

        # 语料里版本链只有摘要，不伪造正文（伪造正文会让"依据片段"出现重复条目）
        return cls(doc_id=vid, title="Attention Is All You Need",
                   abstract=corpus.VERSION_ABSTRACTS.get(vid, ""), sections=[], version=vid)

    @property
    def body_texts(self) -> list[str]:
        return [s.text for s in self.sections]


def s_id(s: Any, i: int) -> str:
    return f"s{i}"


# ────────────────────────── 向量化 ──────────────────────────
def _hash_embed(text: str, dim: int = 768) -> list[float]:
    """无 Key 时的确定性兜底向量：字符 bigram + 词 token 的签名哈希。

    这不是语义模型，只能保证"字面/用词越接近分数越高"。因此结果里会标注
    engine=local-hash，提示线上必须换成真实 embedding 才能用于生产判定。
    """
    vec = [0.0] * dim
    t = (text or "").lower()
    toks: list[str] = []
    toks += re.findall(r"[a-z0-9]+", t)
    compact = re.sub(r"\s+", "", t)
    toks += [compact[i:i + 2] for i in range(len(compact) - 1)]
    if not toks:
        return vec
    for tok in toks:
        h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
        vec[h % dim] += 1.0 if (h >> 63) & 1 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def _cfg(name: str) -> str:
    """读 .env / 环境变量；显式传参优先于这里（见 embed_texts 的 None 哨兵）。"""
    try:
        from app.config import get_settings

        return str(getattr(get_settings(), name, "") or "")
    except Exception:  # noqa: BLE001 —— 直接以脚本方式运行且未装依赖时仍可降级
        return ""


def embed_texts(texts: Sequence[str], *, client: Any = None, model: str | None = None,
                base_url: str | None = None, api_key: str | None = None,
                timeout: float = 30.0, batch: int = 64,
                errors: list[str] | None = None) -> tuple[list[list[float]], str]:
    """返回 (vectors, engine)。engine ∈ openai-compatible | local-hash。

    ``api_key`` / ``base_url`` / ``model`` 传 ``None`` 表示读配置；显式传 ``""`` 表示
    强制离线（测试用）。调用失败的原因写进 ``errors``，不再静默吞掉。
    """
    texts = list(texts)
    if not texts:
        return [], "local-hash"
    errs = errors if errors is not None else []

    if client is None:
        key = api_key if api_key is not None else (_cfg("EMBEDDING_API_KEY") or _cfg("LLM_API_KEY"))
        url = base_url if base_url is not None else (_cfg("EMBEDDING_BASE_URL") or _cfg("LLM_BASE_URL"))
        mdl = model if model is not None else (_cfg("EMBEDDING_MODEL") or "text-embedding-3-small")
        if key and url:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=key, base_url=url, timeout=timeout)
                model = mdl
            except Exception as e:  # noqa: BLE001
                errs.append(f"embedding 客户端初始化失败（{type(e).__name__}），已降级本地哈希")
                client = None
        else:
            client = None

    if client is not None:
        try:
            out: list[list[float]] = []
            for i in range(0, len(texts), batch):
                resp = client.embeddings.create(model=model or "text-embedding-3-small",
                                                input=texts[i:i + batch])
                out += [d.embedding for d in resp.data]
            if len(out) == len(texts):
                return [_norm(v) for v in out], "openai-compatible"
            errs.append(f"embedding 返回条数不符（{len(out)}/{len(texts)}），已降级本地哈希")
        except Exception as e:  # noqa: BLE001 —— 配额/网络/模型不支持，统一降级
            errs.append(f"embedding 调用失败（{type(e).__name__}），已降级本地哈希")
    return [_hash_embed(t) for t in texts], "local-hash"


def _norm(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else [0.0 for _ in v]


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


# ────────────────────────── 相似度 ──────────────────────────
def _sim_matrix(va: list[list[float]], vb: list[list[float]]) -> list[list[float]]:
    """一次算完所有段落对的余弦；正文比对与逐段对齐共用，避免 O(n·m) 算两遍。"""
    return [[_cos(a, b) for b in vb] for a in va]


def best_match_mean(va: list[list[float]], vb: list[list[float]],
                    matrix: list[list[float]] | None = None) -> float:
    """正文整体相似度 = **双向最佳匹配的均值**。

    单向匹配会被"新版新增/删除段落"拖低（v1→v7 加了消融实验，v1 的段落仍能匹配到，
    但 v7 的段落匹配不到 v1），双向取均值才能反映"同一篇论文的不同版本"。
    """
    if not va or not vb:
        return 0.0
    m = matrix if matrix is not None else _sim_matrix(va, vb)
    fwd = sum(max(row) for row in m) / len(va)
    bwd = sum(max(m[i][j] for i in range(len(va))) for j in range(len(vb))) / len(vb)
    return (fwd + bwd) / 2


def align_sections(va: list[list[float]], vb: list[list[float]],
                   threshold: float = ALIGN_TH,
                   matrix: list[list[float]] | None = None) -> list[tuple[int, int, float]]:
    """贪心一对一对齐：全局相似度最高的段对优先配对。"""
    m = matrix if matrix is not None else _sim_matrix(va, vb)
    pairs = [(i, j, m[i][j]) for i in range(len(va)) for j in range(len(vb))
             if m[i][j] >= threshold]
    pairs.sort(key=lambda p: -p[2])
    used_a: set[int] = set()
    used_b: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for i, j, s in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j, s))
    out.sort(key=lambda p: p[0])
    return out


# ────────────────────────── 版本 ──────────────────────────
def base_id(raw: str) -> str:
    """去掉尾部版本号：arxiv:1706.03762v7 → arxiv:1706.03762。"""
    return re.sub(r"[vV]\d+$", "", (raw or "").strip())


def parse_version(raw: str) -> tuple[str, tuple[int, ...]]:
    """解析版本号 → (kind, 可比较元组)。kind ∈ numeric | date | none。"""
    s = (raw or "").strip()
    if not s:
        return "none", ()
    m = _VER_RE.search(s)
    if m:
        return "numeric", _nums(m.group(1))
    m = _DATE_RE.search(s)
    if m:
        return "date", tuple(int(g) for g in (m.group(1), m.group(2), m.group(3) or 1))
    m = _PLAIN_NUM_RE.match(s)
    if m:
        return "numeric", _nums(m.group(1))
    return "none", ()


def _nums(s: str) -> tuple[int, ...]:
    parts = [p for p in s.split(".") if p.isdigit()]
    return tuple(int(p) for p in parts) or ()


def compare_versions(a_raw: str, b_raw: str) -> int | None:
    """B 相对 A：1=更新，0=相同，-1=更旧，None=不可比较。"""
    ka, va = parse_version(a_raw)
    kb, vb = parse_version(b_raw)
    if ka == "none" or kb == "none" or ka != kb or not va or not vb:
        return None
    n = max(len(va), len(vb))
    va = va + (0,) * (n - len(va))
    vb = vb + (0,) * (n - len(vb))
    return (vb > va) - (vb < va)


def version_advice(a_ver: str, b_ver: str, verdict: str) -> str:
    """覆盖建议。A=库中已有，B=用户新上传。"""
    if verdict != DUPLICATE:
        return "非重复文献（不同论文），不涉及覆盖决策，建议作为独立条目入库。"
    cmp = compare_versions(a_ver, b_ver)
    a_disp, b_disp = a_ver or "未标注", b_ver or "未标注"
    if cmp is None:
        return (f"版本不可比较（库中 {a_disp}，新上传 {b_disp}）："
                f"建议人工确认后再决定是否覆盖。")
    if cmp > 0:
        return (f"新上传版本更新（库中 {a_disp} → 新上传 {b_disp}）："
                f"建议覆盖库中旧版本，并将旧版本存档以便回溯。")
    if cmp < 0:
        return (f"库中已有更新版本建议保留（库中 {a_disp}，新上传 {b_disp} 更旧）："
                f"是否覆盖？建议保留库中版本，仅将 {b_disp} 作为历史版本留存。")
    return f"版本相同（{a_disp}）：无需覆盖。"


# ────────────────────────── 差异 ──────────────────────────
_DIFF_SYSTEM = """你是学术文献的版本比对助手，只做**事实陈述**。

硬性约束：
1. 禁止任何价值判断，不得出现"更好/更差/更清晰/改进/优化/提升/完善/严谨"等评价词；
2. 只描述实质变化：新增了什么、删除了什么、数值或实验设置从什么变为什么、哪些表述被改写；
3. 若仅为措辞、语序或排版差异，输出且仅输出：无明显实质变化（仅措辞或排版差异）；
4. 输出不超过 2 句中文，不要编号，不要解释你的推理。"""


def llm_diff(old: str, new: str, *, client: Any = None, model: str | None = None,
             base_url: str | None = None, api_key: str | None = None,
             timeout: float = 30.0) -> str | None:
    key = api_key if api_key is not None else _cfg("LLM_API_KEY")
    url = base_url if base_url is not None else _cfg("LLM_BASE_URL")
    mdl = model if model is not None else (_cfg("LLM_MODEL") or "qwen-plus")
    if not key or not url:
        return None
    try:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=url, timeout=timeout)
        resp = client.chat.completions.create(
            model=mdl, temperature=0.0, timeout=timeout,
            messages=[{"role": "system", "content": _DIFF_SYSTEM},
                      {"role": "user", "content": f"【旧段落】\n{_q(old, 1200)}\n\n【新段落】\n{_q(new, 1200)}"}],
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def rule_diff(old: str, new: str) -> str:
    """无 LLM 时的规则 diff：difflib + 数值变化，同样只陈述事实。"""
    facts: list[str] = []
    if len(old) != len(new):
        facts.append(f"字数 {len(old)} → {len(new)}")

    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    added: list[str] = []
    deleted: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        seg_new, seg_old = new[j1:j2].strip(), old[i1:i2].strip()
        if tag in ("insert", "replace") and len(seg_new) >= 6:
            added.append(seg_new)
        if tag in ("delete", "replace") and len(seg_old) >= 6:
            deleted.append(seg_old)
    if added:
        top = max(added, key=len)
        facts.append(f"新增表述：{_q(top, 60)}")
    if deleted:
        top = max(deleted, key=len)
        facts.append(f"删除表述：{_q(top, 60)}")

    na = {x.replace(",", "") for x in _NUM_RE.findall(old)}
    nb = {x.replace(",", "") for x in _NUM_RE.findall(new)}
    gone, come = sorted(na - nb), sorted(nb - na)
    if gone and come:
        facts.append(f"数值变化：{'/'.join(gone[:3])} → {'/'.join(come[:3])}")
    elif come:
        facts.append(f"新增数值：{'/'.join(come[:3])}")
    elif gone:
        facts.append(f"移除数值：{'/'.join(gone[:3])}")

    if not facts:
        return "无明显实质变化（仅措辞或排版差异）"
    return "；".join(facts)


# ────────────────────────── 主流程 ──────────────────────────
def compare_papers(
    a: PaperDoc | dict[str, Any],
    b: PaperDoc | dict[str, Any],
    *,
    client: Any = None,
    embed_model: str | None = None,
    chat_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    dup_threshold: float = DEFAULT_DUP_TH,
    rel_threshold: float = DEFAULT_REL_TH,
    use_llm: bool = True,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """论文 A（库中已有） vs 论文 B（新上传）。

    返回固定四键 `verdict / similarity / version_advice / diff_summary`，
    另附 `evidence / field_scores / diff_detail / versions / engine / warnings` 供追溯。
    """
    A = a if isinstance(a, PaperDoc) else PaperDoc.from_dict(a)
    B = b if isinstance(b, PaperDoc) else PaperDoc.from_dict(b)
    warnings: list[str] = []

    # ① 向量化：title/abstract + 正文段落，一次批量，保证同一向量空间可比
    pool: list[str] = [A.title, B.title, A.abstract, B.abstract, *A.body_texts, *B.body_texts]
    embed_errors: list[str] = []
    vecs, engine = embed_texts(pool, client=client, model=embed_model,
                               base_url=base_url, api_key=api_key, timeout=timeout,
                               errors=embed_errors)
    warnings.extend(embed_errors)
    if engine == "local-hash" and not embed_errors:
        warnings.append("未配置 EMBEDDING_API_KEY/LLM_API_KEY，相似度由确定性哈希向量计算（仅供联调，生产请换真实 embedding）")

    n = 4
    va_title, vb_title = vecs[0], vecs[1]
    va_abs, vb_abs = vecs[2], vecs[3]
    va_body = vecs[n:n + len(A.body_texts)]
    vb_body = vecs[n + len(A.body_texts):n + len(A.body_texts) + len(B.body_texts)]

    # ② 分字段相似度 + 证据片段
    body_matrix = _sim_matrix(va_body, vb_body) if (va_body and vb_body) else []
    s_title = _cos(va_title, vb_title)
    s_abs = _cos(va_abs, vb_abs)
    s_body = best_match_mean(va_body, vb_body, body_matrix) if (va_body and vb_body) else 0.0
    field_scores = {"title": round(s_title, 4), "abstract": round(s_abs, 4), "body": round(s_body, 4)}

    weights = dict(WEIGHTS)
    if not A.abstract or not B.abstract:        # 缺摘要时把权重让给正文
        weights = {"title": 0.3, "abstract": 0.0, "body": 0.7}
    elif not va_body or not vb_body:            # 缺正文时只用元信息
        weights = {"title": 0.4, "abstract": 0.6, "body": 0.0}
    total_w = sum(weights.values()) or 1.0
    similarity = round(sum(field_scores[k] * weights[k] for k in weights) / total_w, 4)

    # ③ 元数据规则优先：同一主干（arXiv ID 去掉 vN）直接判重复。
    # 但**不覆盖** semantic_similarity，避免把元数据规则伪装成语义分数。
    semantic_similarity = similarity
    same_base = bool(base_id(A.doc_id) and base_id(A.doc_id) == base_id(B.doc_id))
    metadata_override = bool(same_base and similarity < dup_threshold)
    if metadata_override:
        warnings.append(f"同一主干 {base_id(A.doc_id)}（元数据规则优先于语义相似度 {similarity}）")
        similarity = max(similarity, dup_threshold)
    verdict = DUPLICATE if similarity >= dup_threshold else (
        RELATED if similarity >= rel_threshold else DISTINCT)

    # ④ 证据：每个字段给出命中的原文片段
    evidence: list[dict[str, Any]] = [
        {"field": "title", "score": field_scores["title"],
         "a_quote": _q(A.title, 60), "b_quote": _q(B.title, 60), "a_ref": A.doc_id, "b_ref": B.doc_id},
    ]
    if A.abstract and B.abstract:
        evidence.append({"field": "abstract", "score": field_scores["abstract"],
                         "a_quote": _q(A.abstract, 90), "b_quote": _q(B.abstract, 90),
                         "a_ref": f"{A.doc_id}#abstract", "b_ref": f"{B.doc_id}#abstract"})
    pairs = align_sections(va_body, vb_body, matrix=body_matrix) if (va_body and vb_body) else []
    for i, j, s in sorted(pairs, key=lambda p: -p[2])[:3]:
        evidence.append({"field": "body", "score": round(s, 4),
                         "a_quote": _q(A.sections[i].text, 90), "b_quote": _q(B.sections[j].text, 90),
                         "a_ref": A.sections[i].sid, "b_ref": B.sections[j].sid})

    # ⑤ 版本建议
    a_ver = A.version or A.doc_id
    b_ver = B.version or B.doc_id
    advice = version_advice(a_ver, b_ver, verdict)

    # ⑥ 逐段差异（仅同一篇论文时才有意义）
    diff_detail: list[dict[str, Any]] = []
    if verdict == DUPLICATE:
        # 摘要本身也是一个比对单元（很多预印本只有摘要可比对）
        if A.abstract and B.abstract and s_abs < UNCHANGED_TH:
            _sum = (llm_diff(A.abstract, B.abstract, client=client, model=chat_model,
                             base_url=base_url, api_key=api_key, timeout=timeout)
                    if use_llm else None)
            diff_detail.append({"change": "modified", "similarity": round(s_abs, 4),
                                "a_ref": f"{A.doc_id}#abstract", "b_ref": f"{B.doc_id}#abstract",
                                "heading": "Abstract",
                                "summary": _sum or rule_diff(A.abstract, B.abstract),
                                "engine": "llm" if _sum else "difflib"})

        matched_a = {i for i, _, _ in pairs}
        matched_b = {j for _, j, _ in pairs}
        for i, j, s in pairs:
            if s >= UNCHANGED_TH:
                diff_detail.append({"change": "unchanged", "similarity": round(s, 4),
                                    "a_ref": A.sections[i].sid, "b_ref": B.sections[j].sid,
                                    "heading": A.sections[i].heading or B.sections[j].heading,
                                    "summary": "内容一致"})
                continue
            summary = (llm_diff(A.sections[i].text, B.sections[j].text, client=client,
                                model=chat_model, base_url=base_url, api_key=api_key,
                                timeout=timeout) if use_llm else None)
            diff_detail.append({"change": "modified", "similarity": round(s, 4),
                                "a_ref": A.sections[i].sid, "b_ref": B.sections[j].sid,
                                "heading": A.sections[i].heading or B.sections[j].heading,
                                "summary": summary or rule_diff(A.sections[i].text, B.sections[j].text),
                                "engine": "llm" if summary else "difflib"})
        for j, sec in enumerate(B.sections):
            if j not in matched_b:
                diff_detail.append({"change": "added", "similarity": 0.0, "b_ref": sec.sid,
                                    "heading": sec.heading, "engine": "align",
                                    "summary": f"新版本新增该段（约 {len(sec.text)} 字）"})
        for i, sec in enumerate(A.sections):
            if i not in matched_a:
                diff_detail.append({"change": "removed", "similarity": 0.0, "a_ref": sec.sid,
                                    "heading": sec.heading, "engine": "align",
                                    "summary": f"新版本删除该段（约 {len(sec.text)} 字）"})

    diff_summary = build_diff_summary(diff_detail, verdict)
    return {
        "verdict": verdict,
        "similarity": similarity,
        "semantic_similarity": semantic_similarity,   # 原始语义分；similarity 可能被元数据规则抬到阈值
        "metadata_override": metadata_override,
        "version_advice": advice,
        "diff_summary": diff_summary,
        "evidence": evidence,
        "field_scores": field_scores,
        "weights": weights,          # 缺摘要/缺正文时权重会重分配，必须回传以免"分数对不上权重"
        "diff_detail": diff_detail,
        "versions": {"a": a_ver, "b": b_ver, "cmp": compare_versions(a_ver, b_ver),
                     "same_base": same_base},
        "engine": {"embedding": engine, "diff": _diff_engine(diff_detail)},
        "warnings": warnings,
    }


def _diff_engine(detail: list[dict[str, Any]]) -> str:
    engines = {d.get("engine") for d in detail if d.get("engine")}
    return ",".join(sorted(e for e in engines if e)) or "none"


def build_diff_summary(detail: list[dict[str, Any]], verdict: str) -> str:
    if verdict != DUPLICATE:
        return "两篇为不同论文（或仅主题相关），不做版本级逐段比对。"
    if not detail:
        return "两篇内容一致，未检出段落级差异。"

    cnt = {"unchanged": 0, "modified": 0, "added": 0, "removed": 0}
    for d in detail:
        cnt[d["change"]] = cnt.get(d["change"], 0) + 1
    head = (f"共 {len(detail)} 段比对：未变 {cnt['unchanged']} / 修订 {cnt['modified']} / "
            f"新增 {cnt['added']} / 删除 {cnt['removed']}。")

    lines: list[str] = []
    for d in detail:
        if d["change"] == "unchanged":
            continue
        tag = {"modified": "修订", "added": "新增", "removed": "删除"}[d["change"]]
        where = d.get("heading") or d.get("b_ref") or d.get("a_ref") or "-"
        lines.append(f"- [{tag}] {where}：{d['summary']}")
    unchanged = [d.get("heading") or d.get("a_ref") for d in detail if d["change"] == "unchanged"]
    if unchanged:
        lines.append(f"- [未变] {'、'.join(str(u) for u in unchanged[:6])} 内容与库中版本一致")
    return "\n".join([head, *lines])


# ────────────────────────── 工具信封桥接 ──────────────────────────
def to_tool_envelope(result: dict[str, Any]) -> dict[str, Any]:
    """转成本系统统一的 ToolEnvelope（供 semantic_dedup 工具直接返回）。"""
    from app.agents.tools import corpus
    from app.schemas import EvidenceCard

    # 两条硬约束：
    # ① block_id 必须是**文档内局部 ID**，否则 cite_key 会拼成 doc#doc#block；
    # ② 只有**字面可校验**的片段才能提升为可引用证据卡，否则会污染"引文忠实性"这道闸。
    cards: list[EvidenceCard] = []
    uncitable: list[str] = []
    for e in result.get("evidence", []):
        doc_id, _, block = str(e.get("a_ref") or "").partition("#")
        quote = _q(e.get("a_quote", ""), 120)
        if not doc_id or not corpus.find_quote(quote):
            uncitable.append(str(e.get("a_ref") or e.get("field") or "?"))
            continue
        cards.append(EvidenceCard(
            doc_id=doc_id,
            block_id=block or "abstract",
            quote=quote,
            score=float(e.get("score", 0.0)),
            meta={"field": e.get("field"), "b_ref": e.get("b_ref"),
                  "b_quote": _q(e.get("b_quote", ""), 120)},
        ))
    data = {k: result[k] for k in ("verdict", "similarity", "version_advice", "diff_summary")}
    data["semantic_similarity"] = result.get("semantic_similarity", result.get("similarity"))
    data["metadata_override"] = bool(result.get("metadata_override", False))
    data["warnings"] = list(result.get("warnings") or [])   # 降级/失败原因必须让上层看到
    return {
        "ok": True,
        "tool": "semantic_dedup",
        "data": data,
        "cited": [c.model_dump(mode="json") for c in cards],
        "metrics": {"engine": result.get("engine", {}), "field_scores": result.get("field_scores", {}),
                    "versions": result.get("versions", {}), "uncitable_evidence": uncitable,
                    "semantic": result.get("engine", {}).get("embedding") == "openai-compatible"},
    }


# ────────────────────────── CLI / 自测 ──────────────────────────
def _load(spec: str) -> PaperDoc:
    """入参可以是 JSON 文件路径、JSON 字符串，或内置语料 ID。"""
    p = spec.strip()
    if p.startswith("{"):
        return PaperDoc.from_dict(json.loads(p))
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return PaperDoc.from_dict(json.load(f))
    if p.startswith("arxiv:"):
        return PaperDoc.from_version_chain(p)
    return PaperDoc.from_corpus(p)


def _selftest() -> int:
    checks: list[tuple[str, bool, str]] = []

    # ① 同论文不同版本（v1 库内 / v7 新上传）
    A = PaperDoc.from_version_chain("arxiv:1706.03762v1")
    B = PaperDoc.from_version_chain("arxiv:1706.03762v7")
    r1 = compare_papers(A, B, use_llm=False, api_key="", base_url="")
    checks.append(("同主干判重复", r1["verdict"] == DUPLICATE, f"{r1['verdict']} sim={r1['similarity']}"))
    checks.append(("新版本更新→建议覆盖", "新上传版本更新" in r1["version_advice"], r1["version_advice"][:34]))
    checks.append(("逐段差异含事实陈述", "新增" in r1["diff_summary"] or "修订" in r1["diff_summary"],
                   r1["diff_summary"].split("\n")[1][:40] if "\n" in r1["diff_summary"] else r1["diff_summary"][:40]))

    # ② 反向：库内是 v7，新上传 v1
    r2 = compare_papers(B, A, use_llm=False, api_key="", base_url="")
    checks.append(("更旧版本→提示保留库中版本", "库中已有更新版本建议保留" in r2["version_advice"],
                   r2["version_advice"][:34]))

    # ③ 同版本号
    r3 = compare_papers(A, PaperDoc.from_version_chain("arxiv:1706.03762v1"), use_llm=False, api_key="", base_url="")
    checks.append(("版本相同→无需覆盖", "版本相同" in r3["version_advice"], r3["version_advice"][:24]))

    # ④ 不同论文（Transformer vs BERT）
    r4 = compare_papers(PaperDoc.from_corpus("doc_transformer"),
                        PaperDoc.from_corpus("doc_bert"), use_llm=False, api_key="", base_url="")
    checks.append(("不同论文不判重复", r4["verdict"] != DUPLICATE, f"{r4['verdict']} sim={r4['similarity']}"))
    checks.append(("不同论文不给覆盖建议", "不涉及覆盖决策" in r4["version_advice"], r4["version_advice"][:24]))

    # ⑤ 完全无关内容
    far = PaperDoc.from_dict({"doc_id": "doc_recipe", "title": "家常番茄炒蛋的火候控制",
                              "abstract": "讨论中式家常菜番茄炒蛋的火候与调味顺序。",
                              "sections": [{"heading": "步骤", "text": "先炒蛋后炒番茄，控制油温六成热。"}]})
    r5 = compare_papers(PaperDoc.from_corpus("doc_transformer"), far, use_llm=False, api_key="", base_url="")
    checks.append(("无关内容判 distinct", r5["verdict"] == DISTINCT, f"{r5['verdict']} sim={r5['similarity']}"))
    checks.append(("distinct 不做段落比对", "不做版本级逐段比对" in r5["diff_summary"], r5["diff_summary"][:24]))

    # ⑥ 证据必须两侧都有原文字段
    ok_evidence = all(e.get("a_quote") and e.get("b_quote") for e in r1["evidence"])
    checks.append(("相似度给出依据片段", ok_evidence and len(r1["evidence"]) >= 2,
                   f"{len(r1['evidence'])} 条，最高分 {max(e['score'] for e in r1['evidence']):.2f}"))

    # ⑦ 数值变化被如实记录
    a6 = PaperDoc.from_dict({"doc_id": "p_v2", "version": "v2", "title": "Efficient Attention",
                             "sections": [{"heading": "Results", "text": "本方法在 EN-DE 上取得 28.4 BLEU。"}]})
    b6 = PaperDoc.from_dict({"doc_id": "p_v3", "version": "v3", "title": "Efficient Attention",
                             "sections": [{"heading": "Results", "text": "本方法在 EN-DE 上取得 29.1 BLEU，并补充了消融实验。"}]})
    r6 = compare_papers(a6, b6, use_llm=False, api_key="", base_url="")
    joined = " ".join(d["summary"] for d in r6["diff_detail"])
    checks.append(("数值变化 28.4→29.1 被记录", "28.4" in joined and "29.1" in joined,
                   r6["diff_detail"][0]["summary"][:44] if r6["diff_detail"] else "-"))
    checks.append(("v3 更新于 v2", "新上传版本更新" in r6["version_advice"], r6["version_advice"][:24]))

    # ⑧ diff 不得出现评价词
    banned = ("更好", "更优", "更差", "改进", "优化", "提升", "更清晰", "完善", "严谨")
    bad = [w for w in banned if w in r6["diff_summary"]]
    checks.append(("diff 不含价值判断词", not bad, f"命中={bad}"))

    # ⑨ 输出四键 + 数值合法
    keys_ok = all(k in r6 for k in ("verdict", "similarity", "version_advice", "diff_summary"))
    checks.append(("输出四键完整且分数合法", keys_ok and 0.0 <= r6["similarity"] <= 1.0,
                   f"sim={r6['similarity']}"))

    # ⑩ 版本号解析
    checks.append(("版本号解析 vN/日期",
                   parse_version("arxiv:1706.03762v7") == ("numeric", (7,))
                   and parse_version("2024-05-01")[0] == "date"
                   and compare_versions("v2", "v10") == 1,
                   f"v7={parse_version('arxiv:1706.03762v7')} v2→v10={compare_versions('v2', 'v10')}"))

    width = max(len(c[0]) for c in checks)
    passed = 0
    for name, ok, detail in checks:
        passed += ok
        print(f"[{'OK' if ok else 'XX'}] {name:<{width}}  {detail}")
    print("─" * 78)
    print(f"通过 {passed}/{len(checks)}（离线：哈希向量 + difflib，未联网）")
    return 0 if passed == len(checks) else 1


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="文献语义去重与版本比对")
    ap.add_argument("--a", default="", help="论文A（库中已有）：JSON 文件 / JSON 串 / 语料 doc_id / arxiv:xxx vN")
    ap.add_argument("--b", default="", help="论文B（新上传）：同上")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    ap.add_argument("--selftest", action="store_true", help="离线自测")
    ap.add_argument("--demo", action="store_true", help="内置演示：1706.03762 v1 vs v7")
    ap.add_argument("--embed-model", default="", help="embedding 模型名（默认读 EMBEDDING_MODEL）")
    ap.add_argument("--model", default="", help="用于 diff 的对话模型（默认读 LLM_MODEL）")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--dup-th", type=float, default=DEFAULT_DUP_TH)
    ap.add_argument("--rel-th", type=float, default=DEFAULT_REL_TH)
    ap.add_argument("--no-llm", action="store_true", help="只用规则 diff，不调 LLM")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    if args.demo or not args.b:
        a, b = PaperDoc.from_version_chain("arxiv:1706.03762v1"), PaperDoc.from_version_chain("arxiv:1706.03762v7")
    else:
        a, b = _load(args.a), _load(args.b)

    res = compare_papers(a, b, embed_model=args.embed_model or None, chat_model=args.model or None,
                         base_url=args.base_url or None, api_key=args.api_key or None,
                         dup_threshold=args.dup_th, rel_threshold=args.rel_th,
                         use_llm=not args.no_llm)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    print(f"A = {a.doc_id}   B = {b.doc_id}")
    print(f"判定        : {res['verdict']}  similarity={res['similarity']}")
    print(f"字段分      : {res['field_scores']}  实际权重={res['weights']}")
    print(f"版本建议    : {res['version_advice']}")
    print("依据片段:")
    for e in res["evidence"]:
        print(f"  [{e['field']:<8}] {e['score']:.3f}  A: {e['a_quote'][:44]}")
        print(f"  {'':<8}          B: {e['b_quote'][:44]}")
    print("差异总结:")
    print("  " + res["diff_summary"].replace("\n", "\n  "))
    for w in res["warnings"]:
        print(f"[!] {w}")


if __name__ == "__main__":
    main()
