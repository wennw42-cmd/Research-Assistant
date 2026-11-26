"""基于检索片段的有据问答（Grounded QA）：只用片段，输出可溯源的 JSON。

    python -m app.grounded_qa --selftest
    python -m app.grounded_qa --demo -q "LLM 微调用了多少张 A100？"
    python -m app.grounded_qa -q "..." --context ctx.json

输出契约（仅 JSON）：
    {"answer": str, "citations": [{"file","page","chunk_id","quote"}],
     "confidence": float, "missing": str}

四条硬规则的落地方式（不是只写在 prompt 里）：
1. **出处**：answer 里的 [来源:文件,页码,片段ID] 必须能解析到本次输入的片段；
   citations 逐条校验 chunk_id 存在、file/page 一致；
2. **片段不足**：模型拒答 → 原样返回"现有文献无法确定"，missing 为空时自动补齐；
3. **数值逐字**：answer 中每个数值 token 必须在本片段的原文中逐字出现（不做单位换算/四舍五入），
   只出现在非引用片段里也算不通过——防止"从别处搬数"；
4. **原文 vs 归纳**：每个结论句必须带可用出处或 [推断]，否则判为不合规。

模型输出不合规时先带错误反馈重试一次；仍不合规则降级为**原文摘录**（逐字引用 + 出处），
绝不返回无出处的自由生成。未配置 LLM Key 时直接走摘录模式（MockLLM 做不了有据问答）。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):                       # 允许 python app/grounded_qa.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers import BaseLLM, get_llm
from app.utils import extract_json, get_logger, short

log = get_logger("ra.grounded_qa")

REFUSAL = "现有文献无法确定"
MAX_FRAG_CHARS = 1500          # 单片段进 prompt 的上限（原文仍保留完整用于校验）
MAX_QUOTE_CHARS = 400

# ── 引用与数值的解析规则 ──
_CITE_RE = re.compile(r"\[来源[:：]\s*([^\[\]]+?)\]")
_NUM_RE = re.compile(r"(?<![a-z0-9\-])\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.I)
_SECTION_REF = re.compile(r"第\s*\d+(?:\.\d+)*\s*[章节表图]")      # "第 5.2 节"不是数值结论
_LIST_MARK = re.compile(r"(?m)^\s*\d+[.、)]\s*")
_CJK = "一-鿿"

SYSTEM_PROMPT = """你是严谨的学术问答助手。你只能使用用户提供的检索片段，禁止使用外部知识、常识或记忆补全。

必须遵守：
1. 每个结论后紧跟 [来源:文件名,页码,片段ID]，一条结论可标多个来源。
2. 片段不足以回答时，answer 必须严格为"现有文献无法确定"，并在 missing 中逐条列出缺少哪些信息。
3. 数值、公式、实验设置必须与原文逐字一致：不得换算单位、不得四舍五入、不得改写公式。
4. 区分"原文陈述"与"你的归纳"：归纳句必须以 [推断] 结尾；没有 [推断] 的句子会被当作原文陈述校验。
5. citations[].quote 必须逐字摘自对应片段（不超过 200 字），不得改写、不得拼接。
只输出 JSON，不要代码块、不要解释、不要 Markdown。"""

USER_TMPL = """问题：{question}

检索片段（唯一允许的信息来源）：
{context}

输出 JSON：
{{"answer":"...","citations":[{{"file":"...","page":0,"chunk_id":"...","quote":"..."}}],"confidence":0.0,"missing":"..."}}"""


# ══════════════════════════ 数据契约 ══════════════════════════
@dataclass(frozen=True)
class Fragment:
    """一条可溯源的检索片段。chunk_id 必填——没有它就无法校验出处。"""

    chunk_id: str
    text: str
    file: str = ""
    page: int = 0
    score: float = 0.0

    @classmethod
    def from_any(cls, obj: Any) -> "Fragment":
        if isinstance(obj, cls):
            return obj
        if not isinstance(obj, dict):
            raise TypeError(f"片段必须是 dict 或 Fragment，收到 {type(obj).__name__}")
        # 兼容 app.retrieval 的 hit 结构（id / title / doc_id / page_start）
        cid = str(obj.get("chunk_id") or obj.get("id") or "").strip()
        text = str(obj.get("text") or obj.get("content") or "")
        if not cid:
            raise ValueError("片段缺少 chunk_id（或 id）：没有片段 ID 就无法标注与校验出处")
        if not text.strip():
            raise ValueError(f"片段 {cid} 的 text 为空")
        return cls(
            chunk_id=cid,
            text=text,
            file=str(obj.get("file") or obj.get("title") or obj.get("doc_id") or ""),
            page=int(obj.get("page", obj.get("page_start", 0)) or 0),
            score=float(obj.get("score", 0.0) or 0.0),
        )


def fragments_from_hits(hits: Sequence[Any]) -> list[Fragment]:
    """直接接 app.retrieval.pipeline 的 hits（含 id / doc_id / page_start / text）。"""
    return [Fragment.from_any(h) for h in hits]


# ══════════════════════════ 文本工具 ══════════════════════════
def _norm(s: str) -> str:
    """比对用规范化：只折叠空白，不改字符。引用必须逐字（空白差异属排版噪声）。"""
    return re.sub(r"\s+", "", s or "").lower()


def _clamp(x: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return lo
    if math.isnan(v):
        return lo
    return max(lo, min(hi, v))


def _sentences(text: str, *, english_periods: bool = False) -> list[str]:
    pat = r"(?<=[。！？；!?;])|(?<=\.)\s+|\n+" if english_periods else r"(?<=[。！？；!?;])|\n+"
    return [s.strip() for s in re.split(pat, text or "") if s.strip()]


def _terms(text: str) -> set[str]:
    t = (text or "").lower()
    out = set(re.findall(r"[a-z0-9][a-z0-9\-\._]*", t))
    for seg in re.findall(f"[{_CJK}]+", t):
        out.update(seg[i:i + 2] for i in range(len(seg) - 1))
    return out


def _numbers(text: str) -> list[str]:
    clean = _SECTION_REF.sub("", _LIST_MARK.sub("", text or ""))
    return [m.group(0).lower() for m in _NUM_RE.finditer(clean)]


def _parse_marker(body: str) -> tuple[str, int, str] | None:
    """拆 [来源:file,page,chunk_id]。文件路径里可能有逗号，所以从右往左拆。"""
    parts = body.replace("，", ",").rsplit(",", 2)
    if len(parts) != 3:
        return None
    file, page_s, cid = (p.strip() for p in parts)
    try:
        page = int(page_s)
    except ValueError:
        return None
    return (file, page, cid) if cid else None


# ══════════════════════════ 主入口 ══════════════════════════
def answer_question(question: str, context: Sequence[Any], *,
                    llm: BaseLLM | None = None, retries: int = 1) -> dict[str, Any]:
    """仅依据 context 回答问题，返回固定 JSON 结构。

    context: [{"chunk_id","text","file","page"}...]，也接受 app.retrieval 的 hits。
    llm: 可注入；默认用配置的真实模型，未配 Key 时降级为原文摘录模式。
    """
    frags = [Fragment.from_any(c) for c in (context or [])]
    if not frags:
        return _refusal("未提供任何检索片段，无法回答")
    q = (question or "").strip()
    if not q:
        return _refusal("问题为空")

    llm = llm if llm is not None else _default_llm()
    if llm is None:
        return _extractive(q, frags, reason="未配置 LLM，降级为原文摘录")

    payload = _ask(llm, q, frags)
    if payload is None:
        return _extractive(q, frags, reason="模型未返回可解析的 JSON")

    report = _validate(payload, frags)
    used_retry = False
    if report["issues"] and retries > 0:
        used_retry = True
        payload2 = _ask(llm, q, frags, feedback=report["issues"])
        if payload2 is not None:
            payload = payload2
            report = _validate(payload, frags)

    if report["issues"]:
        log.warning("有据问答校验失败 %d 项：%s", len(report["issues"]),
                    short("; ".join(report["issues"]), 200))
        return _extractive(q, frags, reason="模型输出未通过出处/数值校验")

    return _finalize(payload, report, used_retry=used_retry)


def _default_llm() -> BaseLLM | None:
    """MockLLM 是脚本化的，做不了有据问答：没配 Key 就直接走摘录模式。"""
    from app import config

    s = config.get_settings()
    return None if s.use_mock_llm else get_llm(s)


def _ask(llm: BaseLLM, question: str, frags: list[Fragment],
         feedback: list[str] | None = None) -> dict[str, Any] | None:
    ctx = "\n\n".join(
        f"[F{i}] file={f.file} page={f.page} chunk_id={f.chunk_id}\n{f.text[:MAX_FRAG_CHARS]}"
        for i, f in enumerate(frags, 1))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TMPL.format(question=question, context=ctx)},
    ]
    if feedback:
        messages.append({
            "role": "user",
            "content": "上一版输出未通过校验，必须修正后重新输出 JSON：\n- "
                       + "\n- ".join(feedback[:8]),
        })
    try:
        parsed = llm.chat_json(messages)
    except Exception as e:                                  # noqa: BLE001
        log.warning("LLM 调用失败：%s", e)
        return None
    if isinstance(parsed, str):
        parsed = extract_json(parsed)
    return parsed if isinstance(parsed, dict) else None


# ══════════════════════════ 校验 ══════════════════════════
def _validate(payload: dict[str, Any], frags: list[Fragment]) -> dict[str, Any]:
    """返回 issues（必须为空才可采纳）与清洗后的 citations。"""
    issues: list[str] = []
    by_id = {f.chunk_id: f for f in frags}

    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return {"issues": ["answer 缺失或为空"], "citations": [], "refusal": False}

    refusal = answer.strip().startswith(REFUSAL)
    raw_cites = payload.get("citations")
    if raw_cites is None:
        raw_cites = []
    if not isinstance(raw_cites, list):
        issues.append("citations 必须是数组")
        raw_cites = []

    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for c in raw_cites:
        if not isinstance(c, dict):
            issues.append("citations 中存在非对象条目")
            continue
        cid = str(c.get("chunk_id", "")).strip()
        quote = str(c.get("quote", "")).strip()
        frag = by_id.get(cid)
        if frag is None:
            issues.append(f"引用了不存在的片段：{cid or '(空)'}")
            continue
        if not quote:
            issues.append(f"片段 {cid} 的 quote 为空")
            continue
        if _norm(quote) not in _norm(frag.text):
            issues.append(f"quote 未逐字出现在片段 {cid} 的原文中")
            continue
        if str(c.get("file", frag.file)) != frag.file or \
                int(c.get("page", frag.page) or 0) != frag.page:
            issues.append(f"片段 {cid} 的 file/page 与原文不一致")
            continue
        key = (cid, quote)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"file": frag.file, "page": frag.page,
                        "chunk_id": cid, "quote": quote[:MAX_QUOTE_CHARS]})

    if refusal:
        return {"issues": issues, "citations": cleaned, "refusal": True}

    # ① answer 里的 [来源:...] 必须都能解析到本次片段，且列入 citations
    markers = _CITE_RE.findall(answer)
    cited_ids = {c["chunk_id"] for c in cleaned}
    if not markers and not cleaned:
        issues.append("answer 中没有任何出处，且 citations 为空")
    for body in markers:
        parsed = _parse_marker(body)
        if parsed is None:
            issues.append(f"出处格式无法解析：{short(body, 60)}")
            continue
        file, page, cid = parsed
        frag = by_id.get(cid)
        if frag is None:
            issues.append(f"answer 标注了不存在的片段：{cid}")
            continue
        if (file, page) != (frag.file, frag.page):
            issues.append(f"answer 标注的 file/page 与片段 {cid} 不一致")
        if cid not in cited_ids:
            issues.append(f"出处 {cid} 未列入 citations 数组")

    # ② 每个结论句必须有可用出处或 [推断]
    for sent in _sentences(answer):
        if not _is_claim(sent):
            continue
        if "[推断]" in sent:
            continue
        if any(_parse_marker(b) and _parse_marker(b)[2] in by_id      # type: ignore[index]
               for b in _CITE_RE.findall(sent)):
            continue
        issues.append(f"结论缺少出处：{short(sent, 40)}")

    # ③ 数值/公式逐字对齐：只在与引用相关的片段里找，防止从别处搬数
    pool_frags = [f for f in frags if not cited_ids or f.chunk_id in cited_ids]
    pool = "".join(_norm(f.text) for f in pool_frags)
    body = _SECTION_REF.sub("", _CITE_RE.sub("", answer))
    for num in _numbers(body):
        if _norm(num) not in pool:
            issues.append(f"数值/公式未在引用片段中逐字出现：{num}")

    # issues 去重并保持顺序
    uniq = list(dict.fromkeys(issues))
    return {"issues": uniq, "citations": cleaned, "refusal": False}


def _is_claim(sentence: str) -> bool:
    s = sentence.strip()
    if not s or s.startswith("#") or s.startswith("（以下为原文摘录"):
        return False
    if s.startswith(REFUSAL):
        return False
    if len(s) <= 20 and s.endswith(("：", ":")):       # 小标题，不是结论
        return False
    if re.match(r"^(缺失|缺少|注[:：]|说明[:：]|注：)", s):
        return False
    return True


def _finalize(payload: dict[str, Any], report: dict[str, Any], *,
              used_retry: bool) -> dict[str, Any]:
    answer = str(payload["answer"]).strip()
    conf = _clamp(payload.get("confidence"))
    missing = str(payload.get("missing") or "").strip()

    if report.get("refusal"):
        if not missing:
            missing = "检索片段未包含回答该问题所需的信息"
        conf = min(conf, 0.2)
        return {"answer": REFUSAL, "citations": report["citations"],
                "confidence": round(conf, 3), "missing": missing}

    if used_retry:
        conf = min(conf, 0.8)
    conf = min(conf, 0.95)
    if not report["citations"]:
        conf = min(conf, 0.4)
    return {"answer": answer, "citations": report["citations"],
            "confidence": round(conf, 3), "missing": missing}


def _refusal(missing: str) -> dict[str, Any]:
    return {"answer": REFUSAL, "citations": [], "confidence": 0.0, "missing": missing}


# ══════════════════════════ 降级：原文摘录 ══════════════════════════
def _extractive(question: str, frags: list[Fragment], *, reason: str = "") -> dict[str, Any]:
    """逐字摘录最相关的原句并标注出处；不生成、不改写、不做归纳。

    只取**一个**片段（问句重叠最高的那个）：跨片段拼摘要等于做了未声明的归纳，
    而这条路径恰恰是"无法校验生成"时的兜底，宁窄勿错。
    """
    q_terms = _terms(question)
    need = max(1, math.ceil(0.2 * len(q_terms)))
    best: dict[str, int] = {}
    sents: dict[str, list[tuple[int, str]]] = {}
    for f in frags:
        for sent in _sentences(f.text, english_periods=True):
            s = sent.strip()
            if len(s) < 10:
                continue
            overlap = len(q_terms & _terms(s))
            if overlap >= need:
                sents.setdefault(f.chunk_id, []).append((overlap, s))
                best[f.chunk_id] = max(best.get(f.chunk_id, 0), overlap)
    if not best:
        return _refusal("现有片段与问题没有可定位的重叠，无法抽取可靠答案"
                        + (f"（{reason}）" if reason else ""))

    # 同分取检索顺序更靠前的片段（frags 已按相关度排好）
    rank = {f.chunk_id: i for i, f in enumerate(frags)}
    winner = max(best, key=lambda cid: (best[cid], -rank.get(cid, 0)))
    frag = next(f for f in frags if f.chunk_id == winner)
    picked = sorted(sents[winner], key=lambda x: -x[0])[:2]

    lines, cites = [], []
    for _, s in picked:
        lines.append(f"{s}[来源:{frag.file or '未知文献'},{frag.page},{frag.chunk_id}]")
        cites.append({"file": frag.file, "page": frag.page, "chunk_id": frag.chunk_id,
                      "quote": s[:MAX_QUOTE_CHARS]})
    return {"answer": "（以下为原文摘录）\n" + "\n".join(lines),
            "citations": cites, "confidence": 0.3, "missing": ""}


# ══════════════════════════ 自测 / CLI ══════════════════════════
class _FakeLLM(BaseLLM):
    """脚本化模型：按顺序返回预设 payload，用于离线断言校验逻辑。"""

    model_name = "fake"

    def __init__(self, payloads: list[Any]) -> None:
        self.payloads, self.calls = list(payloads), 0

    def chat(self, messages, tools=None, tool_choice=None, temperature=None, max_tokens=None):
        from app.providers import ChatResponse

        p = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return ChatResponse(content=json.dumps(p, ensure_ascii=False), model=self.model_name)


_FRAGS = [
    {"chunk_id": "p5b1", "file": "attention.pdf", "page": 5,
     "text": "自注意力的计算复杂度为 O(n^2 · d)，其中 n 为序列长度，d 为表示维度。"
             "相比循环层，其每层的顺序操作数仅为 O(1)。"},
    {"chunk_id": "p3b2", "file": "bert.pdf", "page": 3,
     "text": "BERT 仅使用 Transformer 的编码器堆叠，不使用解码器。"},
]
_Q = "Transformer 自注意力的计算复杂度是多少？"


def _selftest() -> int:
    # ① 合规输出：照单采纳，出处与数值均校验通过
    good = {
        "answer": "自注意力的计算复杂度为 O(n^2 · d)[来源:attention.pdf,5,p5b1]，"
                  "每层顺序操作数为 O(1)，比循环层更易并行[推断]。",
        "citations": [{"file": "attention.pdf", "page": 5, "chunk_id": "p5b1",
                       "quote": "自注意力的计算复杂度为 O(n^2 · d)"}],
        "confidence": 0.9, "missing": "",
    }
    r = answer_question(_Q, _FRAGS, llm=_FakeLLM([good]))
    assert r["citations"][0]["chunk_id"] == "p5b1", r
    assert 0.0 <= r["confidence"] <= 1.0 and r["missing"] == ""
    assert "O(n^2 · d)" in r["answer"]

    # ② 编造出处 + 编造数值 + 无出处结论：重试仍失败 → 摘录模式，引用必须全部可校验
    bad = {"answer": "该模型使用了 1e-4 的学习率。",
           "citations": [{"file": "fake.pdf", "page": 1, "chunk_id": "nope", "quote": "x"}],
           "confidence": 0.9, "missing": ""}
    r = answer_question(_Q, _FRAGS, llm=_FakeLLM([bad, bad]))
    assert r["confidence"] <= 0.3, r
    assert r["citations"], "摘录模式必须给出可校验的出处"
    ids = {f["chunk_id"] for f in _FRAGS}
    for c in r["citations"]:
        assert c["chunk_id"] in ids
        text = next(f["text"] for f in _FRAGS if f["chunk_id"] == c["chunk_id"])
        assert _norm(c["quote"]) in _norm(text), "摘录必须是原文子串"
    assert "1e-4" not in r["answer"], "不得把编造数值带进答案"

    # ③ 模型主动拒答：原样返回，missing 自动补齐
    refuse = {"answer": REFUSAL, "citations": [], "confidence": 0.0, "missing": ""}
    r = answer_question("这篇论文用了多少块 H100？", _FRAGS, llm=_FakeLLM([refuse]))
    assert r["answer"] == REFUSAL and r["missing"], r
    assert r["confidence"] <= 0.2

    # ④ 空片段：直接拒答，不调用模型
    fake = _FakeLLM([good])
    r = answer_question(_Q, [], llm=fake)
    assert r["answer"] == REFUSAL and fake.calls == 0, r

    # ⑤ 出处格式/页码不符会被拦下
    wrong = dict(good)
    wrong["answer"] = "自注意力复杂度为 O(n^2 · d)[来源:attention.pdf,8,p5b1]。"
    wrong["citations"] = [{"file": "attention.pdf", "page": 8, "chunk_id": "p5b1",
                           "quote": "自注意力的计算复杂度为 O(n^2 · d)"}]
    r = answer_question(_Q, _FRAGS, llm=_FakeLLM([wrong, wrong]))
    assert r["confidence"] <= 0.3, r
    assert all(c["page"] == 5 for c in r["citations"]), r

    # ⑥ 数值是单位换算（原文 80GB，答案 81920MB）→ 判不合规
    frags = [{"chunk_id": "p1", "file": "a.pdf", "page": 1,
              "text": "训练使用 8 张 A100（80GB），学习率 1e-3。"}]
    converted = {"answer": "训练显存为 81920MB[来源:a.pdf,1,p1]。",
                 "citations": [{"file": "a.pdf", "page": 1, "chunk_id": "p1",
                                "quote": "训练使用 8 张 A100（80GB）"}],
                 "confidence": 0.9, "missing": ""}
    r = answer_question("训练显存多大？", frags, llm=_FakeLLM([converted, converted]))
    assert "81920" not in r["answer"], "换算后的数值不得出现在答案里"

    # ⑦ 接检索管线的 hits 结构
    hits = [{"id": "p5b1", "doc_id": "attention.pdf", "page_start": 5,
             "text": _FRAGS[0]["text"]}]
    got = fragments_from_hits(hits)
    assert got[0].chunk_id == "p5b1" and got[0].file == "attention.pdf" and got[0].page == 5

    print("selftest: 7/7 通过")
    return 0


def _demo(question: str, top_k: int) -> None:
    """检索 → 有据问答，端到端。无 Key 时自动走原文摘录。"""
    from app.retrieval.pipeline import AcademicRetrievalPipeline
    from app.retrieval.sample_corpus import build_sample_chunks

    pipe = AcademicRetrievalPipeline(build_sample_chunks()).build()
    hits = pipe.search(question, top_k=top_k)["hits"]
    result = answer_question(question, fragments_from_hits(hits))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="基于检索片段的有据问答（仅输出 JSON）")
    ap.add_argument("--question", "-q", default="", help="问题")
    ap.add_argument("--context", default="", help="片段 JSON 文件（数组或 {hits:[...]}）")
    ap.add_argument("--demo", action="store_true", help="用内置语料跑 检索→问答")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())
    if args.demo:
        if not args.question:
            raise SystemExit("--demo 需要 -q/--question")
        _demo(args.question, args.top_k)
        return
    if not args.question:
        raise SystemExit("需要 -q/--question（或 --selftest / --demo）")

    context: Any = []
    if args.context:
        data = json.loads(Path(args.context).read_text(encoding="utf-8"))
        context = data.get("hits", data) if isinstance(data, dict) else data
    print(json.dumps(answer_question(args.question, context), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
