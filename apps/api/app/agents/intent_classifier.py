"""意图分类器（12 类科研意图 → 结构化 JSON）。

OpenAI 兼容接口 + JSON 模式。开箱即用：

    python app/agents/intent_classifier.py "帮我解析 data/paper.pdf"
    python app/agents/intent_classifier.py "对比 Transformer 和 BERT" --model qwen-plus

也可直接跑内置自测（离线，不联网）：

    python app/agents/intent_classifier.py --selftest

输出契约（仅 JSON，不解释）：
{
  "intent": "document_parse|dedup_check|rag_qa|cross_paper_reasoning|translate|paper_search|"
            "arxiv_track|citation_graph|survey_generate|writing_assist|data_viz|chitchat",
  "confidence": 0.0-1.0,
  "entities": {"paper_ids": [], "query": "", "file_path": ""},
  "need_clarify": true/false,
  "clarify_question": ""
}
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

# ═══════════════════════════════ 意图集合 ═══════════════════════════════
INTENTS: list[str] = [
    "document_parse",         # 解析 PDF/扫描件/OCR，产出结构化内容
    "dedup_check",            # 判断两篇（或多篇）是否重复/同一版本链
    "rag_qa",                 # 基于文献库的单篇/单主题问答
    "cross_paper_reasoning",  # 跨多篇文献对比、归纳、推理
    "translate",              # 版式感知翻译
    "paper_search",           # 检索/推荐论文
    "arxiv_track",            # 长期追踪 arXiv/期刊更新与订阅
    "citation_graph",         # 引用/共引网络、主路径、社区
    "survey_generate",        # 生成综述（多节、带引用）
    "writing_assist",         # 润色/改写/续写/投稿信
    "data_viz",               # 表格/数据 → 图表
    "chitchat",               # 寒暄、与科研无关、信息不足
]

# 需要 file_path 才能执行的意图
FILE_INTENTS = {"document_parse", "dedup_check", "translate", "data_viz"}
# 需要论文标识或检索词才能执行的意图
PAPER_INTENTS = {"dedup_check", "rag_qa", "cross_paper_reasoning", "paper_search",
                 "arxiv_track", "citation_graph", "survey_generate"}

# ↘ 与本系统 Supervisor 的路由桥接（分类器 → 六专业 Agent）
INTENT_TO_AGENT: dict[str, str] = {
    "document_parse": "parse",
    "dedup_check": "parse",
    "rag_qa": "rag",
    "cross_paper_reasoning": "rag",
    "translate": "translation",
    "paper_search": "retrieval",
    "arxiv_track": "retrieval",
    "citation_graph": "graph",
    "survey_generate": "writing",
    "writing_assist": "writing",
    "data_viz": "writing",
    "chitchat": "clarify",
}


def intent_to_route(intent: str) -> str:
    return INTENT_TO_AGENT.get(intent, "rag")


# ═══════════════════════════════ 提示词 ═══════════════════════════════
SYSTEM_PROMPT = """你是科研助手系统的意图分类器。只输出 JSON，不要任何解释文字。

## 可选意图（12 类，只能选一个）
- document_parse：解析/上传/转换 PDF、扫描件 OCR、抽取表格公式
- dedup_check：判断两篇文献是否重复、是否为同一论文的 v1/v7 等版本
- rag_qa：基于文献库回答单个知识点（是什么/为什么/怎么做）
- cross_paper_reasoning：跨多篇文献对比、归纳、找共识与分歧
- translate：翻译文献内容（保留版式、术语一致）
- paper_search：检索/推荐论文（一次性）
- arxiv_track：长期追踪、订阅 arXiv/期刊新论文
- citation_graph：引用网络、共引、主路径、社区发现
- survey_generate：生成整篇综述（含大纲、分节、引用）
- writing_assist：润色、改写、续写、摘要、投稿信等写作辅助
- data_viz：把表格或数据画成图（柱状/折线/散点）
- chitchat：寒暄、无关话题、信息严重不足

## 判别规则（优先级从高到低）
1. 出现"解析/上传/转结构化/扫描件/OCR" → document_parse
2. 出现"是不是重复/同一篇/去重/v1 v7 版本" → dedup_check
3. 出现"对比/区别/差异/优缺点/哪个更好"且涉及≥2 个对象 → cross_paper_reasoning
4. 出现"翻译/译成/中英对照" → translate
5. 出现"综述/survey/写一篇综述" → survey_generate
6. 出现"引用网络/共引/引用图谱/主路径" → citation_graph
7. 出现"追踪/订阅/最新/持续关注" → arxiv_track
8. 出现"找论文/检索/推荐相关论文" → paper_search
9. 出现"画成图/柱状图/折线图/可视化"且对象是数据或表格 → data_viz
10. 出现"润色/改写/续写/投稿信/更学术" → writing_assist
11. 其余基于文献的提问 → rag_qa
12. 与科研无关或信息不足 → chitchat

## 实体抽取
- paper_ids：arXiv ID（如 1706.03762、arXiv:2101.00001v3）、DOI、库内 doc_id、文献标题
- query：检索/问答/检索用的核心关键词短语（无则空串）
- file_path：路径或文件名（含 .pdf/.docx/.tex/.png/.jpg/.json 等），没有则空串

## 置信度校准
- 0.9~1.0：关键词明确且实体完整
- 0.7~0.9：意图明确但实体缺失（如要翻译却没说哪篇）
- 0.6~0.7：有多类线索混用
- <0.6：无法判断，此时 need_clarify 必须为 true 并给出澄清问题

## 输出 JSON Schema
{"intent":"<上述之一>","confidence":<0.0-1.0>,
 "entities":{"paper_ids":[<字符串>],"query":"<关键词>","file_path":"<路径或空串>"},
 "need_clarify":<true|false>,"clarify_question":"<需要澄清时的问句，否则空串>"}"""


# ═══════════════════════════════ 校验与兜底 ═══════════════════════════════
_PATH_RE = re.compile(r"[\w\-./\\一-龥]+\.(?:pdf|docx?|tex|pptx?|png|jpe?g|json|md|txt|csv)\b", re.I)
_ARXIV_RE = re.compile(r"(?:arxiv\s*[:：]?\s*)?(\d{4}\.\d{4,5})(?:v(\d+))?", re.I)
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;)\]]+", re.I)
_DOCID_RE = re.compile(r"\bdoc_[A-Za-z0-9_]+\b")
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

CLARIFY_QUESTIONS: dict[str, str] = {
    "document_parse": "需要解析哪个文件？请提供 PDF/扫描件路径（或先上传）。",
    "dedup_check": "要比较哪两篇文献？请给出 arXiv ID／DOI／文件名（例如 1706.03762v1 与 v7）。",
    "rag_qa": "请补充要查询的问题对象：是针对哪篇文献或哪个主题提问？",
    "cross_paper_reasoning": "请明确要对比的对象（≥2 个）与对比维度，例如“Transformer 与 BERT 的复杂度和预训练目标”。",
    "translate": "要翻译哪篇文献的哪些部分？请给出文件/论文 ID 与目标语言（如“第 3 段译成中文”）。",
    "paper_search": "想检索什么主题？请给出关键词与期望的时间范围（如“近两年 高效注意力”）。",
    "arxiv_track": "要长期追踪哪条线？请给出关键词或种子论文 ID，以及推送频率。",
    "citation_graph": "以哪些论文为种子构建引用网络？请给出 paper_id 或主题。",
    "survey_generate": "综述的主题与范围是什么？请给出主题词、篇幅与是否限定年份。",
    "writing_assist": "需要辅助哪段文字？请粘贴原文并说明目标（润色/改写/续写）。",
    "data_viz": "要可视化哪些数据？请给出数据来源（表格/文件）与图表类型。",
    "chitchat": "我可以帮你解析文献、检索追踪、对比问答、翻译、做引用图谱或写综述，想先做哪一步？",
}


def _empty_result(intent: str = "chitchat", confidence: float = 0.0) -> dict[str, Any]:
    return {
        "intent": intent,
        "confidence": confidence,
        "entities": {"paper_ids": [], "query": "", "file_path": ""},
        "need_clarify": True,
        "clarify_question": CLARIFY_QUESTIONS.get(intent, CLARIFY_QUESTIONS["chitchat"]),
    }


def _extract_entities(user_input: str) -> dict[str, Any]:
    """规则兜底抽取：模型漏抽时用，保证实体不为空（尤其是 file_path）。"""
    text = user_input or ""
    path_m = _PATH_RE.search(text)
    file_path = path_m.group(0).strip() if path_m else ""

    paper_ids: list[str] = []
    for m in _ARXIV_RE.finditer(text):
        pid = f"arXiv:{m.group(1)}" + (f"v{m.group(2)}" if m.group(2) else "")
        if pid not in paper_ids:
            paper_ids.append(pid)
    for m in _DOI_RE.finditer(text):
        if m.group(0) not in paper_ids:
            paper_ids.append(m.group(0))
    for m in _DOCID_RE.finditer(text):
        if m.group(0) not in paper_ids:
            paper_ids.append(m.group(0))
    # 显式带 vN 的版本号（如 v1 与 v7），补齐版本链实体
    versions = re.findall(r"\bv(\d{1,2})\b", text)
    if versions and paper_ids:
        base = re.sub(r"v\d+$", "", paper_ids[0])
        for v in dict.fromkeys(versions):
            pid = f"{base}v{v}"
            if pid not in paper_ids:
                paper_ids.append(pid)
    return {"paper_ids": paper_ids, "file_path": file_path}


def _fallback_query(user_input: str, intent: str) -> str:
    """模型没给 query 时，用输入本身做保守兜底（去掉路径与指令词）。"""
    text = (_PATH_RE.sub("", user_input or "") or user_input or "").strip(" ，,。.!?：:")
    stop = ("帮我", "请", "一下", "能不能", "可以吗", "我想", "我要", "需要", "麻烦", "的")
    if len(text) > 40 or intent in {"chitchat"}:
        return ""
    for s in stop:
        text = text.replace(s, "")
    return text.strip()[:40]


def normalize_result(raw: Any, user_input: str) -> dict[str, Any]:
    """把任意模型输出收敛成契约 JSON：类型校正 + 实体兜底 + 澄清决策。"""
    if not isinstance(raw, dict):
        return _empty_result()

    intent = str(raw.get("intent", "")).strip()
    if intent not in INTENTS:
        intent = "chitchat"

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    raw_ent = raw.get("entities") if isinstance(raw.get("entities"), dict) else {}
    fallback = _extract_entities(user_input)

    paper_ids = raw_ent.get("paper_ids")
    paper_ids = [str(p).strip() for p in paper_ids if str(p).strip()] if isinstance(paper_ids, list) else []
    paper_ids = list(dict.fromkeys(paper_ids + fallback["paper_ids"]))

    file_path = str(raw_ent.get("file_path") or "").strip() or fallback["file_path"]
    query = str(raw_ent.get("query") or "").strip() or _fallback_query(user_input, intent)

    # ── 澄清决策 ──
    need_clarify = bool(raw.get("need_clarify")) or confidence < 0.6
    clarify_question = str(raw.get("clarify_question") or "").strip()

    # paper_ids（库内 doc_id / arXiv ID）与 file_path 等价可定位，有其一即无需澄清
    missing_file = intent in FILE_INTENTS and not file_path and not paper_ids
    missing_paper = intent in PAPER_INTENTS and not paper_ids and not query
    if missing_file or missing_paper:
        need_clarify = True
    if need_clarify and not clarify_question:
        if confidence < 0.6:
            clarify_question = "没太理解你的需求，" + CLARIFY_QUESTIONS.get(intent, CLARIFY_QUESTIONS["chitchat"])
        elif missing_file:
            clarify_question = CLARIFY_QUESTIONS.get(intent, CLARIFY_QUESTIONS["chitchat"])
        else:
            clarify_question = CLARIFY_QUESTIONS.get(intent, CLARIFY_QUESTIONS["chitchat"])

    return {
        "intent": intent,
        "confidence": confidence,
        "entities": {"paper_ids": paper_ids, "query": query, "file_path": file_path},
        "need_clarify": bool(need_clarify),
        "clarify_question": clarify_question if need_clarify else "",
    }


def parse_llm_text(text: str, user_input: str) -> dict[str, Any]:
    """从模型文本里稳健解析 JSON（先整体、再代码块、再子串）。"""
    text = (text or "").strip()
    for candidate in (text, _strip_fence(text)):
        try:
            return normalize_result(json.loads(candidate), user_input)
        except Exception:  # noqa: BLE001
            continue
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return normalize_result(json.loads(m.group(0)), user_input)
        except Exception:  # noqa: BLE001
            pass
    return _empty_result()


def _strip_fence(text: str) -> str:
    m = _JSON_BLOCK.search(text)
    return m.group(1).strip() if m else text


# ═══════════════════════════════ 离线兜底（无 Key / 自测）═══════════════════════════════
_RULES: list[tuple[str, str, float]] = [
    (r"(解析|上传|转(?:成)?结构化|版面|版式|扫描件|ocr|识别一下文件)", "document_parse", 0.92),
    (r"(是不是(?:同一|重复)|是否重复|去重|重复文献|同一篇|v\d.{0,6}v\d)", "dedup_check", 0.9),
    (r"(对比|比较|区别|差异|优缺点|哪个更好|versus|vs\b)", "cross_paper_reasoning", 0.88),
    (r"(翻译|译成|中英|双语|译文)", "translate", 0.92),
    (r"(综述|survey|写一篇(?:关于)?\S{0,10}(?:综述|调研))", "survey_generate", 0.9),
    (r"(引用(?:网络|图谱)|共引|citation graph|主路径|文献脉络)", "citation_graph", 0.9),
    (r"(追踪|订阅|持续关注|推送|每周|每月|最新进展|新论文)", "arxiv_track", 0.9),
    (r"(检索|搜索|找.{0,14}论文|推荐.{0,14}论文|相关论文|关于.{0,12}的论文)", "paper_search", 0.85),
    (r"(画成图|柱状图|折线图|饼图|散点图|可视化|画个图)", "data_viz", 0.9),
    (r"(润色|改写|续写|投稿信|更学术|精简|缩写|扩写)", "writing_assist", 0.85),
    (r"(是什么|为什么|如何|怎么(?:做|实现)|原理|解释|总结|多少|哪一年)", "rag_qa", 0.75),
]


def mock_classify(user_input: str) -> dict[str, Any]:
    """规则分类器：零依赖、零成本、可断言，用于 CI / 离线演示 / 模型不可用兜底。"""
    text = (user_input or "").lower()
    best: tuple[str, float] = ("chitchat", 0.3)
    for pattern, intent, weight in _RULES:
        if re.search(pattern, text, re.I):
            if weight > best[1]:
                best = (intent, weight)
    raw = {"intent": best[0], "confidence": best[1], "entities": {}, "need_clarify": best[1] < 0.6}
    return normalize_result(raw, user_input)


# ═══════════════════════════════ 主函数 ═══════════════════════════════
def classify_intent(
    user_input: str,
    *,
    client: Any | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    strategy: str = "auto",
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """分类一条用户输入。

    strategy:
      - "json_object"  ：JSON 模式（OpenAI / DeepSeek / Qwen 等支持 response_format）
      - "function_call"：用 function calling 强制结构化（兼容性最好）
      - "auto"         ：先 json_object，失败自动降级 function_call，再降级本地规则
    无 API Key 时直接走 mock_classify，保证永远返回合法契约 JSON。
    """
    user_input = (user_input or "").strip()
    if not user_input:
        return _empty_result("chitchat", 0.0)

    model = model or os.getenv("LLM_MODEL", "qwen-plus")
    base_url = base_url or os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:  # 离线兜底
        return mock_classify(user_input)

    if client is None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("需要安装 openai：pip install openai>=1.40") from e
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"用户输入：{user_input}\n只输出 JSON："},
    ]

    attempts = [strategy] if strategy != "auto" else ["json_object", "function_call"]
    for mode in attempts:
        try:
            if mode == "json_object":
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    response_format={"type": "json_object"},
                )
                return parse_llm_text(resp.choices[0].message.content or "", user_input)
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                tools=[{"type": "function", "function": {
                    "name": "record_intent",
                    "description": "记录识别到的科研意图与实体",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "intent": {"type": "string", "enum": INTENTS},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "entities": {
                                "type": "object",
                                "properties": {
                                    "paper_ids": {"type": "array", "items": {"type": "string"}},
                                    "query": {"type": "string"},
                                    "file_path": {"type": "string"},
                                },
                                "required": ["paper_ids", "query", "file_path"],
                            },
                            "need_clarify": {"type": "boolean"},
                            "clarify_question": {"type": "string"},
                        },
                        "required": ["intent", "confidence", "entities", "need_clarify", "clarify_question"],
                    },
                }}],
                tool_choice={"type": "function", "function": {"name": "record_intent"}},
            )
            calls = resp.choices[0].message.tool_calls or []
            if not calls:
                continue
            return normalize_result(json.loads(calls[0].function.arguments or "{}"), user_input)
        except Exception as e:  # noqa: BLE001 —— 降级而不是抛炸上游链路
            last_error = f"{mode} 失败：{type(e).__name__}: {e}"

    # 模型链路全失败 → 本地规则，绝不让调用方拿到异常
    result = mock_classify(user_input)
    result["clarify_question"] = result.get("clarify_question") or ""
    return result


def classify_batch(inputs: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    return [classify_intent(i, **kwargs) for i in inputs]


# ═══════════════════════════════ 自测 ═══════════════════════════════
SELFTEST: list[tuple[str, str]] = [
    ("帮我解析 data/paper.pdf", "document_parse"),
    ("arXiv:1706.03762v1 和 v7 是不是同一篇论文？", "dedup_check"),
    ("Transformer 的位置编码是怎么做的？", "rag_qa"),
    ("对比 Transformer 和 BERT 在长文本建模上的差异", "cross_paper_reasoning"),
    ("把 doc_bert 的第 3 段翻译成中文", "translate"),
    ("帮我找最近两年关于高效注意力的论文", "paper_search"),
    ("订阅 Longformer 相关的新论文，每周推送", "arxiv_track"),
    ("以 doc_transformer 为种子画出引用网络", "citation_graph"),
    ("生成一份关于高效 Transformer 的综述", "survey_generate"),
    ("帮我把这一段改写得更学术一点", "writing_assist"),
    ("把表 2 的实验结果画成柱状图", "data_viz"),
    ("你好啊", "chitchat"),
    ("嗯……", "chitchat"),
]


def _selftest() -> int:
    passed = 0
    print(f"{'输入':<44} {'意图':<22} {'置信':<6} {'澄清':<6} 判定")
    print("─" * 96)
    for text, expected in SELFTEST:
        r = classify_intent(text)
        okk = r["intent"] == expected
        passed += okk
        # 标记用 ASCII：Windows 控制台默认 GBK，emoji 会触发 UnicodeEncodeError
        print(f"{text[:42]:<44} {r['intent']:<22} {r['confidence']:<6.2f} "
              f"{str(r['need_clarify']):<6} {'[OK]' if okk else f'[XX] expected={expected}'}")
        if r["need_clarify"]:
            print(f"{'':<44} └ 澄清：{r['clarify_question']}")
    print("─" * 96)
    print(f"通过 {passed}/{len(SELFTEST)}（离线规则路径，未联网）")
    return 0 if passed == len(SELFTEST) else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="科研助手意图分类器")
    ap.add_argument("text", nargs="?", help="待分类的用户输入")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--strategy", default="auto", choices=["auto", "json_object", "function_call"])
    ap.add_argument("--selftest", action="store_true", help="运行离线自测")
    args = ap.parse_args()

    if args.selftest or not args.text:
        raise SystemExit(_selftest())

    result = classify_intent(args.text, model=args.model, base_url=args.base_url, strategy=args.strategy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
