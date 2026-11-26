"""Supervisor：意图识别与路由。

路由策略 = **三层漏斗**，越靠前越便宜、越确定：

    ① 显式指令   /translate /graph ...           → 零延迟、零成本、100% 确定
    ② 规则打分   领域关键词 → route + confidence   → 可解释、可回归测试（CI 断言）
    ③ LLM 分类   不确定时才调用模型               → 成本兜底，且要求输出理由便于排查

为什么不让 LLM 全权路由：高频简单查询占比 >70%，
纯 LLM 路由会带来 300ms~2s 延迟和大量"过度路由到 planner"的误判。
"""
from __future__ import annotations

import re
from typing import Any

from app.schemas import ALL_ROUTES
from app.schemas.state import GraphState
from app.utils import get_logger

log = get_logger("ra.supervisor")

# ① 显式指令
COMMANDS: dict[str, str] = {
    "/parse": "parse", "/dedup": "parse", "/search": "retrieval", "/find": "retrieval",
    "/ask": "rag", "/qa": "rag", "/translate": "translation", "/graph": "graph",
    "/write": "writing", "/survey": "planner",
}

# ② 领域规则：(正则, route, 权重)
RULES: list[tuple[str, str, float]] = [
    (r"(解析|上传|版面|版式|去重|去重掉|重复文献|v1|v7|扫描件|ocr|ocr识别)", "parse", 0.9),
    (r"(翻译|translate|中英|双语|译文)", "translation", 0.9),
    (r"(引用图谱|引用网络|共引|citation graph|主路径|文献脉络)", "graph", 0.9),
    (r"(检索|搜索|找论文|推荐论文|追踪|订阅|最新进展)", "retrieval", 0.85),
    (r"(综述|survey|写一篇|撰写|成稿|latex|大纲)", "writing", 0.85),
    (r"(对比|区别|差异|versus|vs\.|优缺点)", "rag", 0.7),
    (r"(是什么|为什么|如何|解释|原理|总结|复杂度|区别在哪)", "rag", 0.75),
]

# 复合意图信号：一句指令里出现 ≥2 个不同领域动词 ⇒ 需要 Planner 拆 DAG
COMPOSITE_PATTERNS = [
    r"(对比|比较|分析).{0,20}(并|且|然后|再).{0,10}(综述|总结|写成|生成|撰)",
    r"(检索|调研).{0,20}(并|且|然后|再).{0,10}(综述|写成|生成)",
    r"(综述|survey).{0,10}(并|且|含).{0,10}(图|表|对比|latex)",
]
VAGUE_PATTERNS = [r"^\s*$", r"^(你好|hi|hello|在吗)\s*$"]


def rule_route(query: str) -> tuple[str, float, str, set[str]]:
    """返回 (候选 route, 置信度, 理由, 命中的领域集合)。"""
    scores: dict[str, float] = {}
    hits: set[str] = set()
    for pattern, route, weight in RULES:
        if re.search(pattern, query, re.I):
            scores[route] = max(scores.get(route, 0.0), weight)
            hits.add(route)
    if not scores:
        return "", 0.0, "", hits
    best = max(scores.items(), key=lambda kv: kv[1])
    reason = f"规则命中：{sorted(hits)}"
    return best[0], best[1], reason, hits


def looks_composite(query: str, hits: set[str]) -> bool:
    if any(re.search(p, query, re.I) for p in COMPOSITE_PATTERNS):
        return True
    return len(hits) >= 2


def supervisor_node(state: GraphState) -> dict[str, Any]:
    """LangGraph 节点：只做路由决策，不做任何重活。"""
    query = (state.get("query") or "").strip()
    attached = bool((state.get("attachment_context") or "").strip())
    trace_entry = {"node": "supervisor", "query_len": len(query)}

    # ── ① 显式指令 ──
    lowered = query.lower()
    for cmd, route in COMMANDS.items():
        if lowered.startswith(cmd):
            log.info("命中显式指令 %s → %s", cmd, route)
            reason = "用户提供斜杠命令，直接路由"
            if attached and route == "parse":
                route = "rag"
                reason = "附件已由系统解析，改用 rag 基于附件回答"
            return {
                "route": route,
                "complexity": "composite" if route == "planner" else "simple",
                "confidence": 1.0,
                "intent": f"显式指令 {cmd}",
                "route_reason": reason,
                "trace": [trace_entry | {"hit": "command", "route": route}],
            }

    # ── 含糊输入 → 反问（附件轮次除外：附件本身就是上下文）──
    if not attached and (any(re.search(p, query, re.I) for p in VAGUE_PATTERNS) or len(query) < 4):
        return {
            "route": "clarify",
            "complexity": "simple",
            "confidence": 0.9,
            "intent": "寒暄或信息不足",
            "route_reason": "查询过短或无实质诉求，优先澄清而不是硬答",
            "trace": [trace_entry | {"hit": "vague", "route": "clarify"}],
        }

    # ── ② 规则路由 ──
    route, conf, reason, hits = rule_route(query)
    if attached and route == "parse":
        route, conf = "rag", max(conf, 0.8)
        reason += "；附件已由系统解析，改用 rag 基于附件回答"
        hits.discard("parse")
    if attached and not hits:
        route, conf = "rag", 0.75
        reason = "附件轮次：问题未命中特定领域，默认基于附件回答"
    if len(hits) >= 2 or looks_composite(query, hits):
        log.info("复合意图 → planner（命中领域 %s）", sorted(hits))
        return {
            "route": "planner",
            "complexity": "composite",
            "confidence": max(conf, 0.9),
            "intent": f"多步复合任务，涉及 {sorted(hits)}",
            "route_reason": reason + "；多领域动词同时命中，交由 Planner 拆解 DAG",
            "trace": [trace_entry | {"hit": "rules-composite", "domains": sorted(hits), "route": "planner"}],
        }
    if conf >= 0.7:
        log.info("规则路由 → %s（conf=%.2f）", route, conf)
        return {
            "route": route,
            "complexity": "simple",
            "confidence": conf,
            "intent": reason,
            "route_reason": reason,
            "trace": [trace_entry | {"hit": "rules", "route": route, "confidence": conf}],
        }

    # ── ③ LLM 兜底分类 ──
    from app.providers.llm import get_llm

    llm = get_llm()
    prompt = (
        "你是科研助手的路由器。请判断用户意图属于哪一类，并说明 JSON 理由。\n"
        f"可选路由：{', '.join(ALL_ROUTES)}\n"
        "规则：需要多步协作（如先检索对比、再成稿）→ route=planner, complexity=composite；"
        "单步任务 → 对应单 Agent, complexity=simple。\n"
        f"用户查询：{query}\n只输出 JSON："
        '{"route":"...","complexity":"simple|composite","confidence":0.0-1.0,"intent":"...","reason":"..."}'
    )
    parsed = llm.chat_json([{"role": "user", "content": prompt}])
    if isinstance(parsed, dict) and parsed.get("route") in ALL_ROUTES:
        r = str(parsed["route"])
        reason = str(parsed.get("reason", ""))
        if attached and r == "parse":
            r = "rag"
            reason += "；附件已由系统解析，改用 rag 基于附件回答"
        return {
            "route": r,
            "complexity": str(parsed.get("complexity", "simple")),
            "confidence": float(parsed.get("confidence", 0.6)),
            "intent": str(parsed.get("intent", "")),
            "route_reason": reason,
            "trace": [trace_entry | {"hit": "llm", "route": r}],
        }

    return {
        "route": "rag",
        "complexity": "simple",
        "confidence": 0.4,
        "intent": "未识别，落到默认 RAG",
        "route_reason": "规则与模型均未给出可信路由",
        "trace": [trace_entry | {"hit": "fallback", "route": "rag"}],
    }
