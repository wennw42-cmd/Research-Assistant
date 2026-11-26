"""Function Calling 执行循环（所有专业子 Agent 共用的"手"）。

一次 run_tool_loop = 单条 ReAct 微循环：
    LLM(带 tools) → tool_calls → 逐个执行（统一信封）→ 结果回灌 → 再问 LLM
直到：① 模型不再调用工具；② 达到步数上限（防止预算失控）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from app import config
from app.agents.tools.registry import RunContext, registry
from app.providers.llm import BaseLLM
from app.utils import get_logger, short

log = get_logger("ra.loop")


@dataclass
class LoopResult:
    assistant_text: str = ""
    turns: list[dict[str, Any]] = field(default_factory=list)   # 本轮新增的全部消息
    evidence: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_summary: str = ""


def _dedup_evidence(pool: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {e.get("doc_id", "") + "#" + e.get("block_id", "") for e in pool}
    out = list(pool)
    for e in new:
        key = e.get("doc_id", "") + "#" + e.get("block_id", "")
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def run_tool_loop(
    llm: BaseLLM,
    messages: Sequence[dict[str, Any]],
    tools_specs: Sequence[dict[str, Any]],
    ctx: RunContext,
    evidence_pool: list[dict[str, Any]] | None = None,
    max_steps: int | None = None,
) -> LoopResult:
    """执行带工具的多步推理，返回本轮新增消息、累计证据与执行步骤。"""
    settings = config.get_settings()
    max_steps = max_steps or settings.AGENT_MAX_TOOL_STEPS
    result = LoopResult()
    msgs: list[dict[str, Any]] = list(messages)
    pool: list[dict[str, Any]] = list(evidence_pool or [])

    for step in range(1, max_steps + 1):
        resp = llm.chat(msgs, tools=tools_specs)
        if not resp.has_calls:
            result.final_summary = resp.content
            if resp.content:
                result.turns.append({"role": "assistant", "content": resp.content})
            break

        msgs.append({"role": "assistant", "content": resp.content or "", "tool_calls": resp.tool_calls})
        result.turns.append({"role": "assistant", "content": resp.content or "", "tool_calls": resp.tool_calls})

        for call in resp.tool_calls:
            name = call["function"]["name"]
            args_raw = call["function"].get("arguments", "{}")
            ctx.extra["evidence_pool"] = pool           # 让 write_section 等工具吃到已有证据
            env = registry.invoke(name, args_raw, ctx)
            tool_msg = {
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": json.dumps(env, ensure_ascii=False),
            }
            msgs.append(tool_msg)
            result.turns.append(tool_msg)
            new_cards = env.get("cited") or []
            pool = _dedup_evidence(pool, new_cards)
            result.steps.append({
                "step": step, "tool": name, "ok": env.get("ok"),
                "args": args_raw if len(args_raw) < 300 else args_raw[:300] + "…",
                "cited": len(new_cards),
                "latency_ms": (env.get("metrics") or {}).get("latency_ms"),
            })
            log.info("step=%d tool=%-20s ok=%s cited=%d", step, name, env.get("ok"), len(new_cards))

    # 收口：若最后一步是工具结果（模型还没说话），强制要一次无工具的总结
    if not result.final_summary:
        closing = llm.chat(msgs + [{"role": "user", "content": "基于上述工具结果给出结构化结论，保留引用锚点。"}],
                           tools=None)
        result.final_summary = closing.content
        result.turns.append({"role": "assistant", "content": closing.content})

    result.evidence = pool
    result.assistant_text = short(result.final_summary, 4000)
    return result
