"""Runner：把「记忆 → 图 → 记忆」串起来的唯一对外入口。

上层（FastAPI / CLI / Demo）只认这一个函数，便于以后替换传输协议。
"""
from __future__ import annotations

from typing import Any, Mapping

from app import config, history
from app.agents.graphs import get_graph
from app.memory import memory
from app.schemas.state import new_state
from app.utils import get_logger, timer_ms

log = get_logger("ra.runner")


def _thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}


def _build_result(out: Mapping[str, Any], *, thread_id: str, graph_thread: str,
                  turn: int, query: str, status: str,
                  latency_ms: int | None) -> dict[str, Any]:
    settings = config.get_settings()
    return {
        "status": status,
        "thread_id": thread_id,
        "graph_thread": graph_thread,
        "turn": turn,
        "query": query,
        "route": out.get("route"),
        "complexity": out.get("complexity"),
        "confidence": out.get("confidence"),
        "route_reason": out.get("route_reason"),
        "intent": out.get("intent"),
        "plan": out.get("plan") or [],
        "task_results": out.get("task_results") or {},
        "answer": out.get("answer", ""),
        "verify": out.get("verify") or {},
        "evidence": out.get("evidence") or [],
        "trace": out.get("trace") or [],
        "latency_ms": latency_ms,
        "model": settings.LLM_PROVIDER if not settings.use_mock_llm else "mock",
    }


def _persist_run(result: Mapping[str, Any]) -> None:
    """Agent 执行落库；失败只告警，不影响回答。"""
    try:
        history.save_agent_run(result)
    except Exception as exc:  # noqa: BLE001
        log.warning("agent run 落库失败（%s）：%s: %s",
                    result.get("graph_thread"), type(exc).__name__, exc)


def _turn_of(graph_thread: str) -> int:
    tail = graph_thread.rsplit("#t", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def run(query: str, thread_id: str = "default", user_id: str = "anonymous",
        attachment: str = "") -> dict[str, Any]:
    """执行一轮对话。返回结构化结果，供 API 直接序列化。"""
    graph = get_graph()
    attachment = str(attachment or "")[:20_000]

    memory.append(thread_id, "user", query)
    state = new_state(query, thread_id=thread_id, user_id=user_id,
                      attachment_context=attachment)
    state["literature"] = memory.build_literature_context(user_id)

    # 注入历史：最近若干轮原文 + 更早的滚动摘要
    history = memory.get_history(thread_id, limit=8)
    summary = memory.get_summary(thread_id)
    if history:
        state["messages"] = [{"role": m["role"], "content": m["content"]} for m in history[:-1]] + state["messages"]
    if summary:
        state["literature"]["conversation_summary"] = summary

    # 每个新问题开一个新的图线程：checkpointer 会累积带 reducer 的字段，
    # 复用同一 thread 会让上一轮的 evidence/trace 混入本轮，污染证据与校验。
    turn = memory.next_turn(thread_id)
    graph_thread = f"{thread_id}#t{turn}"
    state.pop("evidence", None)   # 证据池按轮次重建，不跨轮继承

    elapsed = timer_ms()
    out = graph.invoke(state, config=_thread_config(graph_thread))
    interrupted = "__interrupt__" in out

    result = _build_result(
        out, thread_id=thread_id, graph_thread=graph_thread, turn=turn, query=query,
        status="awaiting_human" if interrupted else "ok", latency_ms=elapsed(),
    )
    # 先落库（等待态也要留档），再做进程内记忆；/task 在人工确认期间能看到 DAG。
    _persist_run(result)
    memory.save_task_state(thread_id, result["plan"], result["task_results"])

    # HITL：被 interrupt_before 拦下时返回等待态
    if interrupted:
        log.info("graph_thread=%s 等待人工确认（HITL）", graph_thread)
        return {"status": "awaiting_human", "thread_id": thread_id, "graph_thread": graph_thread,
                "interrupt": str(out["__interrupt__"])[:300]}

    memory.append(thread_id, "assistant", result["answer"])
    log.info("完成 route=%s plan=%d evidence=%d latency=%dms",
             result["route"], len(result["plan"]), len(result["evidence"]), result["latency_ms"])
    return result


def resume(graph_thread: str, decision: str = "approved", feedback: str = "") -> dict[str, Any]:
    """HITL 恢复执行。graph_thread 为 run() 返回的图线程 ID；decision ∈ approved | rejected。"""
    from langgraph.types import Command

    graph = get_graph()
    out = graph.invoke(
        Command(resume={"decision": decision, "feedback": feedback}),
        config=_thread_config(graph_thread),
    )
    thread_id = graph_thread.split("#t", 1)[0]
    result = _build_result(
        out, thread_id=thread_id, graph_thread=graph_thread,
        turn=_turn_of(graph_thread), query=str(out.get("query") or ""),
        status="ok", latency_ms=None,
    )
    _persist_run(result)  # 同 graph_thread 原地更新等待中的那条记录
    memory.append(thread_id, "assistant", result["answer"])
    memory.save_task_state(thread_id, result["plan"], result["task_results"])
    return result


def get_task_state(thread_id: str) -> dict[str, Any]:
    return memory.load_task_state(thread_id)


def reset(thread_id: str) -> None:
    memory.clear(thread_id)
