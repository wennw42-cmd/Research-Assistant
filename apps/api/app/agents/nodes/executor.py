"""Executor：按 DAG 拓扑序执行子任务。

当前实现在同一节点内按层**串行**执行便于调试；生产要做真并发时，
把 for 循环换成 LangGraph 的 Send API 做扇出（每个 subtask 一个分支），
再由 join 节点汇总 —— 其余逻辑无需改动。
"""
from __future__ import annotations

from typing import Any

from app.agents.dag import from_dicts, is_finished, ready_subtasks, to_dicts
from app.agents.nodes.specialists import dispatch
from app.schemas import SubTask
from app.schemas.state import GraphState
from app.utils import get_logger, jdump, short, timer_ms

log = get_logger("ra.executor")


def executor_node(state: GraphState) -> dict[str, Any]:
    plan_raw = list(state.get("plan") or [])
    if not plan_raw:
        return {"trace": [{"node": "executor", "warning": "空计划，跳过"}]}

    tasks: list[SubTask] = from_dicts(plan_raw)
    ready = ready_subtasks(tasks)
    if not ready:
        for t in tasks:
            if t.status == "pending":
                t.status = "failed"
                log.warning("子任务 %s 因依赖未满足被标记 failed", t.id)
        return {
            "plan": to_dicts(tasks),
            "trace": [{"node": "executor", "event": "no_ready_tasks"}],
        }

    evidence_delta: list[dict[str, Any]] = []
    messages_delta: list[dict[str, Any]] = []
    trace_delta: list[dict[str, Any]] = []
    answers: list[str] = []
    results: dict[str, Any] = dict(state.get("task_results") or {})

    # 关键：同一节点内前序子任务的产出要立刻对后序任务可见（否则 writing 拿不到上游证据）
    running_state = dict(state)
    local_pool: list[dict[str, Any]] = list(state.get("evidence") or [])

    for t in ready:
        t.status = "running"
        running_state["evidence"] = local_pool
        running_state["task_results"] = results
        elapsed = timer_ms()
        log.info(">> subtask %s [%s] %s", t.id, t.agent, short(t.title, 40))
        try:
            update = dispatch(t.agent, running_state, subtask=t.model_dump(mode="json"))
            local_pool += update.get("evidence", [])
            evidence_delta += update.get("evidence", [])
            messages_delta += update.get("messages", [])
            trace_delta += update.get("trace", [])
            answer = update.get("answer", "")
            t.status = "done"
            answers.append(f"### {t.id} {t.title}\n{answer}")
            results[t.id] = {
                "title": t.title,
                "agent": t.agent,
                "answer": short(answer, 600),
                "cited": [e.get("doc_id", "") + "#" + e.get("block_id", "") for e in update.get("evidence", [])],
                "latency_ms": elapsed(),
            }
        except Exception as e:  # noqa: BLE001
            t.status = "failed"
            t.retry += 1
            log.error("subtask %s 执行失败：%s: %s", t.id, type(e).__name__, e)
            results[t.id] = {"error": f"{type(e).__name__}: {short(str(e), 200)}"}
        log.info("<< subtask %s %s (%dms)", t.id, t.status, elapsed())

    return {
        "plan": to_dicts(tasks),
        "task_results": results,
        "current_subtask_id": ready[-1].id,
        "answer": "\n\n".join(answers) if answers else state.get("answer", ""),
        "evidence": evidence_delta,
        "messages": messages_delta,
        "trace": trace_delta + [{"node": "executor", "executed": [t.id for t in ready],
                                 "status": {t.id: t.status for t in tasks},
                                 "progress": f"{sum(1 for t in tasks if t.status in ('done', 'failed', 'skipped'))}/{len(tasks)}"}],
    }


def route_after_executor(state: GraphState) -> str:
    tasks = from_dicts(state.get("plan") or [])
    if not tasks or is_finished(tasks):
        return "verifier"
    if not ready_subtasks(tasks):
        log.warning("DAG 无法继续推进，强制进入校验")
        return "verifier"
    return "executor"
