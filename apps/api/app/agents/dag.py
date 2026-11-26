"""任务 DAG 的工具：校验、修复、拓扑分批。

为什么不用"链式 pipeline"而用 DAG：复杂科研意图天然存在**可并行分支**
（例如"检索 A"与"检索 B"互不依赖），DAG 能显著缩短端到端延迟，
也是我们"Task Planning"能力相对朴素 ReAct 的核心差异点。
"""
from __future__ import annotations

from typing import Any

from app.schemas import PlanDAG, SubTask
from app.utils import get_logger

log = get_logger("ra.dag")

KNOWN_AGENTS = {"parse", "retrieval", "rag", "translation", "graph", "writing"}


def validate_and_repair(raw: dict[str, Any] | None, goal: str = "") -> PlanDAG | None:
    """把 LLM 输出的任意 JSON 变成**一定可执行**的 DAG。

    修复策略（宁可降级也不让链路崩）：
      · 子任务 id 缺失/重复 → 自动编号 t{i}
      · agent 非法 → 回落到 rag
      · depends_on 指向不存在的 id → 丢弃该依赖
      · 成环 → 从环上最后一个依赖断开
    """
    if not isinstance(raw, dict) or not raw.get("subtasks"):
        return None
    seen_id: dict[str, str] = {}
    tasks: list[SubTask] = []
    for i, raw_t in enumerate(raw.get("subtasks", [])):
        if not isinstance(raw_t, dict):
            continue
        tid = str(raw_t.get("id") or f"t{i + 1}")
        if tid in seen_id:
            tid = f"{tid}_{i}"
        seen_id.setdefault(tid, tid)
        agent = str(raw_t.get("agent") or "rag")
        if agent not in KNOWN_AGENTS:
            agent = "rag"
        tasks.append(SubTask(
            id=tid,
            title=str(raw_t.get("title") or f"子任务 {i + 1}"),
            agent=agent,
            description=str(raw_t.get("description") or ""),
            inputs=raw_t.get("inputs") or {},
            depends_on=[str(d) for d in (raw_t.get("depends_on") or [])],
            tool_hints=[str(t) for t in (raw_t.get("tool_hints") or [])],
        ))

    ids = {t.id for t in tasks}
    for t in tasks:
        t.depends_on = [d for d in t.depends_on if d in ids and d != t.id]
    _break_cycles(tasks)

    plan = PlanDAG(goal=str(raw.get("goal") or goal), rationale=str(raw.get("rationale") or ""), subtasks=tasks)
    if not plan.subtasks:
        return None
    return plan


def _break_cycles(tasks: list[SubTask]) -> None:
    """DFS 检测环，发现即断开回溯边（保留尽量多的并行度）。"""
    on_stack: list[str] = []
    visiting: set[str] = set()

    def dfs(t: SubTask) -> None:
        visiting.add(t.id)
        on_stack.append(t.id)
        for dep in list(t.depends_on):
            nxt = next((x for x in tasks if x.id == dep), None)
            if nxt is None:
                continue
            if nxt.id in on_stack:  # 回溯边 → 断开
                log.warning("检测到环形依赖，断开 %s -> %s", t.id, nxt.id)
                t.depends_on.remove(dep)
                continue
            if nxt.id not in visiting:
                dfs(nxt)
        on_stack.pop()

    for task in tasks:
        if task.id not in visiting:
            dfs(task)


def topo_batches(subtasks: list[SubTask]) -> list[list[SubTask]]:
    """按依赖分层：同一层内部可并行（生产可换成 LangGraph Send 做真并发扇出）。"""
    remaining = list(subtasks)
    done: set[str] = {t.id for t in subtasks if t.status == "done"}
    batches: list[list[SubTask]] = []
    guard = 0
    while remaining and guard < len(subtasks) + 2:
        guard += 1
        ready = [t for t in remaining if all(d in done for d in t.depends_on)]
        if not ready:  # 理论上已无环，兜底：强制取第一个，避免死循环
            ready = remaining[:1]
        for t in ready:
            done.add(t.id)
        batches.append(ready)
        remaining = [t for t in remaining if t not in ready]
    return batches


def ready_subtasks(subtasks: list[SubTask]) -> list[SubTask]:
    done = {t.id for t in subtasks if t.status in ("done", "skipped")}
    return [
        t for t in subtasks
        if t.status == "pending" and all(d in done for d in t.depends_on)
    ]


def is_finished(subtasks: list[SubTask]) -> bool:
    return all(t.status in ("done", "failed", "skipped") for t in subtasks)


def to_dicts(subtasks: list[SubTask]) -> list[dict[str, Any]]:
    return [t.model_dump(mode="json") for t in subtasks]


def from_dicts(items: list[dict[str, Any]]) -> list[SubTask]:
    return [SubTask.model_validate(i) for i in items]
