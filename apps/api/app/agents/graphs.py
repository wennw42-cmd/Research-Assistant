"""LangGraph 主图装配（Supervisor + DAG Planner + 六专家 + 校验反思闭环）。

图结构：

                        ┌────────────┐
                        │ supervisor │  意图路由（规则优先 / LLM 兜底）
                        └─────┬──────┘
              ┌───────────────┼────────────────┬──────────────┐
              ▼               ▼                ▼              ▼
        ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐
        │ planner  │   │ specialist │   │ clarify  │   │ (direct) │
        │ 拆 DAG   │   │ 单专家直答 │   └────┬─────┘   └────┬─────┘
        └────┬─────┘   └─────┬──────┘        │              │
             ▼               │               │              │
        ┌──────────┐         │               │              │
        │ executor │◄───┐    │               │              │
        │ 拓扑执行 │    │    │               │              │
        └────┬─────┘    │    │               │              │
             │ DAG未完 ─┘    │               │              │
             ▼               ▼               │              │
        ┌──────────────────────────────┐     │              │
        │            verifier          │◄────┘              │
        │ 引用/覆盖/引文字面/模型复核   │◄───────────────────┘
        └───────┬──────────────┬───────┘
          通过  │              │ 不通过且 revision < max
                ▼              ▼
         ┌────────────┐  ┌───────────┐
         │ responder  │  │ reflector │ → planner（重规划）/ specialist（重跑）
         └─────┬──────┘  └───────────┘
               ▼
              END
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app import config
from app.agents.nodes import (
    clarify_node,
    executor_node,
    planner_node,
    reflector_node,
    responder_node,
    route_after_executor,
    route_after_reflector,
    route_after_verifier,
    route_and_dispatch,
    supervisor_node,
    verify_node,
)
from app.schemas.state import GraphState
from app.utils import get_logger

log = get_logger("ra.graph")

SPECIALIST_ROUTES = {"parse", "retrieval", "rag", "translation", "graph", "writing"}


def route_after_supervisor(state: GraphState) -> str:
    r = (state.get("route") or "rag").strip()
    if r == "planner":
        return "planner"
    if r in ("clarify", "direct"):
        return "clarify"
    if r in SPECIALIST_ROUTES:
        return "specialist"
    log.warning("未知路由 %s，回落 rag", r)
    return "specialist"


def build_graph(checkpointer: Any | None = None, interrupt_before: list[str] | None = None):
    """构建并编译主图。

    checkpointer：传入 MemorySaver / PostgresSaver 即可获得
      · 断点续跑（刷新页面不丢上下文）
      · HITL（配合 interrupt_before=["executor"]）
    生产建议用 langgraph.checkpoint.postgres.PostgresSaver。
    """
    g = StateGraph(GraphState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("specialist", route_and_dispatch)
    g.add_node("clarify", clarify_node)
    g.add_node("verifier", verify_node)
    g.add_node("reflector", reflector_node)
    g.add_node("responder", responder_node)

    g.set_entry_point("supervisor")

    g.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"planner": "planner", "specialist": "specialist", "clarify": "clarify"},
    )
    g.add_edge("planner", "executor")
    g.add_conditional_edges("executor", route_after_executor, {"executor": "executor", "verifier": "verifier"})
    g.add_edge("specialist", "verifier")
    g.add_edge("clarify", "responder")
    g.add_conditional_edges("verifier", route_after_verifier, {"responder": "responder", "reflector": "reflector"})
    g.add_conditional_edges("reflector", route_after_reflector, {"planner": "planner", "specialist": "specialist"})
    g.add_edge("responder", END)

    settings = config.get_settings()
    if interrupt_before is None and settings.AGENT_HITL:
        interrupt_before = ["executor"]
    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def default_checkpointer():
    """优先新版 InMemorySaver，兼容旧版 MemorySaver。"""
    try:  # langgraph >= 0.2.6
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except ImportError:  # pragma: no cover
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


_graph_cache: Any | None = None


def get_graph(rebuild: bool = False):
    global _graph_cache
    if _graph_cache is None or rebuild:
        _graph_cache = build_graph(checkpointer=default_checkpointer())
        log.info("主图已编译")
    return _graph_cache
