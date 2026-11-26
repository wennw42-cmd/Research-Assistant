"""LangGraph 状态定义。

要点：
1) 只给少数几个字段加 reducer（追加语义），其余字段用"整体替换"，
   避免出现"半新半旧"的中间态；
2) 全部字段可 JSON 序列化 —— 这是 checkpoint 持久化 / trace 回放的前提；
3) messages 使用 OpenAI 原生 dict 协议（role/content/tool_calls/tool_call_id），
   不绑定任何厂商 SDK 的消息类。
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

MsgRole = Literal["system", "user", "assistant", "tool"]
Route = Literal[
    "parse", "retrieval", "rag", "translation", "graph", "writing",
    "planner", "clarify", "direct",
]


class GraphState(TypedDict, total=False):
    # ── 身份与输入 ──
    thread_id: str
    user_id: str
    query: str
    attachment_context: str          # 用户上传附件已解析的文本（本轮有效）

    # ── 对话（追加）──
    messages: Annotated[list[dict[str, Any]], operator.add]

    # ── 路由决策 ──
    intent: str
    route: str
    complexity: Literal["simple", "composite"]
    confidence: float
    route_reason: str

    # ── 任务规划中间态（整体替换）──
    plan: list[dict[str, Any]]        # list[SubTask-as-dict]
    current_subtask_id: str
    task_results: dict[str, Any]      # subtask_id -> 摘要结果

    # ── 证据池（追加）──
    evidence: Annotated[list[dict[str, Any]], operator.add]

    # ── 产出 ──
    answer: str
    verify: dict[str, Any]
    revision: int

    # ── 上下文与观测 ──
    literature: dict[str, Any]
    scratchpad: dict[str, Any]
    trace: Annotated[list[dict[str, Any]], operator.add]


def new_state(query: str, thread_id: str = "default", user_id: str = "anonymous",
              attachment_context: str = "") -> GraphState:
    """构造一次对话的初始状态。"""
    return GraphState(
        thread_id=thread_id,
        user_id=user_id,
        query=query,
        attachment_context=attachment_context,
        messages=[{"role": "user", "content": query}],
        intent="",
        route="rag",
        complexity="simple",
        confidence=0.0,
        route_reason="",
        plan=[],
        current_subtask_id="",
        task_results={},
        evidence=[],
        answer="",
        verify={},
        revision=0,
        literature={},
        scratchpad={},
        trace=[],
    )
