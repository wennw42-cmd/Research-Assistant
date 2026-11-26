"""Planner：把复合意图拆成可执行 DAG。

输入示例：「对比 Transformer 与 BERT 并生成综述」
输出示例（依赖而非链）：
        t1 检索             ← 根
       /  \
    t2 对比  t3 图谱        ← 可并行
       \  /
        t4 撰写综述
             |
        t5 导出 LaTeX

设计要点：
1. **模型产出 + 硬校验**：LLM 只负责"想"，结果的合法性由 validate_and_repair 兜底；
2. **可 replan**：反思节点把 critique 注入后重跑本节点，产出修订后的 DAG；
3. **粒度约束**：子任务 3~7 个，太少退化成单步、太多会拖垮延迟与预算。
"""
from __future__ import annotations

from typing import Any

from app import config
from app.agents.dag import KNOWN_AGENTS, to_dicts, validate_and_repair
from app.agents.tools.builtin import AGENT_SCOPES
from app.agents.tools.registry import registry
from app.schemas.state import GraphState
from app.utils import get_logger, short

log = get_logger("ra.planner")

AGENT_DESC = {
    "parse": "负责文档解析、OCR、语义去重与版本链判定",
    "retrieval": "负责文献检索、外部元数据获取、追踪与推荐",
    "rag": "负责基于检索证据的多文问答、对比与推理",
    "translation": "负责版式感知翻译与双语锚点维护",
    "graph": "负责引用关系挖掘、社区发现与主路径分析",
    "writing": "负责章节撰写、图表生成与 LaTeX/BibTeX 导出",
}


def _tool_menu() -> str:
    lines = []
    for agent in sorted(KNOWN_AGENTS):
        names = [t.name for t in registry._tools.values() if set(t.scopes) & set(AGENT_SCOPES.get(agent, ()))]
        lines.append(f"- {agent}（{AGENT_DESC[agent]}）：可用工具 {names}")
    return "\n".join(lines)


AGENT_MENU = "Agent 与工具：\n" + _tool_menu()


PLANNER_SYSTEM = """你是科研助手的任务规划器。把用户的复合科研指令拆解为 3~7 个有依赖关系的子任务 DAG。

硬性约束：
1. 每个子任务的 agent 必须来自：{agents}
2. depends_on 只能引用已定义的 id；能并行的必须并行（不要拉成一条链）
3. 每个子任务要写清 inputs，下游需要的字段必须能在上游产出里找到
4. 术语/引用类交付物必须由带证据的工具产出，禁止凭空撰写

输出严格 JSON（不要代码块、不要解释）：
{{"goal":"...","rationale":"...","subtasks":[{{"id":"t1","title":"...","agent":"rag","description":"...",
  "inputs":{{}},"depends_on":[],"tool_hints":["hybrid_search"]}}]}}
"""


def planner_node(state: GraphState) -> dict[str, Any]:
    query = state.get("query", "")
    critique = (state.get("verify") or {}).get("issues") or []
    library_context = state.get("literature") or {}
    attached = (state.get("attachment_context") or "").strip()
    attach_note = (
        f"\n用户附件原文（前 6000 字，已解析，可直接用于规划）：\n{short(attached, 6000)}\n"
        if attached else ""
    )

    from app.providers.llm import get_llm

    llm = get_llm()
    replan_note = ""
    if critique:
        replan_note = (
            "\n\n【上一轮执行存在问题，请在本次规划中修正】\n"
            + "\n".join(f"- {c}" for c in critique[:6])
            + "\n请据此调整子任务分解、工具选择或依赖顺序。"
        )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM.format(agents=", ".join(sorted(KNOWN_AGENTS)))},
        {"role": "user", "content": (
            f"用户目标：{query}\n"
            f"文献库上下文：{short(str(library_context), 400)}\n"
            f"{attach_note}"
            f"可用 {AGENT_MENU}\n"
            f"请规划 DAG。{replan_note}"
        )},
    ]
    raw = llm.chat_json(messages)
    plan = validate_and_repair(raw, goal=query) if raw else None
    source = "llm"

    if plan is None or not plan.subtasks:
        log.warning("LLM 规划不可用，回落启发式模板")
        from app.providers.llm import _mock_plan

        plan = validate_and_repair(_mock_plan(query), goal=query)
        source = "heuristic"

    plan_dicts = to_dicts(plan.subtasks)
    log.info("生成计划 source=%s 子任务=%s", source, [(t["id"], t["agent"], t["depends_on"]) for t in plan_dicts])
    return {
        "plan": plan_dicts,
        "current_subtask_id": "",
        "complexity": "composite",
        "intent": plan.goal,
        "route_reason": f"{state.get('route_reason', '')} | plan({source}): {short(plan.rationale, 80)}",
        "trace": [{"node": "planner", "source": source, "n_subtasks": len(plan_dicts),
                   "edges": [[t["id"], d] for t in plan_dicts for d in t["depends_on"]]}],
    }

