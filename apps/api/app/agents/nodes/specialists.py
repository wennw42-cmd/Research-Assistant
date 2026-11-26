"""六个专业子 Agent（子图的最小可用形态）。

每个子 Agent = 专属 System Prompt + **被授权的工具子集** + 共享的执行循环。
共享执行循环、隔离工具与提示词 —— 这是"多 Agent"而不是"一个万能 prompt"的关键：

    ┌──────────┬──────────────────────────┬──────────────────────────────┐
    │ Agent    │ 授权 scope               │ 交付物                        │
    ├──────────┼──────────────────────────┼──────────────────────────────┤
    │ parse    │ m1                       │ DocIR、去重报告、版本链        │
    │ retrieval│ m4 + shared              │ 候选论文、元数据、推荐理由      │
    │ rag      │ m2 + shared              │ 带引用锚点的结论               │
    │ translation│ m3 + shared            │ block 级译文 + bbox 锚点       │
    │ graph    │ m5 + shared              │ 节点/边/社区/主路径            │
    │ writing  │ m6 + shared              │ Markdown、Vega-Lite、LaTeX    │
    └──────────┴──────────────────────────┴──────────────────────────────┘
"""
from __future__ import annotations

from typing import Any

from app.agents.tools.builtin import AGENT_SCOPES
from app.agents.tools.registry import RunContext, registry
from app.agents.toolloop import run_tool_loop
from app.providers.llm import get_llm
from app.schemas.state import GraphState
from app.utils import get_logger, jdump, short

log = get_logger("ra.specialists")

SPECIALIST_PROMPTS: dict[str, str] = {
    "parse": (
        "你是文档解析专家。负责把 PDF/扫描件转成结构化 DocIR，并完成语义去重。\n"
        "要求：① 扫描件先做质量分级再选 OCR 链路；② 去重时优先用元数据版本规则（如 arXiv v1/v7 属同一篇），"
        "语义相似度只作兜底；③ 任何结论都要给出可解释的 diff 或详情。"
    ),
    "retrieval": (
        "你是文献检索专家。负责从库内与外部学术库（OpenAlex/Crossref/S2）找论文。\n"
        "要求：① 优先返回与查询最相关的条目并说明推荐理由；② 检索结果必须去重（与文献库已有条目比对）；"
        "③ 无结果时明确说明，不要编造标题或 DOI。"
    ),
    "rag": (
        "你是文献问答与多文推理专家。所有结论必须基于 hybrid_search 取回的证据。\n"
        "铁律：① 每条论断后必须附 [[ev:doc_id#block_id]] 锚点；② 证据不足时必须说'库内暂无可靠依据'，"
        "禁止用先验知识补全；③ 不要生成任何文献标题、作者或 DOI。"
    ),
    "translation": (
        "你是学术翻译专家。按语义块（block）翻译，保持术语一致与版式锚点。\n"
        "要求：① 以 block_id 为单元，保留 page/bbox 以支持双向联动；② 套用用户术语表；"
        "③ 公式、图表标题、参考文献不翻译。"
    ),
    "graph": (
        "你是引用图谱挖掘专家。负责构建引用/共引网络并识别研究脉络。\n"
        "要求：① 区分 cites 与 co-cited 边；② 给出社区划分与主路径；③ 用节点/边结构返回，便于前端渲染。"
    ),
    "writing": (
        "你是学术写作专家。基于已有证据撰写章节并导出 LaTeX/BibTeX。\n"
        "铁律：① 引用只能来自证据池中的 doc_id#block_id，不得新增任何未见参考文献；"
        "② 无证据支撑的要点必须留空并标注待补充；③ 需要配图时输出 Vega-Lite 规范。"
    ),
}


def dispatch(agent: str, state: GraphState, subtask: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行一个专业子 Agent，返回可写回 GraphState 的增量。"""
    scopes = AGENT_SCOPES.get(agent, ("shared",))
    specs = registry.specs(scopes=scopes)
    llm = get_llm()

    goal = (subtask or {}).get("description") or state.get("query", "")
    upstream = ""
    tr = state.get("task_results") or {}
    deps = (subtask or {}).get("depends_on") or []
    if deps:
        upstream = "\n上游产出摘要：\n" + "\n".join(
            f"- [{d}] {short(jdump(tr.get(d, {})), 300)}" for d in deps if d in tr
        )

    ctx = RunContext(
        thread_id=state.get("thread_id", "default"),
        user_id=state.get("user_id", "anonymous"),
        trace_id=state.get("thread_id", ""),
    )
    attached = (state.get("attachment_context") or "").strip()
    system_prompt = SPECIALIST_PROMPTS.get(agent, "你是科研助手。")
    attach_block = ""
    if attached:
        system_prompt += (
            "\n本轮用户上传了附件。附件文本是用户提供的一手材料，可直接作为回答依据；"
            "引用时注明“据附件”，不得编造页码、作者或文献信息。"
            "附件已由系统解析，不要调用 parse_document 解析这段文本。"
            "本轮不强制 [[ev:doc#block]] 锚点，但不得声称附件内容来自库内检索。"
        )
        attach_block = (
            "\n\n【用户附件原文（系统已解析，直接可用）】\n"
            f"{short(attached, 12000)}\n【附件结束】"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"任务：{goal}\n"
            f"输入参数：{jdump((subtask or {}).get('inputs', {}), 400)}\n"
            f"{upstream}\n"
            f"当前可用证据 {len(state.get('evidence') or [])} 条。"
            f"{attach_block}\n开始执行。"
        )},
    ]

    log.info(">> dispatch agent=%s tools=%s", agent, [s["function"]["name"] for s in specs])
    result = run_tool_loop(
        llm=llm,
        messages=messages,
        tools_specs=specs,
        ctx=ctx,
        evidence_pool=list(state.get("evidence") or []),
    )
    # state.evidence 走 append reducer，这里只回传**新增**部分
    old_keys = {e.get("doc_id", "") + "#" + e.get("block_id", "") for e in (state.get("evidence") or [])}
    new_evidence = [e for e in result.evidence if e.get("doc_id", "") + "#" + e.get("block_id", "") not in old_keys]

    return {
        "messages": result.turns,
        "evidence": new_evidence,
        "answer": result.final_summary,
        "trace": [{"node": f"agent:{agent}",
                   "subtask": (subtask or {}).get("id", ""),
                   "steps": result.steps,
                   "new_evidence": len(new_evidence)}],
    }


def route_and_dispatch(state: GraphState) -> dict[str, Any]:
    """非复合意图：Supervisor 直接把活派给单个专业子 Agent。"""
    return dispatch(state.get("route", "rag"), state)
