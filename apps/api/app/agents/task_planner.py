"""任务规划器：目标 + 可用工具 → 可执行子任务 DAG。

设计宗旨（两条硬规则，写入提示词并在代码里二次强制）：
  R1 **优先复用已有文献库**：任何分析/写作类任务之前，必须先有本地检索任务；
  R2 **外部检索本地优先**：用到 arXiv/OpenAlex 的任务，必须依赖一个本地检索任务。

输出契约（可执行 JSON）：
{"goal": "...",
 "tasks": [{"id":"t1","action":"做什幺","tool":"工具名|none",
            "depends_on":["t0"],"output":"预期产出（可被下游引用的字段）"}]}

用法：
  python app/agents/task_planner.py --selftest
  python app/agents/task_planner.py --goal "对比 Transformer 与 BERT 并生成综述"
  python app/agents/task_planner.py --goal "扫描件 OCR 后去重" --tools parse_document,extract_scanned_ocr
  python app/agents/task_planner.py --goal "生成高效 Transformer 综述" --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

# ═══════════════════════════════ 工具集与策略常量 ═══════════════════════════════
DEFAULT_TOOLS: list[dict[str, str]] = [
    {"name": "hybrid_search", "desc": "本地文献库混合检索（BM25+向量+精排），返回带 page/bbox 的证据卡"},
    {"name": "parse_document", "desc": "解析 PDF/文档为结构化 DocIR（版式块、阅读序）"},
    {"name": "extract_scanned_ocr", "desc": "扫描件质量分级 + OCR + 版面恢复"},
    {"name": "semantic_dedup", "desc": "语义去重与 arXiv 版本链判定"},
    {"name": "fetch_paper", "desc": "外部学术库检索（arXiv/OpenAlex/Crossref/S2）"},
    {"name": "compare_claims", "desc": "多维度跨文献对照，每格结论带引用锚点"},
    {"name": "build_citation_graph", "desc": "构建引用/共引异构图，输出社区与主路径"},
    {"name": "translate_block", "desc": "按 block 翻译并保持 bbox 锚点"},
    {"name": "write_section", "desc": "按大纲撰写章节，引用强制来自证据池"},
    {"name": "render_chart", "desc": "生成 Vega-Lite 图表规范"},
    {"name": "export_latex", "desc": "导出 LaTeX 正文与 BibTeX"},
    {"name": "none", "desc": "无需工具（纯推理、汇总、格式化）"},
]

LOCAL_TOOLS = {"hybrid_search", "parse_document", "extract_scanned_ocr", "semantic_dedup"}
EXTERNAL_TOOLS = {"fetch_paper", "arxiv_search", "web_search", "openalex_search"}
ANALYSIS_TOOLS = {"compare_claims", "build_citation_graph", "write_section", "render_chart", "export_latex"}
NO_TOOL = "none"
MAX_TASKS = 7
MIN_TASKS = 2

# 模型常自造工具名，这里做别名归一（映射目标必须真实存在于 allowed）
TOOL_ALIASES: dict[str, str] = {
    "arxiv": "fetch_paper", "arxiv_search": "fetch_paper", "arxiv_api": "fetch_paper",
    "openalex": "fetch_paper", "openalex_search": "fetch_paper", "crossref_search": "fetch_paper",
    "web_search": "fetch_paper", "paper_fetch": "fetch_paper",
    "search_paper": "hybrid_search", "library_lookup": "hybrid_search", "local_search": "hybrid_search",
    "vector_search": "hybrid_search", "rag_search": "hybrid_search", "retrieve": "hybrid_search",
    "ocr": "extract_scanned_ocr", "parse": "parse_document", "pdf_parse": "parse_document",
    "dedup": "semantic_dedup", "compare": "compare_claims", "translate": "translate_block",
    "citation_graph": "build_citation_graph", "write": "write_section",
    "chart": "render_chart", "plot": "render_chart", "latex": "export_latex",
}

# 工具 → 专业子 Agent（与本系统 Supervisor 的六个 Agent 对齐）
TOOL_TO_AGENT: dict[str, str] = {
    "parse_document": "parse",
    "extract_scanned_ocr": "parse",
    "semantic_dedup": "parse",
    "hybrid_search": "retrieval",
    "fetch_paper": "retrieval",
    "arxiv_search": "retrieval",
    "compare_claims": "rag",
    "translate_block": "translation",
    "build_citation_graph": "graph",
    "write_section": "writing",
    "render_chart": "writing",
    "export_latex": "writing",
    "none": "rag",
}


def normalize_tools(tool_list: Any) -> list[dict[str, str]]:
    """入参兼容三种写法：None（用默认集）/ ["a","b"] / [{"name":..,"desc":..}]。"""
    if not tool_list:
        return list(DEFAULT_TOOLS)
    out: list[dict[str, str]] = []
    for t in tool_list:
        if isinstance(t, str):
            known = next((d for d in DEFAULT_TOOLS if d["name"] == t), None)
            out.append(known or {"name": t, "desc": "自定义工具"})
        elif isinstance(t, dict) and t.get("name"):
            out.append({"name": str(t["name"]), "desc": str(t.get("desc") or "")})
    return out or list(DEFAULT_TOOLS)


# ═══════════════════════════════ 提示词 ═══════════════════════════════
SYSTEM_PROMPT = """你是科研助手的任务规划器。把用户目标拆解为有序子任务 DAG，输出**可执行 JSON**。

## 硬性规划规则（违反即不合格）
R1 优先复用已有文献库：任何分析/写作类任务（对比、图谱、撰写、成图、导出）之前，
   必须先有一个本地检索任务（hybrid_search / parse_document / semantic_dedup）。
R2 外部检索本地优先：凡使用外部检索工具（fetch_paper 等）的任务，
   必须 depends_on 一个本地检索任务，且 output 中注明"补充外部证据"。
R3 能并行就并行：互不依赖的任务不要拉成一条链（例如检索 A 与检索 B 应并列）。
R4 tool 必须取自给定可用工具列表；确实不需要工具时填 "none"。
R5 depends_on 只能引用已定义的 id，且不得成环。
R6 任务数控制在 2~7 个；每个任务产出必须描述"可被下游引用的字段"，而不是空话。

## 输出 JSON Schema（只输出 JSON，不要解释、不要代码块）
{"goal":"<原始目标>",
 "tasks":[{"id":"t1","action":"<具体做什么，含关键参数>",
           "tool":"<工具名或 none>","depends_on":["t0"],
           "output":"<预期产出与关键字段>"}]}

## 典型范式（仅供参考，要按实际目标调整）
综述：t1 本地检索 → t2 外部补充(依赖 t1) → t3 对比/图谱(依赖 t1,t2) → t4 撰写(依赖 t3) → t5 导出(依赖 t4)
对比：t1 检索 A、t2 检索 B（并列）→ t3 compare_claims(依赖 t1,t2) → t4 撰写(依赖 t3)
解析：t1 parse_document → t2 OCR(依赖 t1，仅扫描件) → t3 semantic_dedup(依赖 t1)
翻译：t1 本地定位原文 → t2 translate_block(依赖 t1)
图谱：t1 本地检索 → t2 fetch_paper 扩展(依赖 t1) → t3 build_citation_graph(依赖 t1,t2)"""


def build_messages(goal: str, tools: list[dict[str, str]]) -> list[dict[str, str]]:
    menu = "\n".join(f"- {t['name']}：{t['desc']}" for t in tools)
    local = ", ".join(sorted(LOCAL_TOOLS & {t['name'] for t in tools})) or "hybrid_search"
    external = ", ".join(sorted(EXTERNAL_TOOLS & {t['name'] for t in tools})) or "（无外部工具）"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"用户目标：{goal}\n\n"
            f"可用工具：\n{menu}\n\n"
            f"其中【本地工具】= {local}；【外部工具】= {external}。\n"
            f"提示：本地工具优先；外部检索必须排在至少一个本地检索任务之后。\n"
            f"只输出 JSON："
        )},
    ]


# ═══════════════════════════════ 解析 / 校验 / 修复 ═══════════════════════════════
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _strip_fence(text: str) -> str:
    m = _JSON_BLOCK.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def _norm_task(raw: dict[str, Any], idx: int, allowed: set[str]) -> dict[str, Any]:
    tid = str(raw.get("id") or f"t{idx + 1}").strip()
    tool = str(raw.get("tool") or NO_TOOL).strip()
    tool = TOOL_ALIASES.get(tool.lower(), tool)
    if tool not in allowed:
        # 近似匹配（模型可能写 fetch_papers 之类），否则降级 none
        hit = next((a for a in allowed if a != NO_TOOL and (a in tool or tool in a)), None)
        tool = hit or NO_TOOL
    if tool == NO_TOOL:
        # 用 action 语义回填，避免把"查 arXiv"这类关键动作抹掉
        act = str(raw.get("action") or "").lower()
        if any(k in act for k in ("arxiv", "外部", "最新", "openalex", "crossref")) and "fetch_paper" in allowed:
            tool = "fetch_paper"
        elif any(k in act for k in ("检索", "搜索", "库内", "本地")) and "hybrid_search" in allowed:
            tool = "hybrid_search"
    deps = raw.get("depends_on")
    deps = [str(d).strip() for d in deps if str(d).strip()] if isinstance(deps, list) else []
    return {
        "id": tid,
        "action": str(raw.get("action") or "").strip() or f"执行步骤 {idx + 1}",
        "tool": tool,
        "depends_on": deps,
        "output": str(raw.get("output") or "").strip() or "结构化结果（供下游引用）",
    }


def _break_cycles(tasks: list[dict[str, Any]]) -> list[str]:
    """DFS 断环，返回修复说明。"""
    notes: list[str] = []
    by_id = {t["id"]: t for t in tasks}
    on_stack: list[str] = []
    visiting: set[str] = set()

    def dfs(t: dict[str, Any]) -> None:
        visiting.add(t["id"])
        on_stack.append(t["id"])
        for d in list(t["depends_on"]):
            if d not in by_id:
                continue
            if d in on_stack:
                t["depends_on"].remove(d)
                notes.append(f"断开环形依赖 {t['id']} -> {d}")
                continue
            if d not in visiting:
                dfs(by_id[d])
        on_stack.pop()

    for t in tasks:
        if t["id"] not in visiting:
            dfs(t)
    return notes


def _topo_order(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保证输出顺序满足依赖（同层保持稳定），便于下游顺序执行。"""
    remaining = list(tasks)
    done: set[str] = set()
    ordered: list[dict[str, Any]] = []
    guard = 0
    while remaining and guard <= len(tasks) + 2:
        guard += 1
        ready = [t for t in remaining if all(d in done for d in t["depends_on"])]
        if not ready:
            ready = remaining[:1]
        for t in ready:
            done.add(t["id"])
            ordered.append(t)
        remaining = [t for t in remaining if t not in ready]
    return ordered


def validate_and_repair(raw: Any, goal: str, tools: list[dict[str, str]]) -> dict[str, Any]:
    """把任意模型输出收敛成一定可执行的 DAG（模型不可信，代码必须兜底）。"""
    allowed = {t["name"] for t in tools} | {NO_TOOL}
    notes: list[str] = []
    items = raw.get("tasks") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        notes.append("模型未产出可用计划，改用启发式模板")
        return heuristic_plan(goal, tools)

    known: dict[str, str] = {}
    tasks: list[dict[str, Any]] = []
    for i, raw_t in enumerate(items):
        if not isinstance(raw_t, dict):
            continue
        t = _norm_task(raw_t, i, allowed)
        if t["id"] in known:  # id 重复 → 重编号
            new_id = f"{t['id']}_{i}"
            notes.append(f"重复 id {t['id']} → {new_id}")
            t["id"] = new_id
        known[t["id"]] = t["id"]
        tasks.append(t)

    # 悬空依赖 / 自依赖
    for t in tasks:
        keep = [d for d in t["depends_on"] if d in known and d != t["id"]]
        if keep != t["depends_on"]:
            notes.append(f"{t['id']} 丢弃无效依赖 {set(t['depends_on']) - set(keep)}")
            t["depends_on"] = keep

    notes += _break_cycles(tasks)

    # 任务数截断（保留前面的，清理指向被删任务的依赖）
    if len(tasks) > MAX_TASKS:
        dropped = {t["id"] for t in tasks[MAX_TASKS:]}
        notes.append(f"截断至 {MAX_TASKS} 个任务，丢弃 {sorted(dropped)}")
        tasks = tasks[:MAX_TASKS]
        for t in tasks:
            t["depends_on"] = [d for d in t["depends_on"] if d not in dropped]

    # ── R1：本地优先（缺本地检索任务则补一个根节点）──
    has_local = any(t["tool"] in LOCAL_TOOLS for t in tasks)
    needs_library = any(t["tool"] in ANALYSIS_TOOLS or t["tool"] in EXTERNAL_TOOLS for t in tasks)
    if not has_local and needs_library:
        root_id = _fresh_id(tasks)
        roots = [t for t in tasks if not t["depends_on"]]
        tasks.insert(0, {
            "id": root_id,
            "action": f"先在本地文献库中检索与「{goal[:40]}」相关的已有文献",
            "tool": "hybrid_search" if "hybrid_search" in allowed else NO_TOOL,
            "depends_on": [],
            "output": "库内相关证据卡列表 doc_id#block_id（供后续分析/写作复用）",
        })
        for t in roots:
            t["depends_on"].append(root_id)
        notes.append(f"按 R1 插入本地检索根节点 {root_id}")

    # ── R2：外部任务必须依赖某个本地任务 ──
    local_ids = [t["id"] for t in tasks if t["tool"] in LOCAL_TOOLS]
    if local_ids:
        for t in tasks:
            if t["tool"] in EXTERNAL_TOOLS and not (set(t["depends_on"]) & set(local_ids)):
                first_local = next((lid for lid in local_ids if lid != t["id"]), None)
                if first_local:
                    t["depends_on"].append(first_local)
                    notes.append(f"按 R2 让 {t['id']}（外部检索）依赖本地任务 {first_local}")

    tasks = _topo_order(tasks)
    return {"goal": goal, "tasks": tasks, "notes": notes}


def _fresh_id(tasks: list[dict[str, Any]]) -> str:
    used = {t["id"] for t in tasks}
    i = 0
    while f"t{i}" in used:
        i += 1
    return f"t{i}"


# ═══════════════════════════════ 启发式（离线/降级）═══════════════════════════════
def heuristic_plan(goal: str, tools: list[dict[str, str]]) -> dict[str, Any]:
    """规则模板规划器：零依赖、可断言，用于无 Key 场景与模型失败兜底。"""
    allowed = {t["name"] for t in tools} | {NO_TOOL}
    g = (goal or "").lower()

    def pick(*names: str) -> str:
        for n in names:
            if n in allowed:
                return n
        return NO_TOOL

    def has(*kw: str) -> bool:
        return any(k in g for k in kw)

    if has("扫描", "ocr", "图片"):
        tasks = [
            {"id": "t0", "action": f"解析并做质量分级：{goal[:40]}", "tool": pick("parse_document"),
             "depends_on": [], "output": "DocIR 版式块 + 是否需 OCR 的判定"},
            {"id": "t1", "action": "对扫描页执行 OCR 与版面恢复", "tool": pick("extract_scanned_ocr"),
             "depends_on": ["t0"], "output": "OCR 文本 + 低置信块清单"},
            {"id": "t2", "action": "对解析结果做去重与版本链判定", "tool": pick("semantic_dedup"),
             "depends_on": ["t1"], "output": "去重分组 canonical/versions/diff"},
        ]
    elif has("去重", "重复", "同一篇", "v1", "v7"):
        tasks = [
            {"id": "t0", "action": "在本地文献库中检索目标论文及其历史版本", "tool": pick("hybrid_search"),
             "depends_on": [], "output": "库内候选 doc_id 与版本信息"},
            {"id": "t1", "action": "执行三级去重与版本链判定", "tool": pick("semantic_dedup"),
             "depends_on": ["t0"], "output": "clusters/canonical/version diff"},
        ]
    elif has("翻译", "translate", "中英", "译文"):
        tasks = [
            {"id": "t0", "action": "定位待翻译文献的语义块（含 bbox 锚点）", "tool": pick("hybrid_search", "parse_document"),
             "depends_on": [], "output": "block_id + page + bbox 列表"},
            {"id": "t1", "action": "按 block 翻译并保持术语表与版式锚点", "tool": pick("translate_block"),
             "depends_on": ["t0"], "output": "译文与原文对齐映射（block_id 级）"},
        ]
    elif has("引用图谱", "引用网络", "共引", "citation", "主路径", "文献脉络"):
        tasks = [
            {"id": "t0", "action": "本地检索种子文献", "tool": pick("hybrid_search"),
             "depends_on": [], "output": "种子 doc_id 列表"},
            {"id": "t1", "action": "补充外部引用关系（本地不足时）", "tool": pick("fetch_paper"),
             "depends_on": ["t0"], "output": "外部引文边与元数据，补充外部证据"},
            {"id": "t2", "action": "构建引用异构图并识别社区与主路径", "tool": pick("build_citation_graph"),
             "depends_on": ["t0", "t1"], "output": "nodes/edges/communities/main_path"},
        ]
    elif has("图", "柱状", "折线", "可视化"):
        tasks = [
            {"id": "t0", "action": "本地检索含目标数据的表格与结论", "tool": pick("hybrid_search"),
             "depends_on": [], "output": "数据表与数值块 block_id"},
            {"id": "t1", "action": "生成图表规范", "tool": pick("render_chart"),
             "depends_on": ["t0"], "output": "Vega-Lite spec（含数据来源锚点）"},
        ]
    elif has("综述", "survey", "调研报告", "写成"):
        tasks = [
            {"id": "t0", "action": f"本地文献库检索：{goal[:40]}", "tool": pick("hybrid_search"),
             "depends_on": [], "output": "库内证据卡（优先复用已有文献）"},
            {"id": "t1", "action": "外部补充最新进展（本地不足时）", "tool": pick("fetch_paper"),
             "depends_on": ["t0"], "output": "外部候选论文，标注补充外部证据"},
            {"id": "t2", "action": "汇总对比并梳理脉络", "tool": pick("compare_claims", "build_citation_graph"),
             "depends_on": ["t0", "t1"], "output": "对比矩阵/引用脉络 + 引用锚点"},
            {"id": "t3", "action": "按大纲撰写综述各章节", "tool": pick("write_section"),
             "depends_on": ["t2"], "output": "带 [[ev:doc#block]] 的 Markdown 章节"},
            {"id": "t4", "action": "导出 LaTeX 与 BibTeX", "tool": pick("export_latex"),
             "depends_on": ["t3"], "output": "可编译 .tex + .bib"},
        ]
    elif has("对比", "区别", "差异", "versus", "vs"):
        tasks = [
            {"id": "t0", "action": "本地检索对比对象 A 的证据", "tool": pick("hybrid_search"),
             "depends_on": [], "output": "A 侧证据卡"},
            {"id": "t1", "action": "本地检索对比对象 B 的证据（与 t0 并行）", "tool": pick("hybrid_search"),
             "depends_on": [], "output": "B 侧证据卡"},
            {"id": "t2", "action": "多维度对照分析", "tool": pick("compare_claims"),
             "depends_on": ["t0", "t1"], "output": "对比矩阵（每格带引用锚点）"},
            {"id": "t3", "action": "整理为结论性文字", "tool": pick("write_section", NO_TOOL),
             "depends_on": ["t2"], "output": "带引用的对比结论"},
        ]
    elif has("解析", "上传", "版面"):
        tasks = [
            {"id": "t0", "action": f"解析文档：{goal[:40]}", "tool": pick("parse_document"),
             "depends_on": [], "output": "DocIR 版式块与阅读序"},
            {"id": "t1", "action": "入库并与已有文献去重", "tool": pick("semantic_dedup"),
             "depends_on": ["t0"], "output": "去重结果与 canonical 文档"},
        ]
    else:  # 默认：本地检索 → 回答
        tasks = [
            {"id": "t0", "action": f"本地文献库检索：{goal[:40]}", "tool": pick("hybrid_search"),
             "depends_on": [], "output": "库内证据卡列表"},
            {"id": "t1", "action": "基于检索证据组织回答", "tool": NO_TOOL,
             "depends_on": ["t0"], "output": "带 [[ev:doc#block]] 锚点的结论"},
        ]

    tasks = [t for t in tasks if t["tool"] in allowed]
    tasks = _topo_order(tasks)
    return {"goal": goal, "tasks": tasks, "notes": ["启发式模板生成（模型不可用或未配置 Key）"]}


# ═══════════════════════════════ 主函数 ═══════════════════════════════
def plan(
    goal: str,
    tool_list: Any = None,
    *,
    client: Any | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    strategy: str = "auto",
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """生成可执行子任务 DAG。

    strategy: json_object | function_call | auto（失败逐级降级，最终回启发式模板）
    无 API Key 时直接走 heuristic_plan()，保证永远返回合法 JSON。
    """
    goal = (goal or "").strip()
    tools = normalize_tools(tool_list)
    if not goal:
        return {"goal": "", "tasks": [], "notes": ["目标为空"]}

    model = model or os.getenv("LLM_MODEL", "qwen-plus")
    base_url = base_url or os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        return heuristic_plan(goal, tools)

    if client is None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("需要安装 openai：pip install openai>=1.40") from e
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    messages = build_messages(goal, tools)
    attempts = [strategy] if strategy != "auto" else ["json_object", "function_call"]

    for mode in attempts:
        try:
            if mode == "json_object":
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    response_format={"type": "json_object"},
                )
                raw = _loads(resp.choices[0].message.content or "")
                if raw is not None:
                    return validate_and_repair(raw, goal, tools)
                continue
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                tools=[{"type": "function", "function": {
                    "name": "submit_plan",
                    "description": "提交子任务 DAG",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string"},
                            "tasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "action": {"type": "string"},
                                        "tool": {"type": "string"},
                                        "depends_on": {"type": "array", "items": {"type": "string"}},
                                        "output": {"type": "string"},
                                    },
                                    "required": ["id", "action", "tool", "depends_on", "output"],
                                },
                            },
                        },
                        "required": ["goal", "tasks"],
                    },
                }}],
                tool_choice={"type": "function", "function": {"name": "submit_plan"}},
            )
            calls = resp.choices[0].message.tool_calls or []
            if calls:
                raw = _loads(calls[0].function.arguments or "")
                if raw is not None:
                    return validate_and_repair(raw, goal, tools)
        except Exception:  # noqa: BLE001 —— 降级而非抛炸
            continue

    return heuristic_plan(goal, tools)


def _loads(text: str) -> Any:
    for candidate in ((text or "").strip(), _strip_fence(text)):
        try:
            return json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None
    return None


# ═══════════════════════════════ 与 LangGraph 执行器的桥接 ═══════════════════════════════
def to_subtasks(p: dict[str, Any], thread_id: str = "default") -> list[dict[str, Any]]:
    """转成 app.schemas.SubTask 兼容的字典，可直接喂给 executor_node 的 DAG。"""
    out: list[dict[str, Any]] = []
    for t in p.get("tasks", []):
        out.append({
            "id": t["id"],
            "title": t["action"][:60],
            "agent": TOOL_TO_AGENT.get(t["tool"], "rag"),
            "description": t["action"],
            "inputs": {"tool": t["tool"], "expected_output": t["output"], "thread_id": thread_id},
            "depends_on": list(t["depends_on"]),
            "tool_hints": [t["tool"]] if t["tool"] != NO_TOOL else [],
            "status": "pending",
            "result": None,
            "retry": 0,
            "max_retry": 1,
        })
    return out


def describe(p: dict[str, Any]) -> str:
    """把 DAG 画成可读文本（答辩/日志用）。"""
    lines = [f"目标：{p.get('goal', '')}"]
    by_id = {t["id"]: t for t in p.get("tasks", [])}
    for t in p.get("tasks", []):
        deps = ",".join(t["depends_on"]) or "-"
        lines.append(f"  {t['id']:<4} [{t['tool']:<22}] deps={deps:<8} {t['action']}")
        lines.append(f"       └ 产出：{t['output']}")
    for n in p.get("notes", []):
        lines.append(f"  * {n}")
    _ = by_id
    return "\n".join(lines)


# ═══════════════════════════════ 自测 ═══════════════════════════════
SELFTEST: list[tuple[str, str]] = [
    ("对比 Transformer 与 BERT 并生成综述", "compare_claims"),
    ("生成一份关于高效 Transformer 的综述", "export_latex"),
    ("帮我解析 data/scan.pdf 的扫描件并去重", "extract_scanned_ocr"),
    ("arXiv:1706.03762v1 和 v7 是不是同一篇", "semantic_dedup"),
    ("把 doc_bert 第 3 段翻译成中文", "translate_block"),
    ("画出 doc_transformer 的引用网络", "build_citation_graph"),
    ("把表 2 数据画成柱状图", "render_chart"),
    ("Transformer 的注意力复杂度是多少", "hybrid_search"),
]


def _selftest() -> int:
    passed = 0
    for goal, expect_tool in SELFTEST:
        p = plan(goal)
        tasks = p["tasks"]
        ids = {t["id"] for t in tasks}
        tools = [t["tool"] for t in tasks]
        ok_tool = expect_tool in tools
        ok_deps = all(d in ids for t in tasks for d in t["depends_on"])
        ok_local_first = not any(
            t["tool"] in EXTERNAL_TOOLS and not (set(t["depends_on"]) & {x["id"] for x in tasks if x["tool"] in LOCAL_TOOLS})
            for t in tasks
        )
        ok = ok_tool and ok_deps and ok_local_first and 2 <= len(tasks) <= MAX_TASKS
        passed += ok
        print(f"[{'OK' if ok else 'XX'}] {goal[:28]:<30} tasks={len(tasks)} tools={tools}")
        if not ok:
            print(f"      tool={ok_tool} deps={ok_deps} local_first={ok_local_first}")
    print(f"\n通过 {passed}/{len(SELFTEST)}（离线启发式路径）")
    return 0 if passed == len(SELFTEST) else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="科研助手任务规划器")
    ap.add_argument("--goal", "-g", help="用户目标")
    ap.add_argument("--tools", "-t", default="", help="逗号分隔的工具名（默认用内置 11 个）")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    ap.add_argument("--model", default=None)
    ap.add_argument("--strategy", default="auto", choices=["auto", "json_object", "function_call"])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest or not args.goal:
        raise SystemExit(_selftest())

    tool_list = [x.strip() for x in args.tools.split(",") if x.strip()] or None
    p = plan(args.goal, tool_list, model=args.model, strategy=args.strategy)
    if args.json:
        print(json.dumps({"goal": p["goal"], "tasks": p["tasks"]}, ensure_ascii=False, indent=2))
    else:
        print(describe(p))
        print()
        print(json.dumps({"goal": p["goal"], "tasks": p["tasks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
