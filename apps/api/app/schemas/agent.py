"""Agent 层数据契约。

设计原则：**一切中间产物都是可序列化、可入库、可回放的结构体**，
EvidenceCard 是整个系统的"货币"，RAG/写作/综述/可视化全部围绕它结算。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# 六类专业子 Agent + 三个元路由
AgentName = Literal[
    "parse",        # 文档解析与语义去重
    "retrieval",    # 论文检索 / 追踪 / 推荐
    "rag",          # 抗幻觉多文问答与推理
    "translation",  # 版式感知双向联动翻译
    "graph",        # 引用图谱挖掘
    "writing",      # 学术写作与数据可视化
]
META_ROUTE = Literal["planner", "clarify", "direct"]
ALL_ROUTES = (
    "parse",
    "retrieval",
    "rag",
    "translation",
    "graph",
    "writing",
    "planner",
    "clarify",
    "direct",
)


# ────────────────────────────── 工具层 ──────────────────────────────
class EvidenceCard(BaseModel):
    """抗幻觉的最小证据单元：必须能回溯到原文字符级位置。"""

    doc_id: str
    block_id: str
    quote: str
    page: int | None = None
    bbox: list[float] | None = None  # [x0, y0, x1, y1]，支撑 UI 高亮跳转
    score: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def cite_key(self) -> str:
        """写作阶段的引用锚点：[[ev:doc_id#block_id]]"""
        return f"{self.doc_id}#{self.block_id}"


class ToolEnvelope(BaseModel):
    """**所有工具的唯一返回结构**。

    禁止工具直接返回裸数据：LLM 需要稳定的信封才能自纠错，
    链路需要 cited[] 才能做证据绑定，观测需要 metrics 才能做性能归因。
    """

    ok: bool
    tool: str
    data: dict[str, Any] = Field(default_factory=dict)
    cited: list[EvidenceCard] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    hint: str | None = None


# ────────────────────────────── 规划层 ──────────────────────────────
class SubTask(BaseModel):
    """DAG 中的一个节点。"""

    id: str
    title: str
    agent: str  # AgentName | "planner"
    description: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    tool_hints: list[str] = Field(default_factory=list)
    status: Literal["pending", "ready", "running", "done", "failed", "skipped"] = "pending"
    result: dict[str, Any] | None = None
    retry: int = 0
    max_retry: int = 1


class PlanDAG(BaseModel):
    """一个可执行的有向无环图。"""

    goal: str
    rationale: str = ""
    subtasks: list[SubTask] = Field(default_factory=list)

    @property
    def roots(self) -> list[SubTask]:
        return [t for t in self.subtasks if not t.depends_on]


class VerifyResult(BaseModel):
    """校验器输出：决定是否放行、是否触发反思。"""

    passed: bool = False
    evidence_coverage: float = 0.0   # 被证据支撑的 claim 比例
    citation_valid: bool = False     # 引用锚点是否都能在库里命中
    quote_faithful: bool = False     # quote 是否字面存在于原文（防编造）
    issues: list[str] = Field(default_factory=list)
    fixed: list[str] = Field(default_factory=list)


# 先声明后引用需要在模块末尾 rebuild（前向引用）
ToolEnvelope.model_rebuild()
