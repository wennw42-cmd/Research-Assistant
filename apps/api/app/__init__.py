"""Research Assistant · Agent Layer (LangGraph)。

目录约定：
    app/schemas/   —— 状态与数据契约（DocIR / EvidenceCard / SubTask / PlanDAG）
    app/providers/ —— 外部模型服务的 Adapter（可降级为 mock）
    app/memory/    —— 会话记忆 / 文献库上下文 / 任务中间态
    app/agents/    —— Supervisor + 规划 + 六个专业子 Agent + 校验 + 反思
"""
__version__ = "0.1.0"
