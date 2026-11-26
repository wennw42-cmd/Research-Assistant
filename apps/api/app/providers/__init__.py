"""外部服务适配层（L5）出口。

当前只有 LLM；生产化时在此平行扩展：
    EmbeddingProvider / RerankProvider / AcademicAPIProvider / OCRProvider
所有 Adapter 必须实现"主 → 备 → 离线快照"三级降级，且不得在业务代码里硬编码厂商。
"""
from app.providers.llm import BaseLLM, ChatResponse, MockLLM, OpenAICompatibleLLM, ToolSpec, get_llm

__all__ = ["BaseLLM", "ChatResponse", "MockLLM", "OpenAICompatibleLLM", "ToolSpec", "get_llm"]
