from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """所有外部依赖的配置入口。

    读取顺序：环境变量 > .env 文件 > 默认值。
    生产接入时只需替换 ProviderAdapter，不需要改任何业务代码。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM（任意 OpenAI 兼容接口：DeepSeek / Qwen / GLM / vLLM / Ollama）──
    LLM_PROVIDER: str = "mock"  # mock | openai_compatible
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 4096
    LLM_TIMEOUT: int = 60

    # ── 多模态（阿里云百炼）──
    VISION_MODEL: str = "qwen3-vl-plus"
    IMAGE_MODEL: str = "wanx2.1-t2i-turbo"
    IMAGE_API_BASE: str = "https://dashscope.aliyuncs.com/api/v1"
    IMAGE_TIMEOUT: int = 180

    # ── Embedding / Rerank ──
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_API_KEY: str = ""
    RERANK_MODEL: str = "bge-reranker-v2-m3"
    RERANK_BASE_URL: str = ""

    # ── Agent 运行时 ──
    AGENT_MAX_REVISION: int = 2        # 反思重做上限，防止无限循环
    AGENT_MAX_TOOL_STEPS: int = 4      # 单轮 ReAct 最多工具调用步数
    AGENT_HITL: bool = False           # 是否在 executor 前中断等待人工确认
    RETRIEVAL_TOP_K: int = 6

    # ── 存储 ──
    SQLITE_PATH: str = "data/research_assistant.db"   # 翻译历史；相对路径按 apps/api 解析
    LIBRARY_PATH: str = "data/library"                 # 文献库原文件 / 解析结果 / 分块 / 译文
    RECOMMEND_INTERVAL_HOURS: int = 0                  # >0 时后台定时刷新 arXiv 推荐；0=仅手动
    REDIS_URL: str = "redis://localhost:6379/0"
    POSTGRES_DSN: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/research_assistant"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"

    # ── 外部学术服务 ──
    OPENALEX_MAILTO: str = ""
    CROSSREF_MAILTO: str = ""
    SEMANTIC_SCHOLAR_KEY: str = ""
    GITHUB_TOKEN: str = ""            # 可选：提高 GitHub 检索配额，绝不落盘/打印

    # ── 可观测 ──
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LOG_LEVEL: str = "INFO"

    # ── OCR（Tesseract）──
    OCR_LANG: str = "eng"             # 中文扫描件用 chi_sim+eng
    TESSERACT_CMD: str = ""            # 留空则自动查找 PATH / 默认安装目录
    TESSDATA_PREFIX: str = ""          # 留空则用默认 tessdata；自定义语言包时指向其目录

    @property
    def use_mock_llm(self) -> bool:
        """没配 key 或显式指定 mock 时自动降级，保证代码零密钥可跑。"""
        return self.LLM_PROVIDER.strip().lower() == "mock" or not self.LLM_API_KEY


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
