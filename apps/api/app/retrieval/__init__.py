"""学术 RAG 混合检索子系统（Dense + BM25 → RRF → Rerank）。

    app/retrieval/types.py      Chunk / Filter / Hit 数据契约（过滤条件与后端无关）
    app/retrieval/stores.py     向量库抽象 + memory / qdrant / chroma / pgvector
    app/retrieval/embed.py      Embedding 抽象（远端 OpenAI 兼容 / 离线哈希兜底）
    app/retrieval/bm25.py       稀疏检索（中英混排分词，先筛后打分）
    app/retrieval/fusion.py     RRF 与加权融合
    app/retrieval/query.py      意图识别 / 查询改写 / 元数据过滤
    app/retrieval/rerank.py     启发式重排 + cross-encoder 重排
    app/retrieval/pipeline.py   主链路（含过滤降级）
    app/retrieval/adapters.py   从 app/rag 入库产物 / 离线语料 / dict 构建 chunk
    app/retrieval/cli.py        CLI + 自测 + 演示
"""

__all__ = ["pipeline", "stores", "query", "rerank", "bm25", "fusion", "embed", "adapters", "types"]
