"""M2 RAG 入库与长上下文加载子系统。

    app/rag/params.py     切块参数与 token 估算
    app/rag/memory.py     内存/显存监控与阶段计量
    app/rag/store.py      JSONL 落盘 + 偏移索引 + LRU 缓存淘汰
    app/rag/chunker.py    句子切分 / 流式打包 / 抽取式 lead
    app/rag/ingest.py     541 页级流式入库（两阶段 + 磁盘换内存）
    app/rag/retrieval.py  摘要层 → 章节层 → 细节层 的分级加载
    app/rag/cli.py        CLI + 自测 + 合成教材生成
"""
