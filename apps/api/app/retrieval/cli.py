"""学术 RAG 混合检索 CLI：自测 + 演示。

    cd apps/api
    python -m app.retrieval.cli --selftest                 # 断言核心不变量（零依赖）
    python -m app.retrieval.cli                            # 跑内置 4 个示例问题
    python -m app.retrieval.cli --query "用了多少张 A100？" --top-k 3
    python -m app.retrieval.cli --params                   # 参数建议表
    python -m app.retrieval.cli --store qdrant --no-recreate   # 接真实向量库（不重建集合）
    python -m app.retrieval.cli --corpus rag-index --index idx/  # 接 app/rag 入库产物

真实模型：配好 EMBEDDING_* / RERANK_* 后自动启用；没配就用确定性哈希 + 启发式重排，
两条路都能跑通，只是能力不同（`engines` 字段会如实标注）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):                       # 允许 python app/retrieval/cli.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.retrieval.adapters import chunks_from_corpus, chunks_from_dicts, chunks_from_rag_index
from app.retrieval.embed import build_embedder
from app.retrieval.fusion import rrf
from app.retrieval.pipeline import AcademicRetrievalPipeline, RetrievalConfig
from app.retrieval.query import plan_query
from app.retrieval.rerank import HeuristicReranker, RerankContext, build_reranker, extract_numbers
from app.retrieval.sample_corpus import build_sample_chunks
from app.retrieval.stores import InMemoryVectorStore, build_store
from app.retrieval.types import Chunk, Filter, Hit

DEMO_QUERIES = [
    "LLM 微调用了多少张 A100，学习率和 batch size 是多少？",
    "Conformer 语音识别用了什么显卡，显存占用多少？",
    "ViT 在 CIFAR-100 上的准确率是多少？",
    "ASR 实验里 warmup 设成多少步，训练多少个 epoch？",
]

PARAMS = """
参数建议（学术 RAG，默认值来自 RetrievalConfig）
┌──────────────────┬─────────┬───────────────────────────────────────────────────────────┐
│ 参数             │ 默认    │ 取值理由                                                   │
├──────────────────┼─────────┼───────────────────────────────────────────────────────────┤
│ dense_top_k      │ 50      │ 召回要宽。RRF 后只留前几条，召回窄了 rerank 救不回来       │
│ sparse_top_k     │ 50      │ 同上；BM25 对型号/数值/稀有符号是主信号                    │
│ fusion_top_k     │ 40      │ 送进 cross-encoder 的条数；40 是延迟/效果的平衡点          │
│ rerank_top_k     │ 8       │ 重排后保留的候选，再按 final_top_k 截断                    │
│ final_top_k      │ 5       │ 进 LLM 上下文，约 2.5k token                               │
│ rrf_k            │ 60      │ Cormack 2009 推荐；k∈[20,100] 不敏感，换 k 不如换 top_k    │
│ min_hits         │ 3       │ 严格过滤后低于此值自动降级为小节软过滤                     │
│ weights          │ 1:1     │ dense/sparse 等权。领域术语多 → 提到 1.0:1.2               │
└──────────────────┴─────────┴───────────────────────────────────────────────────────────┘
调参顺序：先扩 dense_top_k/sparse_top_k 看召回是否够 → 再调 rrf_k → 最后才动 weights。
"""


# ────────────────────────── 构件工厂 ──────────────────────────
def build_chunks(corpus: str, index_dir: str = "") -> list[Chunk]:
    if corpus == "sample":
        return build_sample_chunks()
    if corpus == "agents":
        return chunks_from_corpus()
    if corpus == "rag-index":
        if not index_dir:
            raise SystemExit("--corpus rag-index 需要 --index <ingest 输出目录>")
        return chunks_from_rag_index(index_dir)
    raise SystemExit(f"未知语料 {corpus}，可选：sample | agents | rag-index")


def build_pipeline(chunks: list[Chunk], args: argparse.Namespace) -> AcademicRetrievalPipeline:
    store_kwargs = {
        "memory": lambda: {"collection": "chunks"},
        "qdrant": lambda: {"url": os.getenv("QDRANT_URL", "http://localhost:6333"),
                           "api_key": os.getenv("QDRANT_API_KEY", "")},
        "chroma": lambda: {"path": os.getenv("CHROMA_PATH", "./chroma_db")},
        "pgvector": lambda: {"dsn": os.getenv("PGVECTOR_DSN", "")},
    }
    if args.store not in store_kwargs:
        raise SystemExit(f"未知向量库 {args.store}，可选：{sorted(store_kwargs)}")
    cfg = RetrievalConfig(final_top_k=args.top_k,
                          use_rerank=not args.no_rerank,
                          use_filter=not args.no_filter)
    kwargs: dict = {"config": cfg}
    if args.store != "memory":
        kwargs["store"] = build_store(args.store, **store_kwargs[args.store]())
    if args.embedder:
        kwargs["embedder"] = build_embedder(args.embedder)
    if args.reranker:
        kwargs["reranker"] = build_reranker(args.reranker)
    pipe = AcademicRetrievalPipeline(chunks, **kwargs)
    pipe.build(recreate=not args.no_recreate)
    return pipe


def print_result(res: dict) -> None:
    plan = res["plan"]
    print("\n" + "═" * 84)
    print(f"  Q: {res['query']}")
    print("═" * 84)
    print(f"  意图      : {plan['intent']}   实体: {plan['entities']}   数值: {plan['numbers']}")
    print(f"  改写      : {len(plan['rewrites'])} 条")
    for i, rw in enumerate(plan["rewrites"], 1):
        print(f"      [{i}] {rw[:96]}")
    print(f"  过滤      : {res['stages']['filter']}")
    if res["stages"]["filter_relaxed"]:
        print("      ↳ 严格过滤召回不足，已自动降级为小节软过滤")
    print(f"  引擎      : {res['engines']}")
    print(f"  各阶段    : 改写{res['stages']['rewrites']} · 召回run{res['stages']['runs']} · "
          f"dense{res['stages']['dense_hits']} · sparse{res['stages']['sparse_hits']} · "
          f"融合{res['stages']['fused']} · 重排{res['stages']['reranked']}")
    print(f"  耗时      : {res['elapsed_ms']}ms")
    for h in res["hits"]:
        print(f"\n  #{h['rank']} [{h['id']}] {h['doc_id']} | {h['section_path']}")
        print(f"      p.{h['page_start']}-{h['page_end']} · section={h['section_type']} · "
              f"rrf={h['rrf']} · ranks={h['ranks']}")
        print(f"      why: {h['why']}")
        print(f"      {h['text'][:150]}…")


# ────────────────────────── 自测 ──────────────────────────
def _selftest() -> int:
    chunks = build_sample_chunks()
    assert len(chunks) == 18, f"样例语料应为 18 块，实际 {len(chunks)}"
    assert all(c.metadata.get("section_type") for c in chunks), "每个 chunk 必须有 section_type"

    # 1) 数值抽取：型号不能当数值（A100 ≠ 100）
    assert "100" not in extract_numbers("A100 80GB 训练 8 卡"), extract_numbers("A100 80GB 训练 8 卡")
    assert "80gb" in extract_numbers("A100 80GB 训练 8 卡")

    # 2) 过滤契约：三种后端翻译不报错，且与内存匹配语义一致
    flt = Filter(any_of={"section_type": ["experimental_setup"]},
                 contains_any={"hardware": ["A100"]}, ranges={"n_gpus": (8, 8)})
    assert flt.is_empty() is False
    assert flt.to_chroma() and flt.to_qdrant_spec() and flt.to_sql()[0]
    hit = [c for c in chunks if flt.matches(c.payload())]
    assert [c.id for c in hit] == ["llm:setup"], [c.id for c in hit]

    # 3) 双路召回都非空，且 RRF 融合可复现
    pipe = AcademicRetrievalPipeline(chunks).build()
    q = DEMO_QUERIES[0]
    plan = plan_query(q)
    runs: dict[str, list[Hit]] = {}
    for i, rw in enumerate(plan.rewrites):
        v = pipe.embedder.embed_one(rw)
        runs[f"dense:{i}"] = pipe.store.search(v, top_k=50, flt=plan.filters)
        runs[f"sparse:{i}"] = pipe.bm25.search(rw, top_k=50, flt=plan.filters)
    assert runs["sparse:0"], "BM25 路必须有召回"
    assert runs["dense:0"], "Dense 路必须有召回"
    fused_a = rrf({k: v for k, v in runs.items() if v}, k=60)
    fused_b = rrf({k: v for k, v in runs.items() if v}, k=60)
    assert [(f.chunk.id, f.rrf) for f in fused_a] == [(f.chunk.id, f.rrf) for f in fused_b], "融合必须可复现"
    assert all(f.rrf > 0 for f in fused_a)

    # 4) 端到端：超参/硬件问题必须命中正确的实验设置块
    res = pipe.search(q)
    assert res["hits"], "端到端不得返回空"
    assert res["hits"][0]["id"] == "llm:setup", res["hits"][0]["id"]
    assert res["stages"]["filter_relaxed"] is True, "严格过滤只剩 1 条，应触发降级"

    # 5) 重排：特征不为零（run 名带改写序号，取分要按前缀聚合），分数单调不增
    runs_all = {**{f"dense:{i}": pipe.store.search(pipe.embedder.embed_one(rw), top_k=50)
                   for i, rw in enumerate(plan.rewrites)},
                **{f"sparse:{i}": pipe.bm25.search(rw, top_k=50)
                   for i, rw in enumerate(plan.rewrites)}}
    fused_all = rrf({k: v for k, v in runs_all.items() if v}, k=60)
    assert len(fused_all) >= 10
    ranked = HeuristicReranker().rerank(fused_all[:10], RerankContext(
        query=q, intent="hyperparameter", entities=["A100"], numbers=[],
        preferred_sections=["experimental_setup"]), top_k=5)
    assert len(ranked) == 5
    assert ranked[0].features["dense"] > 0 or ranked[0].features["bm25"] > 0, ranked[0].features
    assert all(a.score >= b.score for a, b in zip(ranked, ranked[1:])), "重排必须单调"

    # 6) 向量库抽象：覆盖写不产生重复
    store = InMemoryVectorStore()
    store.create_collection(4, recreate=True)
    c = Chunk(id="x", text="a")
    store.upsert([c], [[1.0, 0.0, 0.0, 0.0]])
    store.upsert([c], [[0.0, 1.0, 0.0, 0.0]])
    assert store.count() == 1, "同 id 覆盖写后数量应仍为 1"
    assert store.search([0.0, 1.0, 0.0, 0.0], top_k=1)[0].score == 1.0

    # 7) 通用适配入口
    got = chunks_from_dicts([{"id": "d1", "text": "hello", "section_path": "Experimental Setup"}])
    assert got[0].metadata["section_type"] == "experimental_setup"

    print("selftest: 7/7 通过")
    print(f"  语料 {len(chunks)} 块 · 引擎 embedder={pipe.embedder.name} "
          f"store={pipe.store.name} reranker={pipe.reranker.name}")
    print(f"  示例: {q} → {res['hits'][0]['id']} "
          f"(dense {res['engines']['vector_store']}, {res['elapsed_ms']}ms)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="学术 RAG 混合检索（Dense+BM25→RRF→Rerank）")
    ap.add_argument("--query", "-q", default="", help="单条查询；不传则跑内置示例")
    ap.add_argument("--top-k", type=int, default=5, help="最终返回条数（默认 5）")
    ap.add_argument("--corpus", default="sample", choices=["sample", "agents", "rag-index"])
    ap.add_argument("--index", default="", help="rag-index 语料的 ingest 输出目录")
    ap.add_argument("--store", default="memory", choices=["memory", "qdrant", "chroma", "pgvector"])
    ap.add_argument("--no-recreate", action="store_true", help="不重建集合（接真实库时必加）")
    ap.add_argument("--embedder", default="", choices=["", "auto", "hash", "openai"])
    ap.add_argument("--reranker", default="", choices=["", "auto", "heuristic", "cross-encoder"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--json", action="store_true", help="输出 JSON（给上游 Agent 用）")
    ap.add_argument("--params", action="store_true", help="打印参数建议表")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())
    if args.params:
        print(PARAMS)
        return

    chunks = build_chunks(args.corpus, args.index)
    pipe = build_pipeline(chunks, args)
    queries = [args.query] if args.query else DEMO_QUERIES
    for q in queries:
        res = pipe.search(q)
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            print_result(res)


if __name__ == "__main__":
    main()
