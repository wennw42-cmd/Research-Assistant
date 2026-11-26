"""学术 RAG 检索主链路：改写 → 双路召回 → RRF 融合 → Rerank → 带溯源输出。

    query
      │ plan_query()          意图 / 改写 / 元数据过滤
      ├── dense  : 每条改写 → embedding → 向量库 ANN（带 filter）
      └── sparse : 每条改写 → BM25（先筛后打分）
           │
        RRF 融合（每条改写 × 每一路 = 一条 run，名次加权求和）
           │
        Top-N → Rerank（cross-encoder / 启发式）
           │
        Top-K 输出（含 doc_id / section_path / 页码 / block_ids，可直接做引用卡）

两个工程细节，都是踩过的坑：
1. **过滤要有退路**：严格过滤（如 hyperparams∩[batch_size]）可能把整篇文档的候选清空，
   因此召回数低于 `min_hits` 时自动降级为"小节类型软过滤"，并在 trace 里标注；
2. **每条改写是独立的 run**，而不是把改写拼成一个长 query 再召回——
   后者会让长尾改写淹没主查询的信号。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.bm25 import BM25Index
from app.retrieval.embed import Embedder, HashEmbedder, build_embedder
from app.retrieval.fusion import rrf, weighted_score
from app.retrieval.query import QueryPlan, plan_query
from app.retrieval.rerank import RerankContext, Reranker, build_reranker
from app.retrieval.stores import InMemoryVectorStore, VectorStore, build_store
from app.retrieval.types import Chunk, Filter, FusedHit, Hit


@dataclass
class RetrievalConfig:
    """参数建议（默认值为学术 RAG 的常用起点，不是拍脑袋）：

    dense_top_k / sparse_top_k = 50   : 召回要宽。RRF 之后只留前几条，召回窄了再好的 rerank 也救不回来
    fusion_top_k = 40                 : 送进 rerank 的条数。cross-encoder 逐对打分，40 条是延迟/效果的平衡点
    final_top_k = 5                   : 进 LLM 上下文的条数，约 2.5k token
    rrf_k = 60                        : Cormack 2009 推荐值；k∈[20,100] 不敏感
    min_hits = 3                      : 过滤后的召回下限，低于则降级过滤
    """

    dense_top_k: int = 50
    sparse_top_k: int = 50
    fusion_top_k: int = 40
    rerank_top_k: int = 8
    final_top_k: int = 5
    rrf_k: int = 60
    weights: dict[str, float] = field(default_factory=lambda: {"dense": 1.0, "sparse": 1.0})
    min_hits: int = 3
    use_rewrite: bool = True
    use_filter: bool = True
    use_rerank: bool = True
    fusion: str = "rrf"                 # rrf | weighted（weighted 仅作对照/消融）
    embed_batch: int = 64
    snippet_chars: int = 320


class AcademicRetrievalPipeline:
    def __init__(self, chunks: list[Chunk], *, store: VectorStore | None = None,
                 embedder: Embedder | None = None, reranker: Reranker | None = None,
                 config: RetrievalConfig | None = None) -> None:
        self.chunks = list(chunks)
        self.config = config or RetrievalConfig()
        self.store: VectorStore = store or InMemoryVectorStore()
        self.embedder: Embedder = embedder or build_embedder("auto")
        self.reranker: Reranker | None = reranker if reranker is not None else build_reranker("auto")
        self.bm25 = BM25Index()
        self.built = False

    # ── 建库 ──
    def build(self, *, recreate: bool = True) -> "AcademicRetrievalPipeline":
        """recreate=True 会清空同名集合；接生产库时传 False（只 upsert，不动既有数据）。"""
        if not self.chunks:
            raise ValueError("没有可入库的 chunk")
        self.bm25.build(self.chunks)
        vecs: list[list[float]] = []
        for i in range(0, len(self.chunks), self.config.embed_batch):
            batch = self.chunks[i:i + self.config.embed_batch]
            vecs += self.embedder.embed([c.text for c in batch])
        dim = len(vecs[0]) if vecs else 0
        self.store.create_collection(dim, recreate=recreate)
        self.store.upsert(self.chunks, vecs)
        self.built = True
        return self

    # ── 检索 ──
    def search(self, query: str, *, top_k: int | None = None,
               plan: QueryPlan | None = None) -> dict[str, Any]:
        if not self.built:
            raise RuntimeError("请先调用 build()")
        cfg = self.config
        final_k = top_k or cfg.final_top_k
        t0 = time.perf_counter()

        plan = plan or plan_query(query, enable_rewrite=cfg.use_rewrite, enable_filter=cfg.use_filter)
        flt = plan.filters if cfg.use_filter else Filter()

        # 每条改写只 embed 一次：过滤降级会重跑召回，重复调远端 embedding 是纯浪费
        qvecs = {rw: self.embedder.embed_one(rw) for rw in plan.rewrites}
        runs: dict[str, list[Hit]] = {}
        for i, rw in enumerate(plan.rewrites):
            runs[f"dense:{i}"] = self.store.search(qvecs[rw], top_k=cfg.dense_top_k, flt=flt)
            runs[f"sparse:{i}"] = self.bm25.search(rw, top_k=cfg.sparse_top_k, flt=flt)

        runs_nonempty = {k: v for k, v in runs.items() if v}
        weights = {name: cfg.weights.get(name.split(":")[0], 1.0) for name in runs_nonempty}
        fused = (rrf(runs_nonempty, k=cfg.rrf_k, weights=weights) if cfg.fusion == "rrf"
                 else weighted_score(runs_nonempty, weights=weights))

        # 过滤降级：严格过滤召回不足 → 退到小节软过滤，绝不返回空结果
        relaxed = False
        if len(fused) < cfg.min_hits and not flt.is_empty():
            soft = plan.relaxed()
            if soft.describe() != flt.describe():
                relaxed = True
                runs2: dict[str, list[Hit]] = {}
                for i, rw in enumerate(plan.rewrites):
                    runs2[f"dense:{i}"] = self.store.search(qvecs[rw], top_k=cfg.dense_top_k, flt=soft)
                    runs2[f"sparse:{i}"] = self.bm25.search(rw, top_k=cfg.sparse_top_k, flt=soft)
                runs_nonempty = {k: v for k, v in runs2.items() if v}
                w2 = {n: cfg.weights.get(n.split(":")[0], 1.0) for n in runs_nonempty}
                fused = (rrf(runs_nonempty, k=cfg.rrf_k, weights=w2) if cfg.fusion == "rrf"
                         else weighted_score(runs_nonempty, weights=w2))
                flt = soft

        cands = fused[:cfg.fusion_top_k]
        ctx = RerankContext(query=query, intent=plan.intent, entities=plan.entities,
                            numbers=plan.numbers, preferred_sections=plan.preferred_sections)
        if cfg.use_rerank and self.reranker is not None:
            ranked = self.reranker.rerank(cands, ctx, top_k=cfg.rerank_top_k)
        else:
            ranked = [FusedHit_to_ranked(f) for f in cands[:cfg.rerank_top_k]]

        hits = [{
            "rank": i + 1,
            "id": h.chunk.id,
            "score": h.score,
            "rrf": round(h.rrf, 6),
            "ranks": h.ranks,
            "why": h.why,
            "doc_id": h.chunk.doc_id,
            "title": h.chunk.title,
            "section_path": h.chunk.section_path,
            "page_start": h.chunk.page_start,
            "page_end": h.chunk.page_end,
            "block_ids": h.chunk.block_ids,
            "section_type": h.chunk.metadata.get("section_type", ""),
            "text": h.chunk.text[:cfg.snippet_chars],
        } for i, h in enumerate(ranked[:final_k])]

        return {
            "query": query,
            "plan": plan.to_dict(),
            "engines": {
                "embedder": self.embedder.name,
                "vector_store": self.store.name,
                "sparse": "bm25(k1=1.5,b=0.75)",
                "fusion": f"{cfg.fusion}(k={cfg.rrf_k})" if cfg.fusion == "rrf" else "weighted-minmax",
                "reranker": self.reranker.name if (cfg.use_rerank and self.reranker) else "none",
            },
            "stages": {
                "rewrites": len(plan.rewrites),
                "runs": len(runs_nonempty),
                "dense_hits": max((len(v) for k, v in runs_nonempty.items() if k.startswith("dense")), default=0),
                "sparse_hits": max((len(v) for k, v in runs_nonempty.items() if k.startswith("sparse")), default=0),
                "fused": len(fused),
                "reranked": len(ranked),
                "filter": flt.describe(),
                "filter_relaxed": relaxed,
            },
            "hits": hits,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }


def FusedHit_to_ranked(f: FusedHit):
    """rerank 关闭时的兜底：直接把融合结果当最终结果（保留 why 字段结构一致）。"""
    from app.retrieval.rerank import RerankedHit

    return RerankedHit(chunk=f.chunk, score=round(f.rrf, 6), features={"rrf": round(f.rrf, 6)},
                       rrf=f.rrf, ranks=dict(f.ranks))
