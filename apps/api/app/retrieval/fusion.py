"""多路召回融合：RRF（默认）+ 归一化加权（对照/消融用）。

为什么用 RRF 而不是"加权求和分数"：
- Dense 与 Sparse 的**分数尺度不可比**（余弦 0~1 vs BM25 0~30），
  加权求和必须先归一化，而归一化对 top-k 截断很敏感；
- RRF 只用**名次**：`score = Σ w_i / (k + rank_i)`，与量纲无关，也天然抑制"某一路分数膨胀"。

k 的取值：60 是原始论文（Cormack et al. 2009）的推荐值，也是多数开源实现的默认。
k 越小 → 越强调头部名次（top-1 权重差距更大）；k 越大 → 越接近"出现即计数"。
实践中 k∈[20,100] 影响不大，**换 k 不如换 top_k 有效**。
"""
from __future__ import annotations

from app.retrieval.types import FusedHit, Hit

RRF_K = 60


def rrf(runs: dict[str, list[Hit]], *, k: int = RRF_K,
        weights: dict[str, float] | None = None) -> list[FusedHit]:
    """倒数排名融合。runs = {"dense": [Hit...], "sparse": [Hit...]}，rank 从 1 开始。"""
    weights = weights or {}
    merged: dict[str, FusedHit] = {}
    for source, hits in runs.items():
        w = float(weights.get(source, 1.0))
        for h in hits:
            if h.chunk is None:
                continue
            fh = merged.get(h.chunk.id)
            if fh is None:
                fh = merged[h.chunk.id] = FusedHit(chunk=h.chunk)
            fh.rrf += w / (k + h.rank)
            fh.ranks[source] = h.rank
            fh.scores[source] = h.score
    out = list(merged.values())
    # 同分时按"最好名次 → id"排序，保证结果可复现（检索系统最忌讳不可复现）
    out.sort(key=lambda f: (-f.rrf, min(f.ranks.values()), f.chunk.id))
    return out


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def weighted_score(runs: dict[str, list[Hit]], *,
                   weights: dict[str, float] | None = None) -> list[FusedHit]:
    """对照方案：各路分数 min-max 归一化后加权求和（供消融实验对比 RRF）。"""
    weights = weights or {}
    merged: dict[str, FusedHit] = {}
    for source, hits in runs.items():
        w = float(weights.get(source, 1.0))
        if not hits:
            continue
        norms = minmax([h.score for h in hits])
        for h, nv in zip(hits, norms):
            if h.chunk is None:
                continue
            fh = merged.get(h.chunk.id)
            if fh is None:
                fh = merged[h.chunk.id] = FusedHit(chunk=h.chunk)
            fh.rrf += w * nv
            fh.ranks[source] = h.rank
            fh.scores[source] = h.score
    out = list(merged.values())
    out.sort(key=lambda f: (-f.rrf, f.chunk.id))
    return out
