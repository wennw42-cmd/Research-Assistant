"""重排（Rerank）：对 RRF 的 Top-N 做精排。

为什么必须有两级：召回要**快而全**（向量 ANN + BM25，允许噪声），
重排要**准而慢**（cross-encoder 逐对打分，只能跑几十条）。
直接用向量相似度当最终排序，学术场景里"方法段落"经常压过"实验设置段落"——
因为它们的措辞更像通用论述，向量上距离问题更近，但答案不在那儿。

本模块两个实现：
    HeuristicReranker    离线兜底：多特征线性加权，权重显式可解释、结果可复现
    CrossEncoderReranker 线上：Cohere 风格 /rerank（BGE-reranker-v2-m3、Qwen3-Reranker 自建服务同构）
"""
from __future__ import annotations

import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.fusion import minmax
from app.retrieval.types import Chunk, FusedHit

# 数值/单位：超参与硬件问题的硬信号。
# 前后边界缺一不可：否则 "A100"/"CIFAR-100" 会被抽成数值 "100"，
# 把每个实验设置段落都误判为"命中数值"（实测：所有 chunk 的 number 特征都变 1.0）。
_NUM_RE = re.compile(
    r"(?<![a-z0-9\-])\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*"
    r"(?:gb|g|tb|mb|k|卡|层|epochs?|w|万)?(?![a-z0-9])", re.I)


def has_number(text: str, num: str) -> bool:
    """整词命中，不是子串命中：查 "100" 不该被 "V100" / "10000" 命中。"""
    return re.search(rf"(?<![a-z0-9]){re.escape(num)}(?![a-z0-9])", text, re.I) is not None
_CJK = re.compile(r"[一-鿿]")
_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9\-\._]*")


def extract_numbers(text: str) -> list[str]:
    return [m.group(0).strip().lower() for m in _NUM_RE.finditer(text or "")]


def run_scores(f: FusedHit, kind: str) -> float:
    """取某一路在**所有改写 run** 上的最高分。

    pipeline 里 run 名是 `dense:0 / sparse:1 / …`（每条改写各算一路），
    只按 `"dense"` 取会永远拿到 0——重排的两个主特征直接失效。
    """
    return max((v for k, v in f.scores.items() if k.split(":")[0] == kind), default=0.0)


def query_terms(query: str) -> list[str]:
    """用于覆盖度打分的词项：ASCII 词 + 中文 2~4 字片段。"""
    q = (query or "").lower()
    words = _ASCII_WORD.findall(q)
    cjk: list[str] = []
    for seg in re.findall(r"[一-鿿]+", q):
        cjk += [seg[i:i + 2] for i in range(len(seg) - 1)] or [seg]
    return words + cjk


@dataclass
class RerankContext:
    """重排所需的查询侧信息（由 query.py 产出）。"""

    query: str = ""
    intent: str = "general"                       # hyperparameter | hardware | method | result | general
    entities: list[str] = field(default_factory=list)    # A100 / learning rate / batch size …
    numbers: list[str] = field(default_factory=list)     # 8 卡 / 80GB / 1e-3 …
    preferred_sections: list[str] = field(default_factory=list)


@dataclass
class RerankedHit:
    chunk: Chunk
    score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    rrf: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def why(self) -> str:
        return " ".join(f"{k}={v:.2f}" for k, v in sorted(self.features.items(), key=lambda x: -x[1]))


class Reranker(ABC):
    name: str = "abstract"

    @abstractmethod
    def rerank(self, candidates: list[FusedHit], ctx: RerankContext,
               top_k: int = 8) -> list[RerankedHit]: ...


# ══════════════════════════ 离线兜底：特征加权 ══════════════════════════
DEFAULT_WEIGHTS: dict[str, float] = {
    "bm25": 0.26,        # 稀疏路分数（归一化）——精确 token 命中
    "dense": 0.16,       # 稠密路分数（归一化）——语义相近
    "rrf": 0.10,         # 融合名次共识
    "coverage": 0.18,    # 查询词项在正文里的覆盖度
    "entity": 0.14,      # 关键实体（A100 / learning rate）字面命中
    "number": 0.10,      # 数值命中：8卡、80GB、1e-3 这类硬信号
    "section": 0.06,     # 小节类型先验（实验设置/超参优先）
}


class HeuristicReranker(Reranker):
    """把"人判断相关性时在看的信号"显式写成特征：可解释、可调、可回归。

    与真 cross-encoder 的差距：**抓不到同义改写**（"显存" vs "GPU memory"），
    线上请换 CrossEncoderReranker，权重不用动。
    """

    name = "heuristic"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def rerank(self, candidates: list[FusedHit], ctx: RerankContext, top_k: int = 8) -> list[RerankedHit]:
        if not candidates:
            return []
        sparse = [run_scores(f, "sparse") for f in candidates]
        dense = [run_scores(f, "dense") for f in candidates]
        rrfs = [f.rrf for f in candidates]
        n_sparse, n_dense, n_rrf = minmax(sparse), minmax(dense), minmax(rrfs)

        qts = query_terms(ctx.query)
        out: list[RerankedHit] = []
        for i, f in enumerate(candidates):
            text = f.chunk.text.lower()
            payload = f.chunk.payload()

            coverage = (sum(1 for t in qts if t in text) / len(qts)) if qts else 0.0
            entity = (sum(1 for e in ctx.entities if str(e).lower() in text) / len(ctx.entities)) \
                if ctx.entities else 0.0
            nums = ctx.numbers
            number = (sum(1 for n in nums if has_number(text, n)) / len(nums)) if nums else 0.0
            meta_ok = 0.0
            if ctx.entities:
                # 元数据里是 learning_rate，查询侧是 learning rate：统一成空格再比
                joined = " ".join(str(v) for v in payload.values()
                                  if isinstance(v, (list, str))).lower().replace("_", " ")
                meta_ok = sum(1 for e in ctx.entities if str(e).lower() in joined) / len(ctx.entities)
            section = 1.0 if (ctx.preferred_sections and
                              payload.get("section_type") in ctx.preferred_sections) else 0.0

            feats = {
                "bm25": n_sparse[i], "dense": n_dense[i], "rrf": n_rrf[i],
                "coverage": coverage, "entity": max(entity, meta_ok), "number": number,
                "section": section,
            }
            score = sum(self.weights.get(k, 0.0) * v for k, v in feats.items())
            out.append(RerankedHit(chunk=f.chunk, score=round(score, 6), features={k: round(v, 4) for k, v in feats.items()},
                                   rrf=f.rrf, ranks=dict(f.ranks)))
        out.sort(key=lambda h: (-h.score, h.chunk.id))
        return out[:top_k]


# ══════════════════════════ 线上：cross-encoder ══════════════════════════
class CrossEncoderReranker(Reranker):
    """Cohere 风格 `POST {base}/rerank`，body={"model","query","documents","top_n"}。

    自建 BGE-reranker-v2-m3 / Qwen3-Reranker 通常也提供同构接口；
    若只有 `/score` 或 `/pair` 形态，改 `_call` 一个方法即可，其余不动。
    """

    name = "cross-encoder"

    def __init__(self, *, base_url: str = "", api_key: str = "", model: str = "bge-reranker-v2-m3",
                 timeout: float = 30.0, fallback: Reranker | None = None) -> None:
        self.base_url = (base_url or os.getenv("RERANK_BASE_URL")
                         or os.getenv("EMBEDDING_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("RERANK_API_KEY") or os.getenv("EMBEDDING_API_KEY") or ""
        self.model = model or os.getenv("RERANK_MODEL") or "bge-reranker-v2-m3"
        self.timeout = timeout
        self.fallback = fallback
        if not self.base_url or not self.api_key:
            raise RuntimeError("未配置 RERANK_BASE_URL / RERANK_API_KEY，无法调用 rerank 服务")

    def _call(self, query: str, documents: list[str]) -> list[float]:
        import httpx                                   # noqa: PLC0415

        with httpx.Client(timeout=self.timeout) as cli:
            r = cli.post(f"{self.base_url}/rerank",
                         headers={"Authorization": f"Bearer {self.api_key}"},
                         json={"model": self.model, "query": query,
                               "documents": documents, "top_n": len(documents)})
            r.raise_for_status()
            data: dict[str, Any] = r.json()
        # 必须拿回**每条**文档的分数再自己排序：只要 top_n 条会让 index 错位
        scores = [0.0] * len(documents)
        for row in data.get("results", []):
            i = int(row.get("index", -1))
            if 0 <= i < len(scores):
                scores[i] = float(row.get("relevance_score", row.get("score", 0.0)))
        return scores

    def rerank(self, candidates: list[FusedHit], ctx: RerankContext, top_k: int = 8) -> list[RerankedHit]:
        if not candidates:
            return []
        texts = [f.chunk.text[:2000] for f in candidates]     # 截断：cross-encoder 有长度上限
        try:
            scores = self._call(ctx.query, texts)
        except Exception:                                     # noqa: BLE001
            if self.fallback is None:
                raise
            out = self.fallback.rerank(candidates, ctx, top_k)
            for h in out:
                h.features["fallback"] = 1.0
            return out
        out = [RerankedHit(chunk=f.chunk, score=round(s, 6),
                           features={"cross_encoder": round(s, 4)}, rrf=f.rrf, ranks=dict(f.ranks))
               for f, s in zip(candidates, scores)]
        out.sort(key=lambda h: (-h.score, h.chunk.id))
        return out[:top_k]


def build_reranker(kind: str = "auto", **kw: Any) -> Reranker:
    """kind: auto（有 Key 走 cross-encoder，否则启发式）| heuristic | cross-encoder。"""
    if kind == "heuristic":
        return HeuristicReranker(**kw)
    if kind == "cross-encoder":
        return CrossEncoderReranker(**kw)
    try:
        return CrossEncoderReranker(fallback=HeuristicReranker(), **kw)
    except Exception:  # noqa: BLE001
        return HeuristicReranker()
