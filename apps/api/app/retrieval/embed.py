"""Embedding 抽象：真实模型 / 离线兜底同一接口。

线上选型建议（学术 RAG 实测口碑）：
    BGE-M3 / Qwen3-Embedding-8B   中英混排、长上下文，学术语料首选
    text-embedding-3-large        英文为主，中文偏弱
    GTE-Qwen2-7B-instruct         中文强，显存占用高

兜底策略：**绝不用随机向量**。随机向量的"相似度"是噪声，会让召回看起来在动、
其实全错——这里用确定性的 hashing trick（字符/词 n-gram → 定长向量 → L2 归一化），
它抓的是词汇重叠，抓不到同义改写，但**可复现、可回归测试**，并在 `engine` 字段里如实标注。
"""
from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from typing import Any

from app.retrieval.bm25 import tokenize


def l2(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


class Embedder(ABC):
    name: str = "abstract"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """文本 → 向量列表（顺序一致）。失败应抛异常，不要返回假向量。"""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashEmbedder(Embedder):
    """离线兜底：hashing trick（unigram 权重 1.0 + bigram 0.5，次线性 tf）→ L2。"""

    name = "local-hash"
    dim = 1024

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        toks = tokenize(text)
        if len(toks) == 1:
            toks = toks + toks
        vec = [0.0] * self.dim
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            vec[self._bucket(t)] += 1.0 + math.log(c)
        for a, b in zip(toks, toks[1:]):
            vec[self._bucket(a + "|" + b)] += 0.5
        return l2(vec)

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % self.dim


class OpenAIEmbedder(Embedder):
    """OpenAI 兼容 `/embeddings`（BGE-M3 / Qwen3 / text-embedding-3 都走这条）。"""

    name = "openai-compatible"

    def __init__(self, *, base_url: str = "", api_key: str = "", model: str = "",
                 dim: int = 0, batch_size: int = 32, timeout: float = 30.0) -> None:
        self.base_url = (base_url or os.getenv("EMBEDDING_BASE_URL")
                         or os.getenv("LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY") or ""
        self.model = model or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        self.dim = dim
        self.batch_size = batch_size
        self.timeout = timeout
        if not self.base_url or not self.api_key:
            raise RuntimeError("未配置 EMBEDDING_BASE_URL / EMBEDDING_API_KEY，无法使用远端 embedding")

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx                       # 延迟导入：离线环境下不必装

        out: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as cli:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                r = cli.post(f"{self.base_url}/embeddings",
                             headers={"Authorization": f"Bearer {self.api_key}"},
                             json={"model": self.model, "input": batch})
                r.raise_for_status()
                data: dict[str, Any] = r.json()
                rows = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
                if len(rows) != len(batch):
                    raise RuntimeError(f"embedding 返回 {len(rows)} 条，期望 {len(batch)} 条")
                out += [list(map(float, row["embedding"])) for row in rows]
        if out and self.dim == 0:
            self.dim = len(out[0])
        return out


def build_embedder(kind: str = "auto", **kw: Any) -> Embedder:
    """kind: auto（有 Key 走远端，否则本地哈希）| hash | openai。"""
    if kind == "hash":
        return HashEmbedder(**kw)
    if kind == "openai":
        return OpenAIEmbedder(**kw)
    try:
        return OpenAIEmbedder(**kw)
    except Exception:  # noqa: BLE001 —— 无 Key / 未装 httpx，一律退到本地确定性向量
        return HashEmbedder(**{k: v for k, v in kw.items() if k == "dim"})
