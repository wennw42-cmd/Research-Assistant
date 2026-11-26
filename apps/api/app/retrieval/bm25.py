"""BM25 稀疏检索（Okapi BM25，零依赖）。

为什么稀疏路不能省：
- **超参数/硬件这类问题靠的是精确 token**——`1e-3`、`A100`、`80GB`、`batch size 32`，
  向量模型会把它们"语义化"成相近的数字，反而召回错版本；
- 稠密模型对论文里的**稀有符号与型号**覆盖不足，BM25 的 IDF 天然会把它们顶上来。

实现要点：
- 中英混排分词 = ASCII 词 + 中文 bigram（`app/parsing` 与 `app/rag` 用的同一套口径）；
- IDF 用 `ln(1 + (N-df+0.5)/(df+0.5))`，恒正，避免小语料下出现负分；
- 过滤是**先筛后打分**（pre-filter），不是打分后再丢，否则 top-k 会被无关文档占满。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from app.retrieval.types import Chunk, Filter, Hit

_ASCII = re.compile(r"[a-z0-9][a-z0-9\-\._]*")
_CJK = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> list[str]:
    t = (text or "").lower()
    grams: list[str] = []
    for seg in _CJK.findall(t):
        grams += [seg[i:i + 2] for i in range(len(seg) - 1)] or [seg]
    return _ASCII.findall(t) + grams


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75

    chunks: list[Chunk] = field(default_factory=list)
    _tf: list[Counter[str]] = field(default_factory=list)
    _df: Counter[str] = field(default_factory=Counter)
    _len: list[int] = field(default_factory=list)
    _avg_len: float = 0.0

    def build(self, chunks: list[Chunk]) -> "BM25Index":
        self.chunks = list(chunks)
        self._tf = []
        self._df = Counter()
        self._len = []
        for c in self.chunks:
            tf = Counter(tokenize(c.text))
            self._tf.append(tf)
            self._len.append(sum(tf.values()))
            self._df.update(tf.keys())
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0
        return self

    def idf(self, term: str) -> float:
        n = len(self.chunks)
        if not n:
            return 0.0
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, q_terms: Counter[str], idx: int) -> float:
        tf, dl = self._tf[idx], self._len[idx]
        s = 0.0
        for term, qtf in q_terms.items():
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1.0))
            s += self.idf(term) * (f * (self.k1 + 1)) / denom * qtf
        return s

    def search(self, query: str, top_k: int = 50, flt: Filter | None = None) -> list[Hit]:
        q_terms = Counter(tokenize(query))
        if not q_terms or not self.chunks:
            return []
        flt = flt or Filter()

        cand = [i for i, c in enumerate(self.chunks) if flt.matches(c.payload())]
        scored = [(self.score(q_terms, i), i) for i in cand]
        scored = [(s, i) for s, i in scored if s > 0]
        # 同分用 chunk_id 兜底，保证结果可复现
        scored.sort(key=lambda p: (-p[0], self.chunks[p[1]].id))
        return [Hit(chunk_id=self.chunks[i].id, score=round(s, 6), source="sparse",
                    rank=r + 1, chunk=self.chunks[i])
                for r, (s, i) in enumerate(scored[:top_k])]
