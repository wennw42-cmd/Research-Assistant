"""检索数据契约：Chunk / Filter / Hit / FusedHit。

`Filter` 是**与向量库无关**的过滤描述，三种后端各自翻译（见 `to_chroma / to_sql / to_qdrant_spec`），
这样换 Qdrant / Chroma / pgvector 时业务逻辑一行不用改。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 建议写入 payload 的字段名（元数据过滤的"契约字段"，非强制）
FIELD_SECTION_TYPE = "section_type"   # abstract|method|experimental_setup|hyperparameters|hardware|results|related_work
FIELD_HYPERPARAMS = "hyperparams"     # ["learning_rate","batch_size",...]
FIELD_HARDWARE = "hardware"           # ["A100","80GB","8卡"]
FIELD_N_GPUS = "n_gpus"               # 数字，支持范围过滤
FIELD_HAS_TABLE = "has_table"         # bool
FIELD_YEAR = "year"


@dataclass
class Chunk:
    """一个可检索单元（对应 app/rag 的 L1 段落块，或 DocIR 的若干 block）。"""

    id: str
    text: str
    doc_id: str = ""
    title: str = ""
    section_path: str = ""          # "第 5 章 反向传播 > 5.2 链式法则"
    page_start: int = 0
    page_end: int = 0
    block_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """进向量库的 payload：元数据与文本分列，便于只回传元数据做过滤。"""
        return {
            "chunk_id": self.id, "doc_id": self.doc_id, "title": self.title,
            "section_path": self.section_path, "page_start": self.page_start,
            "page_end": self.page_end, "block_ids": self.block_ids,
            "text": self.text, **self.metadata,
        }

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> "Chunk":
        known = {"chunk_id", "doc_id", "title", "section_path", "page_start",
                 "page_end", "block_ids", "text"}
        return cls(
            id=str(p.get("chunk_id", "")), text=str(p.get("text", "")),
            doc_id=str(p.get("doc_id", "")), title=str(p.get("title", "")),
            section_path=str(p.get("section_path", "")),
            page_start=int(p.get("page_start", 0) or 0),
            page_end=int(p.get("page_end", 0) or 0),
            block_ids=list(p.get("block_ids") or []),
            metadata={k: v for k, v in p.items() if k not in known},
        )

    @property
    def citation(self) -> str:
        """溯源串：文档 + 标题路径 + 页码。"""
        where = f"p{self.page_start}" + (f"-{self.page_end}" if self.page_end != self.page_start else "")
        path = f" | {self.section_path}" if self.section_path else ""
        return f"{self.doc_id}{path} | {where}"

    @property
    def n_tokens_approx(self) -> int:
        return len(self.text) // 2


@dataclass
class Filter:
    """后端无关的过滤条件。

    must        : 等值匹配（payload 值是 list 时表示"包含该元素"）
    any_of      : 字段值 ∈ 列表（OR）
    contains_any: 字段是数组，且与给定列表**有交集**
    ranges      : 数值范围 (min, max)，端点可为 None；**字段缺失即视为不匹配**
    """

    must: dict[str, Any] = field(default_factory=dict)
    any_of: dict[str, list[Any]] = field(default_factory=dict)
    contains_any: dict[str, list[Any]] = field(default_factory=dict)
    ranges: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.must or self.any_of or self.contains_any or self.ranges)

    def matches(self, payload: dict[str, Any]) -> bool:
        for k, v in self.must.items():
            got = payload.get(k)
            if isinstance(got, list):
                if v not in got:
                    return False
            elif got != v:
                return False
        for k, opts in self.any_of.items():
            got = payload.get(k)
            if isinstance(got, list):
                if not set(got) & set(opts):
                    return False
            elif got not in opts:
                return False
        for k, opts in self.contains_any.items():
            got = payload.get(k)
            if not isinstance(got, list) or not (set(got) & set(opts)):
                return False
        for k, (lo, hi) in self.ranges.items():
            got = payload.get(k)
            if not isinstance(got, (int, float)):      # 缺字段=不匹配（宁可漏，不可错）
                return False
            if lo is not None and got < lo:
                return False
            if hi is not None and got > hi:
                return False
        return True

    def describe(self) -> str:
        parts = [f"{k}={v}" for k, v in self.must.items()]
        parts += [f"{k}∈{v}" for k, v in self.any_of.items()]
        parts += [f"{k}∩{v}" for k, v in self.contains_any.items()]
        parts += [f"{k}∈[{lo},{hi}]" for k, (lo, hi) in self.ranges.items()]
        return " AND ".join(parts) or "（无过滤）"

    # ── 后端翻译 ──
    def to_chroma(self) -> dict[str, Any] | None:
        """Chroma 的 `where`（同一字段多条件需包 $and；数组交集用 $contains 语义近似）。"""
        if self.is_empty():
            return None
        conds: list[dict[str, Any]] = []
        for k, v in self.must.items():
            conds.append({k: {"$eq": v}})
        for k, opts in self.any_of.items():
            conds.append({k: {"$in": list(opts)}})
        for k, opts in self.contains_any.items():
            conds.append({k: {"$in": list(opts)}})      # Chroma 对数组字段按元素匹配
        for k, (lo, hi) in self.ranges.items():
            if lo is not None:
                conds.append({k: {"$gte": lo}})
            if hi is not None:
                conds.append({k: {"$lte": hi}})
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def to_qdrant_spec(self) -> list[dict[str, Any]]:
        """返回**条件规格**（列表），由 `stores.QdrantVectorStore` 转成 models.FieldCondition。

        不在这里 import qdrant_client：否则没装 qdrant 的环境连本模块都导不进来。
        """
        spec: list[dict[str, Any]] = []
        for k, v in self.must.items():
            spec.append({"key": k, "match": {"value": v}})
        for k, opts in self.any_of.items():
            spec.append({"key": k, "match": {"any": list(opts)}})
        for k, opts in self.contains_any.items():
            spec.append({"key": k, "match": {"any": list(opts)}})
        for k, (lo, hi) in self.ranges.items():
            spec.append({"key": k, "range": {"gte": lo, "lte": hi}})
        return spec

    def to_sql(self, payload_col: str = "payload") -> tuple[str, list[Any]]:
        """pgvector：payload 存 jsonb，数值字段用 `(payload->>'x')::float` 取值。"""
        clauses: list[str] = []
        params: list[Any] = []
        for k, v in self.must.items():
            clauses.append(f"{payload_col}->>%s = %s")
            params += [k, str(v)]
        for k, opts in self.any_of.items():
            clauses.append(f"{payload_col}->>%s = ANY(%s)")
            params += [k, [str(o) for o in opts]]
        for k, opts in self.contains_any.items():
            # 数组字段：jsonb ?| 操作符（任一存在）
            clauses.append(f"{payload_col}->%s ?| %s")
            params += [k, [str(o) for o in opts]]
        for k, (lo, hi) in self.ranges.items():
            if lo is not None:
                clauses.append(f"({payload_col}->>%s)::float >= %s")
                params += [k, lo]
            if hi is not None:
                clauses.append(f"({payload_col}->>%s)::float <= %s")
                params += [k, hi]
        return (" AND ".join(clauses) if clauses else "TRUE"), params


@dataclass
class Hit:
    """单路召回结果。"""

    chunk_id: str
    score: float
    source: str                  # dense | sparse
    rank: int = 0                # 从 1 开始
    chunk: Chunk | None = None


@dataclass
class FusedHit:
    """融合后的候选：保留各路原始名次与分数，便于解释"它为什么在这"。"""

    chunk: Chunk
    rrf: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        return sorted(self.ranks)
