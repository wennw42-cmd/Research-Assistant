"""向量库抽象 + 四个后端实现。

接口只有五个动作，业务侧（pipeline）不感知后端：

    create_collection(dim) → upsert(chunks, vectors) → search(vector, top_k, filter)
    count() → close()

选型建议（学术 RAG，百万级 chunk 以内）：

| 后端 | 适用 | 元数据过滤 | 备注 |
|---|---|---|---|
| InMemoryVectorStore | 单测/单机 ≤5 万块 | 全功能（原生 Python 匹配） | 零依赖，结果可复现；进程内 |
| QdrantVectorStore | 生产首选 | **最强**（嵌套 must/should、数组、范围、payload 索引） | 需给过滤字段建 payload index，否则过滤退化成全扫 |
| PgVectorStore | 已有 Postgres | 强（SQL 直接写，可 JOIN 文献表） | HNSW 索引 + jsonb，事务与备份现成 |
| ChromaVectorStore | 原型/小团队 | 弱（数组字段支持有限，见下） | API 最省事，过滤能力是短板 |

⚠️ Chroma 的坑：官方 `where` 对**数组元数据**支持不完整（部分版本无 `$contains`）。
本适配器把数组字段 join 成 `"A100|80GB"` 字符串，用 `$contains` 做子串匹配；
若你的版本不支持，请把数组拆成布尔字段（`has_gpu_a100=True`）或换 Qdrant/pgvector。
"""
from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from typing import Any

from app.retrieval.types import Chunk, Filter, Hit

NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


class VectorStore(ABC):
    """向量库统一接口：**检索侧只依赖这五个方法，换库不动业务代码**。"""

    name: str = "abstract"

    @abstractmethod
    def create_collection(self, dim: int, *, recreate: bool = False) -> None: ...

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    @abstractmethod
    def search(self, vector: list[float], top_k: int = 50,
               flt: Filter | None = None) -> list[Hit]: ...

    @abstractmethod
    def count(self) -> int: ...

    def close(self) -> None:
        return None


# ══════════════════════════ 内存实现（默认，零依赖）══════════════════════════
class InMemoryVectorStore(VectorStore):
    """进程内余弦检索：先按 Filter 预筛，再算余弦。

    入库/查询时都做 L2 归一化——否则换了不归一化的 embedding 模型，
    内存后端的名次会与 Qdrant/pgvector（COSINE）不一致，离线评测就白做了。
    """

    name = "memory"

    def __init__(self, collection: str = "chunks") -> None:
        self.collection = collection
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self._index: dict[str, int] = {}

    @staticmethod
    def _unit(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def create_collection(self, dim: int, *, recreate: bool = False) -> None:
        if recreate:
            self.chunks, self.vectors, self._index = [], [], {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for c, v in zip(chunks, vectors):
            v = self._unit(v)
            if c.id in self._index:                       # 覆盖式写入
                i = self._index[c.id]
                self.chunks[i], self.vectors[i] = c, v
            else:
                self._index[c.id] = len(self.chunks)
                self.chunks.append(c)
                self.vectors.append(v)

    def search(self, vector: list[float], top_k: int = 50, flt: Filter | None = None) -> list[Hit]:
        vector = self._unit(vector)
        flt = flt or Filter()
        scored: list[tuple[float, Chunk]] = []
        for c, v in zip(self.chunks, self.vectors):
            if not flt.matches(c.payload()):
                continue
            s = sum(a * b for a, b in zip(vector, v))
            scored.append((s, c))
        scored.sort(key=lambda p: (-p[0], p[1].id))
        return [Hit(chunk_id=c.id, score=round(s, 6), source="dense", rank=r + 1, chunk=c)
                for r, (s, c) in enumerate(scored[:top_k])]

    def count(self) -> int:
        return len(self.chunks)


# ══════════════════════════ Qdrant ══════════════════════════
class QdrantVectorStore(VectorStore):
    """pip install qdrant-client；docker run -p 6333:6333 qdrant/qdrant"""

    name = "qdrant"

    def __init__(self, *, url: str = "http://localhost:6333", api_key: str = "",
                 collection: str = "chunks", **kw: Any) -> None:
        self.url, self.api_key, self.collection = url, api_key, collection
        self._kw = kw
        try:
            from qdrant_client import QdrantClient  # noqa: PLC0415
        except ImportError as e:                     # pragma: no cover
            raise ImportError("使用 QdrantVectorStore 需要：pip install qdrant-client") from e
        self._cli = QdrantClient(url=url, api_key=api_key or None, **kw)

    def create_collection(self, dim: int, *, recreate: bool = False) -> None:
        from qdrant_client import models               # noqa: PLC0415

        exists = self._cli.collection_exists(self.collection)
        if exists and recreate:
            self._cli.delete_collection(self.collection)
            exists = False
        if not exists:
            self._cli.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        # 过滤字段建 payload 索引：不加这句，过滤会退化成全量扫描（实测差一个数量级）。
        # schema 必须与字段真实类型一致：n_gpus/year 用 keyword 索引，range 过滤会静默失效。
        schemas = {"doc_id": "keyword", "section_type": "keyword",
                   "hyperparams": "keyword", "hardware": "keyword",
                   "year": "integer", "n_gpus": "integer", "has_table": "bool"}
        for field, schema in schemas.items():
            try:
                self._cli.create_payload_index(self.collection, field, field_schema=schema,
                                               wait=True)
            except Exception:  # noqa: BLE001 —— 字段类型不匹配时忽略，不阻断
                pass

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client import models               # noqa: PLC0415

        self._cli.upsert(self.collection, points=[
            models.PointStruct(id=str(uuid.uuid5(NAMESPACE, c.id)), vector=v, payload=c.payload())
            for c, v in zip(chunks, vectors)])

    def search(self, vector: list[float], top_k: int = 50, flt: Filter | None = None) -> list[Hit]:
        from qdrant_client import models               # noqa: PLC0415

        qf = None
        if flt and not flt.is_empty():
            conds = []
            for spec in flt.to_qdrant_spec():
                if "match" in spec:
                    m = spec["match"]
                    conds.append(models.FieldCondition(
                        key=spec["key"],
                        match=models.MatchAny(any=m["any"]) if "any" in m
                        else models.MatchValue(value=m["value"])))
                else:
                    conds.append(models.FieldCondition(key=spec["key"],
                                                       range=models.Range(**spec["range"])))
            qf = models.Filter(must=conds)
        # qdrant-client ≥1.12 用 query_points；老版本才是 search（1.15+ 已删除 search）
        if hasattr(self._cli, "query_points"):
            res = self._cli.query_points(self.collection, query=vector, limit=top_k,
                                         query_filter=qf, with_payload=True).points
        else:                                          # pragma: no cover
            res = self._cli.search(self.collection, query_vector=vector, limit=top_k,
                                   query_filter=qf, with_payload=True)
        hits: list[Hit] = []
        for r, p in enumerate(res, 1):
            c = Chunk.from_payload(p.payload or {})
            hits.append(Hit(chunk_id=c.id, score=round(float(p.score), 6),
                            source="dense", rank=r, chunk=c))
        return hits

    def count(self) -> int:
        return int(self._cli.count(self.collection).count)


# ══════════════════════════ Chroma ══════════════════════════
class ChromaVectorStore(VectorStore):
    """pip install chromadb。数组元数据会被 join 成 '|' 分隔字符串（见模块 docstring）。"""

    name = "chroma"

    def __init__(self, *, collection: str = "chunks", host: str = "", port: int = 0,
                 path: str = "", **kw: Any) -> None:
        self.collection_name = collection
        try:
            import chromadb                              # noqa: PLC0415
            from chromadb.config import Settings          # noqa: PLC0415
        except ImportError as e:                         # pragma: no cover
            raise ImportError("使用 ChromaVectorStore 需要：pip install chromadb") from e
        if host:
            self._cli = chromadb.HttpClient(host=host, port=port, **kw)
        else:
            self._cli = chromadb.Client(Settings(persist_directory=path or "./chroma_db"), **kw)
        self._col = self._cli.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"})

    def create_collection(self, dim: int, *, recreate: bool = False) -> None:
        if recreate:
            self._cli.delete_collection(self.collection_name)
            self._col = self._cli.get_or_create_collection(
                self.collection_name, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _flat(payload: dict[str, Any]) -> dict[str, Any]:
        """Chroma 元数据只收 str/int/float/bool：数组 join，None 直接丢弃（否则写入报错）。"""
        out = {}
        for k, v in payload.items():
            if v is None:
                continue
            out[k] = "|".join(str(x) for x in v) if isinstance(v, list) else v
        return out

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._col.upsert(ids=[c.id for c in chunks], embeddings=list(vectors),
                         documents=[c.text for c in chunks],
                         metadatas=[self._flat(c.payload()) for c in chunks])

    def search(self, vector: list[float], top_k: int = 50, flt: Filter | None = None) -> list[Hit]:
        where = None
        if flt and not flt.is_empty():
            w = flt.to_chroma() or {}
            # contains_any 在 Chroma 里退化为"字符串包含"：拼成 $or
            ors = [{k: {"$contains": str(v)}} for k, opts in flt.contains_any.items() for v in opts]
            if ors:
                w = {"$and": [w, {"$or": ors}]} if w else {"$or": ors}
            where = w or None
        res = self._col.query(query_embeddings=[vector], n_results=top_k, where=where)
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits: list[Hit] = []
        for r, (m, d) in enumerate(zip(metas, dists), 1):
            c = Chunk.from_payload(m or {})
            hits.append(Hit(chunk_id=c.id, score=round(1.0 - float(d), 6),
                            source="dense", rank=r, chunk=c))
        return hits

    def count(self) -> int:
        return int(self._col.count())


# ══════════════════════════ pgvector ══════════════════════════
class PgVectorStore(VectorStore):
    """pip install psycopg2-binary；库里 `CREATE EXTENSION vector`。

    索引建议：小于 1 万行用 ivfflat(lists=100) 够用；更大规模或要求高召回用
    `USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)`，
    查询时 `SET hnsw.ef_search = 100` 调召回/延迟平衡。
    """

    name = "pgvector"

    def __init__(self, *, dsn: str = "", table: str = "rag_chunks", **kw: Any) -> None:
        self.dsn = dsn or __import__("os").getenv("PGVECTOR_DSN", "")
        self.table = table
        if not self.dsn:
            raise RuntimeError("pgvector 需要 dsn（或环境变量 PGVECTOR_DSN）")
        try:
            import psycopg2                              # noqa: PLC0415
        except ImportError as e:                         # pragma: no cover
            raise ImportError("使用 PgVectorStore 需要：pip install psycopg2-binary") from e
        self._conn = psycopg2.connect(self.dsn, **kw)

    def create_collection(self, dim: int, *, recreate: bool = False) -> None:
        with self._conn, self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if recreate:
                cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                " id text PRIMARY KEY, embedding vector(%s), payload jsonb)", (dim,))
            cur.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} "
                        "USING hnsw (embedding vector_cosine_ops)")

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        import json                                       # noqa: PLC0415

        with self._conn, self._conn.cursor() as cur:
            for c, v in zip(chunks, vectors):
                cur.execute(
                    f"INSERT INTO {self.table} (id, embedding, payload) VALUES (%s, %s::vector, %s) "
                    f"ON CONFLICT (id) DO UPDATE SET embedding=EXCLUDED.embedding, "
                    f"payload=EXCLUDED.payload",
                    (c.id, "[" + ",".join(f"{x:.6f}" for x in v) + "]",
                     json.dumps(c.payload(), ensure_ascii=False)))

    def search(self, vector: list[float], top_k: int = 50, flt: Filter | None = None) -> list[Hit]:
        import json                                       # noqa: PLC0415

        where, params = (flt or Filter()).to_sql()
        lit = "[" + ",".join(f"{x:.6f}" for x in vector) + "]"
        sql = (f"SELECT id, payload, 1 - (embedding <=> %s::vector) AS score "
               f"FROM {self.table} WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s")
        with self._conn.cursor() as cur:
            cur.execute(sql, [lit, *params, lit, top_k])
            rows = cur.fetchall()
        hits: list[Hit] = []
        for r, (_id, payload, score) in enumerate(rows, 1):
            c = Chunk.from_payload(payload if isinstance(payload, dict) else json.loads(payload))
            hits.append(Hit(chunk_id=c.id, score=round(float(score), 6),
                            source="dense", rank=r, chunk=c))
        return hits

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return int(cur.fetchone()[0])

    def close(self) -> None:
        self._conn.close()


def build_store(kind: str = "memory", **kw: Any) -> VectorStore:
    """kind: memory | qdrant | chroma | pgvector。"""
    table = {
        "memory": InMemoryVectorStore,
        "qdrant": QdrantVectorStore,
        "chroma": ChromaVectorStore,
        "pgvector": PgVectorStore,
    }
    if kind not in table:
        raise ValueError(f"未知向量库 {kind}，可选：{sorted(table)}")
    return table[kind](**kw)
