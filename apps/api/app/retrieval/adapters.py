"""数据源适配：把现成的语料接进检索 pipeline。

三个入口覆盖本项目已有的三种语料形态：
    chunks_from_rag_index()  app/rag 入库产物（章节路径 + 页码 + block_ids 全都在）
    chunks_from_corpus()     app/agents/tools/corpus.py 的离线语料
    chunks_from_dicts()      任意 JSON/JSONL 记录
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from app.retrieval.types import Chunk, FIELD_SECTION_TYPE

# 语料里的小节名 → 规范 section_type（过滤与重排先验都依赖这个字段）
_SECTION_MAP: list[tuple[str, str]] = [
    (r"abstract|摘要", "abstract"),
    (r"related|相关工作|related work", "related_work"),
    (r"method|approach|architecture|model|方法|架构|模型", "method"),
    (r"experiment|setup|implementation|training|训练|实验设置|实现", "experimental_setup"),
    (r"hyper-?param|超参", "hyperparameters"),
    (r"hardware|设备|硬件", "hardware"),
    (r"result|ablation|评估|结果|消融", "results"),
    (r"conclusion|结论", "conclusion"),
    (r"reference|参考文献", "reference"),
]


def normalize_section(section: str) -> str:
    import re

    s = (section or "").lower()
    for pat, norm in _SECTION_MAP:
        if re.search(pat, s):
            return norm
    return "other"


def chunks_from_rag_index(index_dir: str | Path, *, levels: Iterable[str] = ("L1",),
                          limit: int | None = None) -> list[Chunk]:
    """从 `app/rag/cli.py --ingest` 的产物构建检索语料（保留标题路径与页码）。"""
    from app.rag.store import ChunkIndex, JsonlStore          # 局部导入：避免强依赖

    idx = ChunkIndex.load(Path(index_dir) / "index.json")
    levels = set(levels)
    stores: dict[str, JsonlStore] = {}
    out: list[Chunk] = []
    try:
        for rec in idx.records:
            if levels and rec.get("level") not in levels:
                continue
            lv = str(rec.get("store") or rec.get("level") or "L1")
            # 只读模式：用 "a" 会把缺文件和空文件都创出来，还漏可写句柄
            st = stores.setdefault(lv, JsonlStore(Path(index_dir) / f"chunks_{lv}.jsonl", "r"))
            text = str(st.read(int(rec["offset"]), int(rec["length"])).get("text", ""))
            path = idx.path_of(rec)
            out.append(Chunk(
                id=str(rec.get("id")), text=text, doc_id=str(rec.get("doc_id", "")),
                title=path[0] if path else str(rec.get("doc_id", "")),
                section_path=" > ".join(path),
                page_start=int(rec.get("page_start", 0) or 0),
                page_end=int(rec.get("page_end", 0) or 0),
                block_ids=list(rec.get("block_ids") or []),
                metadata={"level": rec.get("level"), "section_number": rec.get("section_number", ""),
                          # 用完整标题路径判断小节类型：父章节名里常带"实验设置/超参数"
                          FIELD_SECTION_TYPE: normalize_section(" ".join(path))},
            ))
            if limit and len(out) >= limit:
                break
    finally:
        for st in stores.values():
            st.close()
    return out


def chunks_from_corpus() -> list[Chunk]:
    """从 `app/agents/tools/corpus.py`（DocIR 的内存替身）构建语料。"""
    from app.agents.tools.corpus import BLOCKS, DOCS

    out: list[Chunk] = []
    for b in BLOCKS:
        doc = DOCS.get(b["doc_id"], {})
        out.append(Chunk(
            id=f"{b['doc_id']}#{b['block_id']}", text=b["text"], doc_id=b["doc_id"],
            title=str(doc.get("title", b["doc_id"])),
            section_path=str(b.get("section", "")),
            page_start=int(b.get("page", 0)), page_end=int(b.get("page", 0)),
            block_ids=[str(b["block_id"])],
            metadata={"year": doc.get("year"), "venue": doc.get("venue"),
                      "source": doc.get("source"),
                      FIELD_SECTION_TYPE: normalize_section(str(b.get("section", "")))},
        ))
    return out


def chunks_from_dicts(rows: Iterable[dict[str, Any]]) -> list[Chunk]:
    """通用入口：至少需要 id/text，其余字段有则用、无则留空。"""
    out: list[Chunk] = []
    for r in rows:
        meta = dict(r.get("metadata") or {})
        if FIELD_SECTION_TYPE not in meta:
            meta[FIELD_SECTION_TYPE] = normalize_section(str(r.get("section_path", "")))
        out.append(Chunk(
            id=str(r["id"]), text=str(r.get("text", "")), doc_id=str(r.get("doc_id", "")),
            title=str(r.get("title", "")), section_path=str(r.get("section_path", "")),
            page_start=int(r.get("page_start", 0) or 0), page_end=int(r.get("page_end", 0) or 0),
            block_ids=list(r.get("block_ids") or []), metadata=meta,
        ))
    return out
