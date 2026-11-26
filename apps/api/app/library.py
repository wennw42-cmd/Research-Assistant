"""文献库：PDF 原文件 + DocIR/阅读器/分块 + 检索问答 + 页级翻译。

设计取舍
--------
- **原文件不丢**：每篇 PDF 独立目录，原文、解析结果、分块、译文分文件存放；
  SQLite 只存元数据与路径，Navicat 里直接可读。
- **解析走后台线程池**：541 页教材的解析要几十秒到几分钟，HTTP 只负责收文件与轮询状态；
  服务重启时把未完成任务重新排队，不把半成品当成功。
- **检索复用** ``app.retrieval`` 的 BM25 + 稠密 + RRF + rerank 主链路，不另造 RAG。
- **问答只依据证据**：未配置模型或模型失败时，返回检索到的原文片段并明确标注，
  不生成没有出处的答案。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pymupdf as pm

from app.config import get_settings
from app.history import _connect, _write_conn
from app.parsing.pipeline import parse_pdf
from app.rag.chunker import StreamingPacker
from app.rag.params import count_tokens
from app.reader import build_reader
from app.retrieval.pipeline import AcademicRetrievalPipeline, RetrievalConfig
from app.retrieval.query import plan_query
from app.retrieval.types import Chunk, Filter

log = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent.parent

MAX_PDF_BYTES = 100 * 1024 * 1024
CHAPTER_TARGET, CHAPTER_OVERLAP, CHAPTER_MAX = 512, 64, 768
TRANSLATABLE = {"title", "heading", "paragraph", "list_item"}
SKIP_CHUNK = {"title", "header", "footer", "reference"}

PAPER_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id               TEXT PRIMARY KEY,
    title            TEXT NOT NULL DEFAULT '',
    filename         TEXT NOT NULL DEFAULT '',
    sha256           TEXT NOT NULL UNIQUE,
    arxiv_id         TEXT NOT NULL DEFAULT '',
    base_id          TEXT NOT NULL DEFAULT '',
    version          TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'queued',
    progress         REAL NOT NULL DEFAULT 0,
    error            TEXT NOT NULL DEFAULT '',
    page_count       INTEGER NOT NULL DEFAULT 0,
    size_bytes       INTEGER NOT NULL DEFAULT 0,
    authors_json     TEXT NOT NULL DEFAULT '[]',
    abstract         TEXT NOT NULL DEFAULT '',
    year             INTEGER,
    keywords_json    TEXT NOT NULL DEFAULT '[]',
    dir              TEXT NOT NULL DEFAULT '',
    file_path        TEXT NOT NULL DEFAULT '',
    docir_path       TEXT NOT NULL DEFAULT '',
    reader_path      TEXT NOT NULL DEFAULT '',
    chunks_path      TEXT NOT NULL DEFAULT '',
    translations_path TEXT NOT NULL DEFAULT '',
    duplicate_of     TEXT NOT NULL DEFAULT '',
    meta_json        TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv  ON papers(base_id);
"""


class LibraryError(ValueError):
    """文献库参数/状态错误。"""


# ────────────────────────── 路径与连接 ──────────────────────────

def library_dir() -> Path:
    raw = Path(get_settings().LIBRARY_PATH or "data/library")
    return raw if raw.is_absolute() else (APP_DIR / raw).resolve()


def _conn() -> sqlite3.Connection:
    conn = _connect()
    conn.executescript(PAPER_SCHEMA)
    return conn


@contextmanager
def _read_conn():
    conn = _conn()
    try:
        yield conn
    finally:
        conn.close()


def init() -> Path:
    """建表并返回库目录。"""
    root = library_dir()
    root.mkdir(parents=True, exist_ok=True)
    _conn().close()
    return root


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _bump() -> None:
    global _REV
    with _CACHE_LOCK:
        _REV += 1
        _CACHE["pipeline"] = None
        _CACHE["chunks"] = {}


# ────────────────────────── 元数据 ──────────────────────────

def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    meta = _json(row["meta_json"], {})
    stats = meta.get("stats") or {}
    return {
        "id": row["id"],
        "title": row["title"] or row["filename"] or row["id"],
        "filename": row["filename"],
        "arxiv_id": row["arxiv_id"],
        "base_id": row["base_id"],
        "version": row["version"],
        "status": row["status"],
        "progress": round(float(row["progress"] or 0), 3),
        "error": row["error"],
        "page_count": row["page_count"],
        "size_bytes": row["size_bytes"],
        "authors": _json(row["authors_json"], []),
        "abstract": row["abstract"],
        "year": row["year"],
        "keywords": _json(row["keywords_json"], []),
        "duplicate_of": row["duplicate_of"],
        "stats": stats,
        "warnings": meta.get("warnings") or [],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _json(raw: str | None, default: Any) -> Any:
    try:
        return json.loads(raw or "")
    except (TypeError, ValueError):
        return default


def _public(paper: dict[str, Any]) -> dict[str, Any]:
    out = dict(paper)
    out["links"] = {
        "file": f"/api/papers/{paper['id']}/file",
        "reader": f"/api/papers/{paper['id']}/reader",
        "markdown": f"/api/papers/{paper['id']}/markdown",
    }
    return out


def _fetch(pid: str) -> sqlite3.Row | None:
    with _read_conn() as conn:
        return conn.execute("SELECT * FROM papers WHERE id = ?", (pid,)).fetchone()


def get_paper(pid: str) -> dict[str, Any] | None:
    row = _fetch(pid)
    return _public(_row_dict(row)) if row else None


def find_by_arxiv(base_id: str) -> dict[str, Any] | None:
    if not base_id:
        return None
    with _read_conn() as conn:
        row = conn.execute(
            "SELECT * FROM papers WHERE base_id = ? ORDER BY created_at LIMIT 1", (base_id,)
        ).fetchone()
    return _public(_row_dict(row)) if row else None


def _require(pid: str) -> sqlite3.Row:
    row = _fetch(pid)
    if row is None:
        raise LibraryError("文献不存在")
    return row


def list_papers(*, query: str = "", status: str = "", limit: int = 200) -> list[dict[str, Any]]:
    sql = "SELECT * FROM papers"
    where: list[str] = []
    args: list[Any] = []
    if query.strip():
        where.append("(title LIKE ? OR filename LIKE ? OR abstract LIKE ?)")
        like = f"%{query.strip()}%"
        args += [like, like, like]
    if status.strip():
        where.append("status = ?")
        args.append(status.strip())
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    args.append(max(1, min(1000, int(limit))))
    with _read_conn() as conn:
        return [_public(_row_dict(r)) for r in conn.execute(sql, args)]


def library_stats() -> dict[str, Any]:
    with _read_conn() as conn:
        rows = conn.execute("SELECT status, COUNT(*) n FROM papers GROUP BY status").fetchall()
        total = conn.execute("SELECT COUNT(*) n, COALESCE(SUM(page_count),0) p FROM papers").fetchone()
    return {
        "papers": total["n"],
        "pages": total["p"],
        "by_status": {r["status"]: r["n"] for r in rows},
        "dir": str(library_dir()),
    }


def _write_row(pid: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _write_conn() as conn:
        conn.executescript(PAPER_SCHEMA)
        conn.execute(f"UPDATE papers SET {cols} WHERE id = ?", [*fields.values(), pid])


def _delete(pid: str) -> bool:
    row = _fetch(pid)
    if row is None:
        return False
    root = library_dir().resolve()
    target = Path(row["dir"]).resolve()
    if target.is_relative_to(root):
        import shutil

        shutil.rmtree(target, ignore_errors=True)
    with _write_conn() as conn:
        conn.executescript(PAPER_SCHEMA)
        conn.execute("DELETE FROM papers WHERE id = ?", (pid,))
    _bump()
    return True


def rename_paper(pid: str, title: str) -> dict[str, Any]:
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    if not title or len(title) > 300:
        raise LibraryError("标题不能为空且不超过 300 字符")
    _require(pid)
    _write_row(pid, title=title)
    return get_paper(pid) or {}


def delete_paper(pid: str) -> bool:
    return _delete(pid)


def retry_paper(pid: str) -> dict[str, Any]:
    row = _require(pid)
    if row["status"] == "ready":
        return get_paper(pid) or {}
    _write_row(pid, status="queued", progress=0.0, error="")
    _submit(pid)
    return get_paper(pid) or {}


# ────────────────────────── arXiv 版本 / 去重 ──────────────────────────

_ARXIV_RE = re.compile(
    r"(?i)(?:arxiv[:/\s]*)?((?:\d{4}\.\d{4,5})|(?:[a-z][a-z\-]+/\d{7}))(v\d+)?"
)
_NORM_RE = re.compile(r"[^0-9a-z一-鿿]+")


def _arxiv_from(text: str) -> tuple[str, str]:
    m = _ARXIV_RE.search(text or "")
    if not m:
        return "", ""
    return m.group(1), (m.group(2) or "").lower()


def _norm_title(text: str) -> str:
    return _NORM_RE.sub("", (text or "").lower())


def _title_sim(a: str, b: str) -> float:
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _quick_info(path: Path, filename: str) -> dict[str, Any]:
    """只读第一页，上传后立即返回标题/页数/arXiv 版本，不等完整解析。"""
    info: dict[str, Any] = {"pages": 0, "title": Path(filename).stem, "text": "",
                            "authors": [], "year": None}
    try:
        doc = pm.open(str(path))
        try:
            info["pages"] = int(doc.page_count)
            meta = doc.metadata or {}
            text = (doc[0].get_text("text") or "")[:8000] if doc.page_count else ""
            info["text"] = text
            meta_title = re.sub(r"\s+", " ", str(meta.get("title") or "")).strip()
            if meta_title and len(meta_title) >= 6 and "untitled" not in meta_title.lower():
                info["title"] = meta_title[:300]
            else:
                for line in text.splitlines():
                    line = re.sub(r"\s+", " ", line).strip()
                    if len(line) >= 8 and not re.match(r"(?i)arxiv|doi|http", line):
                        info["title"] = line[:300]
                        break
            author = str(meta.get("author") or "").strip()
            if author:
                info["authors"] = [a.strip() for a in re.split(r"[;；]| and ", author) if a.strip()][:30]
            created = str(meta.get("creationDate") or "")
            m = re.search(r"(19|20)\d{2}", created)
            if m:
                info["year"] = int(m.group(0))
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("quick pdf probe failed: %s", exc)
    base, version = _arxiv_from(filename + "\n" + info["text"])
    if base:
        info["arxiv_id"] = f"{base}{version}"
        info["base_id"] = base
        info["version"] = version
        # PDF creationDate 常是文件生成时间，arXiv 编号里的年份更可信
        yy = int(base[:2]) if "." in base else None
        if yy is not None:
            info["year"] = 2000 + yy if yy < 90 else 1900 + yy
    return info


def _duplicate_candidates(*, title: str, base_id: str, exclude: str = "") -> list[dict[str, Any]]:
    with _read_conn() as conn:
        rows = conn.execute("SELECT * FROM papers WHERE id != ? ORDER BY created_at DESC", (exclude,)).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        other = _row_dict(row)
        if base_id and other["base_id"] == base_id:
            out.append({
                "relation": "version",
                "score": 1.0,
                "paper": _public(other),
                "note": (
                    f"同一 arXiv 论文：已有 {other['version'] or '无版本号'}，"
                    f"本次 {other['version'] or ''}；保留两版可对比差异。"
                ),
            })
            continue
        score = _title_sim(title, other["title"])
        if score >= 0.88:
            out.append({
                "relation": "similar",
                "score": round(score, 3),
                "paper": _public(other),
                "note": f"标题相似度 {score:.2f}，可能是同一篇论文的不同文件。",
            })
    return out[:5]


# ────────────────────────── 上传 ──────────────────────────

def create_paper(filename: str, data: bytes) -> dict[str, Any]:
    """保存原文件并排队解析；完全相同的文件直接返回已有文献。"""
    filename = Path(filename or "paper.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise LibraryError("文献库只接收 PDF 文件")
    if len(data) > MAX_PDF_BYTES:
        raise LibraryError(f"PDF 超过 {MAX_PDF_BYTES // 1024 // 1024} MB 上限")
    if not data.startswith(b"%PDF-"):
        raise LibraryError("不是有效的 PDF（缺少 %PDF- 文件头）")

    sha = hashlib.sha256(data).hexdigest()
    with _read_conn() as conn:
        existing = conn.execute("SELECT * FROM papers WHERE sha256 = ?", (sha,)).fetchone()
    if existing:
        paper = _public(_row_dict(existing))
        return {"created": False, "paper": paper,
                "duplicates": [{"relation": "exact", "score": 1.0, "paper": paper,
                                "note": "文件逐字节相同，已返回库中已有文献。"}]}

    pid = f"{_slug(Path(filename).stem)[:44] or 'paper'}-{sha[:10]}"
    root = library_dir() / pid
    root.mkdir(parents=True, exist_ok=True)
    target = root / "original.pdf"
    tmp = root / "original.pdf.part"
    tmp.write_bytes(data)
    tmp.replace(target)

    info = _quick_info(target, filename)
    duplicates = _duplicate_candidates(title=info["title"], base_id=info.get("base_id", ""), exclude=pid)
    now = _now()
    meta = {"source_filename": filename, "quick": {"pages": info["pages"]}}
    with _write_conn() as conn:
        conn.executescript(PAPER_SCHEMA)
        conn.execute(
            """INSERT INTO papers (
                id,title,filename,sha256,arxiv_id,base_id,version,status,progress,error,
                page_count,size_bytes,authors_json,abstract,year,keywords_json,dir,file_path,
                docir_path,reader_path,chunks_path,translations_path,duplicate_of,meta_json,
                created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                pid, info["title"], filename, sha, info.get("arxiv_id", ""), info.get("base_id", ""),
                info.get("version", ""), "queued", 0.0, "", info["pages"], len(data),
                json.dumps(info["authors"], ensure_ascii=False), "", info.get("year"),
                "[]", str(root), str(target),
                str(root / "docir.json"), str(root / "reader.json"), str(root / "chunks.json"),
                str(root / "translations.json"), "", json.dumps(meta, ensure_ascii=False),
                now, now,
            ),
        )
    _bump()
    _submit(pid)
    paper = get_paper(pid) or {}
    return {"created": True, "paper": paper, "duplicates": duplicates}


def _slug(text: str) -> str:
    text = re.sub(r"[^\w一-鿿\-]+", "-", text, flags=re.UNICODE).strip("-_")
    return text[:44] or "paper"


def replace_with(new_id: str, old_id: str) -> dict[str, Any]:
    """用新版本替换旧条目：删旧文件，保留新条目。"""
    if new_id == old_id:
        raise LibraryError("新旧文献不能相同")
    new = _require(new_id)
    _require(old_id)
    _delete(old_id)
    _write_row(new_id, duplicate_of="")
    _bump()
    return get_paper(new_id) or _public(_row_dict(new))


# ────────────────────────── 后台解析入库 ──────────────────────────

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="paper-parse")
_JOBS: set[str] = set()
_JOBS_LOCK = threading.Lock()


def _submit(pid: str) -> None:
    with _JOBS_LOCK:
        if pid in _JOBS:
            return
        _JOBS.add(pid)
    _EXECUTOR.submit(_parse_job, pid)


def _job_done(pid: str) -> None:
    with _JOBS_LOCK:
        _JOBS.discard(pid)


def resume_incomplete() -> None:
    """服务重启后把 queued/parsing 的任务重新跑一遍；半成品不冒充成功。"""
    init()
    with _read_conn() as conn:
        rows = conn.execute("SELECT id,status FROM papers WHERE status IN ('queued','parsing')").fetchall()
    for row in rows:
        if row["status"] == "parsing":
            _write_row(row["id"], status="queued", progress=0.0, error="服务重启，重新解析")
        _submit(row["id"])


def _parse_job(pid: str) -> None:
    try:
        row = _fetch(pid)
        if row is None:
            return
        _write_row(pid, status="parsing", progress=0.05, error="")
        source = Path(row["file_path"])
        assets = Path(row["dir"]) / "assets"
        assets.mkdir(parents=True, exist_ok=True)

        parsed = parse_pdf(source, ocr="auto", assets_dir=assets)
        docir = parsed.model_dump(mode="json")
        Path(row["docir_path"]).write_text(
            json.dumps(docir, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_row(pid, progress=0.42)

        title = _doc_title(docir, row["title"])
        reader = build_reader(docir, title=title)
        _rewrite_assets(reader, assets)
        Path(row["reader_path"]).write_text(
            json.dumps(reader, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_row(pid, progress=0.7)

        chunks = _build_chunks(reader, pid, title)
        Path(row["chunks_path"]).write_text(
            json.dumps([c.__dict__ for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        abstract = _extract_abstract(reader)
        keywords = _keywords_from(reader, docir)
        meta = _json(row["meta_json"], {})
        meta["stats"] = {
            **(docir.get("stats") or {}),
            "chunks": len(chunks),
            "reader": reader.get("stats") or {},
        }
        meta["warnings"] = (docir.get("warnings") or []) + (reader.get("warnings") or [])
        _write_row(
            pid,
            status="ready",
            progress=1.0,
            error="",
            title=title,
            abstract=abstract,
            keywords_json=json.dumps(keywords, ensure_ascii=False),
            page_count=int((docir.get("source") or {}).get("pages") or reader["meta"]["pages"]),
            meta_json=json.dumps(meta, ensure_ascii=False),
        )
        _bump()
        log.info("paper %s parsed: %s blocks, %s chunks", pid, len(reader["blocks"]), len(chunks))
    except Exception as exc:  # noqa: BLE001
        log.exception("paper parse failed: %s", pid)
        _write_row(pid, status="failed", progress=0.0, error=f"解析失败：{exc}")
        _bump()
    finally:
        _job_done(pid)


def _doc_title(docir: dict[str, Any], fallback: str) -> str:
    for block in docir.get("blocks") or []:
        if block.get("type") == "title" and (block.get("text") or "").strip():
            return re.sub(r"\s+", " ", block["text"]).strip()[:300]
    return fallback[:300] or "未命名文献"


def _rewrite_assets(reader: dict[str, Any], assets: Path) -> None:
    root = assets.resolve()
    for block in reader.get("blocks") or []:
        raw = block.get("image_path")
        if not raw:
            continue
        p = Path(raw)
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p
        block["image_path"] = p.name if resolved.is_relative_to(root) else ""


def _extract_abstract(reader: dict[str, Any]) -> str:
    blocks = reader.get("blocks") or []
    start = -1
    for i, b in enumerate(blocks):
        if b.get("type") == "heading" and re.search(r"(?i)\babstract\b|摘\s*要", b.get("text") or ""):
            start = i
            break
    if start >= 0:
        parts: list[str] = []
        for b in blocks[start + 1:]:
            if b.get("type") in ("heading", "title"):
                break
            if b.get("type") == "paragraph" and (b.get("text") or "").strip():
                parts.append(b["text"].strip())
                if sum(len(x) for x in parts) > 4000:
                    break
        if parts:
            return "\n".join(parts)[:4000]
    for b in blocks[:40]:
        if b.get("type") == "paragraph":
            text = (b.get("text") or "").strip()
            if len(text) >= 200:
                return text[:4000]
    return ""


def _keywords_from(reader: dict[str, Any], docir: dict[str, Any]) -> list[str]:
    raw = str((docir.get("source") or {}).get("keywords") or "")
    words = [x.strip() for x in re.split(r"[;,；、]", raw) if x.strip()]
    if words:
        return words[:20]
    # 只从原文的 "Index Terms / Keywords" 行取，不凭空生成
    for b in (reader.get("blocks") or [])[:80]:
        text = (b.get("text") or "").strip()
        m = re.match(r"(?i)(?:index terms|keywords?)\s*[:：\-—]\s*(.+)", text)
        if m:
            return [x.strip() for x in re.split(r"[;,；、]", m.group(1)) if x.strip()][:20]
    return []


def _section_type(crumb: str) -> str:
    low = (crumb or "").lower()
    for key, name in (
        ("abstract", "abstract"), ("introduction", "introduction"), ("related", "related_work"),
        ("method", "method"), ("approach", "method"), ("experiment", "experimental_setup"),
        ("result", "results"), ("ablation", "results"), ("conclusion", "conclusion"),
        ("future", "future_work"), ("appendix", "appendix"), ("reference", "references"),
    ):
        if key in low:
            return name
    return "body"


def _build_chunks(reader: dict[str, Any], doc_id: str, title: str) -> list[Chunk]:
    """按标题层级切节，节内用流式打包器切块；块保留页码/block 锚点供前端跳转。"""
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    state: dict[str, Any] = {"crumb": title, "type": "body", "packer": None}

    def crumb() -> str:
        return " > ".join([title] + [t for _, t in stack if t and t != title]).strip(" >")

    def make_packer() -> StreamingPacker:
        c = crumb()
        st = _section_type(c)

        def emit(units: list[tuple[str, dict[str, Any]]]) -> None:
            if not units:
                return
            body = " ".join(u[0] for u in units if u[0])
            if not body.strip():
                return
            pages = [int(u[1].get("page") or 0) for u in units]
            ids = list(dict.fromkeys(str(u[1].get("block_id")) for u in units if u[1].get("block_id")))
            bbox = next((u[1].get("bbox") for u in units if u[1].get("bbox")), [])
            text = f"【{c}】\n{body}" if c else body
            chunks.append(Chunk(
                id=f"{doc_id}:c{len(chunks):05d}", text=text, doc_id=doc_id, title=title,
                section_path=c, page_start=min(pages) if pages else 0,
                page_end=max(pages) if pages else 0, block_ids=ids,
                metadata={"section_type": st, "bbox": bbox,
                          "first_block": ids[0] if ids else ""},
            ))

        return StreamingPacker(CHAPTER_TARGET, CHAPTER_OVERLAP, CHAPTER_MAX, emit)

    state["packer"] = make_packer()
    for block in reader.get("blocks") or []:
        btype = block.get("type") or ""
        if btype in ("heading", "title"):
            state["packer"].flush()
            if btype == "heading":
                level = int(block.get("level") or 1)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, (block.get("text") or "").strip()))
            state["packer"] = make_packer()
            continue
        if btype in SKIP_CHUNK:
            continue
        text = (block.get("markdown") or block.get("latex") or block.get("caption")
                or block.get("text") or "").strip()
        if not text:
            continue
        if btype == "unknown" and float(block.get("confidence") or 0) < 0.35:
            continue
        state["packer"].add(text, {
            "block_id": block.get("id"), "page": block.get("page"),
            "bbox": block.get("bbox") or [], "type": btype,
        })
    state["packer"].flush()

    refs = reader.get("references") or []
    if refs:
        text = "\n".join(r.get("text") or "" for r in refs)[:12000]
        chunks.append(Chunk(
            id=f"{doc_id}:refs", text=f"【References】\n{text}", doc_id=doc_id, title=title,
            section_path="References", page_start=min(int(r.get("page") or 0) for r in refs),
            page_end=max(int(r.get("page") or 0) for r in refs),
            block_ids=[r.get("block_id") for r in refs if r.get("block_id")],
            metadata={"section_type": "references", "first_block": refs[0].get("block_id", "")},
        ))
    return chunks


# ────────────────────────── 检索 ──────────────────────────

_CACHE_LOCK = threading.Lock()
_REV = 0
_CACHE: dict[str, Any] = {"pipeline": None, "chunks": {}}


def _load_chunks() -> dict[str, Chunk]:
    out: dict[str, Chunk] = {}
    with _read_conn() as conn:
        rows = conn.execute(
            "SELECT id,chunks_path FROM papers WHERE status = 'ready' ORDER BY created_at"
        ).fetchall()
    for row in rows:
        try:
            raw = json.loads(Path(row["chunks_path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("load chunks failed %s: %s", row["id"], exc)
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                out[str(item.get("id"))] = Chunk(
                    id=str(item.get("id")), text=str(item.get("text") or ""),
                    doc_id=str(item.get("doc_id") or row["id"]),
                    title=str(item.get("title") or ""),
                    section_path=str(item.get("section_path") or ""),
                    page_start=int(item.get("page_start") or 0),
                    page_end=int(item.get("page_end") or 0),
                    block_ids=list(item.get("block_ids") or []),
                    metadata=dict(item.get("metadata") or {}),
                )
            except (TypeError, ValueError):
                continue
    return out


def _pipeline() -> tuple[AcademicRetrievalPipeline, dict[str, Chunk]]:
    with _CACHE_LOCK:
        if _CACHE["pipeline"] is not None:
            return _CACHE["pipeline"], _CACHE["chunks"]
        chunks = _load_chunks()
        if not chunks:
            raise LibraryError("文献库里还没有解析完成的论文")
        pipe = AcademicRetrievalPipeline(
            list(chunks.values()),
            config=RetrievalConfig(final_top_k=8, rerank_top_k=10, fusion_top_k=50,
                                   use_rewrite=True, use_filter=True),
        ).build()
        _CACHE["pipeline"], _CACHE["chunks"] = pipe, chunks
        return pipe, chunks


def search(query: str, *, paper_ids: Iterable[str] | None = None, top_k: int = 6) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        raise LibraryError("问题不能为空")
    pipe, by_id = _pipeline()
    plan = plan_query(query, enable_rewrite=True, enable_filter=True)
    ids = [x for x in (paper_ids or []) if x]
    if ids:
        plan.filters.any_of["doc_id"] = ids
    result = pipe.search(query, top_k=max(1, min(20, int(top_k))), plan=plan)
    evidence: list[dict[str, Any]] = []
    for hit in result.get("hits") or []:
        chunk = by_id.get(hit.get("id"))
        if chunk is None:
            continue
        evidence.append({
            "id": hit.get("id"),
            "paper_id": chunk.doc_id,
            "title": chunk.title,
            "section": chunk.section_path,
            "page": chunk.page_start,
            "page_end": chunk.page_end,
            "block_id": (chunk.block_ids or [""])[0],
            "anchor": f"blk-{chunk.block_ids[0]}" if chunk.block_ids else "",
            "bbox": (chunk.metadata or {}).get("bbox") or [],
            "section_type": (chunk.metadata or {}).get("section_type") or "",
            "score": round(float(hit.get("score") or 0), 4),
            "quote": _clean_chunk_text(chunk.text)[:1400],
            "url": f"/api/papers/{chunk.doc_id}/reader?anchor={chunk.block_ids[0] if chunk.block_ids else ''}",
        })
    return {
        "query": query,
        "evidence": evidence,
        "plan": result.get("plan") or {},
        "engines": result.get("engines") or {},
        "stages": result.get("stages") or {},
        "elapsed_ms": result.get("elapsed_ms") or 0,
    }


def _clean_chunk_text(text: str) -> str:
    return re.sub(r"^【[^】]*】\n?", "", text or "").strip()


def answer(query: str, *, paper_ids: Iterable[str] | None = None, top_k: int = 6) -> dict[str, Any]:
    found = search(query, paper_ids=paper_ids, top_k=top_k)
    evidence = found["evidence"]
    warnings: list[str] = []
    if not evidence:
        return {**found, "answer": "文献库中没有检索到与该问题相关的原文片段。", "mode": "none",
                "warnings": ["没有证据，未调用模型生成答案。"]}

    settings = get_settings()
    if settings.use_mock_llm:
        answer_text = _extractive_answer(evidence)
        warnings.append("未配置可用大模型，返回检索证据原文（未生成摘要）。")
        mode = "extractive"
    else:
        try:
            from app.providers.llm import get_llm

            messages = [
                {"role": "system", "content": (
                    "你是科研文献问答助手。只能依据用户提供的证据回答，"
                    "每个结论后用 [E1] 这样的编号标注依据；证据不足就明确说“文献中没有足够信息”。"
                    "不要使用证据之外的文献、DOI 或数字，不要编造引用。"
                    "用与问题相同的语言回答，先给结论，再给必要的细节。"
                )},
                {"role": "user", "content": _evidence_prompt(query, evidence)},
            ]
            resp = get_llm().chat(messages, temperature=0.1, max_tokens=1600)
            answer_text = (getattr(resp, "content", "") or "").strip()
            if not answer_text:
                raise RuntimeError("模型返回空答案")
            mode = "llm"
            if "[E" not in answer_text:
                warnings.append("模型未在答案中标注证据编号，请以下方原文片段为准。")
        except Exception as exc:  # noqa: BLE001
            log.warning("grounded answer failed: %s", exc)
            answer_text = _extractive_answer(evidence)
            warnings.append(f"模型调用失败，已降级为原文证据：{exc}")
            mode = "extractive"

    return {**found, "answer": answer_text, "mode": mode, "warnings": warnings,
            "model": get_settings().LLM_MODEL}


def _evidence_prompt(query: str, evidence: list[dict[str, Any]]) -> str:
    lines = [f"问题：{query}", "", "证据："]
    for i, e in enumerate(evidence, 1):
        lines.append(
            f"[E{i}] 论文《{e['title']}》 p{e['page']} · {e['section'] or '正文'}\n{e['quote']}\n"
        )
    return "\n".join(lines)


def _extractive_answer(evidence: list[dict[str, Any]]) -> str:
    lines = ["以下为文献库中检索到的原文证据（未生成摘要）：", ""]
    for i, e in enumerate(evidence[:4], 1):
        lines.append(f"[E{i}]《{e['title']}》 p{e['page']}：{e['quote'][:700]}")
        lines.append("")
    return "\n".join(lines).strip()


# ────────────────────────── 阅读器 / 原文 / 资源 ──────────────────────────

def reader(pid: str) -> dict[str, Any]:
    row = _require(pid)
    if row["status"] != "ready":
        raise LibraryError(f"文献尚未解析完成（{row['status']}）")
    try:
        data = json.loads(Path(row["reader_path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LibraryError(f"阅读器数据读取失败：{exc}") from None
    data["paper"] = _public(_row_dict(row))
    data["translations"] = _translations(pid)
    return data


def file_path(pid: str) -> Path:
    row = _require(pid)
    path = Path(row["file_path"]).resolve()
    root = library_dir().resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise LibraryError("原文件不存在")
    return path


def asset_path(pid: str, name: str) -> Path:
    row = _require(pid)
    base = (Path(row["dir"]) / "assets").resolve()
    path = (base / Path(name).name).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise LibraryError("资源不存在")
    return path


def _doc_view(row: sqlite3.Row) -> dict[str, Any]:
    try:
        reader = json.loads(Path(row["reader_path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LibraryError(f"阅读器数据读取失败：{exc}") from None
    sections: list[dict[str, Any]] = []
    heading = ""
    for b in reader.get("blocks") or []:
        kind = b.get("type")
        if kind in ("heading", "title"):
            text = (b.get("text") or "").strip()
            # 页眉的 arXiv 印章也常被判成标题，别让它冒充章节名
            if text and not text.lower().startswith("arxiv:"):
                heading = text
        elif kind in ("paragraph", "list_item") and (b.get("text") or "").strip():
            sections.append({"id": b.get("id") or f"b{len(sections)}",
                             "heading": heading, "text": b["text"]})
    # doc_id 用 arXiv 号（含 vN），dedup 靠它判定"同一主干"；没有才退回本地 id
    return {"doc_id": row["arxiv_id"] or row["id"], "title": row["title"],
            "abstract": row["abstract"] or "", "sections": sections,
            "version": row["version"] or ""}


def compare_versions(pid: str, other_id: str) -> dict[str, Any]:
    """两版差异：语义判定 + 逐段规则 diff（不调用 LLM，避免上百次请求）。"""
    if pid == other_id:
        raise LibraryError("请选择两篇不同的文献")
    a, b = _require(pid), _require(other_id)
    if a["status"] != "ready" or b["status"] != "ready":
        raise LibraryError("两篇文献都需要解析完成")
    # 统一按旧 → 新排序，版本建议的"新上传"才指向真正的较新版本
    def vnum(row: sqlite3.Row) -> int:
        m = re.search(r"(\d+)\s*$", row["version"] or "")
        return int(m.group(1)) if m else -1

    if vnum(a) < 0 or (vnum(b) >= 0 and vnum(a) > vnum(b)):
        a, b = b, a
    from app.agents.dedup import compare_papers

    return compare_papers(_doc_view(a), _doc_view(b), use_llm=False)


def _translations(pid: str) -> dict[str, Any]:
    row = _require(pid)
    try:
        return json.loads(Path(row["translations_path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def translate_blocks(
    pid: str,
    *,
    block_ids: Iterable[str] | None = None,
    page: int | None = None,
    limit: int = 16,
    force: bool = False,
) -> dict[str, Any]:
    """按块翻译并缓存；一次最多处理 ``limit`` 块，前端按页推进可覆盖整篇。"""
    row = _require(pid)
    if row["status"] != "ready":
        raise LibraryError("文献尚未解析完成")
    try:
        reader_data = json.loads(Path(row["reader_path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LibraryError(f"阅读器数据读取失败：{exc}") from None

    wanted = set(block_ids or [])
    blocks = [
        b for b in reader_data.get("blocks") or []
        if b.get("type") in TRANSLATABLE
        and (not wanted or b.get("id") in wanted)
        and (page is None or int(b.get("page") or 0) == int(page))
        and (b.get("text") or "").strip()
    ]
    cache = _translations(pid)
    todo = [b for b in blocks if force or not _cached(cache, b)]
    batch = todo[:max(1, min(40, int(limit)))]
    if not batch:
        return {"translations": cache, "translated": 0, "remaining": 0}

    from app.translate import LLMTranslator, translate_paragraphs

    try:
        translator = LLMTranslator()
    except Exception as exc:  # noqa: BLE001
        raise LibraryError(f"翻译不可用：{exc}") from None

    done = 0
    warnings: list[str] = []
    for block in batch:
        text = (block.get("text") or "").strip()
        try:
            result = translate_paragraphs(text, translator, max_unit_chars=600)
            zh = "\n\n".join(str(p.get("zh") or "") for p in result.get("pairs") or []).strip()
            if zh:
                cache[block["id"]] = {"zh": zh, "src_sha": _sha_text(text), "at": _now()}
                done += 1
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{block.get('id')}: {exc}")
    Path(row["translations_path"]).write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    remaining = len(todo) - done
    if done == 0 and warnings:
        raise LibraryError("翻译失败：" + warnings[0])
    return {"translations": cache, "translated": done, "remaining": remaining,
            "warnings": warnings[:10]}


def _cached(cache: dict[str, Any], block: dict[str, Any]) -> bool:
    item = cache.get(block.get("id") or "")
    return bool(item and item.get("src_sha") == _sha_text(block.get("text") or ""))


def _sha_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def translate_selection(pid: str, text: str) -> dict[str, str]:
    _require(pid)
    text = (text or "").strip()
    if not text:
        raise LibraryError("选中文本为空")
    if len(text) > 6000:
        raise LibraryError("选中文本过长（上限 6000 字符）")
    from app.translate import LLMTranslator, translate_paragraphs

    try:
        result = translate_paragraphs(text, LLMTranslator(), max_unit_chars=600)
    except Exception as exc:  # noqa: BLE001
        raise LibraryError(f"翻译失败：{exc}") from None
    return {"zh": "\n\n".join(str(p.get("zh") or "") for p in result.get("pairs") or []).strip()}


def markdown(pid: str, *, lang: str = "en") -> str:
    data = reader(pid)
    cache = data.get("translations") or {}
    lines = [f"# {data['meta'].get('title') or pid}", ""]
    for block in data.get("blocks") or []:
        btype = block.get("type")
        text = (block.get("text") or "").strip()
        zh = (cache.get(block.get("id") or "") or {}).get("zh", "")
        if btype == "heading":
            level = min(6, max(2, int(block.get("level") or 1) + 1))
            lines += ["#" * level + " " + text, ""]
        elif btype == "table" and block.get("markdown"):
            lines += [block["markdown"], ""]
        elif btype == "formula":
            lines += ["$$", block.get("latex") or text, "$$", ""]
        elif btype in ("figure", "figure_caption", "table_caption"):
            cap = block.get("caption") or text
            if cap:
                lines += [f"*{cap}*", ""]
        elif text:
            if lang in ("bi", "bilingual") and zh:
                lines += [text, "", f"> {zh}", ""]
            elif lang == "zh" and zh:
                lines += [zh, ""]
            else:
                lines += [text, ""]
    if data.get("references"):
        lines += ["## References", ""]
        lines += [f"- {r.get('text')}" for r in data["references"]]
        lines.append("")
    return "\n".join(lines).strip() + "\n"
