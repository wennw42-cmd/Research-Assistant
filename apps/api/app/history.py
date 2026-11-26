"""历史落库（SQLite，零依赖，Navicat 可直接打开）。

表结构
------
``translations``       一次翻译一行：方向、原文、译文、模型、耗时、统计、校验提示；
``translation_terms``  术语表按行存储，外键指向 ``translations.id``；
``chat_threads``       会话：标题、滚动摘要、创建/更新时间；
``chat_messages``      会话消息（user / assistant / system / tool），按 id 排序即时间序；
``agent_runs``         一轮 Agent 执行的完整结果（路由、计划、答案、校验、证据、耗时）；
``agent_tasks``        该轮 DAG 的每个子任务（状态、输入、结果、错误）；
``generated_images``   文生图记录（提示词、实际提示词、模型、落盘图片、任务 ID）。

设计约束
--------
1. 写库失败**不能**影响主流程：调用方捕获异常并提示，绝不伪造“已保存”；
2. 只存本地文件，不对外发送；路径默认 ``apps/api/data/research_assistant.db``；
3. 只做追加与读取，不静默改历史；删除会话由用户显式触发（级联删除消息与执行记录）。

运行（在 apps/api 下）：
    python -m app.history --selftest
    python -m app.history --list 20
    python -m app.history --db data/demo.db --list
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import threading
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from app.config import get_settings

APP_DIR = Path(__file__).resolve().parent.parent
_SCHEMA_LOCK = threading.Lock()
_READY: set[str] = set()
DIRECTIONS = ("zh2en", "en2zh")
ROLES = ("user", "assistant", "system", "tool")

SCHEMA = """
CREATE TABLE IF NOT EXISTS translations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT    NOT NULL,
    direction          TEXT    NOT NULL,
    source_text        TEXT    NOT NULL,
    translation        TEXT    NOT NULL,
    source_chars       INTEGER NOT NULL DEFAULT 0,
    translation_chars  INTEGER NOT NULL DEFAULT 0,
    model              TEXT    NOT NULL DEFAULT '',
    latency_ms         INTEGER,
    stats_json         TEXT    NOT NULL DEFAULT '{}',
    issues_json        TEXT    NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_translations_created
    ON translations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_translations_direction
    ON translations(direction, id DESC);

CREATE TABLE IF NOT EXISTS translation_terms (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    translation_id  INTEGER NOT NULL REFERENCES translations(id) ON DELETE CASCADE,
    source_term     TEXT    NOT NULL,
    target_term     TEXT    NOT NULL,
    note            TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_terms_translation
    ON translation_terms(translation_id);

CREATE TABLE IF NOT EXISTS chat_threads (
    thread_id   TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL DEFAULT 'agent',
    title       TEXT NOT NULL DEFAULT '',
    summary     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_threads_updated
    ON chat_threads(updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    meta_json   TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread
    ON chat_messages(thread_id, id);

CREATE TABLE IF NOT EXISTS agent_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_thread      TEXT NOT NULL UNIQUE,
    thread_id         TEXT NOT NULL REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
    turn              INTEGER NOT NULL DEFAULT 0,
    query             TEXT NOT NULL DEFAULT '',
    route             TEXT,
    complexity        TEXT,
    confidence        REAL,
    route_reason      TEXT,
    intent            TEXT,
    answer            TEXT NOT NULL DEFAULT '',
    plan_json         TEXT NOT NULL DEFAULT '[]',
    task_results_json TEXT NOT NULL DEFAULT '{}',
    verify_json       TEXT NOT NULL DEFAULT '{}',
    evidence_json     TEXT NOT NULL DEFAULT '[]',
    trace_json        TEXT NOT NULL DEFAULT '[]',
    latency_ms        INTEGER,
    model             TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'ok',
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_thread
    ON agent_runs(thread_id, id DESC);

CREATE TABLE IF NOT EXISTS agent_tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    task_id     TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    agent       TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT '',
    inputs_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    latency_ms  INTEGER,
    error       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_run
    ON agent_tasks(run_id);

CREATE TABLE IF NOT EXISTS generated_images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    actual_prompt   TEXT NOT NULL DEFAULT '',
    negative_prompt TEXT NOT NULL DEFAULT '',
    size            TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    task_id         TEXT NOT NULL DEFAULT '',
    images_json     TEXT NOT NULL DEFAULT '[]',
    latency_ms      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_generated_images_created
    ON generated_images(created_at DESC);
"""


class HistoryError(ValueError):
    """历史库参数或读写错误。"""


@dataclass(frozen=True)
class SavedTranslation:
    id: int
    created_at: str
    direction: str


def db_path() -> Path:
    """返回绝对路径；相对路径按 apps/api 解析，避免受启动目录影响。"""
    raw = Path(get_settings().SQLITE_PATH or "data/research_assistant.db")
    return raw if raw.is_absolute() else (APP_DIR / raw).resolve()


def _connect(path: Path | None = None) -> sqlite3.Connection:
    target = Path(path) if path else db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 3000")
    key = str(target.resolve()).lower()
    with _SCHEMA_LOCK:
        if key not in _READY:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            _migrate(conn)
            _READY.add(key)
    return conn


@contextmanager
def _write_conn(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """写事务 + 收尾 checkpoint：把 WAL 页并回主库文件。

    Navicat 等 GUI 有的不识别 ``-wal``，只看主文件会以为新表不存在；
    每次写完做一次 PASSIVE checkpoint（不阻塞读），主文件始终是最新的。
    """
    conn = _connect(path)
    try:
        with conn:
            yield conn
    finally:
        try:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.Error:
            pass
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """老库补列：CREATE TABLE IF NOT EXISTS 不会修改已存在的表。"""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chat_threads)")}
    if cols and "kind" not in cols:
        conn.execute("ALTER TABLE chat_threads ADD COLUMN kind TEXT NOT NULL DEFAULT 'agent'")


def init_db(path: Path | None = None) -> Path:
    """建库建表；返回实际文件路径。"""
    target = Path(path) if path else db_path()
    _connect(target).close()
    return target


def _term_rows(
    glossary: Sequence[Mapping[str, Any]] | Mapping[str, str] | None,
    direction: str,
) -> list[tuple[str, str, str]]:
    """统一成 (source_term, target_term, note)；术语表方向随翻译方向。"""
    rows: list[tuple[str, str, str]] = []
    if not glossary:
        return rows
    if isinstance(glossary, Mapping):
        # 映射本身就是 {source: target}：中译英是 {中文: English}，英译中是 {English: 中文}
        for src, dst in glossary.items():
            src, dst = str(src).strip(), str(dst).strip()
            if src and dst:
                rows.append((src[:200], dst[:200], ""))
        return rows
    items: Iterable[Any] = glossary
    for item in items:
        if isinstance(item, Mapping):
            zh = str(item.get("zh") or "").strip()
            en = str(item.get("en") or "").strip()
            note = str(item.get("note") or "").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            zh, en = str(item[0]).strip(), str(item[1]).strip()
            note = str(item[2]).strip() if len(item) >= 3 else ""
        else:
            continue
        if not zh or not en:
            continue
        src, dst = (en, zh) if direction == "en2zh" else (zh, en)
        rows.append((src, dst, note[:500]))
    return rows


def save_translation(
    *,
    direction: str,
    source_text: str,
    translation: str,
    glossary: Sequence[Mapping[str, Any]] | Mapping[str, str] | None = None,
    issues: Sequence[str] | None = None,
    stats: Mapping[str, Any] | None = None,
    model: str = "",
    latency_ms: int | None = None,
    path: Path | None = None,
) -> SavedTranslation:
    """追加一条翻译记录与对应术语；返回记录 ID。"""
    if direction not in DIRECTIONS:
        raise HistoryError(f"direction 只能是 {' / '.join(DIRECTIONS)}")
    source = str(source_text or "")
    target_text = str(translation or "")
    if not source.strip():
        raise HistoryError("原文为空，未写入历史")
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    terms = _term_rows(glossary, direction)
    stats_json = json.dumps(dict(stats or {}), ensure_ascii=False)
    issues_json = json.dumps([str(x) for x in (issues or [])], ensure_ascii=False)
    with _write_conn(path) as conn:
        cur = conn.execute(
            """INSERT INTO translations
               (created_at, direction, source_text, translation,
                source_chars, translation_chars, model, latency_ms,
                stats_json, issues_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                created_at, direction, source, target_text,
                len(source), len(target_text), str(model or ""),
                int(latency_ms) if latency_ms is not None else None,
                stats_json, issues_json,
            ),
        )
        record_id = int(cur.lastrowid)
        conn.executemany(
            """INSERT INTO translation_terms
               (translation_id, source_term, target_term, note)
               VALUES (?, ?, ?, ?)""",
            [(record_id, src, dst, note) for src, dst, note in terms],
        )
    return SavedTranslation(record_id, created_at, direction)


def list_translations(
    limit: int = 20,
    offset: int = 0,
    direction: str = "",
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """按时间倒序列出记录摘要；长文只返回预览。"""
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    where, params = "", []
    if direction:
        if direction not in DIRECTIONS:
            raise HistoryError(f"direction 只能是 {' / '.join(DIRECTIONS)}")
        where = "WHERE t.direction = ?"
        params.append(direction)
    sql = f"""
        SELECT t.id, t.created_at, t.direction, t.model, t.latency_ms,
               t.source_chars, t.translation_chars,
               substr(t.source_text, 1, 120)      AS source_preview,
               substr(t.translation, 1, 200)      AS translation_preview,
               (SELECT COUNT(*) FROM translation_terms x
                 WHERE x.translation_id = t.id)   AS term_count
          FROM translations t
          {where}
         ORDER BY t.id DESC
         LIMIT ? OFFSET ?
    """
    with closing(_connect(path)) as conn:
        rows = conn.execute(sql, (*params, limit, offset)).fetchall()
    return [dict(row) for row in rows]


def get_translation(record_id: int, *, path: Path | None = None) -> dict[str, Any] | None:
    """读取单条完整记录（含术语）。"""
    with closing(_connect(path)) as conn:
        row = conn.execute("SELECT * FROM translations WHERE id = ?", (int(record_id),)).fetchone()
        if row is None:
            return None
        terms = conn.execute(
            """SELECT source_term, target_term, note
                 FROM translation_terms WHERE translation_id = ? ORDER BY id""",
            (int(record_id),),
        ).fetchall()
    item = dict(row)
    item["stats"] = json.loads(item.pop("stats_json") or "{}")
    item["issues"] = json.loads(item.pop("issues_json") or "[]")
    item["terms"] = [dict(t) for t in terms]
    return item


# ══════════════════════════════ 会话 / Agent 执行 ══════════════════════════════
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _dumps(value: Any, default: Any) -> str:
    return json.dumps(default if value is None else value, ensure_ascii=False, default=str)


def _opt(value: Any) -> str | None:
    return None if value is None else str(value)[:2000]


def _opt_num(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _ensure_thread(conn: sqlite3.Connection, thread_id: str, *,
                   user_id: str = "", title: str = "", kind: str = "agent") -> None:
    """会话不存在则建；已有会话只补空字段并刷新 updated_at（kind 不覆盖）。"""
    now = _now()
    conn.execute(
        """INSERT INTO chat_threads (thread_id, user_id, kind, title, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(thread_id) DO UPDATE SET
             user_id    = CASE WHEN excluded.user_id <> '' THEN excluded.user_id
                               ELSE chat_threads.user_id END,
             title      = CASE WHEN chat_threads.title = '' THEN excluded.title
                               ELSE chat_threads.title END,
             updated_at = excluded.updated_at""",
        (thread_id, str(user_id or ""), str(kind or "agent")[:20],
         str(title or "")[:120], now, now),
    )


def ensure_thread(thread_id: str, *, user_id: str = "", title: str = "",
                  path: Path | None = None) -> None:
    tid = str(thread_id or "").strip()
    if not tid:
        raise HistoryError("thread_id 不能为空")
    with _write_conn(path) as conn:
        _ensure_thread(conn, tid, user_id=user_id, title=title)


def append_chat_message(
    thread_id: str,
    role: str,
    content: str,
    *,
    user_id: str = "",
    meta: Mapping[str, Any] | None = None,
    kind: str = "agent",
    path: Path | None = None,
) -> int:
    """追加一条会话消息；首条用户消息作为会话标题。返回消息 ID。

    kind 区分来源：``agent``（Agent 多步任务）或 ``general``（通用大模型问答）。
    """
    tid = str(thread_id or "").strip()
    if not tid:
        raise HistoryError("thread_id 不能为空")
    role = str(role or "").strip()
    if role not in ROLES:
        raise HistoryError(f"role 只能是 {' / '.join(ROLES)}")
    text = str(content or "")
    with _write_conn(path) as conn:
        _ensure_thread(conn, tid, user_id=user_id, kind=kind,
                       title=text[:60] if role == "user" else "")
        cur = conn.execute(
            """INSERT INTO chat_messages (thread_id, role, content, meta_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (tid, role, text, _dumps(meta, {}), _now()),
        )
        return int(cur.lastrowid)


def thread_messages(thread_id: str, limit: int = 0, *,
                    path: Path | None = None) -> list[dict[str, Any]]:
    """按时间正序返回消息；limit>0 时只取最近 N 条（取完再正序）。"""
    if limit and limit > 0:
        sql = """SELECT * FROM (
                     SELECT id, thread_id, role, content, meta_json, created_at
                       FROM chat_messages WHERE thread_id = ? ORDER BY id DESC LIMIT ?
                 ) ORDER BY id"""
        params: tuple[Any, ...] = (thread_id, int(limit))
    else:
        sql = """SELECT id, thread_id, role, content, meta_json, created_at
                   FROM chat_messages WHERE thread_id = ? ORDER BY id"""
        params = (thread_id,)
    with closing(_connect(path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    out = [dict(r) for r in rows]
    for item in out:
        item["meta"] = json.loads(item.pop("meta_json") or "{}")
    return out


def thread_turn(thread_id: str, *, path: Path | None = None) -> int:
    """已记录的轮次上界（进程重启后 next_turn 接着数）。"""
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn), 0) AS t FROM agent_runs WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
    return int(row["t"] or 0)


def thread_summary(thread_id: str, *, path: Path | None = None) -> str:
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT summary FROM chat_threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    return str(row["summary"] or "") if row else ""


def set_thread_summary(thread_id: str, summary: str, *, path: Path | None = None) -> None:
    with _write_conn(path) as conn:
        conn.execute(
            "UPDATE chat_threads SET summary = ? WHERE thread_id = ?",
            (str(summary or "")[-4000:], thread_id),
        )


def latest_task_state(thread_id: str, *, path: Path | None = None) -> dict[str, Any]:
    """最近一轮的 DAG 计划与子任务结果（重启后恢复任务中间态）。"""
    with closing(_connect(path)) as conn:
        row = conn.execute(
            """SELECT plan_json, task_results_json FROM agent_runs
                WHERE thread_id = ? ORDER BY id DESC LIMIT 1""",
            (thread_id,),
        ).fetchone()
    if row is None:
        return {"plan": [], "results": {}}
    return {
        "plan": json.loads(row["plan_json"] or "[]"),
        "results": json.loads(row["task_results_json"] or "{}"),
    }


def save_agent_run(result: Mapping[str, Any], *, path: Path | None = None) -> int:
    """写入/覆盖一轮 Agent 执行（按 graph_thread 去重），并刷新子任务行。

    HITL 场景：中断时先以 status=awaiting_human 落库，恢复后用同一 graph_thread
    再次调用即可原地更新为最终答案，不会产生重复行。
    """
    graph_thread = str(result.get("graph_thread") or "").strip()
    thread_id = str(result.get("thread_id") or "").strip()
    if not graph_thread or not thread_id:
        raise HistoryError("graph_thread / thread_id 不能为空")
    plan = result.get("plan") or []
    task_results = result.get("task_results") or {}
    created_at = _now()
    latency = result.get("latency_ms")
    values = (
        graph_thread, thread_id, int(result.get("turn") or 0),
        str(result.get("query") or ""), _opt(result.get("route")),
        _opt(result.get("complexity")), _opt_num(result.get("confidence")),
        _opt(result.get("route_reason")), _opt(result.get("intent")),
        str(result.get("answer") or ""),
        _dumps(plan, []), _dumps(task_results, {}),
        _dumps(result.get("verify"), {}), _dumps(result.get("evidence"), []),
        _dumps(result.get("trace"), []),
        int(latency) if isinstance(latency, (int, float)) else None,
        str(result.get("model") or ""), str(result.get("status") or "ok"), created_at,
    )
    with _write_conn(path) as conn:
        _ensure_thread(conn, thread_id)
        conn.execute(
            """INSERT INTO agent_runs
               (graph_thread, thread_id, turn, query, route, complexity, confidence,
                route_reason, intent, answer, plan_json, task_results_json,
                verify_json, evidence_json, trace_json, latency_ms, model, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(graph_thread) DO UPDATE SET
                 thread_id = excluded.thread_id, turn = excluded.turn,
                 query = excluded.query, route = excluded.route,
                 complexity = excluded.complexity, confidence = excluded.confidence,
                 route_reason = excluded.route_reason, intent = excluded.intent,
                 answer = excluded.answer, plan_json = excluded.plan_json,
                 task_results_json = excluded.task_results_json,
                 verify_json = excluded.verify_json, evidence_json = excluded.evidence_json,
                 trace_json = excluded.trace_json, latency_ms = excluded.latency_ms,
                 model = excluded.model, status = excluded.status""",
            values,
        )
        run_id = int(conn.execute(
            "SELECT id FROM agent_runs WHERE graph_thread = ?", (graph_thread,)
        ).fetchone()["id"])
        conn.execute("DELETE FROM agent_tasks WHERE run_id = ?", (run_id,))
        rows: list[tuple[Any, ...]] = []
        for task in plan if isinstance(plan, list) else []:
            if not isinstance(task, Mapping):
                continue
            tid = str(task.get("id") or "")
            res = task_results.get(tid) if isinstance(task_results, Mapping) else None
            if not isinstance(res, Mapping):
                res = task.get("result") if isinstance(task.get("result"), Mapping) else {}
            task_latency = res.get("latency_ms")
            rows.append((
                run_id, tid, str(task.get("title") or ""), str(task.get("agent") or ""),
                str(task.get("status") or ""), _dumps(task.get("inputs"), {}),
                _dumps(res, {}),
                int(task_latency) if isinstance(task_latency, (int, float)) else None,
                str(res.get("error") or "")[:1000], created_at,
            ))
        conn.executemany(
            """INSERT INTO agent_tasks
               (run_id, task_id, title, agent, status, inputs_json, result_json,
                latency_ms, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return run_id


def save_generated_image(
    *,
    prompt: str,
    images: Sequence[Any] | None,
    model: str = "",
    size: str = "",
    negative_prompt: str = "",
    actual_prompt: str = "",
    task_id: str = "",
    latency_ms: int | None = None,
    path: Path | None = None,
) -> int:
    """记录一次文生图（提示词 + 落盘图片），失败由调用方决定是否提示。"""
    if not str(prompt or "").strip():
        raise HistoryError("提示词为空，未写入记录")
    with _write_conn(path) as conn:
        cur = conn.execute(
            """INSERT INTO generated_images
               (created_at, prompt, actual_prompt, negative_prompt, size,
                model, task_id, images_json, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now(), str(prompt), str(actual_prompt or ""), str(negative_prompt or ""),
             str(size or ""), str(model or ""), str(task_id or ""),
             _dumps(list(images or []), []),
             int(latency_ms) if latency_ms is not None else None),
        )
        return int(cur.lastrowid)


def list_threads(limit: int = 20, offset: int = 0, *,
                 path: Path | None = None) -> list[dict[str, Any]]:
    """会话列表（按最近活动倒序），带消息数、执行轮数与最后一条预览。"""
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    sql = """
        SELECT t.thread_id, t.user_id, t.kind, t.title, t.created_at, t.updated_at,
               (SELECT COUNT(*) FROM chat_messages m
                 WHERE m.thread_id = t.thread_id) AS message_count,
               (SELECT COUNT(*) FROM agent_runs r
                 WHERE r.thread_id = t.thread_id) AS run_count,
               (SELECT substr(m.content, 1, 120) FROM chat_messages m
                 WHERE m.thread_id = t.thread_id ORDER BY m.id DESC LIMIT 1) AS last_message
          FROM chat_threads t
         ORDER BY t.updated_at DESC, t.thread_id
         LIMIT ? OFFSET ?
    """
    with closing(_connect(path)) as conn:
        rows = conn.execute(sql, (limit, offset)).fetchall()
    return [dict(r) for r in rows]


def get_chat_thread(thread_id: str, *, message_limit: int = 200, run_limit: int = 50,
                    path: Path | None = None) -> dict[str, Any] | None:
    """会话详情：元信息 + 消息 + 最近若干轮执行（含子任务）。"""
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT * FROM chat_threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if row is None:
            return None
        messages = thread_messages(thread_id, limit=message_limit, path=path)
        runs = [dict(r) for r in conn.execute(
            """SELECT * FROM agent_runs WHERE thread_id = ?
                ORDER BY id DESC LIMIT ?""",
            (thread_id, max(1, min(int(run_limit), 200))),
        ).fetchall()]
        run_ids = [r["id"] for r in runs]
        tasks: dict[int, list[dict[str, Any]]] = {}
        if run_ids:
            marks = ",".join("?" * len(run_ids))
            for t in conn.execute(
                f"""SELECT * FROM agent_tasks WHERE run_id IN ({marks}) ORDER BY id""",
                run_ids,
            ).fetchall():
                tasks.setdefault(int(t["run_id"]), []).append(dict(t))
    for run in runs:
        for key in ("plan_json", "task_results_json", "verify_json",
                    "evidence_json", "trace_json"):
            run[key.removesuffix("_json")] = json.loads(run.pop(key) or "null")
        run["tasks"] = []
        for t in tasks.get(run["id"], []):
            t["inputs"] = json.loads(t.pop("inputs_json") or "{}")
            t["result"] = json.loads(t.pop("result_json") or "{}")
            run["tasks"].append(t)
    return {"thread": dict(row), "messages": messages, "runs": runs}


def delete_thread(thread_id: str, *, path: Path | None = None) -> bool:
    """删除会话及其消息、执行记录、子任务（外键级联）；返回是否删到了行。"""
    with _write_conn(path) as conn:
        cur = conn.execute("DELETE FROM chat_threads WHERE thread_id = ?", (thread_id,))
        return cur.rowcount > 0


# ══════════════════════════════ 自检 / CLI ══════════════════════════════
def _selftest() -> int:
    failures: list[str] = []
    total = 0

    def check(name: str, ok: bool, detail: Any = "") -> None:
        nonlocal total
        total += 1
        if ok:
            print(f"[OK] {name}")
        else:
            failures.append(f"{name}: {detail}")
            print(f"[FAIL] {name}: {detail}")

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "history.db"
        first = save_translation(
            direction="zh2en",
            source_text="稀疏注意力用于长文档建模。",
            translation="Sparse attention is used for long-document modeling.",
            glossary=[{"zh": "稀疏注意力", "en": "sparse attention", "note": "固定译法"}],
            stats={"paragraphs": 1, "terms": 1},
            model="qwen-plus",
            latency_ms=1234,
            path=db,
        )
        second = save_translation(
            direction="en2zh",
            source_text="The model uses self-attention.",
            translation="该模型使用自注意力。",
            glossary={"self-attention": "自注意力"},
            issues=["示例提示"],
            stats={"paragraphs": 1},
            path=db,
        )
        check("建库文件存在", db.exists(), db)
        check("ID 递增", second.id == first.id + 1, (first.id, second.id))
        check("时间带时区", "T" in first.created_at and ("+" in first.created_at or "Z" in first.created_at),
              first.created_at)

        items = list_translations(path=db)
        check("列出两条", len(items) == 2, len(items))
        check("倒序", items[0]["id"] == second.id, items[0]["id"])
        check("术语计数", items[0]["term_count"] == 1 and items[1]["term_count"] == 1,
              [i["term_count"] for i in items])

        only_zh = list_translations(direction="zh2en", path=db)
        check("按方向过滤", len(only_zh) == 1 and only_zh[0]["id"] == first.id, only_zh)
        check("预览不超长", len(items[0]["source_preview"]) <= 120, len(items[0]["source_preview"]))

        full = get_translation(first.id, path=db)
        check("读取完整记录", full is not None and full["translation"].startswith("Sparse"), full)
        check("术语方向 zh→en",
              full is not None and full["terms"][0]["source_term"] == "稀疏注意力"
              and full["terms"][0]["target_term"] == "sparse attention",
              full and full["terms"])
        check("统计 JSON 还原", full is not None and full["stats"].get("terms") == 1,
              full and full["stats"])

        full_en = get_translation(second.id, path=db)
        check("英译中术语方向",
              full_en is not None and full_en["terms"][0]["source_term"] == "self-attention"
              and full_en["terms"][0]["target_term"] == "自注意力",
              full_en and full_en["terms"])
        check("issues 还原", full_en is not None and full_en["issues"] == ["示例提示"],
              full_en and full_en["issues"])
        check("不存在的记录返回 None", get_translation(99999, path=db) is None)

        # ── 会话 / Agent 执行 ──
        append_chat_message("t-1", "user", "第一问：解释注意力", path=db)
        append_chat_message("t-1", "assistant", "第一答", path=db)
        append_chat_message("t-1", "user", "第二问：继续", path=db)
        msgs = thread_messages("t-1", path=db)
        check("消息按时间正序", [m["role"] for m in msgs] == ["user", "assistant", "user"],
              [m["role"] for m in msgs])
        threads = list_threads(path=db)
        check("会话列表与标题", len(threads) == 1 and threads[0]["title"] == "第一问：解释注意力"
              and threads[0]["message_count"] == 3, threads)
        check("最近消息预览", threads[0]["last_message"] == "第二问：继续", threads[0]["last_message"])

        run = {
            "graph_thread": "t-1#t1", "thread_id": "t-1", "turn": 1,
            "query": "第一问：解释注意力", "route": "research", "complexity": "medium",
            "confidence": 0.8, "intent": "qa", "answer": "初稿",
            "plan": [{"id": "st1", "title": "检索", "agent": "retriever", "status": "done",
                      "inputs": {"q": "attention"}}],
            "task_results": {"st1": {"answer": "检索结果", "latency_ms": 12}},
            "verify": {"passed": True}, "evidence": [{"doc_id": "d1", "block_id": "b1"}],
            "trace": [{"node": "planner"}], "latency_ms": 100, "model": "qwen-plus",
            "status": "ok",
        }
        save_agent_run(run, path=db)
        save_agent_run({**run, "answer": "终稿", "status": "ok"}, path=db)  # 同 graph_thread 覆盖
        detail = get_chat_thread("t-1", path=db)
        check("执行按 graph_thread 去重", detail is not None and len(detail["runs"]) == 1,
              detail and len(detail["runs"]))
        check("覆盖后答案更新", detail is not None and detail["runs"][0]["answer"] == "终稿")
        check("子任务不重复", detail is not None and len(detail["runs"][0]["tasks"]) == 1)
        check("JSON 字段还原", detail is not None
              and detail["runs"][0]["verify"] == {"passed": True}
              and detail["runs"][0]["tasks"][0]["result"]["latency_ms"] == 12)
        check("轮次上界", thread_turn("t-1", path=db) == 1, thread_turn("t-1", path=db))
        check("任务中间态可恢复",
              latest_task_state("t-1", path=db)["results"]["st1"]["answer"] == "检索结果")
        set_thread_summary("t-1", "早期对话摘要", path=db)
        check("摘要落库", thread_summary("t-1", path=db) == "早期对话摘要")

        img_id = save_generated_image(prompt="量子计算示意图", images=[{"url": "/static/generated/a.png"}],
                                      model="wanx2.1-t2i-turbo", size="1024*1024",
                                      task_id="task-1", path=db)
        with closing(_connect(db)) as conn:
            img = conn.execute("SELECT * FROM generated_images WHERE id = ?", (img_id,)).fetchone()
        check("生图记录", img is not None and json.loads(img["images_json"])[0]["url"].endswith("a.png"),
              img and img["images_json"])

        check("删除会话", delete_thread("t-1", path=db) is True)
        check("级联删除消息与执行", get_chat_thread("t-1", path=db) is None
              and thread_messages("t-1", path=db) == [] and thread_turn("t-1", path=db) == 0)

        try:
            save_translation(direction="xx", source_text="a", translation="b", path=db)
        except HistoryError:
            check("非法方向被拒绝", True)
        else:
            check("非法方向被拒绝", False)

        long_text = "长" * 50_000
        long_id = save_translation(direction="zh2en", source_text=long_text,
                                   translation=long_text, path=db)
        check("超长原文可写入", get_translation(long_id.id, path=db) is not None)

    print(f"\nselftest: {total - len(failures)}/{total} 通过")
    for item in failures:
        print(f"  - {item}")
    return 0 if not failures else 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="翻译历史 SQLite 存储（Navicat 可直接打开）")
    ap.add_argument("--selftest", action="store_true", help="跑内置自检")
    ap.add_argument("--init", action="store_true", help="只建库建表并打印路径")
    ap.add_argument("--list", type=int, nargs="?", const=20, metavar="N",
                    help="列出最近 N 条记录（默认 20）")
    ap.add_argument("--direction", choices=DIRECTIONS, help="按翻译方向过滤")
    ap.add_argument("--db", help="覆盖 SQLITE_PATH（默认读 .env）")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    if args.selftest:
        return _selftest()

    target = Path(args.db) if args.db else db_path()
    if args.init:
        print(f"database: {init_db(target)}")
        return 0
    if args.list is not None:
        try:
            items = list_translations(args.list, direction=args.direction or "", path=target)
        except HistoryError as exc:
            print(f"读取失败：{exc}", file=sys.stderr)
            return 1
        if not items:
            print("（暂无记录）")
            return 0
        for item in items:
            print(f"#{item['id']} {item['created_at']} {item['direction']} "
                  f"{item['source_chars']}→{item['translation_chars']} 字 · "
                  f"{item['term_count']} 术语 · {item['model'] or '—'}")
            print(f"    {item['source_preview'].replace(chr(10), ' ')}")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
