"""聊天与会话记录落库：建表、顺序、覆盖更新、级联删除、重启恢复。"""
from __future__ import annotations

import sqlite3

from app import history
from app.memory.store import MemoryStore

TABLES = {
    "translations", "translation_terms",
    "chat_threads", "chat_messages", "agent_runs", "agent_tasks", "generated_images",
}


def _run(thread_id: str = "t1", turn: int = 1, **extra) -> dict:
    base = {
        "graph_thread": f"{thread_id}#t{turn}", "thread_id": thread_id, "turn": turn,
        "query": "为什么注意力机制有效？", "route": "research", "complexity": "medium",
        "confidence": 0.82, "intent": "qa", "answer": "草稿",
        "plan": [{"id": "st1", "title": "检索", "agent": "retriever", "status": "done",
                  "inputs": {"q": "attention"}}],
        "task_results": {"st1": {"answer": "证据", "latency_ms": 7}},
        "verify": {"passed": True}, "evidence": [{"doc_id": "d1", "block_id": "b1"}],
        "trace": [{"node": "planner"}], "latency_ms": 120, "model": "qwen-plus", "status": "ok",
    }
    return {**base, **extra}


def test_schema_created(tmp_path):
    db = tmp_path / "chat.db"
    history.init_db(db)
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert TABLES <= names


def test_messages_order_and_thread_summary(tmp_path):
    db = tmp_path / "chat.db"
    history.append_chat_message("t1", "user", "第一问", path=db)
    history.append_chat_message("t1", "assistant", "第一答", path=db)
    history.append_chat_message("t1", "user", "第二问", path=db)

    msgs = history.thread_messages("t1", path=db)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [m["content"] for m in msgs] == ["第一问", "第一答", "第二问"]
    assert [m["id"] for m in history.thread_messages("t1", limit=2, path=db)] == [2, 3]

    [thread] = history.list_threads(path=db)
    assert thread["title"] == "第一问"
    assert thread["message_count"] == 3
    assert thread["last_message"] == "第二问"


def test_agent_run_upsert_and_tasks(tmp_path):
    db = tmp_path / "chat.db"
    run_id = history.save_agent_run(_run(), path=db)
    history.save_agent_run(_run(answer="终稿", status="ok"), path=db)

    detail = history.get_chat_thread("t1", path=db)
    assert detail is not None
    assert len(detail["runs"]) == 1
    assert detail["runs"][0]["id"] == run_id
    assert detail["runs"][0]["answer"] == "终稿"
    assert detail["runs"][0]["verify"] == {"passed": True}
    assert detail["runs"][0]["plan"][0]["id"] == "st1"
    [task] = detail["runs"][0]["tasks"]
    assert (task["task_id"], task["agent"], task["status"]) == ("st1", "retriever", "done")
    assert task["result"]["latency_ms"] == 7

    assert history.thread_turn("t1", path=db) == 1
    assert history.latest_task_state("t1", path=db)["results"]["st1"]["answer"] == "证据"


def test_cascade_delete(tmp_path):
    db = tmp_path / "chat.db"
    history.append_chat_message("t1", "user", "问", path=db)
    history.save_agent_run(_run(), path=db)
    assert history.delete_thread("t1", path=db) is True

    assert history.get_chat_thread("t1", path=db) is None
    assert history.thread_messages("t1", path=db) == []
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM agent_tasks").fetchone()[0] == 0


def test_memory_survives_restart(tmp_path, monkeypatch):
    db = tmp_path / "chat.db"
    monkeypatch.setattr(history, "db_path", lambda: db)

    store = MemoryStore()
    store.append("t1", "user", "第一问")
    store.append("t1", "assistant", "第一答")
    assert store.next_turn("t1") == 1
    history.save_agent_run(_run(), path=db)

    # 新进程：空字典 + 同一个库
    fresh = MemoryStore()
    assert [m["content"] for m in fresh.get_history("t1")] == ["第一问", "第一答"]
    assert fresh.next_turn("t1") == 2
    assert fresh.load_task_state("t1")["plan"][0]["id"] == "st1"

    fresh.clear("t1")
    assert MemoryStore().get_history("t1") == []


def test_thread_kind_and_migration(tmp_path):
    db = tmp_path / "chat.db"
    history.append_chat_message("gen-1", "user", "你好", kind="general", path=db)
    history.append_chat_message("agent-1", "user", "你好", path=db)
    kinds = {t["thread_id"]: t["kind"] for t in history.list_threads(path=db)}
    assert kinds == {"gen-1": "general", "agent-1": "agent"}

    old = tmp_path / "old.db"
    with sqlite3.connect(old) as conn:
        conn.execute(
            """CREATE TABLE chat_threads (
                   thread_id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT '',
                   title TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '',
                   created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
        )
        conn.execute("INSERT INTO chat_threads VALUES ('old', '', '老会话', '', '2026-01-01', '2026-01-01')")
    [thread] = history.list_threads(path=old)
    assert thread["kind"] == "agent"


def test_generated_image_record(tmp_path):
    db = tmp_path / "chat.db"
    history.save_generated_image(
        prompt="量子计算示意图", images=[{"url": "static/generated/a.png"}],
        model="wanx2.1-t2i-turbo", size="1024*1024", actual_prompt="量子芯片",
        task_id="task-1", latency_ms=900, path=db,
    )
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT prompt, actual_prompt, model, task_id, images_json FROM generated_images"
        ).fetchone()
    assert row[0] == "量子计算示意图" and row[1] == "量子芯片"
    assert row[2] == "wanx2.1-t2i-turbo" and row[3] == "task-1"
    assert "a.png" in row[4]
