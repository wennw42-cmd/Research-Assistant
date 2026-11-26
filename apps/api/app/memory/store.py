"""状态管理：会话记忆 / 文献库上下文 / 任务中间态 / 长期资产。

四类记忆的职责边界（这是多 Agent 系统不失控的关键）：

┌──────────────┬─────────────────────────┬──────────────────────────────┐
│ 记忆类型      │ 放什么                   │ 生命周期 / 谁写谁读           │
├──────────────┼─────────────────────────┼──────────────────────────────┤
│ 会话记忆      │ 用户提问、Agent 结论     │ 按 thread_id；超窗后压缩摘要  │
│ 文献库上下文  │ 文献清单、指纹、可用工具  │ 按 user_id；解析流水线写      │
│ 任务中间态    │ SubTask DAG、每步 result │ 按 thread_id；Planner/Executor│
│ 长期资产      │ 术语表、翻译记忆、画像    │ 按 user_id；跨会话累积复用    │
└──────────────┴─────────────────────────┴──────────────────────────────┘

进程内字典做热缓存，同时写 SQLite（``app.history``）保证重启后不丢：
会话消息、滚动摘要、轮次、任务中间态都能从库里恢复；写库失败只告警，
不影响当轮回答。把 backend 换成 Redis 时替换 `_sessions` 等字典即可，接口不变。
"""
from __future__ import annotations

import time
from typing import Any

from app import history
from app.utils import get_logger, short

log = get_logger("ra.memory")

# ── 演示用初始文献库：真实系统中由 M1 解析流水线写入 ──
_DEFAULT_LIBRARY: dict[str, Any] = {
    "docs": [
        {"doc_id": "doc_transformer", "title": "Attention Is All You Need", "year": 2017, "pages": 15},
        {"doc_id": "doc_bert", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "year": 2018, "pages": 16},
        {"doc_id": "doc_survey_2020", "title": "Efficient Transformers: A Survey", "year": 2020, "pages": 22},
    ],
    "terms": 3,
    "last_sync": 0,
}

_DEFAULT_GLOSSARY: dict[str, str] = {
    "self-attention": "自注意力",
    "masked language model": "掩码语言模型",
    "fine-tuning": "微调",
    "encoder-decoder": "编码器-解码器",
    "positional encoding": "位置编码",
}


class MemoryStore:
    """进程内记忆 + SQLite 持久化（高并发场景再把字典换成 Redis / PG）。"""

    def __init__(self, max_turns: int = 12, summary_after: int = 16) -> None:
        self.max_turns = max_turns
        self.summary_after = summary_after
        self._sessions: dict[str, list[dict[str, Any]]] = {}     # thread_id -> messages
        self._summaries: dict[str, str] = {}                     # thread_id -> 滚动摘要
        self._tasks: dict[str, dict[str, Any]] = {}              # thread_id -> {"plan":..., "results":...}
        self._turns: dict[str, int] = {}                         # thread_id -> 轮次计数
        self._libraries: dict[str, dict[str, Any]] = {}          # user_id -> library
        self._glossaries: dict[str, dict[str, str]] = {}         # user_id -> 术语表（翻译记忆）
        self._hydrated: set[str] = set()                         # 已尝试从库里恢复的线程

    # ───────────── 持久化 ─────────────
    def _hydrate(self, thread_id: str) -> None:
        """首次触碰某线程时从 SQLite 恢复（重启后接着聊）；失败只告警。"""
        if thread_id in self._hydrated:
            return
        self._hydrated.add(thread_id)
        try:
            msgs = history.thread_messages(thread_id, limit=self.summary_after)
            state = history.latest_task_state(thread_id)
            summary = history.thread_summary(thread_id)
            turn = history.thread_turn(thread_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("thread=%s 从库恢复失败：%s: %s", thread_id, type(exc).__name__, exc)
            return
        if msgs:
            self._sessions[thread_id] = [{"role": m["role"], "content": m["content"]} for m in msgs]
        if summary:
            self._summaries[thread_id] = summary
        if turn:
            self._turns[thread_id] = turn
        if state.get("plan") or state.get("results"):
            self._tasks[thread_id] = {**state, "updated_at": int(time.time())}

    # ───────────── 会话记忆 ─────────────
    def get_history(self, thread_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        self._hydrate(thread_id)
        msgs = self._sessions.get(thread_id, [])
        return msgs[-limit:] if limit else list(msgs)

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._hydrate(thread_id)
        self._sessions.setdefault(thread_id, []).append(
            {"role": role, "content": content, "ts": int(time.time())}
        )
        try:
            history.append_chat_message(thread_id, role, content)
        except Exception as exc:  # noqa: BLE001
            log.warning("thread=%s 消息落库失败：%s: %s", thread_id, type(exc).__name__, exc)
        self._maybe_compress(thread_id)

    def _maybe_compress(self, thread_id: str) -> None:
        """超窗压缩：把早期对话折叠成摘要，只保留最近 max_turns 条原文。

        比"滑动窗口直接丢弃"更好：摘要保留了早期约束（如"只用法语回答"），
        又避免 prompt 无限膨胀导致 541 页级上下文溢出。
        """
        msgs = self._sessions.get(thread_id, [])
        if len(msgs) <= self.summary_after:
            return
        old, keep = msgs[:-self.max_turns], msgs[-self.max_turns:]
        digest = "；".join(f"{m['role']}:{short(str(m['content']), 60)}" for m in old)
        self._summaries[thread_id] = (self._summaries.get(thread_id, "") + " " + digest).strip()[-2000:]
        self._sessions[thread_id] = keep
        try:
            history.set_thread_summary(thread_id, self._summaries[thread_id])
        except Exception as exc:  # noqa: BLE001
            log.warning("thread=%s 摘要落库失败：%s: %s", thread_id, type(exc).__name__, exc)
        log.info("thread=%s 会话压缩：丢弃 %d 条原文，保留摘要 %d 字", thread_id, len(old), len(self._summaries[thread_id]))

    def get_summary(self, thread_id: str) -> str:
        self._hydrate(thread_id)
        return self._summaries.get(thread_id, "")

    def next_turn(self, thread_id: str) -> int:
        """轮次计数器。

        LangGraph 的 checkpointer 会按 thread_id 累积带 reducer 的字段（如 evidence），
        因此**每一轮新问题必须用新的图线程**，否则上一轮的证据池会污染本轮校验。
        """
        self._hydrate(thread_id)
        self._turns[thread_id] = self._turns.get(thread_id, 0) + 1
        return self._turns[thread_id]

    def clear(self, thread_id: str) -> None:
        self._sessions.pop(thread_id, None)
        self._summaries.pop(thread_id, None)
        self._tasks.pop(thread_id, None)
        self._turns.pop(thread_id, None)
        self._hydrated.add(thread_id)  # 已删除，避免下次再从库里恢复
        try:
            history.delete_thread(thread_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("thread=%s 删除落库记录失败：%s: %s", thread_id, type(exc).__name__, exc)

    # ───────────── 文献库上下文 ─────────────
    def get_library(self, user_id: str) -> dict[str, Any]:
        if user_id not in self._libraries:
            self._libraries[user_id] = dict(_DEFAULT_LIBRARY)
        return self._libraries[user_id]

    def upsert_doc(self, user_id: str, doc: dict[str, Any]) -> None:
        lib = self.get_library(user_id)
        docs = [d for d in lib["docs"] if d.get("doc_id") != doc.get("doc_id")]
        docs.append(doc)
        lib["docs"] = docs
        lib["last_sync"] = int(time.time())

    def get_glossary(self, user_id: str) -> dict[str, str]:
        return self._glossaries.setdefault(user_id, dict(_DEFAULT_GLOSSARY))

    def add_terms(self, user_id: str, terms: dict[str, str]) -> None:
        self.get_glossary(user_id).update(terms)

    def build_literature_context(self, user_id: str) -> dict[str, Any]:
        """注入给 Agent 的"文献库感知"上下文：让 Agent 知道库里有什么、能用什么。"""
        lib = self.get_library(user_id)
        return {
            "doc_ids": [d["doc_id"] for d in lib["docs"]],
            "doc_count": len(lib["docs"]),
            "titles": {d["doc_id"]: d.get("title", "") for d in lib["docs"]},
            "glossary_size": len(self.get_glossary(user_id)),
        }

    # ───────────── 任务中间态 ─────────────
    def save_task_state(self, thread_id: str, plan: list[dict[str, Any]], results: dict[str, Any]) -> None:
        self._tasks[thread_id] = {"plan": plan, "results": results, "updated_at": int(time.time())}

    def load_task_state(self, thread_id: str) -> dict[str, Any]:
        self._hydrate(thread_id)
        return self._tasks.get(thread_id, {"plan": [], "results": {}})

    def resume_hint(self, thread_id: str) -> str:
        """给 Agent 的"断点续跑"提示词：未完成 DAG 有哪些。"""
        st = self.load_task_state(thread_id)
        pending = [t["id"] for t in st.get("plan", []) if t.get("status") == "pending"]
        return f"未完成子任务：{pending}" if pending else ""


memory = MemoryStore()
