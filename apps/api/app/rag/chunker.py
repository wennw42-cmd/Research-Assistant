"""流式切块器：句子级 / 段落级打包 + 有界 lead 缓冲。

为什么必须是**流式**而不是"先攒整章再切"：
541 页教材里一章可达数万 token，攒在内存里峰值直接失控。
流式打包器只持有"当前块 + 重叠尾巴"，内存占用与**单块大小**同阶，与文档长度无关。
"""
from __future__ import annotations

from typing import Any, Callable

from app.rag.params import count_tokens, hard_split

Unit = tuple[str, dict[str, Any]]     # (文本, 元数据)


class StreamingPacker:
    """按 token 预算贪心打包，块间保留 overlap（回溯尾部单元实现）。

    - `target` 是软目标：达到后就不再往里加（除非当前块还是空的）；
    - `max_tokens` 是硬上限：任何块都不会超过它，超长单元先硬切再入块。
    """

    def __init__(self, target: int, overlap: int, max_tokens: int,
                 on_emit: Callable[[list[Unit]], None]) -> None:
        self.target = target
        self.overlap = overlap
        self.max_tokens = max_tokens
        self.on_emit = on_emit
        self.cur: list[Unit] = []
        self.cur_tokens = 0
        self.emitted = 0
        self._last_emitted: list[Unit] = []

    def add(self, text: str, meta: dict[str, Any] | None = None) -> None:
        unit = (text, meta or {})
        t = count_tokens(text)
        if t > self.max_tokens:
            if self.cur:
                self._emit()
            for piece in hard_split(text, self.max_tokens):
                self.add(piece, meta)
            return
        if self.cur and (self.cur_tokens + t > self.max_tokens or self.cur_tokens >= self.target):
            self._emit()
            self._carry_overlap()
        self.cur.append(unit)
        self.cur_tokens += t

    def flush(self) -> None:
        if self.cur:
            self._emit()

    @property
    def pending_tokens(self) -> int:
        return self.cur_tokens

    # ── 内部 ──
    def _emit(self) -> None:
        if not self.cur:
            return
        self._last_emitted = list(self.cur)   # 供重叠回溯
        self.on_emit(list(self.cur))
        self.emitted += 1
        self.cur = []
        self.cur_tokens = 0

    def _carry_overlap(self) -> None:
        """把上一块尾部的若干单元带进新块，作为重叠（已发出的块不受影响）。"""
        tail: list[Unit] = []
        acc = 0
        for unit in reversed(self._last_emitted):
            if acc >= self.overlap:
                break
            tail.append(unit)
            acc += count_tokens(unit[0])
        tail.reverse()
        self.cur = tail
        self.cur_tokens = acc


class LeadBuffer:
    """有界的"开头 N token"缓冲：用于章节 lead 与章摘要（抽取式，不生成）。"""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.parts: list[str] = []
        self._tokens = 0

    def add(self, text: str) -> bool:
        """加入文本；返回 False 表示已满（不再接收）。"""
        if self.full:
            return False
        t = count_tokens(text)
        if self._tokens + t <= self.max_tokens:
            self.parts.append(text)
            self._tokens += t
            return True
        room = self.max_tokens - self._tokens
        if room > 8:  # 只截到token上限，避免把一个字切成半句
            import re

            cjk = re.findall(r"[㐀-鿿]", text)
            keep = min(len(text), room if len(cjk) > len(text) / 2 else room * 3)
            self.parts.append(text[:keep])
            self._tokens += count_tokens(text[:keep])
        return False

    @property
    def full(self) -> bool:
        return self._tokens >= self.max_tokens

    @property
    def tokens(self) -> int:
        return self._tokens

    def text(self) -> str:
        return "".join(self.parts) if any("\n" in p for p in self.parts) else " ".join(self.parts)


def breadcrumb(path: list[str] | None, sep: str = " > ") -> str:
    """标题路径面包屑：溯源的一等公民，前置到 chunk 文本里提升无上下文可读性。"""
    return sep.join(p for p in (path or []) if p)


def make_chunk_id(doc_id: str, level: str, seq: int) -> str:
    return f"{doc_id}:{level}:{seq:05d}"
