"""分级加载：摘要层(T0) → 章节层(L0) → 段落层(L1) → 细节层(L2)。

关键点：**不是"先加载全部再排序"，而是逐层收敛、逐层才读盘**。

    第 1 层：只用常驻内存的**元信息**（标题路径）给章/节打分，全程零 I/O；
    第 2 层：仅加载入选章节的 L0 文本（数量有上界）；
    第 3 层：仅加载入选小节的 L1 段落文本；
    第 4 层：只为最优段落加载它的 L2 句子块（精确引用）。

每一步的 I/O 都与"上一层选出的数量"成正比，而不是与 541 页成正比。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.rag.memory import MemoryMonitor
from app.rag.store import ChunkIndex, JsonlStore, LRUCache

_CJK = "一-鿿"


def terms(text: str) -> set[str]:
    """中英混排词项：CJK 取 bigram，拉丁取词。用于无依赖的离线打分。"""
    t = (text or "").lower()
    out = set(re.findall(r"[a-z0-9]+", t))
    cjk = re.findall(f"[{_CJK}]", t)
    # 注意：列表切片是 list（不可哈希），必须 join 成字符串
    out |= {"".join(cjk[i:i + 2]) for i in range(len(cjk) - 1)} or set(cjk)
    return out


def score(query_terms: set[str], doc_terms: set[str]) -> float:
    if not query_terms or not doc_terms:
        return 0.0
    inter = len(query_terms & doc_terms)
    return inter / ((len(query_terms) ** 0.5) * (len(doc_terms) ** 0.5) + 1e-9)


@dataclass
class TierItem:
    tier: str
    id: str
    title_path: str
    page_start: int
    page_end: int
    n_tokens: int
    score: float
    text: str


@dataclass
class TieredLoader:
    index: ChunkIndex
    base_dir: Path
    cache: LRUCache = field(default_factory=lambda: LRUCache(max_bytes=32 << 20))
    monitor: MemoryMonitor | None = None
    loaded_bytes: int = 0
    loaded_items: int = 0      # text_of 调用次数（含缓存命中）
    disk_reads: int = 0        # 真正读盘次数（不含缓存命中）

    _stores: dict[str, JsonlStore] = field(default_factory=dict)

    def store(self, level: str) -> JsonlStore:
        if level not in self._stores:
            # 只读用途：不要用 "a"，否则会在只读场景创建/持有写句柄。
            self._stores[level] = JsonlStore(self.base_dir / f"chunks_{level}.jsonl", "r")
        return self._stores[level]

    def close(self) -> None:
        for st in self._stores.values():
            st.close()
        self._stores.clear()

    def __enter__(self) -> "TieredLoader":
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()

    # ── 标题路径（字典编码，按需还原）──
    def _p(self, meta: dict[str, Any]) -> str:
        return self.index.path_str_of(meta)

    # ── 按需加载（走 LRU，正文不常驻）──
    def text_of(self, meta: dict[str, Any]) -> str:
        key = str(meta.get("id"))

        def loader() -> str:
            rec = self.store(str(meta.get("store", "L1"))).read(int(meta["offset"]), int(meta["length"]))
            self.loaded_bytes += int(meta.get("length", 0))
            self.disk_reads += 1
            return str(rec.get("text", ""))

        text = self.cache.get(key, loader) or ""
        self.loaded_items += 1
        return text

    # ── 主入口 ──
    def load(self, query: str = "", *, seed_ids: list[str] | None = None,
             token_budget: int = 4000, top_chapters: int = 2, top_sections: int = 4,
             top_paras: int = 6, detail_per_para: int = 3) -> dict[str, Any]:
        qt = terms(query)
        items: list[TierItem] = []
        dropped: list[str] = []
        used = 0

        def push(tier: str, meta: dict[str, Any], sc: float) -> bool:
            nonlocal used
            if used + int(meta.get("n_tokens", 0)) > token_budget:
                dropped.append(f"{tier}:{meta.get('id')}")
                return False
            text = self.text_of(meta)
            used += int(meta.get("n_tokens", 0))
            items.append(TierItem(
                tier=tier, id=str(meta.get("id")), title_path=self._p(meta),
                page_start=int(meta.get("page_start", 0)), page_end=int(meta.get("page_end", 0)),
                n_tokens=int(meta.get("n_tokens", 0)), score=round(sc, 4), text=text))
            return True

        seeds = [self.index.get(i) for i in (seed_ids or [])]
        seeds = [s for s in seeds if s]

        # ── 第 1 层：摘要层（仅用元信息打分，零正文 I/O）──
        chapters = self.index.by_level("T0")
        if query:
            scored = sorted(chapters, key=lambda m: -score(qt, terms(self._p(m))))
        else:
            scored = chapters
        for m in seeds if seeds else scored[:top_chapters]:
            push("T0", m, score(qt, terms(self._p(m))))
        chosen_chapters = [m["id"] for m in (seeds if seeds else scored[:top_chapters])]

        # ── 第 2 层：章节层 ──
        # 按标题路径前缀判断归属，不能用固定 8 字符前缀（章号变长后会串章）。
        chapter_paths = [self.index.path_of(self.index.get(c) or {}) for c in chosen_chapters]
        chapter_paths = [p for p in chapter_paths if p]

        def under_selected(m: dict[str, Any]) -> bool:
            p = self.index.path_of(m)
            return any(p[:len(cp)] == cp for cp in chapter_paths)

        sections = [m for m in self.index.by_level("L0")
                    if not chosen_chapters or m.get("parent_id") in chosen_chapters
                    or under_selected(m)]
        sections = sorted(sections, key=lambda m: -score(qt, terms(self._p(m))))[:top_sections]
        for m in sections:
            push("L0", m, score(qt, terms(self._p(m))))
        chosen_sections = [str(m["id"]) for m in sections]

        # ── 第 3 层：段落层（只加载入选小节的段落）──
        paras = [m for m in self.index.by_level("L1") if m.get("section_id") in chosen_sections]
        scored_paras: list[tuple[dict[str, Any], float]] = []
        for m in paras:
            sc = score(qt, terms(self.text_of(m)))
            scored_paras.append((m, sc))
        scored_paras.sort(key=lambda p: -p[1])
        for m, sc in scored_paras[:top_paras]:
            push("L1", m, sc)

        # ── 第 4 层：细节层（只为最优段落展开句子块）──
        for m, _ in scored_paras[:max(1, detail_per_para)]:
            for child in self.index.children(str(m["id"]))[:detail_per_para]:
                push("L2", child, score(qt, terms(self.text_of(child))))

        return {
            "query": query,
            "budget": token_budget,
            "used_tokens": used,
            "items": [i.__dict__ for i in items],
            "dropped_by_budget": dropped,
            "tiers": {t: sum(1 for i in items if i.tier == t) for t in ("T0", "L0", "L1", "L2")},
            "io": {"loaded_items": self.loaded_items, "disk_reads": self.disk_reads,
                   "loaded_bytes": self.loaded_bytes},
            "cache": self.cache.stats(),
        }

    # ── 手动展开（前端点"看细节"时用）──
    def expand(self, chunk_id: str, *, depth: int = 1) -> dict[str, Any]:
        meta = self.index.get(chunk_id)
        if not meta:
            return {"error": f"未找到 {chunk_id}"}
        out: list[dict[str, Any]] = [{"id": chunk_id, "level": meta.get("level"),
                                      "title_path": self._p(meta), "text": self.text_of(meta)}]
        level = str(meta.get("level"))
        if level in ("T0", "L0") and depth >= 1:
            for child in self.index.children(chunk_id)[:5]:
                out.append({"id": child.get("id"), "level": child.get("level"),
                            "title_path": self._p(child), "text": self.text_of(child)})
        return {"items": out, "cache": self.cache.stats()}
