"""落盘存储 + 偏移索引 + 缓存淘汰。

541 页教材如果"全文常驻内存"，峰值轻松上 G；这里用三件事压住：

1. **JSONL 顺序追加 + (offset, length) 索引**：正文只落盘，索引里只有元信息；
   取正文按字节 `seek`，不读整个文件。
2. **LRU + 字节预算**：缓存只装"最近真的被读过"的 chunk 正文，超预算淘汰最久未用。
3. **索引与正文分离**：检索只扫元数据（约 300B/块，2 万块 ≈ 6MB），命中后才加载正文。
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


def byte_size(value: Any) -> int:
    """近似字节占用（用于缓存预算，不追求精确）。"""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (dict, list, tuple)):
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    return len(str(value).encode("utf-8"))


class JsonlStore:
    """追加写 + 偏移寻址读。一条记录 = 一行 JSON。"""

    def __init__(self, path: str | Path, mode: str = "a") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = mode.lower()
        if self._mode not in {"r", "a", "w"}:
            raise ValueError(f"不支持的 JsonlStore 模式：{mode}")
        # 写句柄用二进制：offset 必须是真实字节偏移，不能依赖文本模式的 tell() cookie。
        self._fh = (open(self.path, "r", encoding="utf-8", newline="\n")
                    if self._mode == "r" else
                    open(self.path, "ab" if self._mode == "a" else "wb"))

    def append(self, record: dict[str, Any]) -> tuple[int, int]:
        """写入一条，返回 (offset, length) —— 这就是按需加载的"地址"。"""
        if self._mode == "r":
            raise ValueError("只读 JsonlStore 不能写入")
        raw = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        offset = self._fh.tell()
        self._fh.write(raw)
        self._fh.flush()
        return offset, len(raw)

    def read(self, offset: int, length: int) -> dict[str, Any]:
        with open(self.path, "rb") as f:
            f.seek(offset)
            return json.loads(f.read(length).decode("utf-8"))

    def iter_all(self) -> Iterator[dict[str, Any]]:
        """流式全量遍历（只读句柄，不影响写位置）。"""
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "JsonlStore":
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()

    def __del__(self) -> None:      # 异常路径的兜底：GC 时释放句柄
        self.close()


@dataclass
class LRUCache:
    """字节预算的 LRU 缓存：只淘汰"最久未读"，且淘汰到预算之内。"""

    max_bytes: int
    name: str = "cache"
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    _bytes: int = 0
    _data: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    _sizes: dict[str, int] = field(default_factory=dict)

    def get(self, key: str, loader: Callable[[], Any] | None = None) -> Any:
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        if loader is None:
            self.misses += 1
            return None
        value = loader()
        self.put(key, value)
        self.misses += 1
        return value

    def put(self, key: str, value: Any) -> None:
        size = byte_size(value)
        if size > self.max_bytes:      # 单条就超预算：不进缓存，直接返回给调用方
            return
        if key in self._data:
            self._bytes -= self._sizes.pop(key)
            self._data.pop(key, None)
        self._data[key] = value
        self._sizes[key] = size
        self._bytes += size
        self._evict_until(self.max_bytes)

    def _evict_until(self, limit: int) -> None:
        while self._bytes > limit and self._data:
            k = next(iter(self._data))   # 迭代 OrderedDict 得到的是**键**，不能解包
            self._bytes -= self._sizes.pop(k)
            self._data.pop(k)
            self.evictions += 1

    def shrink_to(self, max_bytes: int) -> int:
        """外部（预算守卫）主动缩容，返回释放的字节数。"""
        before = self._bytes
        self.max_bytes = max_bytes
        self._evict_until(max_bytes)
        return before - self._bytes

    def clear(self) -> None:
        self._data.clear()
        self._sizes.clear()
        self._bytes = 0

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {"name": self.name, "items": len(self._data), "bytes": self._bytes,
                "max_bytes": self.max_bytes, "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}


@dataclass
class ChunkIndex:
    """常驻内存的**元信息**索引（不含正文）。

    标题路径做**字典编码**：同一小节的几十个 chunk 共享一份路径字符串，
    否则重复的长字符串会把索引撑到与正文同量级（实测 465 块时 222KB vs 339KB）。
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    paths: dict[str, list[str]] = field(default_factory=dict)   # section_id -> 标题路径
    _by_id: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ── 路径解析 ──
    def path_of(self, meta: dict[str, Any]) -> list[str]:
        return self.paths.get(str(meta.get("section_id")), [])

    def path_str_of(self, meta: dict[str, Any]) -> str:
        return " > ".join(self.path_of(meta))

    def set_path(self, section_id: str, path: list[str]) -> None:
        self.paths[str(section_id)] = list(path)

    def add(self, meta: dict[str, Any]) -> None:
        self.records.append(meta)
        self._by_id[str(meta.get("id"))] = meta

    def get(self, cid: str) -> dict[str, Any] | None:
        return self._by_id.get(cid)

    def by_level(self, level: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("level") == level]

    def children(self, parent_id: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("parent_id") == parent_id]

    def in_section(self, section_id: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("section_id") == section_id]

    def memory_bytes(self) -> int:
        # paths 与 records 一起常驻；只算 records 会低估索引实际占用。
        return byte_size(self.records) + byte_size(self.paths)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"records": self.records, "paths": self.paths}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "ChunkIndex":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        idx = cls(records=data.get("records", []), paths=data.get("paths", {}))
        idx._by_id = {str(r.get("id")): r for r in idx.records}
        return idx

    def stats(self) -> dict[str, Any]:
        by_level: dict[str, int] = {}
        tokens = 0
        for r in self.records:
            by_level[r.get("level", "?")] = by_level.get(r.get("level", "?"), 0) + 1
            tokens += int(r.get("n_tokens", 0) or 0)
        return {"chunks": len(self.records), "by_level": by_level, "tokens": tokens,
                "index_bytes": self.memory_bytes()}
