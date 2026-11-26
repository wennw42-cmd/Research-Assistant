from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

_JSON_FENCE = re.compile(r"```(?:json|JSON)?\s*(?P<body>.*?)```", re.S)
_JSON_OBJECT = re.compile(r"\{.*\}|\[.*\]", re.S)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


log = get_logger("ra.agent")


def jdump(obj: Any, limit: int = 800) -> str:
    """统一序列化：中文可读、异常安全、超长截断（避免把原文全塞进工具返回）。"""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        s = repr(obj)
    return s if len(s) <= limit else s[:limit] + f"…[truncated {len(s) - limit} chars]"


def extract_json(text: str, default: Any = None) -> Any:
    """从 LLM 输出里稳健抽取 JSON：优先代码块，其次裸对象/数组。

    LLM 经常输出 ```json ... ``` 或前后带解释性文字，这里做两级兜底。
    """
    if text is None:
        return default
    m = _JSON_FENCE.search(text)
    raw = m.group("body") if m else None
    candidates = [raw] if raw else []
    m2 = _JSON_OBJECT.search(text)
    if m2:
        candidates.append(m2.group(0))
    for c in candidates:
        try:
            return json.loads(c.strip())
        except Exception:  # noqa: BLE001
            continue
    return default


def short(text: str, n: int = 200) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "…"


def timer_ms() -> Callable[[], int]:
    """轻量计时器：with-less 用法 t = timer_ms(); ...; ms = t()。"""
    start = time.perf_counter()

    def _elapsed() -> int:
        return int((time.perf_counter() - start) * 1000)

    return _elapsed


def role_of(msg: dict) -> str:
    return str(msg.get("role", ""))
