"""内存/显存监控与阶段计量。

三层可观测，缺哪层都能跑：
    tracemalloc  → Python 堆分配（一定有）
    psutil       → 进程 RSS（装了才有）
    torch.cuda   → 显存（有 GPU 才有，可选）

阶段计量用 `with monitor.stage("phase1_parse")` 包住代码块，自动记录
起止 RSS / 分配量 / 增量 / 峰值，最后 `report()` 打成表格——**没有监控就没有优化依据**。
"""
from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

try:
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # noqa: BLE001
    torch = None  # type: ignore

MB = 1024 * 1024


@dataclass
class Snapshot:
    label: str = ""
    rss_mb: float | None = None
    alloc_mb: float | None = None
    peak_mb: float | None = None
    cuda_mb: float | None = None
    elapsed_ms: int = 0


@dataclass
class StageStat:
    name: str
    start: Snapshot
    end: Snapshot
    delta_rss_mb: float | None = None
    delta_alloc_mb: float | None = None
    # tracemalloc 只提供全局峰值，没有“区间峰值”API；这里记录的是截至阶段末的累计峰值。
    peak_stage_mb: float | None = None


class MemoryMonitor:
    """进程级内存计量器。"""

    def __init__(self, *, trace: bool = True, label: str = "") -> None:
        self.label = label
        self._on = trace
        self._proc = psutil.Process() if psutil else None
        self.stages: list[StageStat] = []
        self.peak_rss_mb = 0.0
        self.peak_alloc_mb = 0.0
        if trace and not tracemalloc.is_tracing():
            tracemalloc.start()

    # ── 采样 ──
    def snapshot(self, label: str = "", elapsed_ms: int = 0) -> Snapshot:
        rss = None
        if self._proc:
            try:
                rss = self._proc.memory_info().rss / MB
                self.peak_rss_mb = max(self.peak_rss_mb, rss)
            except Exception:  # noqa: BLE001
                rss = None
        alloc = peak = None
        if tracemalloc.is_tracing():
            cur, pk = tracemalloc.get_traced_memory()
            alloc, peak = cur / MB, pk / MB
            self.peak_alloc_mb = max(self.peak_alloc_mb, peak)
        cuda = None
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    cuda = torch.cuda.memory_allocated() / MB
            except Exception:  # noqa: BLE001
                cuda = None
        return Snapshot(label=label, rss_mb=rss, alloc_mb=alloc, peak_mb=peak,
                        cuda_mb=cuda, elapsed_ms=elapsed_ms)

    # ── 阶段计量 ──
    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        s0 = self.snapshot(name)
        try:
            yield
        finally:
            dt = int((time.perf_counter() - t0) * 1000)
            s1 = self.snapshot(name, dt)
            self.stages.append(StageStat(
                name=name, start=s0, end=s1,
                delta_rss_mb=_d(s1.rss_mb, s0.rss_mb),
                delta_alloc_mb=_d(s1.alloc_mb, s0.alloc_mb),
                peak_stage_mb=s1.peak_mb,
            ))

    # ── 限额守卫 ──
    def check(self, limit_mb: float) -> tuple[bool, str]:
        """超限时返回 (False, 说明)；不抛异常，由调用方决定降级策略。"""
        snap = self.snapshot("check")
        cur = snap.rss_mb or snap.alloc_mb or 0.0
        if cur > limit_mb:
            return False, (f"内存 {cur:.1f}MB 超过限额 {limit_mb:.1f}MB"
                           f"（RSS={_f(snap.rss_mb)} 分配={_f(snap.alloc_mb)} 峰值={_f(snap.peak_mb)}）")
        return True, f"内存 {cur:.1f}MB 在限额 {limit_mb:.1f}MB 内"

    # ── 报告 ──
    def report(self) -> str:
        rows = [f"{'阶段':<20} {'耗时ms':>8} {'RSS MB':>9} {'增量 RSS':>10} {'堆分配 MB':>11} {'堆峰值 MB':>11}"]
        rows.append("─" * 74)
        for st in self.stages:
            rows.append(
                f"{st.name:<20} {st.end.elapsed_ms:>8} {_f(st.end.rss_mb):>9} "
                f"{_f(st.delta_rss_mb, plus=True):>10} {_f(st.end.alloc_mb):>11} {_f(st.peak_stage_mb):>11}"
            )
        last = self.snapshot("final")
        rows.append("─" * 74)
        rows.append(f"峰值 RSS {self.peak_rss_mb:.1f} MB | 峰值堆分配 {self.peak_alloc_mb:.1f} MB | "
                    f"末态 RSS {_f(last.rss_mb)} | 显存 {_f(last.cuda_mb)}")
        if psutil is None:
            rows.append("[i] 未安装 psutil，RSS 列为空（pip install psutil 可补齐）")
        if torch is None:
            rows.append("[i] 未安装 torch，显存列为空")
        return "\n".join(rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "peak_alloc_mb": round(self.peak_alloc_mb, 2),
            "stages": [
                {"name": s.name, "ms": s.end.elapsed_ms,
                 "rss_mb": _num(s.end.rss_mb), "delta_rss_mb": _num(s.delta_rss_mb),
                 "alloc_mb": _num(s.end.alloc_mb), "peak_mb": _num(s.peak_stage_mb)}
                for s in self.stages
            ],
            "backends": {"tracemalloc": tracemalloc.is_tracing(),
                         "psutil": psutil is not None, "torch_cuda": bool(torch)},
        }


def _d(a: float | None, b: float | None) -> float | None:
    return None if (a is None or b is None) else a - b


def _num(v: float | None) -> float | None:
    return None if v is None else round(v, 2)


def _f(v: float | None, plus: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:+.1f}" if plus else f"{v:.1f}"


@dataclass
class BudgetGuard:
    """预算守卫：超限即触发回调（默认清理缓存），并累计触发次数。"""

    limit_mb: float
    on_exceed: Any = None
    monitor: MemoryMonitor | None = None
    hits: int = 0
    log: list[str] = field(default_factory=list)

    def check(self) -> bool:
        mon = self.monitor or MemoryMonitor(trace=False)
        ok, msg = mon.check(self.limit_mb)
        if not ok:
            self.hits += 1
            self.log.append(msg)
            if self.on_exceed:
                self.on_exceed(msg)
        return ok
