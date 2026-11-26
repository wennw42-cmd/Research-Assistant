"""pytest 共享夹具：合成 PDF、检测 OCR 后端、统一的指标输出。

所有夹具都**离线**：不访问网络、不调用真实 LLM。
需要 Tesseract 的用例由 ``require_tesseract`` 显式跳过，其余用例照常执行。

运行（在 apps/api 下）：
    python -m pytest tests -q
    python -m pytest tests -q -s          # 同时打印内存/耗时指标
"""
from __future__ import annotations

import os
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

TOTAL_PAGES = 541
PAGES_PER_CHAPTER = 10

# 阈值可用环境变量覆盖，避免不同机器/CI 因绝对耗时波动而误报。
INGEST_TIME_LIMIT_S = float(os.getenv("RA_TEST_INGEST_SECONDS", "240"))
INGEST_MEMORY_LIMIT_MB = float(os.getenv("RA_TEST_MEMORY_MB", "1024"))
LOAD_MEMORY_LIMIT_MB = float(os.getenv("RA_TEST_LOAD_MEMORY_MB", "128"))


def tesseract_available() -> bool:
    try:
        from app.parsing.ocr import _tesseract_cmd

        return bool(_tesseract_cmd())
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def require_tesseract() -> None:
    if not tesseract_available():
        pytest.skip("未找到 Tesseract 可执行文件；扫描件端到端 OCR 用例跳过")


def _build_scanned_pdf(path: Path) -> Path:
    """把一张只有像素、没有文本层的图片放进 PDF —— 模拟真实扫描件。"""
    import pymupdf as pm
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1654, 2339  # A4 @200dpi
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def font(size: int):
        for candidate in ("arial.ttf", "arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:  # noqa: BLE001
                continue
        return ImageFont.load_default(size=size)

    draw.text((110, 360), "RESEARCH ASSISTANT", font=font(110), fill="black")
    draw.text((110, 620), "OCR TEST 2026", font=font(92), fill="black")
    draw.text((110, 860), "STRUCTURED RESTORATION", font=font(86), fill="black")

    png = path.with_suffix(".png")
    image.save(png)

    doc = pm.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, filename=str(png))
    doc.save(str(path))
    doc.close()
    png.unlink(missing_ok=True)
    return path


@pytest.fixture(scope="session")
def scanned_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_scanned_pdf(tmp_path_factory.mktemp("ocr") / "scanned.pdf")


def _build_long_pdf(path: Path) -> Path:
    """541 页合成教材：每 10 页一章，每章 1 个二级节标题，正文 10pt。"""
    import pymupdf as pm

    doc = pm.open()
    for p in range(TOTAL_PAGES):
        ch = p // PAGES_PER_CHAPTER + 1
        page = doc.new_page(width=595, height=842)
        if p % PAGES_PER_CHAPTER == 0:
            # 20pt / 14pt 的编号标题：与正文 10pt 拉开字号，保证能建出章节树。
            page.insert_text((72, 100), f"{ch} Long Document Chapter {ch}",
                             fontsize=20, fontname="hebo")
            page.insert_text((72, 140), f"{ch}.1 Core Section {ch}",
                             fontsize=14, fontname="hebo")
        lines = [
            f"Page {p + 1} marker C{ch}P{p + 1}. This synthetic page belongs to",
            f"chapter {ch} and section {ch}.1. It exercises streaming parsing,",
            "paragraph chunking, sentence chunking, and tiered retrieval.",
            "The quick brown fox jumps over the lazy dog while the",
            "retrieval loader reads only the chunks selected for the query.",
        ]
        y = 190
        for line in lines:
            page.insert_text((72, y), line, fontsize=10, fontname="helv")
            y += 16
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="session")
def long_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_long_pdf(tmp_path_factory.mktemp("pdf541") / "long_541.pdf")


@pytest.fixture(scope="session")
def ingested_long_doc(long_pdf: Path, tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """只入库一次：541 页的解析/切块/索引是重量级步骤，两个用例共享结果。"""
    from app.rag.ingest import ingest_pdf
    from app.rag.memory import MemoryMonitor

    out = tmp_path_factory.mktemp("rag541")
    monitor = MemoryMonitor(trace=True, label="ingest541")
    tracemalloc.reset_peak()
    started = time.perf_counter()
    report = ingest_pdf(
        long_pdf,
        out,
        page_batch=24,
        doc_id="long_541",
        memory_limit_mb=INGEST_MEMORY_LIMIT_MB,
        monitor=monitor,
    )
    elapsed_s = time.perf_counter() - started
    peak_after = monitor.snapshot("after_ingest")
    return SimpleNamespace(
        out=out,
        report=report,
        monitor=monitor,
        elapsed_s=elapsed_s,
        peak_after=peak_after,
    )
