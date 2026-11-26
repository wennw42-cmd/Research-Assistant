"""扫描件 OCR 兜底：PaddleOCR PP-StructureV2/V3 → pytesseract → 裁图保留。

坐标换算：把页面按 dpi 栅格化后做 OCR，得到的是**像素坐标**，
统一按 scale = 72 / dpi 换算回 PDF 的 pt 坐标系（同为左上原点），保证与文本块 bbox 对齐。

三级降级都失败时，**不猜文字**：把该区域裁成 PNG 存到 assets 目录，
产出 `type="unknown"` + image_path，交给前端展示原图或后续人工/模型处理。
"""
from __future__ import annotations

import os
import re
import shutil
from html import unescape
from pathlib import Path
from typing import Any

try:
    import pymupdf as pm
except ImportError:  # pragma: no cover
    import fitz as pm  # type: ignore

DEFAULT_DPI = 200
_TEXT_THRESHOLD = 90  # 页面可见字符数低于此值且含图像 → 判定为扫描页

_CJK = r"一-鿿"


def _compact_cjk(text: str) -> str:
    """Tesseract 中文会在字词间插空格；去掉汉字之间和标点前的空格，保留中英之间的。"""
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text)
    text = re.sub(rf"\s+(?=[，。；：、！？）】”’])", "", text)
    return re.sub(rf"(?<=[（【“‘])\s+", "", text)


TYPE_MAP = {
    "title": "title", "doc_title": "title", "paragraph_title": "heading",
    "text": "paragraph", "plain text": "paragraph", "abstract": "paragraph",
    "content": "paragraph", "figure": "figure", "figure_caption": "figure_caption",
    "table": "table", "table_caption": "table_caption", "table_title": "table_caption",
    "formula": "formula", "equation": "formula", "isolate_formula": "formula",
    "header": "header", "footer": "footer", "list": "list_item", "reference": "reference",
    "number": "paragraph", "seal": "unknown", "aside_text": "paragraph",
    "vision_footnote": "paragraph",
}


def needs_ocr(page: Any) -> bool:
    text = ""
    try:
        text = (page.get_text("text") or "").strip()
    except Exception:  # noqa: BLE001
        pass
    if len(text) >= _TEXT_THRESHOLD:
        return False
    has_image = False
    try:
        has_image = bool(page.get_images(full=False)) or bool(page.get_image_info())
    except Exception:  # noqa: BLE001
        has_image = False
    return has_image


def rasterize(page: Any, dpi: int = DEFAULT_DPI) -> Any:
    return page.get_pixmap(dpi=dpi, alpha=False)


def crop_region(page: Any, bbox: tuple[float, float, float, float], out_path: str | Path,
                dpi: int = DEFAULT_DPI) -> str:
    """按 PDF 坐标裁剪区域并保存（保留原图区域，供前端/人工复核）。"""
    from pymupdf import Rect

    clip = Rect(*bbox)
    pix = page.get_pixmap(dpi=dpi, alpha=False, clip=clip)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    return str(out_path)


# ────────────────────────── 后端 ──────────────────────────
def _backend_paddle(pix: Any, scale: float) -> list[dict[str, Any]] | None:
    try:
        try:
            from paddleocr import PPStructureV3 as _PP  # paddleocr >= 3
        except Exception:  # noqa: BLE001
            from paddleocr import PPStructure as _PP  # type: ignore

        engine = _PP(table=True, ocr=True, show_log=False)  # type: ignore[call-arg]
        buf = pix.tobytes("png")
        result = engine.predict(buf) if hasattr(engine, "predict") else engine(buf)

        blocks: list[dict[str, Any]] = []
        for item in _iter_pp_items(result):
            raw_type = str(item.get("type", "text")).lower()
            bbox = _scale_bbox(item.get("bbox"), scale)
            res = item.get("res") or {}
            text = _pp_text(res)
            block = {
                "type": TYPE_MAP.get(raw_type, "paragraph"),
                "bbox": bbox,
                "text": text,
                "confidence": float(res.get("score", 0.6)) if isinstance(res, dict) else 0.6,
                "source": "paddleocr",
            }
            if raw_type in ("table", "table_caption") and isinstance(res, dict):
                html = res.get("html") or ""
                if html:
                    block["markdown"] = html_table_to_markdown(html)
            if "latex" in (res or {}):
                block["latex"] = res.get("latex")
                block["latex_approx"] = False
            blocks.append(block)
        return blocks or None
    except Exception:  # noqa: BLE001
        return None


def _iter_pp_items(result: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(result, list):
        for r in result:
            if isinstance(r, dict):
                if "type" in r and ("bbox" in r or "res" in r):
                    items.append(r)
                elif "result" in r and isinstance(r["result"], list):
                    items += [x for x in r["result"] if isinstance(x, dict)]
            elif hasattr(r, "json") and isinstance(getattr(r, "json"), dict):
                res = r.json.get("res", r.json)
                for k, v in res.items() if isinstance(res, dict) else []:
                    if isinstance(v, list):
                        items += [x for x in v if isinstance(x, dict)]
    return items


def _pp_text(res: Any) -> str:
    if isinstance(res, str):
        return res.strip()
    if isinstance(res, dict):
        return str(res.get("text") or "").strip()
    if isinstance(res, list):
        return " ".join(str(x.get("text", "")) for x in res if isinstance(x, dict)).strip()
    return ""


def _tesseract_cmd() -> str:
    """找 tesseract.exe：显式配置 > PATH > Windows 默认安装目录。"""
    try:
        from app.config import get_settings
        configured = get_settings().TESSERACT_CMD
    except Exception:  # noqa: BLE001
        configured = os.getenv("TESSERACT_CMD", "")
    if configured and Path(configured).exists():
        return configured
    found = shutil.which("tesseract")
    if found:
        return found
    for p in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"),
    ):
        if Path(p).exists():
            return p
    return ""


def _tessdata_dir() -> str:
    """找 tessdata 目录：显式配置 > 用户目录（无需管理员写 Program Files）。"""
    try:
        from app.config import get_settings
        configured = get_settings().TESSDATA_PREFIX
    except Exception:  # noqa: BLE001
        configured = os.getenv("TESSDATA_PREFIX", "")
    if configured and Path(configured).is_dir():
        return configured
    user_dir = Path.home() / "AppData/Local/tessdata"
    return str(user_dir) if user_dir.is_dir() else ""


def _backend_tesseract(pix: Any, scale: float) -> list[dict[str, Any]] | None:
    try:
        import io

        import pytesseract
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    cmd = _tesseract_cmd()
    if not cmd:
        return None
    pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        from app.config import get_settings
        lang = get_settings().OCR_LANG or "eng"
    except Exception:  # noqa: BLE001
        lang = os.getenv("OCR_LANG", "eng")
    tessdata = _tessdata_dir()
    # pytesseract 会把 config 原样拼进命令行；路径含空格时请放到无空格目录。
    cfg = f"--tessdata-dir {Path(tessdata).as_posix()}" if tessdata else ""
    try:
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            data = pytesseract.image_to_data(
                img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
        except Exception:  # noqa: BLE001
            # 语言包没装时退回英文，避免整页 OCR 直接失效。
            data = pytesseract.image_to_data(
                img, lang="eng", config=cfg, output_type=pytesseract.Output.DICT)
    except Exception:  # noqa: BLE001
        return None

    lines: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    n = len(data.get("text", []))
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:  # noqa: BLE001
            conf = -1.0
        if conf < 30:  # 低置信词不入库，避免把噪声当内容
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        x0, y0 = float(data["left"][i]), float(data["top"][i])
        x1, y1 = x0 + float(data["width"][i]), y0 + float(data["height"][i])
        lines.setdefault(key, []).append({"t": word, "bbox": [x0, y0, x1, y1], "c": conf})

    blocks: list[dict[str, Any]] = []
    for key in sorted(lines):
        ws = lines[key]
        bbox = [min(w["bbox"][0] for w in ws), min(w["bbox"][1] for w in ws),
                max(w["bbox"][2] for w in ws), max(w["bbox"][3] for w in ws)]
        text = _compact_cjk(" ".join(w["t"] for w in ws))
        blocks.append({
            "type": "paragraph",
            "bbox": [round(v * scale, 2) for v in bbox],
            "text": text,
            "confidence": round(sum(w["c"] for w in ws) / len(ws) / 100, 3),
            "source": "tesseract",
        })
    return blocks or None


def _scale_bbox(bbox: Any, scale: float) -> list[float]:
    if not bbox or len(bbox) < 4:
        return []
    vals = [float(v) for v in bbox[:4]]
    if max(vals) <= 100 and scale < 1:  # 已是无量纲比例时不缩放
        return [round(v, 2) for v in vals]
    return [round(v * scale, 2) for v in vals]


# ────────────────────────── HTML 表格 → Markdown ──────────────────────────
def _md_cell(cell: str) -> str:
    """HTML 单元格 → Markdown 单元格：<br> 变空格、去标签、解码实体、转义竖线。"""
    text = re.sub(r"<br\s*/?>", " ", cell, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).replace("|", r"\|").strip()


def html_table_to_markdown(html: str) -> str:
    """HTML 表格 → GitHub Markdown。colspan/rowspan 不展开（保留原始文本顺序）。"""
    rows: list[list[str]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)
        rows.append([_md_cell(c) for c in cells])
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    head = "| " + " | ".join(norm[0]) + " |"
    sep = "| " + " | ".join("---" for _ in range(width)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in norm[1:]]
    return "\n".join([head, sep, *body])


def ocr_page(page: Any, *, dpi: int = DEFAULT_DPI, backend: str = "auto",
             assets_dir: str | Path | None = None, page_no: int = 1) -> tuple[list[dict[str, Any]], str]:
    """单页 OCR。返回 (blocks, backend_used)；backend_used="none" 表示只裁图未识别。"""
    pix = rasterize(page, dpi)
    scale = 72.0 / dpi

    if backend in ("auto", "paddle"):
        blocks = _backend_paddle(pix, scale)
        if blocks:
            return blocks, "paddleocr"
    if backend in ("auto", "tesseract"):
        blocks = _backend_tesseract(pix, scale)
        if blocks:
            return blocks, "tesseract"

    image_path = ""
    if assets_dir:
        image_path = crop_region(page, (0, 0, page.rect.width, page.rect.height),
                                 Path(assets_dir) / f"page{page_no:03d}_scan.png", dpi)
    reason = "no_ocr_backend" if backend == "auto" else f"backend_unavailable:{backend}"
    return [{
        "type": "unknown",
        "bbox": [0.0, 0.0, round(float(page.rect.width), 2), round(float(page.rect.height), 2)],
        "text": "",
        "confidence": 0.0,
        "source": reason,
        "image_path": image_path,
        "meta": {"reason": reason, "dpi": dpi},
    }], "none"
