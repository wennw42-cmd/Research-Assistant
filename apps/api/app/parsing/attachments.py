"""上传附件解析：PDF / DOCX / 图片 / 纯文本 → 可送入对话与翻译的文本。

边界：
- 只接收白名单扩展名，限制 **字节数、PDF 页数、DOCX 解压后大小、返回文本长度**；
- DOCX 用标准库 zipfile + ElementTree 读 `word/document.xml`，不执行宏、不装 Office；
- 图片走 Tesseract（与扫描 PDF 共用可执行文件/语言包/中文空格压缩）；
- 解析不出来就明确报错，**不拿演示语料冒充真实结果**。
"""
from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS = 120_000
MAX_PDF_PAGES = 100
MAX_DOCX_XML_BYTES = 30 * 1024 * 1024
MAX_DOCX_TOTAL_BYTES = 100 * 1024 * 1024

SUPPORTED_SUFFIXES = {
    ".pdf", ".docx", ".png", ".jpg", ".jpeg",
    ".tif", ".tiff", ".bmp", ".webp", ".txt", ".md", ".markdown",
}
KIND_BY_SUFFIX = {
    ".pdf": "pdf", ".docx": "docx",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".tif": "image",
    ".tiff": "image", ".bmp": "image", ".webp": "image",
    ".txt": "text", ".md": "text", ".markdown": "text",
}

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _clean_name(filename: str) -> str:
    name = re.sub(r"[\x00-\x1f\x7f]", " ", Path(filename or "upload").name).strip()
    return name[:200] or "upload"


def _truncate(text: str, notes: list[str]) -> tuple[str, bool]:
    text = text.strip()
    if len(text) <= MAX_TEXT_CHARS:
        return text, False
    notes.append(f"文本超过 {MAX_TEXT_CHARS} 字符，已截断；完整内容请拆分文件后上传。")
    return text[:MAX_TEXT_CHARS], True


def _pdf_text(data: bytes) -> tuple[str, dict[str, Any], list[str]]:
    try:
        import pymupdf as pm
    except ImportError:  # pragma: no cover
        import fitz as pm  # type: ignore

    with tempfile.TemporaryDirectory(prefix="ra_upload_") as tmp:
        path = Path(tmp) / "upload.pdf"
        path.write_bytes(data)
        try:
            doc = pm.open(str(path))
            try:
                pages = doc.page_count
            finally:
                doc.close()
            if pages > MAX_PDF_PAGES:
                raise ValueError(f"PDF 共 {pages} 页，超过 {MAX_PDF_PAGES} 页上限，请拆分后上传")

            from app.parsing.pipeline import parse_pdf

            res = parse_pdf(path, ocr="auto", dpi=180, assets_dir=Path(tmp) / "assets")
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"PDF 解析失败：{type(exc).__name__}: {exc}") from exc

    parts: list[str] = []
    for block in res.blocks:
        if block.type in ("header", "footer", "unknown"):
            continue
        if block.type == "table":
            text = block.markdown or block.text
        elif block.type == "formula":
            text = block.latex or block.text
        elif block.type == "figure":
            text = block.caption or block.text
        else:
            text = block.text or block.caption or ""
        text = (text or "").strip()
        if not text:
            continue
        if block.type in ("title", "heading"):
            text = f"{'#' * min(max(block.level or 1, 1), 6)} {text}"
        parts.append(text)

    scanned = int(res.stats.get("scanned_pages") or 0)
    warnings = list(res.warnings)
    if scanned:
        warnings.append(f"{scanned} 页为扫描页，已使用 OCR（低置信内容可能不准确）")
    meta = {
        "pages": res.source.get("pages", 0),
        "layout": res.layout,
        "blocks": len(res.blocks),
        "scanned_pages": scanned,
        "tables": res.stats.get("tables", 0),
        "formulas": res.stats.get("formulas", 0),
    }
    return "\n\n".join(parts), meta, warnings


def _docx_text(data: bytes) -> tuple[str, dict[str, Any], list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = sum(info.file_size for info in zf.infolist())
            if total > MAX_DOCX_TOTAL_BYTES:
                raise ValueError("DOCX 解压后体积过大，请先清理嵌入的图片/视频")
            try:
                xml = zf.read("word/document.xml")
            except KeyError as exc:
                raise ValueError("不是有效的 .docx（缺少 word/document.xml）") from exc
    except zipfile.BadZipFile as exc:
        raise ValueError("不是有效的 .docx；旧版 .doc 请先另存为 .docx") from exc

    if len(xml) > MAX_DOCX_XML_BYTES:
        raise ValueError("DOCX 正文 XML 过大，请拆分文档")
    if b"<!ENTITY" in xml.upper():
        raise ValueError("DOCX 含 XML 实体声明，已拒绝解析")

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError(f"DOCX 正文 XML 解析失败：{exc}") from exc

    lines: list[str] = []
    paragraphs = 0
    for para in root.iter(f"{_W_NS}p"):
        text = "".join(node.text or "" for node in para.iter(f"{_W_NS}t")).strip()
        if not text:
            continue
        paragraphs += 1
        style = para.find(f"{_W_NS}pPr/{_W_NS}pStyle")
        style_name = (style.get(f"{_W_NS}val") or "").lower() if style is not None else ""
        match = re.match(r"heading\s*([1-6])", style_name)
        if match:
            lines.append(f"{'#' * int(match.group(1))} {text}")
        else:
            lines.append(text)

    meta = {"paragraphs": paragraphs, "pages": None}
    return "\n\n".join(lines), meta, []


def _image_text(data: bytes) -> tuple[str, dict[str, Any], list[str]]:
    try:
        from PIL import Image, ImageOps
        import pytesseract
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"图片 OCR 依赖不可用（pip install pillow pytesseract）：{exc}") from exc

    from app.parsing.ocr import _compact_cjk, _tessdata_dir, _tesseract_cmd

    cmd = _tesseract_cmd()
    if not cmd:
        raise ValueError("未找到 tesseract，可执行文件；请安装 Tesseract 或配置 TESSERACT_CMD")
    pytesseract.pytesseract.tesseract_cmd = cmd

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"图片无法解码：{type(exc).__name__}") from exc

    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    width, height = image.size
    if max(width, height) > 5000:
        image.thumbnail((5000, 5000), Image.Resampling.LANCZOS)

    try:
        from app.config import get_settings
        lang = get_settings().OCR_LANG or "eng"
    except Exception:  # noqa: BLE001
        lang = "eng"
    tessdata = _tessdata_dir()
    cfg = f"--tessdata-dir {Path(tessdata).as_posix()}" if tessdata else ""

    def run(code: str) -> str:
        return pytesseract.image_to_string(image, lang=code, config=cfg)

    try:
        text = run(lang)
    except Exception:  # noqa: BLE001
        try:
            text = run("eng")
            lang = "eng"
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"OCR 失败：{type(exc).__name__}") from exc

    warnings = [] if text.strip() else ["图片中没有识别到文字；可换更清晰的扫描件或截图。"]
    meta = {"lang": lang, "width": width, "height": height, "scanned_pages": 1}
    return _compact_cjk(text), meta, warnings


def _plain_text(data: bytes) -> tuple[str, dict[str, Any], list[str]]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding), {"encoding": encoding, "pages": None}, []
        except UnicodeDecodeError:
            continue
    raise ValueError("文本编码无法识别（支持 UTF-8 / GB18030）")


def extract_attachment(filename: str, data: bytes) -> dict[str, Any]:
    """解析上传字节；失败抛 ValueError（由 HTTP 层转 400）。"""
    name = _clean_name(filename)
    suffix = Path(name).suffix.lower()
    if suffix == ".doc":
        raise ValueError("旧版 .doc 不支持，请用 Word 另存为 .docx 后重试")
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"不支持的文件类型：{suffix or '无扩展名'}；"
            "支持 PDF、DOCX、PNG/JPG/TIFF/BMP/WEBP、TXT/MD"
        )
    if not data:
        raise ValueError("文件为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    kind = KIND_BY_SUFFIX[suffix]
    if kind == "pdf":
        text, meta, warnings = _pdf_text(data)
    elif kind == "docx":
        text, meta, warnings = _docx_text(data)
    elif kind == "image":
        text, meta, warnings = _image_text(data)
    else:
        text, meta, warnings = _plain_text(data)

    text, truncated = _truncate(text, warnings)
    return {
        "filename": name,
        "kind": kind,
        "text": text,
        "chars": len(text),
        "truncated": truncated,
        "meta": meta,
        "warnings": warnings,
    }


def _selftest() -> int:
    """离线自检：文本 / DOCX / PDF / 类型拒绝，不依赖 Tesseract。"""
    import tempfile

    checks: list[tuple[str, bool]] = []

    text = extract_attachment("notes.txt", "中文摘要。\n\nEnglish.".encode("utf-8"))
    checks.append(("text", text["kind"] == "text" and "中文摘要" in text["text"]))

    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = (
        f'<w:document xmlns:w="{w}"><w:body>'
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
        "<w:r><w:t>1 Introduction</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>We propose a framework.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    ).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", xml)
    docx = extract_attachment("paper.docx", buf.getvalue())
    checks.append(("docx", "# 1 Introduction" in docx["text"]))

    try:
        from app.parsing.cli import make_sample_pdf
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "sample.pdf"
            make_sample_pdf(pdf_path)
            pdf = extract_attachment("sample.pdf", pdf_path.read_bytes())
        checks.append(("pdf", "Attention Is All You Need" in pdf["text"] and pdf["meta"]["pages"] >= 1))
    except Exception as exc:  # noqa: BLE001
        checks.append((f"pdf ({type(exc).__name__})", False))

    try:
        extract_attachment("old.doc", b"x")
    except ValueError:
        checks.append(("reject .doc", True))
    else:
        checks.append(("reject .doc", False))

    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"attachments selftest: {sum(ok for _, ok in checks)}/{len(checks)}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
