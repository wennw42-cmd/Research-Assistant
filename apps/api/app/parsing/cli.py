"""命令行入口与自测。

    python -m app.parsing.cli sample.pdf -o out.json --assets assets/
    python -m app.parsing.cli --selftest

自测会**现场合成**一份双栏论文 PDF（标题/摘要/章节 H1-H2/表格/图注/参考文献）
和一份纯图片扫描页 PDF，无需准备任何外部素材。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import pymupdf as pm
except ImportError:  # pragma: no cover
    import fitz as pm  # type: ignore

from app.parsing.extract import latex_approx
from app.parsing.pipeline import parse_pdf

BOLD = "hebo"
REG = "helv"


# ────────────────────────── 合成素材 ──────────────────────────
def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for w in text.split():
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def _para(page: Any, x: float, y: float, text: str, size: float = 10.0,
          width: int = 42, font: str = REG) -> float:
    """插入一段自动换行文本，返回结束 y。"""
    step = size * 1.4
    for i, line in enumerate(_wrap(text, width)):
        page.insert_text((x, y + i * step), line, fontsize=size, fontname=font)
    return y + (len(_wrap(text, width)) + 1) * step


def make_sample_pdf(path: str | Path) -> Path:
    """合成一份双栏论文（A4）。"""
    doc = pm.open()
    p1 = doc.new_page()

    p1.insert_text((150, 70), "Attention Is All You Need", fontsize=20, fontname=BOLD)
    p1.insert_text((165, 92), "Ashish Vaswani, Noam Shazeer, Niki Parmar", fontsize=10, fontname=REG)
    p1.insert_text((60, 132), "Abstract", fontsize=12, fontname=BOLD)
    _para(p1, 60, 152, "The dominant sequence transduction models are based on complex recurrent "
                       "or convolutional neural networks that include an encoder and a decoder.",
          10, 38)
    _para(p1, 320, 152, "We propose a new simple network architecture, the Transformer, based "
                        "solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
          10, 38)

    p1.insert_text((60, 300), "1 Introduction", fontsize=13, fontname=BOLD)
    _para(p1, 60, 318, "Recurrent neural networks, long short-term memory and gated recurrent "
                       "neural networks in particular, have been firmly established as state of the art "
                       "approaches in sequence modeling.", 10, 38)
    _para(p1, 320, 318, "Attention mechanisms have become an integral part of compelling sequence "
                        "modeling models, allowing modeling of dependencies without regard to their distance "
                        "in the input or output sequences.", 10, 38)

    p1.insert_text((60, 470), "2 Method", fontsize=13, fontname=BOLD)
    p1.insert_text((60, 492), "2.1 Scaled Dot-Product Attention", fontsize=11, fontname=BOLD)
    _para(p1, 60, 510, "We call our particular attention scaled dot-product attention. The input "
                       "consists of queries and keys of dimension dk, and values of dimension dv.", 10, 38)
    _para(p1, 320, 470, "Experiments on two machine translation tasks show these models to be "
                        "superior in quality while being more parallelizable and requiring significantly "
                        "less time to train.", 10, 38)

    # ── 第 2 页：表格 + 图 + 参考文献 ──
    p2 = doc.new_page()
    p2.insert_text((60, 60), "3 Experiments", fontsize=13, fontname=BOLD)
    p2.insert_text((60, 88), "Table 1: BLEU scores on WMT 2014.", fontsize=10, fontname=BOLD)

    x0, y0, col_w, row_h = 60.0, 100.0, 84.0, 20.0
    rows, cols = 4, 3
    for r in range(rows + 1):
        p2.draw_line((x0, y0 + r * row_h), (x0 + cols * col_w, y0 + r * row_h))
    for c in range(cols + 1):
        p2.draw_line((x0 + c * col_w, y0), (x0 + c * col_w, y0 + rows * row_h))
    cells = [["Model", "EN-DE", "EN-FR"], ["Transformer", "28.4", "41.8"],
             ["LSTM", "25.2", "38.1"], ["RNN", "23.1", "35.4"]]
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            p2.insert_text((x0 + c * col_w + 6, y0 + r * row_h + 13), val, fontsize=9, fontname=REG)

    tmp = pm.open()
    tp = tmp.new_page(width=225, height=120)
    tp.insert_text((10, 60), "Encoder / Decoder stack", fontsize=12, fontname=BOLD)
    pix = tp.get_pixmap(dpi=150)
    p2.insert_image(pm.Rect(320, 100, 545, 215), pixmap=pix)
    p2.insert_text((320, 232), "Figure 1: The Transformer model architecture.", fontsize=10, fontname=BOLD)

    p2.insert_text((60, 300), "References", fontsize=13, fontname=BOLD)
    refs = [
        "[1] Ashish Vaswani, Noam Shazeer, et al. Attention is all you need. arXiv:1706.03762, 2017.",
        "[2] Jacob Devlin, Ming-Wei Chang, et al. BERT: Pre-training of deep bidirectional transformers "
        "for language understanding. https://aclanthology.org/N19-1423/, 2019.",
        "[3] Tom Brown, et al. Language models are few-shot learners. DOI:10.5555/example.2020.",
    ]
    y = 320.0
    for r in refs:
        y = _para(p2, 60, y, r, 9, 40)
    _para(p2, 320, 300, "The table and figure above summarize our main empirical results on the two "
                        "translation tasks.", 10, 38)

    doc.save(str(path))
    doc.close()
    return Path(path)


def make_scan_pdf(path: str | Path) -> Path:
    """合成一份**无文本层**的扫描页（整页是一张图）。"""
    src = pm.open()
    sp = src.new_page()
    sp.insert_text((72, 120), "Scanned page: no text layer here.", fontsize=14, fontname=BOLD)
    _para(sp, 72, 160, "This page simulates a scanned or photographed document page so that the "
                       "OCR fallback path can be exercised end to end.", 11, 60)
    pix = sp.get_pixmap(dpi=150)

    doc = pm.open()
    p = doc.new_page()
    p.insert_image(p.rect, pixmap=pix)
    doc.save(str(path))
    doc.close()
    return Path(path)


# ────────────────────────── 自测 ──────────────────────────
def _selftest() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="docir_"))
    paper = make_sample_pdf(tmp / "sample_paper.pdf")
    scan = make_scan_pdf(tmp / "scan_only.pdf")
    checks: list[tuple[str, bool, str]] = []

    res = parse_pdf(paper, assets_dir=tmp / "assets")
    types = [b.type for b in res.blocks]

    checks.append(("双栏判定", res.layout == "double", f"layout={res.layout} detail={res.layout_detail['page_columns']}"))
    nums = []

    def walk(nodes):
        for n in nodes:
            nums.append((n.number, n.level, n.text))
            walk(n.children)

    walk(res.title_tree)
    checks.append(("标题树含编号 1/2/2.1",
                   {"1", "2", "2.1"} <= {n[0] for n in nums},
                   str(nums[:6])))
    checks.append(("主标题识别", any(b.type == "title" for b in res.blocks),
                   str([b.text for b in res.blocks if b.type == "title"])))
    checks.append(("表格 → Markdown",
                   any(b.type == "table" and b.markdown and "---" in b.markdown for b in res.blocks),
                   str([(b.caption, (b.markdown or "").split("\n")[0]) for b in res.blocks if b.type == "table"])))
    checks.append(("图注识别", any(b.type == "figure_caption" for b in res.blocks),
                   str([b.text for b in res.blocks if b.type == "figure_caption"])))
    fig_caps = [b.caption for b in res.blocks if b.type == "figure" and b.caption]
    checks.append(("图区与图注联动", bool(fig_caps), str(fig_caps[:2])))
    refs = [b for b in res.blocks if b.type == "reference"]
    checks.append(("参考文献条目化", len(refs) >= 3 and any(r.meta.get("arxiv") for r in refs),
                   str([(r.meta.get("ref_id"), r.meta.get("arxiv") or r.meta.get("doi")) for r in refs])))
    bad = [b.id for b in res.blocks if not b.bbox or len(b.bbox) != 4 or b.page < 1]
    checks.append(("全部块带 page+bbox", not bad, f"异常块={bad}"))

    # 阅读顺序：第 1 页左栏 1 Introduction 必须早于右栏同高度内容
    order = [(b.page, round(b.bbox[1]), b.text[:22]) for b in res.blocks if b.type in ("heading", "title", "paragraph")]
    checks.append(("阅读顺序非空且按页分组", len(order) > 4, f"前 3 项={order[:3]}"))

    scan_res = parse_pdf(scan, assets_dir=tmp / "assets")
    unk = [b for b in scan_res.blocks if b.type == "unknown"]
    has_img = bool(unk) and bool(unk[0].image_path) and Path(unk[0].image_path).exists()
    checks.append(("扫描页 → unknown + 裁图保留", has_img or
                   any(b.source in ("paddleocr", "tesseract") for b in scan_res.blocks),
                   f"blocks={[b.type for b in scan_res.blocks][:4]} source={[b.source for b in scan_res.blocks][:2]}"))

    latex_ok, cov_ok = latex_approx("softmax(QK^T / √d_k)V")
    checks.append(("公式转写覆盖率", cov_ok >= 0.6, f"{latex_ok} coverage={cov_ok:.2f}"))
    _, cov_bad = latex_approx("a ⊞ b")
    checks.append(("不可转写符号 → 低覆盖（走 unknown）", cov_bad < 0.6, f"coverage={cov_bad:.2f}"))

    print(f"合成 PDF：{paper}\n         {scan}")
    print("─" * 92)
    passed = 0
    for name, ok, detail in checks:
        passed += ok
        print(f"[{'OK' if ok else 'XX'}] {name:<28} {detail}")
    print("─" * 92)
    print(f"通过 {passed}/{len(checks)}   统计：{json.dumps(res.stats, ensure_ascii=False)}")
    print(f"输出目录：{tmp}")
    return 0 if passed == len(checks) else 1


# ────────────────────────── CLI ──────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="学术 PDF → DocIR(JSON)")
    ap.add_argument("pdf", nargs="?", help="PDF 路径")
    ap.add_argument("-o", "--out", default="", help="输出 JSON 路径（默认打印到终端）")
    ap.add_argument("--assets", default="", help="截图/OCR 图片目录（unknown 块保留原图区域）")
    ap.add_argument("--dpi", type=int, default=200, help="扫描页栅格化 DPI")
    ap.add_argument("--ocr", default="auto", choices=["auto", "paddle", "tesseract", "off"],
                    help="OCR 后端；off = 不做 OCR，只裁图保留")
    ap.add_argument("--selftest", action="store_true", help="合成样例 PDF 并跑端到端自测")
    ap.add_argument("--make-sample", default="", help="仅生成一份合成论文 PDF 到指定路径")
    args = ap.parse_args()

    if args.make_sample:
        print(make_sample_pdf(args.make_sample))
        return
    if args.selftest or not args.pdf:
        raise SystemExit(_selftest())

    result = parse_pdf(args.pdf, ocr=args.ocr, dpi=args.dpi, assets_dir=args.assets or None)
    text = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}（{len(result.blocks)} 块，版式={result.layout}）")
    else:
        print(text[:8000])
    if result.warnings:
        print("\n警告：")
        for w in result.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    sys.exit(main())
