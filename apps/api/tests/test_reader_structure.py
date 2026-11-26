"""标题/参考文献结构的小样本回归：防止 PDF 折行把结构切碎。

用真实 PDF 里出现的两种版式特征构造 Lines/Blocks，不依赖网络或 OCR：
- "1" 与 "Introduction" 同一基线、被拆成两行 → 应合并成一个 H1；
- 参考文献悬挂缩进的续行（含 "(8):1735–1780" 这类卷期号）→ 应并回上一条。
"""
from __future__ import annotations

from app.parsing.docir import Block
from app.parsing.headings import classify_headings
from app.parsing.layout import Line, PageGeom
from app.parsing.pipeline import _extract_references


def _line(text: str, *, y: float, x0: float = 108.0, size: float = 10.0,
          bold: bool = False, block: int = 0, no: int = 0) -> Line:
    return Line(page=1, block_no=block, line_no=no, bbox=(x0, y, 500.0, y + 9),
                text=text, size=size, bold=bold)


def test_number_and_title_on_same_baseline_merge_and_table_rows_ignored() -> None:
    geom = PageGeom(page=1, width=595, height=842, lines=[
        _line("1", y=100, size=12, bold=True, block=1, no=0),
        _line("Introduction", y=100, x0=125.9, size=12, bold=True, block=1, no=1),
        _line("3.1", y=300, bold=True, block=2, no=0),
        _line("Encoder and Decoder Stacks", y=300, x0=129.0, block=2, no=1),
        _line("23.75 ByteNet [18]", y=500, size=9, block=3, no=0),
        _line("41.29 ConvS2S Ensemble [9]", y=510, size=9, bold=True, block=4, no=0),
    ])
    heads = classify_headings([geom], body_size=10.0)
    assert [(h.text, h.level) for h in heads] == [
        ("1 Introduction", 1),
        ("3.1 Encoder and Decoder Stacks", 2),
    ]


def test_reference_continuations_merge_and_section_ends_at_next_heading() -> None:
    blocks = [
        Block(id="h1", type="heading", page=10, level=1, text="References"),
        Block(id="r1", type="reference", page=10, bbox=[113.0, 100, 504, 109],
              text="[1] Foo Bar. A paper."),
        Block(id="r1c", type="reference", page=10, bbox=[129.6, 110, 264, 119],
              text="continued line."),
        Block(id="r2", type="reference", page=10, bbox=[113.0, 120, 504, 129],
              text="[2] Baz. Another paper."),
        Block(id="r2c", type="reference", page=10, bbox=[129.6, 130, 300, 139],
              text="(2):313-330, 1993."),
        Block(id="h2", type="heading", page=13, level=1, text="Appendix"),
        Block(id="p1", type="paragraph", page=13, text="appendix body text"),
    ]
    warnings: list[str] = []
    refs = _extract_references(blocks, warnings)
    assert len(refs) == 2
    assert refs[0].text == "[1] Foo Bar. A paper. continued line."
    assert refs[1].text == "[2] Baz. Another paper. (2):313-330, 1993."
    assert all("appendix" not in ref.text for ref in refs)
    assert not warnings
