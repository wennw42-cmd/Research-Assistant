"""DocIR：版式无关的中间表示与解析结果契约。

所有块都带 page(1 起) + bbox([x0,y0,x1,y1]，PDF 坐标系，左上原点、单位 pt)，
前端据此做原文高亮与双向联动；识别不出来的区域**一律 type="unknown" 并保留截图**，
绝不臆造内容。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

BlockType = Literal[
    "title",        # 文章主标题
    "heading",      # 章节标题（H1-H4）
    "paragraph",    # 正文段落
    "list_item",    # 列表项
    "formula",      # 公式（含 latex 近似）
    "table",        # 表格（含 markdown）
    "figure",       # 图片/图区（含 caption）
    "figure_caption",  # 图注
    "table_caption",   # 表注
    "reference",    # 参考文献条目
    "header",       # 页眉
    "footer",       # 页脚
    "unknown",      # 无法识别（必带 image_path 保留原图区域）
]

TYPES_WITH_IMAGE = {"figure", "unknown"}


class Block(BaseModel):
    """一个版式块。text 永远保留**抽取到的原文**，不做任何补全。"""

    id: str
    type: BlockType
    page: int = 1
    bbox: list[float] = Field(default_factory=list)   # [x0, y0, x1, y1]
    text: str = ""
    # 标题
    level: int | None = None          # H1-H4
    number: str | None = None         # 原始章节编号，如 "2.1"
    # 公式
    latex: str | None = None
    latex_approx: bool = False        # True = 由真实字形启发式转写，非 OCR/模型保证
    # 表格
    markdown: str | None = None
    caption: str | None = None
    # 未知/图片
    image_path: str | None = None
    # 质量与溯源
    confidence: float = 1.0
    source: str = "pymupdf"           # pymupdf | find_tables | heuristic | paddleocr | tesseract | nougat
    meta: dict[str, Any] = Field(default_factory=dict)


class TitleNode(BaseModel):
    """标题层级树节点（保留章节编号）。"""

    text: str
    level: int
    number: str | None = None
    page: int = 1
    bbox: list[float] = Field(default_factory=list)
    children: list["TitleNode"] = Field(default_factory=list)


class ParseResult(BaseModel):
    source: dict[str, Any] = Field(default_factory=dict)
    layout: Literal["single", "double", "multi", "mixed", "empty"] = "single"
    layout_detail: dict[str, Any] = Field(default_factory=dict)
    title_tree: list[TitleNode] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def to_json(self, path: str | None = None) -> str:
        data = self.model_dump(mode="json")
        return json_dumps(data, path)


def json_dumps(data: Any, path: str | None = None) -> str:
    import json

    text = json.dumps(data, ensure_ascii=False, indent=2)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
