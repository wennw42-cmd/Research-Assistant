"""M1 文档解析：PDF → DocIR（版式感知、坐标可回溯、扫描件 OCR 兜底）。"""
from app.parsing.docir import Block, ParseResult, TitleNode
from app.parsing.pipeline import parse_pdf

__all__ = ["Block", "ParseResult", "TitleNode", "parse_pdf"]
