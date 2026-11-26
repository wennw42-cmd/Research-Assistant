"""切块参数与 token 估算。

参数不是拍脑袋定的，三条约束决定了取值：
- **下界**：一个 chunk 必须能自洽回答一个问题 → 中文 120 token 以下经常丢主语；
- **上界**：top-k=6 时总上下文要压在 4k~8k → 单块 512 token 是常见甜蜜点；
- **重叠**：必须大于"一个完整句子 + 其指代主语"的长度，否则跨块的指代链会断。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict

_CJK_RE = re.compile(r"[㐀-鿿぀-ヿ가-힯]")
_TIKTOKEN = None
try:  # 有 tiktoken 就用品粒更准的估算，没有也能跑
    import tiktoken as _TIKTOKEN  # type: ignore
except Exception:  # noqa: BLE001
    _TIKTOKEN = None


@dataclass(frozen=True)
class ChunkParams:
    # ── L2 句子级（精确引用 / 高分辨率召回）──
    sent_target: int = 128
    sent_overlap: int = 32
    sent_max: int = 192
    # ── L1 段落级（主力检索粒度）──
    para_target: int = 512
    para_overlap: int = 64
    para_max: int = 768
    # ── L0 章节层（章节整体语义，用"抽取式 lead"承载，不拼接全文）──
    section_lead: int = 512
    # ── T0 摘要层（每章一条，只由各节首句抽取，不生成式摘要）──
    summary_lead: int = 256
    # ── 行为开关 ──
    prefix_path: bool = True          # chunk 文本前置面包屑，提升无上下文召回的可读性
    keep_bbox: bool = True            # 保留首块 bbox 供前端跳转
    heading_as_prefix: bool = False   # True=把标题原文并入正文（会重复计入每个块）

    def table_rows(self) -> list[dict[str, str]]:
        return [
            {"层级": "L0 章节层", "单位": "整节（lead 截断）", "目标": str(self.section_lead),
             "重叠": "—", "上限": str(self.section_lead), "用途": "章节整体语义、定位候选章"},
            {"层级": "L1 段落级", "单位": "段落块", "目标": str(self.para_target),
             "重叠": str(self.para_overlap), "上限": str(self.para_max), "用途": "主力检索粒度"},
            {"层级": "L2 句子级", "单位": "句子", "目标": str(self.sent_target),
             "重叠": str(self.sent_overlap), "上限": str(self.sent_max), "用途": "精确引用与细节展开"},
            {"层级": "T0 摘要层", "单位": "章（各节首句）", "目标": str(self.summary_lead),
             "重叠": "—", "上限": str(self.summary_lead), "用途": "541 页级全局导航"},
        ]


PARAM_RATIONALE = [
    ("L2 128/32", "中文一句约 20-40 token；128 覆盖 3-5 句，32 的重叠保住一个完整句子 + 主语，跨块指代不断链"),
    ("L1 512/64", "top-k=6 时 ~3k token，留出答案空间；64 ≈ 1-2 句，足以衔接上一块末尾的论点"),
    ("L0 lead 512", "章节层不拼全文（541 页教材单章可达数万 token），只取开头 lead 承载章节主题"),
    ("不跨节切分", "跨节拼接会让 title_path 失真——溯源元数据是这套 RAG 的一等公民"),
]


def count_tokens(text: str) -> int:
    """token 估算：优先 tiktoken(cl100k)，否则按 CJK 1 字≈1 token、拉丁≈3.6 字符/token。"""
    if not text:
        return 0
    if _TIKTOKEN is not None:
        try:
            enc = _TIKTOKEN.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:  # noqa: BLE001
            pass
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    return cjk + int(math.ceil(rest / 3.6))


# ────────────────────────── 文本切分 ──────────────────────────
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])|\n{1,}")


def split_sentences(text: str, min_len: int = 2) -> list[str]:
    """中英混排句子切分：以句末标点为准，遇到换行也断（版式块本身是段落）。"""
    parts: list[str] = []
    for seg in _SENT_SPLIT.split(text or ""):
        s = (seg or "").strip()
        if len(s) >= min_len:
            parts.append(s)
    return parts


def hard_split(text: str, max_tokens: int) -> list[str]:
    """超长单元的兜底硬切：按句子切，句仍超长则按字符均分（保证不超上限）。"""
    out: list[str] = []
    for s in split_sentences(text):
        if count_tokens(s) <= max_tokens:
            out.append(s)
            continue
        step = max(16, int(max_tokens * 2))  # 中文按字符近似回推
        for i in range(0, len(s), step):
            out.append(s[i:i + step])
    return out or ([text] if text else [])
