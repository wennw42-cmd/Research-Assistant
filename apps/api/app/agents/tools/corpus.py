"""离线语料库（DocIR 的内存替身）。

真实系统中这里应替换为 PostgreSQL(pgvector) + MinIO，
由 M1 解析流水线写入；此处只为让**整条 Agent 链路零外部依赖可跑**。

数据结构对齐 DocIR 的最小子集：
    doc -> blocks[{block_id, page, bbox, section, text}]
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from app.schemas import EvidenceCard

# ────────────────────────────── 文档与语义块 ──────────────────────────────
DOCS: dict[str, dict[str, Any]] = {
    "doc_transformer": {
        "title": "Attention Is All You Need",
        "year": 2017,
        "venue": "NeurIPS",
        "pages": 15,
        "source": "native_pdf",
    },
    "doc_bert": {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "year": 2018,
        "venue": "NAACL",
        "pages": 16,
        "source": "native_pdf",
    },
    "doc_survey_2020": {
        "title": "Efficient Transformers: A Survey",
        "year": 2020,
        "venue": "arXiv",
        "pages": 22,
        "source": "native_pdf",
    },
    "doc_scan_sample": {
        "title": "A Scanned Paper Sample (OCR)",
        "year": 2019,
        "venue": "unknown",
        "pages": 12,
        "source": "scanned",
    },
}

BLOCKS: list[dict[str, Any]] = [
    # ── Transformer ──
    dict(doc_id="doc_transformer", block_id="b000", page=1, bbox=[72, 610, 523, 690],
         section="Abstract",
         text="我们提出 Transformer，一种完全基于注意力机制、摒弃循环与卷积的网络架构。"
              "在 WMT 2014 英德翻译任务上达到 28.4 BLEU，且训练可完全并行化。"),
    dict(doc_id="doc_transformer", block_id="b001", page=3, bbox=[72, 400, 523, 520],
         section="Architecture",
         text="Transformer 采用编码器-解码器结构，编码器由 6 层相同层堆叠，每层含多头自注意力与前馈网络；"
              "解码器额外插入编码器-解码器注意力层。"),
    dict(doc_id="doc_transformer", block_id="b002", page=4, bbox=[72, 220, 523, 300],
         section="Positional Encoding",
         text="由于模型不含循环与卷积，必须注入位置信息，论文使用正弦位置编码（positional encoding）。"),
    dict(doc_id="doc_transformer", block_id="b003", page=5, bbox=[72, 150, 523, 240],
         section="Complexity",
         text="自注意力的计算复杂度为 O(n^2 · d)，其中 n 为序列长度、d 为表示维度；"
              "相比循环层，其每层的顺序操作数仅为 O(1)，因而更易并行。"),
    dict(doc_id="doc_transformer", block_id="b004", page=8, bbox=[72, 300, 523, 380],
         section="Results",
         text="在 WMT 2014 英德与英法任务上分别取得 28.4 与 41.8 BLEU，训练成本显著低于此前最好的循环模型。"),

    # ── BERT ──
    dict(doc_id="doc_bert", block_id="b000", page=1, bbox=[72, 600, 523, 680], section="Abstract",
         text="BERT 通过联合调节左右上下文，从未标注文本中预训练深层的双向 Transformer 表示，"
              "在 11 项 NLP 任务上取得当时最优结果。"),
    dict(doc_id="doc_bert", block_id="b001", page=3, bbox=[72, 380, 523, 500], section="Pre-training",
         text="BERT 使用两个预训练目标：掩码语言模型（Masked LM）随机掩盖部分 token 并预测之，"
              "以及下一句预测（NSP）。"),
    dict(doc_id="doc_bert", block_id="b002", page=4, bbox=[72, 240, 523, 330], section="Architecture",
         text="BERT 仅使用 Transformer 的编码器堆叠，不使用解码器；"
              "区别于 GPT 的单向约束与 ELMo 的浅层拼接，BERT 的双向性是深层的。"),
    dict(doc_id="doc_bert", block_id="b003", page=5, bbox=[72, 180, 523, 270], section="Fine-tuning",
         text="下游只需在预训练表示上追加一层输出层即可微调，"
              "形成预训练-微调（pre-training and fine-tuning）的统一范式。"),
    dict(doc_id="doc_bert", block_id="b004", page=7, bbox=[72, 300, 523, 380], section="Results",
         text="BERT-large 在 GLUE 上取得 80.5 分，在 SQuAD v1.1 上取得 90.9 的 F1，均刷新当时纪录。"),

    # ── Survey ──
    dict(doc_id="doc_survey_2020", block_id="b000", page=1, bbox=[72, 590, 523, 670],
         section="Introduction",
         text="自 Vaswani 等人提出 Transformer 后，大量工作致力于降低其 O(n^2) 的注意力开销，本文系统梳理此类高效变体。"),
    dict(doc_id="doc_survey_2020", block_id="b001", page=6, bbox=[72, 300, 523, 420],
         section="Taxonomy",
         text="高效 Transformer 可分为固定模式（如 Longformer 的滑窗）、可学习模式（Reformer）、"
              "低秩近似（Linformer）、核方法与循环机制等类别。"),
    dict(doc_id="doc_survey_2020", block_id="b002", page=11, bbox=[72, 200, 523, 300],
         section="Comparison",
         text="在长序列场景下，Linformer 与 Longformer 将复杂度降至 O(n) 或 O(n log n)，"
              "但在短任务上精度略逊于标准 Transformer。"),
    dict(doc_id="doc_survey_2020", block_id="b003", page=18, bbox=[72, 160, 523, 240],
         section="Conclusion",
         text="高效注意力已成为长文本建模的主流方向，但仍缺少统一的评测基准。"),

    # ── 扫描件 OCR 样例（刻意保留一处低置信标记）──
    dict(doc_id="doc_scan_sample", block_id="b000", page=2, bbox=[80, 520, 520, 600],
         section="Method",
         text="本文方法在低质量扫描文档上仍保持稳定【OCR低置信】，建议人工校对该段落。"),
]

# ────────────────────────────── 版本链（v1 / v7 去重用例）──────────────────────────────
VERSION_CHAIN: dict[str, list[str]] = {
    "arxiv:1706.03762": [f"arxiv:1706.03762v{i}" for i in range(1, 8)],
}

VERSION_ABSTRACTS: dict[str, str] = {
    "arxiv:1706.03762v1": "提出仅用注意力的序列转换模型 Transformer，在机器翻译上取得新 SOTA。",
    "arxiv:1706.03762v7": "提出仅用注意力的序列转换模型 Transformer，在机器翻译上取得新 SOTA，并补充了消融实验与更多基线对比。",
}

# 版本链同样是知识库的一部分：把摘要注册为**可引用块**。
# 否则去重产出的引文在 find_quote 里查无此句，抗幻觉第一道闸会误报"引文不忠实"。
DOCS.update({
    vid: {"title": "Attention Is All You Need", "year": 2017, "venue": "arXiv",
          "pages": 15, "source": "native_pdf", "version": vid}
    for vid in VERSION_ABSTRACTS
})
BLOCKS.extend(
    dict(doc_id=vid, block_id="abstract", page=1, bbox=[72, 600, 523, 680],
         section="Abstract", text=txt)
    for vid, txt in VERSION_ABSTRACTS.items()
)

# ────────────────────────────── 引用图 ──────────────────────────────
CITATION_EDGES: list[tuple[str, str, str]] = [
    ("doc_bert", "doc_transformer", "cites"),
    ("doc_survey_2020", "doc_transformer", "cites"),
    ("doc_survey_2020", "doc_bert", "cites"),
    ("doc_scan_sample", "doc_bert", "cites"),
]

# ────────────────────────────── 外部文献库（检索/追踪替身）──────────────────────────────
PAPER_DB: list[dict[str, Any]] = [
    {"title": "GPT-3: Language Models are Few-Shot Learners", "year": 2020, "doi": "10.5555/x1", "citations": 12000},
    {"title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "year": 2019, "doi": "10.5555/x2", "citations": 8000},
    {"title": "Longformer: The Long-Document Transformer", "year": 2020, "doi": "10.5555/x3", "citations": 4500},
    {"title": "Linformer: Self-Attention with Linear Complexity", "year": 2020, "doi": "10.5555/x4", "citations": 2100},
]


# ────────────────────────────── 检索内核 ──────────────────────────────
def _terms(text: str) -> list[str]:
    """中英混排分词：ASCII 词 + 中文 bigram。轻量但对本场景足够有效。"""
    text = (text or "").lower()
    ascii_words = re.findall(r"[a-z0-9][a-z0-9\-_\.]+", text)
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    grams: list[str] = []
    for seg in cjk:
        grams += [seg[i:i + 2] for i in range(len(seg) - 1)] or [seg]
    return ascii_words + grams


_IDF: dict[str, float] = {}
_TERM_FREQ: list[Counter[str]] = []
_BLOCK_INDEX: list[dict[str, Any]] = []


def _ensure_index() -> None:
    if _BLOCK_INDEX:
        return
    df: Counter[str] = Counter()
    for b in BLOCKS:
        tf = Counter(_terms(b["text"]))
        _TERM_FREQ.append(tf)
        df.update(tf.keys())
        _BLOCK_INDEX.append(b)
    n = len(BLOCKS)
    _IDF.update({t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()})


def hybrid_search(query: str, top_k: int = 6, doc_scope: list[str] | None = None) -> list[EvidenceCard]:
    """混合检索 = BM25 稀疏 + 伪稠密（字符 n-gram TF-IDF 余弦）→ 融合 → mock rerank。

    真实系统中：稀疏用 pg 全文/pg_trgm，稠密用 BGE-M3，融合后用 bge-reranker-v2-m3 精排。
    """
    _ensure_index()
    q_terms = Counter(_terms(query))
    if not q_terms:
        return []
    q_norm = math.sqrt(sum((v * _IDF.get(t, 1.0)) ** 2 for t, v in q_terms.items())) or 1.0

    scored: list[tuple[float, dict[str, Any]]] = []
    for tf, blk in zip(_TERM_FREQ, _BLOCK_INDEX):
        if doc_scope and blk["doc_id"] not in doc_scope:
            continue
        common = set(q_terms) & set(tf)
        if not common:
            continue
        # 稠密部分（TF-IDF 余弦近似）
        dense = sum(q_terms[t] * tf[t] * (_IDF.get(t, 1.0) ** 2) for t in common) / q_norm
        # 稀疏部分（简化 BM25）
        bm25 = sum(
            _IDF.get(t, 1.0) * (tf[t] * 1.5) / (tf[t] + 1.2) * (q_terms[t] * 1.5) / (q_terms[t] + 1.2)
            for t in common
        )
        # 标题命中加权：很重要的工程技巧，避免正文噪声压过主题文献
        bonus = 0.35 if any(k.lower() in DOCS[blk["doc_id"]]["title"].lower() for k in re.findall(r"[A-Za-z]{3,}", query)) else 0.0
        scored.append((0.55 * dense + 0.45 * bm25 + bonus, blk))

    scored.sort(key=lambda x: -x[0])
    top = scored[: max(top_k * 2, top_k)]
    top = _mock_rerank(query, top)[:top_k]
    return [
        EvidenceCard(
            doc_id=b["doc_id"],
            block_id=b["block_id"],
            quote=b["text"][:180],
            page=b["page"],
            bbox=b["bbox"],
            score=round(float(s), 4),
            meta={"section": b["section"], "title": DOCS[b["doc_id"]]["title"]},
        )
        for s, b in top
    ]


def _mock_rerank(query: str, pairs: list[tuple[float, dict[str, Any]]]) -> list[tuple[float, dict[str, Any]]]:
    """精排替身：阅读 query 中的实体，命中并保持句读通顺的块提权。"""
    ents = re.findall(r"[A-Za-z]{3,}", query) + re.findall(r"[\u4e00-\u9fff]{2,4}", query)
    out = []
    for s, b in pairs:
        hit = sum(1 for e in ents if e and e in b["text"])
        out.append((round(s + 0.12 * hit, 4), b))
    out.sort(key=lambda x: -x[0])
    return out


def get_block(block_id: str, doc_id: str | None = None) -> dict[str, Any] | None:
    """支持两种写法：纯 block_id（需传 doc_id）或 "doc_id#block_id"。

    block_id 在文档内唯一；跨文档定位必须由 (doc_id, block_id) 二元组完成，
    这样 cite_key = "doc#block" 才不会出现 doc#doc#block 的重复前缀。
    """
    if "#" in block_id:
        doc_id, block_id = block_id.split("#", 1)
    for b in BLOCKS:
        if b["block_id"] == block_id and (doc_id is None or b["doc_id"] == doc_id):
            return b
    return None


def find_quote(quote: str) -> dict[str, Any] | None:
    """字面存在性校验 —— 抗幻觉的第一道闸：模型编的引文一定查无此句。"""
    q = (quote or "").strip()
    if len(q) < 8:
        return None
    for b in BLOCKS:
        if q in b["text"]:
            return b
    # 允许轻微截断差异：取前 24 字做前缀匹配
    prefix = q[:24]
    for b in BLOCKS:
        if prefix in b["text"]:
            return b
    return None
