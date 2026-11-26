"""综述分析示例材料（合成，结构照真实论文，内容为演示编写）。

**不是原文摘录**：这里的摘要与段落是为了跑通 `app.survey` 而按真实论文
的公开事实改写的演示材料（Transformer / BERT / GPT-1~3）。
真实使用时把同结构的 JSON 传给 `analyze(papers)` 或 `--papers papers.json`：

    {"paper_id": "...", "title": "...", "year": 2017, "venue": "...",
     "aliases": ["Transformer"],                 # 供"继承关系"识别被引用的模型名
     "abstract": "...",
     "sections": [{"chunk_id": "...", "heading": "...", "page": 3, "text": "..."}]}

每个 section 的 text 会被逐句切分；分析器只引用切出来的原句，不改写。
"""
from __future__ import annotations

from typing import Any


def build_sample_papers() -> list[dict[str, Any]]:
    return [
        # ─────────────────────────────── Transformer (2017) ───────────────────────────────
        {
            "paper_id": "transformer",
            "title": "Attention Is All You Need",
            "year": 2017,
            "venue": "NeurIPS",
            "authors": ["Vaswani et al."],
            "aliases": ["Transformer", "Attention Is All You Need"],
            "abstract": "本文提出 Transformer，一种完全基于注意力机制、摒弃循环与卷积的序列转换架构。",
            "sections": [
                {"chunk_id": "transformer:method", "heading": "3 Model Architecture", "page": 3,
                 "text": "Transformer 采用编码器-解码器结构：编码器由 6 层相同层堆叠，"
                         "每层含多头自注意力与前馈网络；解码器额外插入编码器-解码器注意力层。"},
                {"chunk_id": "transformer:objective", "heading": "5.1 Training Objective", "page": 5,
                 "text": "训练目标为标准的序列到序列最大似然：给定源句，最大化目标句的条件概率。"},
                {"chunk_id": "transformer:data", "heading": "4 Data", "page": 5,
                 "text": "训练数据为 WMT 2014 英德约 450 万句对，英法约 3600 万句对。"
                         "base 模型约 6500 万参数，large 模型约 2.13 亿参数。"},
                {"chunk_id": "transformer:innovation", "heading": "6 Conclusion", "page": 6,
                 "text": "摒弃循环与卷积、仅用自注意力，使训练可完全并行化，是本文的核心创新。"},
            ],
        },
        # ────────────────────────────────── BERT (2018) ──────────────────────────────────
        {
            "paper_id": "bert",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "year": 2018,
            "venue": "NAACL",
            "authors": ["Devlin et al."],
            "aliases": ["BERT"],
            "abstract": "BERT 通过联合调节左右上下文，从未标注文本中预训练深层双向 Transformer 表示。",
            "sections": [
                {"chunk_id": "bert:architecture", "heading": "3 BERT", "page": 3,
                 "text": "BERT 仅使用 Transformer 的编码器堆叠，不使用解码器，因此是双向编码器结构；"
                         "base 模型 1.1 亿参数，large 模型 3.4 亿参数。"},
                {"chunk_id": "bert:objective", "heading": "3.1 Pre-training Tasks", "page": 4,
                 "text": "预训练使用两个目标：掩码语言模型（MLM）随机掩盖部分 token 并预测，"
                         "以及下一句预测（NSP）。"},
                {"chunk_id": "bert:data", "heading": "3.2 Pre-training Data", "page": 4,
                 "text": "预训练语料为 BooksCorpus（约 8 亿词）与英文维基百科（约 25 亿词），合计约 33 亿词。"},
                {"chunk_id": "bert:lineage", "heading": "3 BERT", "page": 3,
                 "text": "BERT 沿用 Transformer 的编码器结构，但去掉解码器并改为双向预训练。"},
                {"chunk_id": "bert:innovation", "heading": "6 Conclusion", "page": 12,
                 "text": "首次将双向 Transformer 预训练用于通用语言理解，在 11 项 NLP 任务上取得当时最优。"},
            ],
        },
        # ───────────────────────────────── GPT-1 (2018) ─────────────────────────────────
        {
            "paper_id": "gpt1",
            "title": "Improving Language Understanding by Generative Pre-Training",
            "year": 2018,
            "venue": "OpenAI Tech Report",
            "authors": ["Radford et al."],
            "aliases": ["GPT", "GPT-1"],
            "abstract": "本文提出生成式预训练加判别式微调的框架，先在无标注语料上预训练语言模型，再迁移到下游任务。",
            "sections": [
                {"chunk_id": "gpt1:architecture", "heading": "3 Model", "page": 3,
                 "text": "GPT 仅使用 Transformer 的解码器（decoder-only），采用 12 层、12 头、768 维的结构，"
                         "共 1.17 亿参数。"},
                {"chunk_id": "gpt1:objective", "heading": "3.1 Unsupervised Pre-training", "page": 3,
                 "text": "训练目标为标准自回归语言模型：根据前文预测下一个 token。"},
                {"chunk_id": "gpt1:data", "heading": "4 Experiments", "page": 4,
                 "text": "预训练语料为 BooksCorpus，约 7000 本未出版书籍、约 8 亿词。"},
                {"chunk_id": "gpt1:lineage", "heading": "2 Related Work", "page": 2,
                 "text": "GPT 基于 Transformer 架构，但只保留解码器，并将训练目标改为单向自回归语言建模。"},
                {"chunk_id": "gpt1:innovation", "heading": "6 Conclusion", "page": 7,
                 "text": "提出“预训练 + 微调”的两阶段范式，证明通用预训练表示可迁移到多种下游任务。"},
            ],
        },
        # ───────────────────────────────── GPT-2 (2019) ─────────────────────────────────
        {
            "paper_id": "gpt2",
            "title": "Language Models are Unsupervised Multitask Learners",
            "year": 2019,
            "venue": "OpenAI Tech Report",
            "authors": ["Radford et al."],
            "aliases": ["GPT-2"],
            "abstract": "本文提出 GPT-2，将模型与语料同时放大，并展示无监督多任务学习的潜力。",
            "sections": [
                {"chunk_id": "gpt2:architecture", "heading": "2 Approach", "page": 3,
                 "text": "GPT-2 沿用 GPT 的解码器结构，层数增至 48 层，参数量达 15 亿。"},
                {"chunk_id": "gpt2:objective", "heading": "2 Approach", "page": 3,
                 "text": "训练目标仍为自回归语言模型，不针对任何下游任务做专门训练，"
                         "下游任务以零样本方式完成。"},
                {"chunk_id": "gpt2:data", "heading": "2.1 Training Dataset", "page": 4,
                 "text": "训练语料 WebText 约 40GB、800 万个网页，来自 Reddit 外链。"},
                {"chunk_id": "gpt2:lineage", "heading": "2 Approach", "page": 3,
                 "text": "GPT-2 直接沿用 GPT 的 decoder-only 架构与自回归目标。"},
                {"chunk_id": "gpt2:innovation", "heading": "5 Discussion", "page": 8,
                 "text": "证明仅靠大规模语言建模即可在零样本设定下完成多任务，是范式上的关键转折。"},
            ],
        },
        # ───────────────────────────────── GPT-3 (2020) ─────────────────────────────────
        {
            "paper_id": "gpt3",
            "title": "Language Models are Few-Shot Learners",
            "year": 2020,
            "venue": "NeurIPS",
            "authors": ["Brown et al."],
            "aliases": ["GPT-3"],
            "abstract": "本文提出 GPT-3，以 1750 亿参数验证上下文学习（in-context learning）的能力。",
            "sections": [
                {"chunk_id": "gpt3:architecture", "heading": "2 Approach", "page": 4,
                 "text": "GPT-3 延续 decoder-only 架构，共 96 层，参数量 1750 亿。"},
                {"chunk_id": "gpt3:objective", "heading": "1 Introduction", "page": 3,
                 "text": "不做梯度更新，仅在提示中给出少量示例，模型通过上下文学习完成任务；"
                         "训练目标仍为自回归语言模型。"},
                {"chunk_id": "gpt3:data", "heading": "2.1 Training Dataset", "page": 4,
                 "text": "训练数据混合 Common Crawl、WebText2、Books1、Books2 与维基百科，"
                         "合计约 3000 亿 token。"},
                {"chunk_id": "gpt3:lineage", "heading": "2 Approach", "page": 4,
                 "text": "GPT-3 沿用 GPT-2 的架构与自回归训练目标，规模扩大约两个数量级。"},
                {"chunk_id": "gpt3:innovation", "heading": "1 Introduction", "page": 3,
                 "text": "提出上下文学习（few-shot / zero-shot），把“微调”替换为“提示”，"
                         "是本文的关键转折。"},
            ],
        },
    ]
