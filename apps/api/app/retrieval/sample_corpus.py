"""自测/演示用学术语料（合成，但结构照真实论文写）。

刻意埋了三个"纯向量检索会翻车"的坑，用来验证改写 + 过滤 + 重排到底有没有起作用：

1. **Related Work 干扰项**：`llm_related` 里写满了 "batch size 256 / 学习率 5e-4 / V100"，
   词面上比正确答案更像问题——没有小节过滤，它一定排在实验设置前面；
2. **_method 段复述超参但不给值**：`llm_method` 提到"余弦学习率调度"，
   却没有具体数值——没有实体/数值特征，重排无从区分；
3. **跨论文同型号硬件**：`asr_hardware` 也是多卡训练（4×V100），
   没有 `n_gpus` 范围过滤就会被"8 卡 A100"的问题召回来。
"""
from __future__ import annotations

from app.retrieval.types import Chunk

P1 = "doc_vit"
P2 = "doc_llm_ft"
P3 = "doc_asr"

TITLES = {
    P1: "Vision Transformer on Small Datasets",
    P2: "Efficient Fine-Tuning of Large Language Models",
    P3: "Streaming ASR with Conformers",
}


def _mk(cid: str, doc: str, section: str, page: int, text: str, *, stype: str,
        hyper: list[str] | None = None, hw: list[str] | None = None,
        n_gpus: int | None = None, has_table: bool = False, year: int = 2023,
        venue: str = "ACL") -> Chunk:
    meta: dict = {"section_type": stype, "year": year, "venue": venue, "has_table": has_table}
    if hyper is not None:
        meta["hyperparams"] = hyper
    if hw is not None:
        meta["hardware"] = hw
    if n_gpus is not None:
        meta["n_gpus"] = n_gpus
    return Chunk(id=cid, text=text, doc_id=doc, title=TITLES[doc],
                 section_path=f"{TITLES[doc]} > {section}",
                 page_start=page, page_end=page, block_ids=[cid.split(":")[-1]], metadata=meta)


def build_sample_chunks() -> list[Chunk]:
    out: list[Chunk] = []

    # ───────────── doc_vit ─────────────
    out.append(_mk("vit:abstract", P1, "Abstract", 1,
                   "我们研究在数据受限场景下训练 Vision Transformer 的问题，提出一种分块嵌入与强正则的组合方案。",
                   stype="abstract", year=2021, venue="CVPR"))
    out.append(_mk("vit:method", P1, "Method", 3,
                   "模型采用 12 层 Transformer 编码器，patch size 为 16，隐藏维度 768，注意力头数 12。",
                   stype="method", year=2021, venue="CVPR"))
    out.append(_mk("vit:related", P1, "Related Work", 6,
                   "此前的 ViT 变体通常需要 JFT-300M 级别的预训练数据，其训练设置普遍采用 batch size 4096 与学习率 1e-3。",
                   stype="related_work", year=2021, venue="CVPR"))
    out.append(_mk("vit:setup", P1, "Experimental Setup", 8,
                   "实验在单张 RTX 3090（24GB）上完成：AdamW 优化器，学习率 3e-4，batch size 64，"
                   "共训练 100 个 epoch，采用余弦学习率调度与 0.05 的权重衰减。",
                   stype="experimental_setup", hyper=["optimizer", "learning_rate", "batch_size",
                                                      "epochs", "scheduler", "weight_decay"],
                   hw=["RTX3090", "24GB"], n_gpus=1, year=2021, venue="CVPR"))
    out.append(_mk("vit:results", P1, "Results", 10,
                   "在 CIFAR-100 上取得 81.2% 的 top-1 准确率，比同等规模的 ResNet 基线高 2.7 个点。",
                   stype="results", year=2021, venue="CVPR"))

    # ───────────── doc_llm_ft ─────────────
    out.append(_mk("llm:abstract", P2, "Abstract", 1,
                   "本文提出一种参数高效的微调方法，在仅训练 0.5% 参数的条件下接近全量微调的效果，并系统报告训练配置。",
                   stype="abstract"))
    out.append(_mk("llm:method", P2, "Method", 3,
                   "我们在注意力投影矩阵上插入低秩适配器，并使用余弦学习率调度；"
                   "优化器与具体超参数取值见实验设置一节。",
                   stype="method", hyper=["scheduler"]))
    out.append(_mk("llm:related", P2, "Related Work", 6,
                   "已有工作通常在 V100 集群上以 batch size 256、学习率 5e-4 训练，"
                   "并采用 8 卡数据并行；本文的训练配置与之存在明显差异。",
                   stype="related_work", hw=["V100"]))
    out.append(_mk("llm:setup", P2, "Experimental Setup", 9,
                   "训练在 8 张 A100（80GB）上进行：AdamW 优化器，学习率 1e-3，batch size 32，"
                   "warmup 2000 步，共训练 3 个 epoch，混合精度 bf16，梯度裁剪 1.0。",
                   stype="experimental_setup",
                   hyper=["optimizer", "learning_rate", "batch_size", "warmup", "epochs"],
                   hw=["A100", "80GB", "bf16"], n_gpus=8, has_table=True))
    out.append(_mk("llm:hardware", P2, "Hardware Environment", 10,
                   "硬件环境：8×NVIDIA A100 80GB，节点内 NVLink 互联，节点间 200Gbps InfiniBand；"
                   "启用 bf16 混合精度后单卡峰值显存占用 62GB，训练吞吐约 11800 tokens/s。",
                   stype="hardware", hw=["A100", "80GB", "bf16"], n_gpus=8))
    out.append(_mk("llm:results", P2, "Results", 12,
                   "在五个数据集上平均取得 89.4 分，与全量微调相差 0.6 分，而可训练参数量减少 99.5%。",
                   stype="results"))
    out.append(_mk("llm:ablation", P2, "Ablation", 13,
                   "消融实验显示：去掉 warmup 会使最终成绩下降 1.8 分；将 batch size 从 32 增至 128 收益不显著。",
                   stype="ablation", hyper=["warmup", "batch_size"]))

    # ───────────── doc_asr ─────────────
    out.append(_mk("asr:abstract", P3, "Abstract", 1,
                   "我们提出面向流式语音识别的 Conformer 结构，在保持低延迟的同时降低词错误率。",
                   stype="abstract", year=2022, venue="ICASSP"))
    out.append(_mk("asr:method", P3, "Method", 4,
                   "编码器由 18 层 Conformer 块组成，卷积核大小 31，采用相对位置编码。",
                   stype="method", year=2022, venue="ICASSP"))
    out.append(_mk("asr:setup", P3, "Experimental Setup", 8,
                   "模型在 4 张 V100（32GB）上训练：学习率 2e-3，batch size 16，warmup 10000 步，共 50 个 epoch。",
                   stype="experimental_setup",
                   hyper=["learning_rate", "batch_size", "warmup", "epochs"],
                   hw=["V100", "32GB"], n_gpus=4, year=2022, venue="ICASSP"))
    out.append(_mk("asr:hardware", P3, "Hardware Environment", 9,
                   "硬件环境为单机 4×V100 32GB，fp16 混合精度，单卡显存占用约 28GB，"
                   "实时率 RTF 为 0.21。",
                   stype="hardware", hw=["V100", "32GB", "fp16"], n_gpus=4,
                   year=2022, venue="ICASSP"))
    out.append(_mk("asr:results", P3, "Results", 11,
                   "在 LibriSpeech test-clean 上词错误率为 3.9%，较基线降低 0.7 个百分点。",
                   stype="results", year=2022, venue="ICASSP"))
    out.append(_mk("asr:conclusion", P3, "Conclusion", 13,
                   "本文证明在有限硬件条件下亦可通过结构改进获得竞争力，后续将探索更大规模预训练。",
                   stype="conclusion", year=2022, venue="ICASSP"))

    return out
