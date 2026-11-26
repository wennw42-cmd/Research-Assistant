import type { Pair } from './CompareReader';

/**
 * 演示数据。中英段长刻意不一致，用于观察段内插值如何消除高度差带来的漂移。
 * 公式、代码、变量名按原文保留，渲染时靠 white-space: pre-wrap 保持换行。
 */
export const samplePairs: Pair[] = [
  {
    id: 'p1',
    en: 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism.',
    zh: '目前主流的序列转换模型基于包含编码器与解码器的复杂循环或卷积神经网络。性能最好的模型还通过注意力机制将编码器与解码器连接起来。',
  },
  {
    id: 'p2',
    en: 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.',
    zh: '我们提出一种全新的简单网络架构 Transformer，它完全基于注意力机制，彻底摒弃了循环与卷积结构。在两个机器翻译任务上的实验表明，该模型质量更优，同时更易于并行化，训练所需时间也显著减少。',
  },
  {
    id: 'p3',
    en: 'Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations.',
    zh: '自注意力（有时称为内部注意力）是一种将单个序列中不同位置相互关联、以计算序列表示的注意力机制。自注意力已成功应用于多种任务，包括阅读理解、生成式摘要、文本蕴含以及学习与任务无关的句子表示。',
  },
  {
    id: 'p4',
    en: 'The scaled dot-product attention is defined as\n\n    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V\n\nwhere d_k is the dimension of the keys. The dot products are scaled by 1/sqrt(d_k) to counteract the effect of large magnitudes.',
    zh: '缩放点积注意力定义为\n\n    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V\n\n其中 d_k 为键的维度。点积除以 sqrt(d_k) 进行缩放，以抵消数值量级过大带来的影响。',
  },
  {
    id: 'p5',
    en: 'For a sequence of length n, the computational complexity of self-attention is O(n^2 d), where d is the representation dimension. This quadratic cost becomes prohibitive for long documents, and prior work has therefore explored sparse attention patterns, which reduce the cost at the price of a lower effective capacity.',
    zh: '对于长度为 n 的序列，自注意力的计算复杂度为 O(n^2 d)，其中 d 为表示维度。这一平方级开销在长文档上变得难以承受，因此已有工作探索了稀疏注意力模式；这类模式以较低的有效容量为代价降低了开销。',
  },
  {
    id: 'p6',
    en: 'In all experiments we set d_model = 512 and use h = 8 attention heads. The following configuration is used by default:\n\n    config = {"d_model": 512, "num_heads": 8, "dropout": 0.1}\n\nThe dropout rate is applied to the output of each sub-layer before it is added to the residual connection.',
    zh: '在所有实验中，我们取 d_model = 512，并使用 h = 8 个注意力头。默认使用如下配置：\n\n    config = {"d_model": 512, "num_heads": 8, "dropout": 0.1}\n\n每个子层的输出在进入残差连接之前都会施加 dropout。',
  },
  {
    id: 'p7',
    en: 'We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding with a shared source-target vocabulary of about 37,000 tokens.',
    zh: '我们在标准的 WMT 2014 英德数据集上训练，该数据集包含约 450 万个句对。句子采用字节对编码，源语言与目标语言共享约 37,000 个词的词表。',
  },
  {
    id: 'p8',
    en: 'There are many other ways to make attention more efficient. For example, we can restrict attention to a local window, or combine global and local attention. These variants are orthogonal to the Transformer and may be combined with it in future work.',
    zh: '提升注意力效率还有许多其他途径，例如将注意力限制在局部窗口内，或结合全局注意力与局部注意力。这些变体与 Transformer 相互正交，未来工作可以将其组合使用。',
  },
];
