"""查询理解：意图识别 → 查询改写 → 元数据过滤条件。

针对"细粒度超参数 / 硬件环境"这类问题，纯向量检索有三个典型失败：

1. **问法与原文措辞不同**：用户问"用了几张卡"，原文写 "We train on 8×A100 (80GB)"；
2. **答案藏在特定小节**：超参/硬件几乎总在 Experimental Setup / Implementation Details / Appendix，
   而这些段落的"通用论述感"弱，向量上离问题句更远；
3. **干扰项措辞更像问题**：Related Work 里复述别人的超参，向量上反而更近。

对策分别是：查询改写（补出原文的规范说法）、元数据过滤（把候选压进正确的小节类型）、
重排阶段的实体/数值硬信号（见 rerank.py）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.retrieval.rerank import extract_numbers
from app.retrieval.types import Filter

# ── 超参数词典：用户说法 → 元数据里的规范字段名（与入库时 hyperparams 字段对齐）──
_HYPER_PATTERNS: list[tuple[str, str]] = [
    (r"learning\s*rate|lr\b|学习率", "learning_rate"),
    (r"batch\s*size|\bbs\b|批[次大]?[大小]|批尺寸", "batch_size"),
    (r"epochs?\b|轮数|训练轮次", "epochs"),
    (r"optimiz[a-z]*|优化器|adamw?\b|sgd\b", "optimizer"),
    (r"weight\s*decay|权重衰减|l2\s*正则", "weight_decay"),
    (r"dropout|丢弃率", "dropout"),
    (r"warmup|预热", "warmup"),
    (r"scheduler|调度|学习率衰减|cosine\s*anneal", "scheduler"),
    (r"seed\b|随机种子", "seed"),
    (r"超参|超参数|hyper-?parameters?", "__any__"),
]

# ── 硬件词典 ──
_GPU_MODELS = r"a100|h100|h800|a800|v100|rtx\s?\d{4}|3090|4090|4090d|\bt4\b|tpu|昇腾|910b"
_HW_PATTERNS: list[tuple[str, str]] = [
    (_GPU_MODELS, "gpu_model"),
    (r"\d+\s?gb|显存|vram|显存占用|memory\s*footprint", "gpu_memory"),
    (r"fp16|bf16|int8|混合精度|precision|量化", "precision"),
    (r"多卡|单机多卡|分布式|ddp|deepspeed|数据并行|张量并行", "distributed"),
]

# 小节类型先验：答案最可能落在哪儿（与入库时 section_type 取值对齐）
_SECTIONS_HYPER = ["experimental_setup", "hyperparameters", "implementation",
                   "training_details", "appendix", "reproducibility"]
_SECTIONS_HW = ["experimental_setup", "hardware", "implementation", "training_details",
                "appendix", "reproducibility"]
_SECTIONS_RESULT = ["results", "ablation", "experiments", "analysis"]

# 改写时补进去的"原文规范说法"
_CANON_TERMS = {
    "hyperparameter": "experimental setup hyperparameters training configuration 实验设置 超参数 训练配置",
    "hardware": "hardware environment GPU memory 硬件环境 显卡 显存 训练设备",
    "result": "experimental results comparison 实验结果 对比",
}


@dataclass
class QueryPlan:
    original: str
    intent: str = "general"                       # hyperparameter | hardware | result | method | general
    entities: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    rewrites: list[str] = field(default_factory=list)
    filters: Filter = field(default_factory=Filter)
    preferred_sections: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def relaxed(self) -> Filter:
        """降级过滤：只保留小节类型软约束（严格过滤召回不足时用）。"""
        return Filter(any_of=dict(self.filters.any_of)) if self.filters.any_of else Filter()

    def to_dict(self) -> dict:
        return {"original": self.original, "intent": self.intent, "entities": self.entities,
                "numbers": self.numbers, "rewrites": self.rewrites,
                "filters": self.filters.describe(), "sections": self.preferred_sections,
                "notes": self.notes}


def _find(patterns: list[tuple[str, str]], text: str) -> list[str]:
    hits: list[str] = []
    for pat, canon in patterns:
        if re.search(pat, text, re.I):
            hits.append(canon)
    return hits


def _gpu_models(text: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(_GPU_MODELS, text, re.I):
        v = re.sub(r"\s+", "", m.group(0)).upper()
        if v not in out:
            out.append(v)
    return out


def _n_gpus(text: str) -> int | None:
    m = re.search(r"(\d+)\s*(?:张|块|个|台)?\s*(?:卡|gpu|a100|h100|v100|rtx)", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*(?:x|×|\*)\s*(?:a100|h100|v100|rtx|gpu)", text, re.I)
    return int(m.group(1)) if m else None


def detect_intent(q: str) -> tuple[str, list[str], list[str]]:
    """返回 (intent, 超参字段, 硬件字段)。两者可并存，intent 取信号更强的那个。"""
    q_l = q.lower()
    hyper = [c for c in _find(_HYPER_PATTERNS, q_l) if c != "__any__"]
    hw = _find(_HW_PATTERNS, q_l)
    if re.search(r"超参|hyper-?parameters?|用了?什么?参|参数配置|怎么训的|训练配置", q_l):
        hyper = hyper or ["__any__"]
    # 注意不要把"对比"算进结果意图："对比 A 与 B 的方法"是方法问题，加了小节过滤会把方法段全筛掉
    if re.search(r"结果|性能|多少分|准确率|错误率|误差|bleu|accuracy|f1|sota|提升", q_l) \
            and not hyper and not hw:
        return "result", [], []
    if hyper and hw:
        return "hyperparameter", hyper, hw          # 例如"8卡 A100 上 batch size 多大"
    if hyper:
        return "hyperparameter", hyper, []
    if hw:
        return "hardware", [], hw
    if re.search(r"方法|架构|模型|原理|怎么实现|method", q_l):
        return "method", [], []
    return "general", [], []


def build_filters(intent: str, hyper: list[str], hw: list[str], q: str) -> Filter:
    flt = Filter()
    sections: list[str] = []
    if intent == "hyperparameter":
        sections = _SECTIONS_HYPER
        flt.contains_any["hyperparams"] = sorted(set(hyper)) if hyper else []
    elif intent == "hardware":
        sections = _SECTIONS_HW
        models = _gpu_models(q)
        if models:
            flt.contains_any["hardware"] = models
    elif intent == "result":
        sections = _SECTIONS_RESULT

    # 硬件约束常常与超参问题共存：两个字段是 AND，别互相覆盖
    if intent == "hyperparameter":
        models = _gpu_models(q)
        if models:
            flt.contains_any["hardware"] = models
            sections = sorted(set(sections) | set(_SECTIONS_HW))
    n = _n_gpus(q)
    if n:
        flt.ranges["n_gpus"] = (n, n)
    if hyper and intent == "hardware":
        flt.contains_any["hyperparams"] = sorted(set(hyper))
    if sections:
        flt.any_of["section_type"] = sections
    # 清掉空列表，避免"contains_any 为空"被当成必须匹配空集
    flt.contains_any = {k: v for k, v in flt.contains_any.items() if v}
    return flt


def rewrite(q: str, intent: str, entities: list[str], numbers: list[str]) -> list[str]:
    """产出多条改写：原句 + 规范术语句 + 关键实体句。多路召回各跑一遍。"""
    outs: list[str] = [q]
    if intent in _CANON_TERMS:
        outs.append(f"{q} {_CANON_TERMS[intent]}")
    if entities or numbers:
        key = " ".join([*entities, *numbers])
        outs.append(f"{key} {_CANON_TERMS.get(intent, '')}".strip())
        outs.append(f"{' '.join(entities)} {' '.join(numbers)} implementation details".strip())
    # 去重并保持顺序（检索系统里"改写数量"是可调的超参，默认 ≤4 条）
    seen, uniq = set(), []
    for s in outs:
        s = re.sub(r"\s+", " ", s).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            uniq.append(s)
    return uniq[:4]


def plan_query(q: str, *, enable_rewrite: bool = True, enable_filter: bool = True) -> QueryPlan:
    """查询理解主入口：纯规则、零依赖、可复现（不调用 LLM，避免改写结果不可控）。"""
    intent, hyper, hw = detect_intent(q)
    # 实体只取"能在正文/元数据里出现的值"：gpu_memory/precision 是字段类别，不是实体
    entities = _gpu_models(q) + [h.replace("_", " ") for h in hyper]
    numbers = extract_numbers(q)
    flt = build_filters(intent, hyper, hw, q) if enable_filter else Filter()
    rewrites = rewrite(q, intent, entities, numbers) if enable_rewrite else [q]
    sections = {"hyperparameter": _SECTIONS_HYPER, "hardware": _SECTIONS_HW,
                "result": _SECTIONS_RESULT}.get(intent, [])
    notes = []
    if not flt.is_empty():
        notes.append("已按意图加元数据过滤；若召回不足会自动降级为小节软过滤")
    if len(rewrites) > 1:
        notes.append(f"生成 {len(rewrites)} 条改写，多路召回各跑一遍后按 RRF 合并")
    return QueryPlan(original=q, intent=intent, entities=sorted(set(entities)),
                     numbers=numbers, rewrites=rewrites, filters=flt,
                     preferred_sections=sections, notes=notes)
