"""LLM Provider Adapter。

统一协议 = OpenAI Chat Completions（DeepSeek / Qwen / GLM / Kimi / vLLM / Ollama 均兼容）。
新增厂商只需加一条 Settings，**不需要碰任何 Agent 代码**。

对外只暴露三个方法：
    chat()       —— 带 Function Calling 的对话
    chat_json()  —— 强制 JSON 输出（内部做抽取与重试）
    embed()      —— 向量化（真实实现接 BGE-M3，mock 用哈希伪向量）
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app import config
from app.utils import extract_json, get_logger, short

log = get_logger("ra.providers")

# tools 传给 LLM 的标准 OpenAI 格式：
# [{"type": "function", "function": {"name":..., "description":..., "parameters": <JSON Schema>}}]
ToolSpec = dict[str, Any]


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""

    @property
    def has_calls(self) -> bool:
        return bool(self.tool_calls)


class BaseLLM:
    model_name: str = "base"

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[ToolSpec] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:  # pragma: no cover - interface
        raise NotImplementedError

    def chat_json(
        self,
        messages: Sequence[dict[str, Any]],
        retries: int = 1,
        tools: Sequence[ToolSpec] | None = None,
    ) -> Any:
        """拿结构化结果：先正常 chat，再稳健抽 JSON；失败则重试并要求"只输出 JSON"。

        这是所有 Planner / Router / Verifier 的统一入口，
        保证即便模型偷懒加装饰文字，链路也不会崩。
        """
        msgs = list(messages)
        last_raw = ""
        for attempt in range(retries + 1):
            resp = self.chat(msgs, tools=tools, temperature=0.0)
            last_raw = resp.content
            parsed = extract_json(resp.content)
            if parsed is not None:
                return parsed
            msgs = msgs + [
                {"role": "assistant", "content": last_raw[:500]},
                {"role": "user", "content": "请严格只输出 JSON，不要任何解释、代码块或前后缀。"},
            ]
        log.warning("chat_json 解析失败，raw=%s", short(last_raw, 120))
        return None

    def embed(self, texts: list[str]) -> list[list[float]]:  # 默认离线伪向量
        return [_hash_embedding(t) for t in texts]


class OpenAICompatibleLLM(BaseLLM):
    """真实模型：任意 OpenAI 兼容端点。"""

    def __init__(self, settings: config.Settings | None = None) -> None:
        from openai import OpenAI  # 延迟导入，保证 mock 模式零依赖可跑

        s = settings or config.get_settings()
        self.settings = s
        self.model_name = s.LLM_MODEL
        self._client = OpenAI(api_key=s.LLM_API_KEY, base_url=s.LLM_BASE_URL, timeout=s.LLM_TIMEOUT)

    def chat(self, messages, tools=None, tool_choice=None, temperature=None, max_tokens=None) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.settings.LLM_MODEL,
            "messages": list(messages),
            "temperature": self.settings.LLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": self.settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = tool_choice or "auto"
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"},
            }
            for c in (msg.tool_calls or [])
        ]
        usage = getattr(resp, "usage", None)
        return ChatResponse(
            content=msg.content or "",
            tool_calls=calls,
            usage={"prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                   "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0},
            model=self.model_name,
        )


class MockLLM(BaseLLM):
    """离线脚本化模型。

    存在的意义：**没有 API Key 也能端到端跑通全部 Agent 链路**，
    用于 CI、离线答辩预演、以及"模型之外"的逻辑回归测试。
    它按关键词选择工具、按 résumé 生成可执行 JSON 计划，
    行为确定、可断言。
    """

    model_name = "mock-llm"

    def chat(self, messages, tools=None, tool_choice=None, temperature=None, max_tokens=None) -> ChatResponse:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break
        system = str(messages[0].get("content", "")) if messages else ""
        tools = list(tools or [])
        tool_calls = self._pick_tools(last_user, system, tools)

        # 还没调工具 → 发起调用；已调过 → 汇总成答案
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if tool_calls and not has_tool_result:
            return ChatResponse(content="", tool_calls=tool_calls, model=self.model_name)

        ask_json = any(k in last_user for k in ("JSON", "json", "只输出", "schema", "路由", "计划")) or (
            "只输出 JSON" in (system + last_user)
        )
        if ask_json:
            return ChatResponse(content=self._canned_json(last_user, system), model=self.model_name)

        return ChatResponse(
            content=self._compose(last_user, messages) if has_tool_result else self._draft(last_user),
            model=self.model_name,
        )

    # ── 内部策略 ──
    @staticmethod
    def _pick_tools(user: str, system: str, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        names = [t["function"]["name"] for t in tools]
        wanted: list[str] = []
        text = user + system
        if any(k in text for k in ("解析", "上传", "PDF", "pdf", "去重", "版本", "v1", "v7", "扫描", "OCR")):
            wanted += ["parse_document", "extract_scanned_ocr", "semantic_dedup"]
        if any(k in text for k in ("检索", "推荐", "追踪", "找论文", "最新")):
            wanted += ["hybrid_search", "fetch_paper"]
        if any(k in text for k in ("翻译", "translate", "中英", "双语")):
            wanted += ["translate_block"]
        if any(k in text for k in ("图谱", "引用网络", "citation", "共引", "主路径")):
            wanted += ["build_citation_graph", "fetch_paper"]
        if any(k in text for k in ("综述", "survey", "写作", "LaTeX", "章节")):
            wanted += ["write_section", "export_latex"]
        if any(k in text for k in ("对比", "区别", "versus", "vs")):
            wanted += ["compare_claims"]
        if any(k in text for k in ("图", "可视化", "柱状", "趋势")):
            wanted += ["render_chart"]
        if not wanted:
            wanted = ["hybrid_search"]

        selected: list[str] = []
        for w in wanted:
            if w in names and w not in selected:
                selected.append(w)
            if len(selected) >= 2:
                break
        return [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": n, "arguments": json.dumps(_mock_args(n, user), ensure_ascii=False)},
            }
            for n in selected
        ]

    @staticmethod
    def _canned_json(user: str, system: str) -> str:
        """给 Router / Planner / Verifier 返回可直接使用的 JSON。"""
        if "路由" in user or "意图" in user or "route" in system:
            if any(k in user for k in ("综述", "对比", "survey", "生成一篇")):
                return json.dumps(
                    {"route": "planner", "complexity": "composite", "confidence": 0.93,
                     "intent": "多步复合任务：需要先检索/对比，再组织成综述",
                     "reason": "查询包含 ≥2 个不同专业领域的动词（对比 + 生成综述）"},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"route": "rag", "complexity": "simple", "confidence": 0.7,
                 "intent": "单步问答", "reason": "未命中强特征，落到默认 RAG"},
                ensure_ascii=False,
            )
        if "计划" in user or "plan" in system or "DAG" in user:
            return json.dumps(_mock_plan(user), ensure_ascii=False)
        if "校验" in user or "幻觉" in user or "verify" in system:
            return json.dumps(
                {"issues": [], "passed": True, "evidence_coverage": 0.9,
                 "comment": "所有关键论断均有库内证据支撑"},
                ensure_ascii=False,
            )
        return json.dumps({"ok": True, "result": "mock"}, ensure_ascii=False)

    @staticmethod
    def _compose(user: str, messages: Sequence[dict[str, Any]]) -> str:
        ev_lines: list[str] = []
        for m in messages:
            if m.get("role") != "tool":
                continue
            body = _loads(m.get("content", ""))
            if not isinstance(body, dict):
                continue
            for c in (body.get("cited") or [])[:3]:
                ev_lines.append(f"[[ev:{c.get('doc_id')}#{c.get('block_id')}]] {short(str(c.get('quote')), 90)}")
            data = body.get("data") or {}
            if isinstance(data, dict) and data:
                ev_lines.append(f"- {body.get('tool')}: {short(str(list(data)[:6]), 140)}")
        ev = "\n".join(dict.fromkeys(ev_lines)) or "（本次未取到证据）"
        return f"针对「{short(user, 60)}」，基于已获取的库内证据：\n{ev}\n\n结论：以上要点均可在原文中定位核对。"

    @staticmethod
    def _draft(user: str) -> str:
        return f"[mock] 已理解任务：{short(user, 120)}。真实接入后由 Planner 拆解为 DAG 并调用工具执行。"


def _mock_args(tool: str, user: str) -> dict[str, Any]:
    """给 mock 工具调用造合法入参（真实实现里由 Planner/上游结果提供）。"""
    if tool == "hybrid_search":
        return {"query": _extract_topic(user), "top_k": 4}
    if tool == "compare_claims":
        return {"topic_a": "Transformer", "topic_b": "BERT",
                "aspects": ["架构", "预训练目标", "上下文建模", "下游范式", "计算复杂度"]}
    if tool == "build_citation_graph":
        return {"seed_doc_ids": ["doc_transformer", "doc_bert"], "depth": 2}
    if tool == "translate_block":
        return {"doc_id": "doc_transformer", "block_ids": ["b000"], "target_lang": "zh"}
    if tool == "write_section":
        return {"title": "引言", "aspects": ["背景", "研究问题", "本文贡献"]}
    if tool == "export_latex":
        return {"title": "A Mini Survey"}
    if tool == "render_chart":
        return {"chart_type": "bar", "x": ["Transformer", "BERT"], "y": [1.0, 0.92]}
    if tool == "parse_document":
        return {"file_path": "sample.pdf", "force_ocr": False}
    if tool in ("semantic_dedup",):
        return {"doc_ids": ["arxiv:1706.03762v1", "arxiv:1706.03762v7"]}
    if tool == "extract_scanned_ocr":
        return {"file_path": "doc_scan_sample.pdf", "engine": "auto"}
    if tool == "fetch_paper":
        return {"query": _extract_topic(user), "source": "openalex", "limit": 3}
    return {}


def _extract_topic(user: str) -> str:
    for kw in ("Transformer", "BERT", "扫描件", "OCR", "去重", "综述", "图谱"):
        if kw in user:
            return kw
    return user[:24] or "research"


def _mock_plan(user: str) -> dict[str, Any]:
    """mock 的复合任务计划模板：对比 A 与 B 并生成综述。"""
    return {
        "goal": user[:80],
        "rationale": "复合意图需先后完成检索、对比、成稿三步，其中成稿依赖前两者的产出。",
        "subtasks": [
            {"id": "t1", "title": "检索双方核心文献", "agent": "retrieval",
             "description": "检索 Transformer 与 BERT 的原始论文与关键变体",
             "inputs": {"query": "Transformer BERT 原始论文"}, "depends_on": [], "tool_hints": ["hybrid_search"]},
            {"id": "t2", "title": "多维度对比分析", "agent": "rag",
             "description": "从架构、预训练目标、上下文建模、下游范式、复杂度五个维度对比",
             "inputs": {"a": "Transformer", "b": "BERT"}, "depends_on": ["t1"], "tool_hints": ["compare_claims"]},
            {"id": "t3", "title": "构建引用关系图", "agent": "graph",
             "description": "以两篇种子论文构建引用网络并识别社区",
             "inputs": {"seed": ["doc_transformer", "doc_bert"]}, "depends_on": ["t1"], "tool_hints": ["build_citation_graph"]},
            {"id": "t4", "title": "撰写综述正文", "agent": "writing",
             "description": "基于对比结果与图谱撰写带引用锚点的综述初稿",
             "inputs": {"sections": ["引言", "方法对比", "文献脉络", "结论"]},
             "depends_on": ["t2", "t3"], "tool_hints": ["write_section"]},
            {"id": "t5", "title": "导出 LaTeX/BibTeX", "agent": "writing",
             "description": "把初稿导出为可编译的 LaTeX 与参考文献",
             "inputs": {}, "depends_on": ["t4"], "tool_hints": ["export_latex"]},
        ],
    }


def _hash_embedding(text: str, dim: int = 64) -> list[float]:
    """离线伪向量：仅用于让向量检索的调用链跑通，真实场景替换为 BGE-M3。"""
    vec = [0.0] * dim
    for token in _tokenize(text):
        h = _fnv1a(token) % dim
        vec[h] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [round(v / norm, 6) for v in vec]


def _tokenize(text: str) -> Iterable[str]:
    buf: list[str] = []
    for ch in (text or "").lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                yield "".join(buf)
                buf = []
            yield ch
    if buf:
        yield "".join(buf)


def _fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _loads(raw: Any) -> Any:
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:  # noqa: BLE001
        return None


def get_llm(settings: config.Settings | None = None) -> BaseLLM:
    s = settings or config.get_settings()
    if s.use_mock_llm:
        log.info("LLM 降级为 MockLLM（未配置 LLM_API_KEY 或 LLM_PROVIDER=mock）")
        return MockLLM()
    return OpenAICompatibleLLM(s)
