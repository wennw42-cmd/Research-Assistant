"""工具注册与 Function Calling 规范。

三条硬性规范（评测时最容易被追问，务必守住）：
1. **单一信封**：所有工具返回 ToolEnvelope，任何异常都收敛为 ok=False + hint，
   让 LLM 能"看到错误并自纠"，而不是把异常抛到链路上炸掉；
2. **Pydantic 即 Schema**：入参 schema 由 Pydantic 模型自动生成 OpenAI tools 格式，
   杜绝手写 JSON Schema 与实现漂移；
3. **作用域隔离**：工具按 Agent 授权（ToolRegistry.specs(scopes=...)），
   解析 Agent 拿不到写作工具，降低误调用与 prompt 膨胀。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from app.schemas import EvidenceCard, ToolEnvelope
from app.utils import get_logger, short

log = get_logger("ra.tools")


@dataclass
class RunContext:
    """一次工具调用的运行时上下文（用于鉴权、追踪、注入 user/thread）。"""

    thread_id: str = "default"
    user_id: str = "anonymous"
    trace_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


ToolFunc = Callable[[BaseModel, RunContext], dict[str, Any] | ToolEnvelope]


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: ToolFunc
    scopes: tuple[str, ...] = ()      # 允许调用它的 Agent
    returns: str = "ToolEnvelope"
    slow: bool = False                # True 表示走异步任务队列而非同步阻塞

    def openai_spec(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"{self.description}\n返回结构：{self.returns}（统一信封 ToolEnvelope）",
                "parameters": schema,
            },
        }


# ────────────────────────── 信封构造器 ──────────────────────────
def ok(tool: str, data: dict[str, Any], cited: list[EvidenceCard] | None = None,
       metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    env = ToolEnvelope(ok=True, tool=tool, data=data, cited=cited or [], metrics=metrics or {})
    return env.model_dump(mode="json")


def fail(tool: str, error: str, hint: str | None = None) -> dict[str, Any]:
    log.warning("tool=%s failed: %s", tool, short(error, 160))
    return ToolEnvelope(ok=False, tool=tool, error=error, hint=hint).model_dump(mode="json")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def tool(self, fn: ToolFunc) -> Callable[..., Tool]:
        """装饰器语法糖（可选用法），本文件内置工具为可读性采用显式注册。"""
        raise NotImplementedError

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self, scopes: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
        """导出给 LLM 的 tools 数组（按 Agent 作用域裁剪）。"""
        scopes = tuple(scopes or ())
        out = []
        for t in self._tools.values():
            if scopes and not (set(t.scopes) & set(scopes)):
                continue
            out.append(t.openai_spec())
        return out

    def invoke(self, name: str, arguments: str | dict[str, Any], ctx: RunContext | None = None) -> dict[str, Any]:
        """执行工具：入参校验 → 调用 → 统一信封返回。

        注意：即使工具内部抛异常也返回信封，这是"抗炸链路"的关键。
        """
        ctx = ctx or RunContext()
        tool = self._tools.get(name)
        started = time.perf_counter()
        if tool is None:
            return fail(name, f"未注册的工具：{name}", hint="可用工具见系统提示中的 tools 列表")
        raw = arguments if isinstance(arguments, dict) else _safe_json(arguments)
        if raw is None:
            return fail(name, "arguments 不是合法 JSON", hint="请重新输出严格 JSON 格式的参数")
        try:
            args = tool.args_model.model_validate(raw)
        except ValidationError as e:
            return fail(name, f"入参校验失败：{short(e.json(), 240)}", hint="按工具 schema 补齐必填字段后重试")
        try:
            result = tool.func(args, ctx)
        except Exception as e:  # noqa: BLE001
            return fail(name, f"执行异常：{type(e).__name__}: {short(str(e), 200)}", hint="稍后重试或换一种参数组合")
        env = result if isinstance(result, dict) and "ok" in result else ok(name, {"result": result})
        env.setdefault("metrics", {})
        env["metrics"].setdefault("latency_ms", int((time.perf_counter() - started) * 1000))
        log.info("tool=%-22s ok=%-5s latency=%sms", name, env.get("ok"), env["metrics"]["latency_ms"])
        return env


def _safe_json(raw: str) -> dict[str, Any] | None:
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {"value": v}
    except Exception:  # noqa: BLE001
        return None


registry = ToolRegistry()
