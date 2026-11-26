"""通用大模型模块：模型问答 + 图片理解 + 文生图 + 图片转文字（OCR）。

与 Agent 完全分离：不路由、不检索文献库、不要求引用锚点。
- 问答：直接调用已配置的 OpenAI 兼容模型（默认阿里云百炼 qwen-plus）。
- 图片理解：qwen3-vl-plus，走 OpenAI 兼容多模态消息。
- 文生图：通义万相，走 DashScope 异步任务接口，生成结果下载到 static/generated。
- 图片转文字：复用 ``app.parsing.attachments`` 的 OCR 流水线，只接受图片。
- 语音：前端使用浏览器原生 Web Speech API（语音识别 / 朗读），后端不接收、不保存音频。

运行（在 apps/api 下）：
    python -m app.general --text "什么是 Transformer？"
    python -m app.general --image figure.png
    python -m app.general --image-chat figure.png --question "这张图说明了什么？"
    python -m app.general --gen-image "极简风格科研助手插图"
    python -m app.general --selftest
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from app import config
from app.parsing.attachments import MAX_UPLOAD_BYTES, extract_attachment
from app.providers import BaseLLM, ChatResponse, get_llm
from app.utils import get_logger
from app.zh2en import _safe_error

log = get_logger("ra.general")

SYSTEM_PROMPT = (
    "你是科研助手的通用问答模块。用你自己的知识直接、准确地回答用户问题；"
    "如果随问题提供了联网搜索结果，以搜索结果为准并注明信息时间。\n"
    "要求：① 使用与用户相同的语言回答；② 不编造文献标题、作者、DOI 或引用；"
    "③ 不编造实时数据（天气、新闻、价格等），拿不到就明确说明；"
    "④ 如果问题需要用户上传的论文或数据，提示改用「对话 / 任务」上传附件。"
)
# 需要联网才答得准的意图。命中才开搜索，普通问答不加延迟、不产生搜索费用。
# ponytail: 关键词启发式；误判只是多一次搜索，漏判时用户可在问题里带上“天气/最新”等词。
LIVE_KEYWORDS = (
    "天气", "气温", "温度", "下雨", "降雨", "降水", "台风", "空气质量", "雾霾",
    "湿度", "风力", "预报", "aqi", "新闻", "最新", "实时", "股价", "汇率",
    "油价", "金价", "比分", "热搜", "路况", "航班",
    "weather", "forecast", "temperature", "rain", "news", "latest",
    "stock", "exchange rate",
)
VISION_SYSTEM_PROMPT = (
    "你是科研助手的图像理解模块。准确描述图片内容、图表趋势和其中文字；"
    "看不清或不确定时明确说明，不编造图中不存在的信息；使用与用户相同的语言回答。"
)

MAX_MESSAGES = 40
MAX_MESSAGE_CHARS = 20_000
MAX_TOTAL_CHARS = 60_000
MAX_SYSTEM_CHARS = 4_000
MAX_PROMPT_CHARS = 2_000
MAX_NEGATIVE_CHARS = 1_000
VISION_MAX_BYTES = 10 * 1024 * 1024
IMAGE_DOWNLOAD_MAX_BYTES = 20 * 1024 * 1024
GENERATED_DIR = Path(__file__).resolve().parent / "static" / "generated"
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}
IMAGE_SUFFIXES = set(IMAGE_MIME)
_SIZE_RE = re.compile(r"^(\d{3,4})\*(\d{3,4})$")


class GeneralError(ValueError):
    """通用模块的可直接展示给用户的错误（不包含密钥）。"""


def _clean_messages(
    messages: Sequence[Any], system: str = ""
) -> list[dict[str, str]]:
    """校验并裁剪对话：限制条数/单条长度/总长度，拒绝空对话。"""
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise GeneralError("messages 必须是消息数组")
    system_text = str(system or "").strip()[:MAX_SYSTEM_CHARS] or SYSTEM_PROMPT
    out: list[dict[str, str]] = [{"role": "system", "content": system_text}]
    total = 0
    for item in messages:
        if not isinstance(item, dict):
            raise GeneralError("每条消息必须是 {role, content} 对象")
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in ("system", "user", "assistant"):
            raise GeneralError(f"不支持的消息角色：{role or '空'}")
        if not content:
            continue
        if len(content) > MAX_MESSAGE_CHARS:
            raise GeneralError(f"单条消息超过 {MAX_MESSAGE_CHARS} 字符")
        total += len(content)
        if total > MAX_TOTAL_CHARS:
            raise GeneralError(f"对话总长度超过 {MAX_TOTAL_CHARS} 字符，请新建对话")
        out.append({"role": role, "content": content})
    if not any(m["role"] == "user" for m in out):
        raise GeneralError("至少需要一条用户消息")
    if len(out) > MAX_MESSAGES + 1:
        out = [out[0]] + out[-MAX_MESSAGES:]
    return out


def _needs_live(text: str) -> bool:
    """问题是否依赖实时数据（天气、新闻、行情等）。"""
    lowered = (text or "").lower()
    return any(word in lowered for word in LIVE_KEYWORDS)


def _last_user_text(messages: Sequence[dict[str, Any]]) -> str:
    return next(
        (str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )


def _search_supported(settings: config.Settings) -> bool:
    """联网搜索只适用于百炼原生端点的 qwen 模型。"""
    return (
        "dashscope.aliyuncs.com" in settings.LLM_BASE_URL
        and settings.LLM_MODEL.lower().startswith("qwen")
    )


def _parse_search_response(data: Any) -> tuple[str, dict[str, Any]]:
    """解析 DashScope 原生文本生成响应（result_format=message）。"""
    if not isinstance(data, dict):
        return "", {}
    choices = (data.get("output") or {}).get("choices") or []
    answer = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        answer = str(message.get("content") or "").strip()
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return answer, usage


def _search_chat(
    messages: Sequence[dict[str, Any]],
    temperature: float,
    settings: config.Settings,
) -> dict[str, Any] | None:
    """带联网搜索的问答；失败返回 None，由调用方降级为普通模型回答。"""
    import httpx

    body = {
        "model": settings.LLM_MODEL,
        "input": {"messages": list(messages)},
        "parameters": {
            "result_format": "message",
            "enable_search": True,
            "temperature": temperature,
            "max_tokens": settings.LLM_MAX_TOKENS,
        },
    }
    try:
        resp = httpx.post(
            f"{settings.IMAGE_API_BASE.rstrip('/')}/services/aigc/text-generation/generation",
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            json=body,
            timeout=max(30, settings.LLM_TIMEOUT),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("live search failed: %s", _safe_error(exc))
        return None
    answer, usage = _parse_search_response(data)
    if not answer:
        log.warning("live search returned no content")
        return None
    plugin = (usage.get("plugins") or {}).get("search") or {}
    return {
        "answer": answer,
        "model": settings.LLM_MODEL,
        "usage": usage,
        "search": {"enabled": True, "count": int(plugin.get("count") or 0)},
    }


def chat(
    messages: Sequence[Any],
    *,
    system: str = "",
    temperature: float = 0.3,
    llm: BaseLLM | None = None,
) -> dict[str, Any]:
    """通用问答：普通问题直接调模型；实时类问题先走百炼联网搜索。"""
    cleaned = _clean_messages(messages, system)
    settings = config.get_settings()
    temp = max(0.0, min(2.0, float(temperature)))
    auto_llm = llm is None
    if auto_llm:
        if settings.use_mock_llm:
            raise GeneralError(
                "通用问答需要真实模型：请在 apps/api/.env 配置 LLM_API_KEY 并重启"
            )
        llm = get_llm()
        if _needs_live(_last_user_text(cleaned)) and _search_supported(settings):
            live = _search_chat(cleaned, temp, settings)
            if live:
                return live
    try:
        resp: ChatResponse = llm.chat(cleaned, temperature=temp)
    except Exception as exc:  # noqa: BLE001
        log.error("general chat failed: %s", _safe_error(exc))
        raise GeneralError(f"模型调用失败：{_safe_error(exc)}") from None
    answer = (resp.content or "").strip()
    if not answer:
        raise GeneralError("模型没有返回内容，请重试")
    result = {
        "answer": answer,
        "model": getattr(llm, "model_name", "") or settings.LLM_MODEL,
        "usage": resp.usage,
    }
    if auto_llm and _needs_live(_last_user_text(cleaned)):
        # 搜索失败时明确告诉用户这条回答不含实时数据，避免误以为查过了。
        result["warnings"] = ["联网搜索不可用，回答未包含实时数据"]
    return result


def _check_image(
    filename: str,
    data: bytes,
    *,
    limit: int = MAX_UPLOAD_BYTES,
    label: str = "图片",
) -> tuple[str, str]:
    """校验图片扩展名/非空/大小，返回 (安全文件名, 小写扩展名)。"""
    name = Path(filename or "image").name
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise GeneralError("通用模块只支持图片（PNG/JPG/TIFF/BMP/WEBP）")
    if not data:
        raise GeneralError(f"{label}为空")
    if len(data) > limit:
        raise GeneralError(f"{label}超过 {limit // (1024 * 1024)} MB")
    return name, suffix


def image_to_text(filename: str, data: bytes) -> dict[str, Any]:
    """图片转文字：只接受图片，复用统一 OCR/限流逻辑。"""
    name, _ = _check_image(filename, data)
    try:
        result = extract_attachment(name, data)
    except ValueError as exc:
        raise GeneralError(str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        log.error("image ocr failed: %s", _safe_error(exc))
        raise GeneralError(f"图片识别失败：{_safe_error(exc)}") from None
    if result.get("kind") != "image":
        raise GeneralError("该文件不是图片")
    return {
        "filename": result.get("filename", name),
        "kind": "image",
        "text": result.get("text", ""),
        "chars": result.get("chars", 0),
        "truncated": result.get("truncated", False),
        "meta": result.get("meta") or {},
        "warnings": result.get("warnings") or [],
    }


def _build_vision_messages(
    filename: str, data: bytes, question: str = ""
) -> tuple[str, str, list[dict[str, Any]]]:
    """构造 OpenAI 兼容的多模态消息（图片用 data URI）。"""
    name, suffix = _check_image(
        filename, data, limit=VISION_MAX_BYTES, label="图片理解文件"
    )
    prompt = (
        str(question or "").strip()[:MAX_MESSAGE_CHARS]
        or "请详细描述这张图片的内容，并提取其中的文字。"
    )
    data_uri = f"data:{IMAGE_MIME[suffix]};base64,{base64.b64encode(data).decode('ascii')}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    return name, prompt, messages


def image_chat(
    filename: str,
    data: bytes,
    *,
    question: str = "",
    model: str = "",
    client: Any = None,
) -> dict[str, Any]:
    """图片理解：把图片和问题一起交给多模态模型。"""
    settings = config.get_settings()
    name, prompt, messages = _build_vision_messages(filename, data, question)
    if settings.use_mock_llm:
        raise GeneralError(
            "图片理解需要真实模型：请在 apps/api/.env 配置 LLM_API_KEY 并重启"
        )
    model_name = model or settings.VISION_MODEL
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
        )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("vision chat failed: %s", _safe_error(exc))
        raise GeneralError(f"图片理解失败：{_safe_error(exc)}") from None
    answer = (resp.choices[0].message.content or "").strip()
    if not answer:
        raise GeneralError("模型没有返回内容，请重试")
    return {
        "answer": answer,
        "model": model_name,
        "usage": getattr(resp, "usage", None),
        "filename": name,
        "question": prompt,
    }


def _image_gen_params(
    prompt: str, size: str, n: int, negative_prompt: str
) -> tuple[str, str, int, str]:
    """校验文生图参数，返回 (prompt, size, n, negative_prompt)。"""
    text = str(prompt or "").strip()
    if not text:
        raise GeneralError("提示词不能为空")
    if len(text) > MAX_PROMPT_CHARS:
        raise GeneralError(f"提示词最多 {MAX_PROMPT_CHARS} 字符")
    size = str(size or "1024*1024").strip()
    match = _SIZE_RE.fullmatch(size)
    if not match:
        raise GeneralError("尺寸格式应为 宽*高，例如 1024*1024")
    width, height = int(match.group(1)), int(match.group(2))
    if (
        width < 512
        or height < 512
        or width > 1440
        or height > 1440
        or width * height > 2_100_000
    ):
        raise GeneralError("尺寸需在 512–1440 像素之间，且总像素不超过约 210 万")
    try:
        count = int(n)
    except (TypeError, ValueError):
        raise GeneralError("生成数量必须是 1–4 的整数") from None
    if not 1 <= count <= 4:
        raise GeneralError("生成数量必须是 1–4 的整数")
    negative = str(negative_prompt or "").strip()
    if len(negative) > MAX_NEGATIVE_CHARS:
        raise GeneralError(f"反向提示词最多 {MAX_NEGATIVE_CHARS} 字符")
    return text, size, count, negative


def _dashscope_error(resp: Any) -> str:
    """从 DashScope 错误响应里取可展示信息，不回显请求头。"""
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return f"HTTP {getattr(resp, 'status_code', '?')}"
    err: Any = data.get("message") or data.get("code") or data.get("error") or ""
    if isinstance(err, dict):
        err = err.get("message") or err.get("code") or str(err)
    return f"HTTP {getattr(resp, 'status_code', '?')}: {str(err)[:300]}"


def _prune_generated(max_files: int = 200, max_age: int = 7 * 24 * 3600) -> None:
    """清理过旧/过多的生成图片，避免 static/generated 无限增长。"""
    try:
        files = sorted(
            GENERATED_DIR.glob("gen_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    now = time.time()
    for index, path in enumerate(files):
        try:
            if index >= max_files or now - path.stat().st_mtime > max_age:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def _save_generated(urls: list[str]) -> list[dict[str, Any]]:
    """下载通义万相结果到本地静态目录；失败时保留远端 URL。"""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    _prune_generated()
    out: list[dict[str, Any]] = []
    import httpx

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for url in urls:
            item: dict[str, Any] = {"url": url, "local_url": "", "file": "", "warnings": []}
            parsed = urlparse(url)
            host = parsed.hostname or ""
            if parsed.scheme not in ("http", "https") or not host.endswith(".aliyuncs.com"):
                item["warnings"].append("图片地址不在阿里云域名下，未下载")
                out.append(item)
                continue
            try:
                resp = client.get(url)
                resp.raise_for_status()
                if len(resp.content) > IMAGE_DOWNLOAD_MAX_BYTES:
                    raise ValueError("图片超过下载上限")
                content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(
                    content_type, ".png"
                )
                filename = f"gen_{uuid.uuid4().hex}{suffix}"
                (GENERATED_DIR / filename).write_bytes(resp.content)
                item["local_url"] = f"/static/generated/{filename}"
                item["file"] = filename
            except Exception as exc:  # noqa: BLE001
                item["warnings"].append(f"下载失败：{_safe_error(exc)[:200]}")
            out.append(item)
    return out


def generate_image(
    prompt: str,
    *,
    size: str = "1024*1024",
    n: int = 1,
    negative_prompt: str = "",
    model: str = "",
    client: Any = None,
) -> dict[str, Any]:
    """文生图：提交 DashScope 异步任务，轮询完成后把图片落到本地。"""
    settings = config.get_settings()
    text, size, count, negative = _image_gen_params(prompt, size, n, negative_prompt)
    if settings.use_mock_llm:
        raise GeneralError(
            "图片生成需要真实模型：请在 apps/api/.env 配置 LLM_API_KEY 并重启"
        )
    model_name = model or settings.IMAGE_MODEL
    base = (settings.IMAGE_API_BASE or "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
    headers = {
        "Authorization": f"Bearer {settings.LLM_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    payload: dict[str, Any] = {
        "model": model_name,
        "input": {"prompt": text},
        "parameters": {
            "size": size,
            "n": count,
            "prompt_extend": True,
            "watermark": False,
        },
    }
    if negative:
        payload["input"]["negative_prompt"] = negative

    import httpx

    owns_client = client is None
    http = client or httpx.Client(timeout=30)
    try:
        resp = http.post(
            f"{base}/services/aigc/text2image/image-synthesis",
            headers=headers,
            json=payload,
        )
        if resp.status_code >= 400:
            raise GeneralError(f"图片生成请求失败：{_dashscope_error(resp)}")
        task_id = (resp.json().get("output") or {}).get("task_id")
        if not task_id:
            raise GeneralError(f"图片生成未返回任务 ID：{_dashscope_error(resp)}")

        deadline = time.monotonic() + max(30, int(settings.IMAGE_TIMEOUT))
        result: dict[str, Any] = {}
        while time.monotonic() < deadline:
            time.sleep(1.5)
            poll = http.get(
                f"{base}/tasks/{task_id}",
                headers={"Authorization": headers["Authorization"]},
            )
            if poll.status_code >= 400:
                raise GeneralError(f"图片生成查询失败：{_dashscope_error(poll)}")
            result = poll.json()
            output = result.get("output") or {}
            status = output.get("task_status")
            if status == "SUCCEEDED":
                break
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                detail = output.get("message") or output.get("code") or status
                raise GeneralError(f"图片生成失败：{str(detail)[:300]}")
        else:
            raise GeneralError(f"图片生成超时（{settings.IMAGE_TIMEOUT}s），请稍后重试")
    finally:
        if owns_client:
            http.close()

    output = result.get("output") or {}
    results = output.get("results") or []
    urls = [item["url"] for item in results if isinstance(item, dict) and item.get("url")]
    if not urls:
        raise GeneralError("图片生成完成但没有返回图片")
    images = _save_generated(urls)
    actual_prompt = output.get("actual_prompt") or next(
        (item.get("actual_prompt") for item in results if item.get("actual_prompt")), ""
    )
    return {
        "images": images,
        "model": model_name,
        "task_id": task_id,
        "prompt": text,
        "actual_prompt": actual_prompt or "",
        "size": size,
        "count": len(images),
        "usage": result.get("usage") or {},
        "warnings": [warning for item in images for warning in item.get("warnings", [])],
    }


class _EchoLLM(BaseLLM):
    """仅用于自检，不联网。"""

    model_name = "selftest"

    def chat(self, messages, tools=None, tool_choice=None, temperature=None, max_tokens=None):
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return ChatResponse(content=f"echo:{last}", model=self.model_name)


def _selftest() -> int:
    checks: list[tuple[str, bool]] = []

    cleaned = _clean_messages([{"role": "user", "content": "你好"}])
    checks.append(("自动注入 system prompt", cleaned[0]["role"] == "system"))
    checks.append(("保留用户消息", cleaned[-1]["content"] == "你好"))

    for bad, label in (
        ([], "拒绝空对话"),
        ([{"role": "tool", "content": "x"}], "拒绝非法角色"),
        ([{"role": "user", "content": "x" * (MAX_MESSAGE_CHARS + 1)}], "拒绝超长消息"),
    ):
        try:
            _clean_messages(bad)
            checks.append((label, False))
        except GeneralError:
            checks.append((label, True))

    out = chat([{"role": "user", "content": "ping"}], llm=_EchoLLM())
    checks.append(("问答返回模型文本", out["answer"] == "echo:ping"))

    checks.append(("天气问题识别为实时", _needs_live("北京今天天气怎么样？")))
    checks.append(("普通问题不触发搜索", not _needs_live("什么是 Transformer？")))
    answer, usage = _parse_search_response(
        {"output": {"choices": [{"message": {"content": " 晴 1–15℃ "}}]},
         "usage": {"plugins": {"search": {"count": 1}}}}
    )
    checks.append(
        ("解析联网搜索结果", answer == "晴 1–15℃" and usage["plugins"]["search"]["count"] == 1)
    )

    try:
        image_to_text("paper.pdf", b"%PDF-1.4")
        checks.append(("图片接口拒绝 PDF", False))
    except GeneralError:
        checks.append(("图片接口拒绝 PDF", True))

    try:
        from io import BytesIO

        from PIL import Image, ImageDraw

        buf = BytesIO()
        img = Image.new("RGB", (120, 40), "white")
        ImageDraw.Draw(img).text((8, 12), "OCR OK", fill="black")
        img.save(buf, format="PNG")
        res = image_to_text("sample.png", buf.getvalue())
        checks.append(("PNG 走 OCR 并返回文本字段", "text" in res and res["kind"] == "image"))
    except Exception as exc:  # noqa: BLE001
        print(f"  OCR 自检异常：{_safe_error(exc)}")
        checks.append(("PNG 走 OCR 并返回文本字段", False))

    _, prompt, messages = _build_vision_messages("sample.png", b"\x89PNG", "这是什么？")
    content = messages[-1]["content"]
    checks.append(
        (
            "图片理解生成 data URI",
            content[0]["image_url"]["url"].startswith("data:image/png;base64,"),
        )
    )
    checks.append(("图片理解保留问题", prompt == "这是什么？" and content[1]["text"] == "这是什么？"))

    try:
        _build_vision_messages("paper.pdf", b"%PDF-1.4")
        checks.append(("图片理解拒绝 PDF", False))
    except GeneralError:
        checks.append(("图片理解拒绝 PDF", True))

    _, size, count, _ = _image_gen_params("画一只猫", "1024*1024", 1, "")
    checks.append(("文生图参数规范化", size == "1024*1024" and count == 1))
    for bad, label in (
        (("", "1024*1024", 1, ""), "文生图拒绝空提示词"),
        (("猫", "9999*9999", 1, ""), "文生图拒绝超大尺寸"),
        (("猫", "1024*1024", 9, ""), "文生图拒绝超量"),
    ):
        try:
            _image_gen_params(*bad)
            checks.append((label, False))
        except GeneralError:
            checks.append((label, True))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"selftest: {len(checks) - len(failed)}/{len(checks)} 通过")
    return 0 if not failed else 1


def _demo() -> None:
    print("通用模块能力：")
    print("  问答：python -m app.general --text \"什么是 Transformer？\"")
    print("  图片理解：python -m app.general --image-chat figure.png --question \"这张图说明了什么？\"")
    print("  文生图：python -m app.general --gen-image \"极简风格科研助手插图\"")
    print("  OCR：python -m app.general --image figure.png")
    print("  语音：由前端使用浏览器 Web Speech API，后端不接收音频。")


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="通用大模型模块（问答 / 图片理解 / 文生图 / OCR）")
    ap.add_argument("--text", help="单轮问题")
    ap.add_argument("--messages", help="JSON 文件：消息数组或 {messages:[...]}")
    ap.add_argument("--image", help="图片路径（OCR）")
    ap.add_argument("--image-chat", help="图片路径（图片理解）")
    ap.add_argument("--question", default="", help="配合 --image-chat 的问题")
    ap.add_argument("--gen-image", help="文生图提示词")
    ap.add_argument("--size", default="1024*1024", help="文生图尺寸，如 1024*1024")
    ap.add_argument("--n", type=int, default=1, help="文生图数量 1–4")
    ap.add_argument("--negative", default="", help="文生图反向提示词")
    ap.add_argument("--system", default="", help="自定义 system prompt")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--out", help="把结果写入 JSON 文件")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.demo:
        _demo()
        return 0

    try:
        if args.gen_image:
            result = generate_image(
                args.gen_image,
                size=args.size,
                n=args.n,
                negative_prompt=args.negative,
            )
        elif args.image_chat:
            with open(args.image_chat, "rb") as handle:
                data = handle.read(VISION_MAX_BYTES + 1)
            result = image_chat(
                Path(args.image_chat).name, data, question=args.question
            )
        elif args.image:
            with open(args.image, "rb") as handle:
                data = handle.read(MAX_UPLOAD_BYTES + 1)
            result = image_to_text(Path(args.image).name, data)
        else:
            messages: list[dict[str, str]] = []
            if args.messages:
                payload = json.loads(Path(args.messages).read_text(encoding="utf-8"))
                messages = payload.get("messages", []) if isinstance(payload, dict) else payload
            if args.text:
                messages.append({"role": "user", "content": args.text})
            result = chat(messages, system=args.system, temperature=args.temperature)
    except (GeneralError, OSError, json.JSONDecodeError) as exc:
        print(f"错误：{_safe_error(exc)}", file=sys.stderr)
        return 2

    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result.get("images"):
        for image in result["images"]:
            print(image.get("local_url") or image.get("url") or "")
    else:
        print(result.get("answer") or result.get("text") or "")
        if result.get("warnings"):
            print("提示：" + "；".join(map(str, result["warnings"])), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
