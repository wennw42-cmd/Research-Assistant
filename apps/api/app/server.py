"""可选的 HTTP 入口 + 内置前端（需要先装 fastapi + uvicorn）。

    pip install fastapi "uvicorn[standard]"
    export  LLM_PROVIDER=mock      # 或用 .env
    python -m app.server           # http://localhost:8000/

仅做薄封装：鉴权、流式（SSE）与前端契约放这里，**业务逻辑一律在 Agent 层**。
前端是零构建的单页应用，直接由本服务托管在 /static。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import library, studio
from app.agents import get_task_state, reset, resume, run
from app.config import get_settings
from app.history import (
    HistoryError,
    append_chat_message,
    db_path as history_db_path,
    get_chat_thread,
    get_translation,
    list_threads,
    list_translations,
    save_generated_image,
    save_translation,
)
from app.general import (
    GeneralError,
    chat as general_chat,
    generate_image,
    image_chat,
    image_to_text,
)
from app.parsing.attachments import MAX_UPLOAD_BYTES, extract_attachment
from app.parsing.ocr import _tessdata_dir, _tesseract_cmd
from app.translate import (
    LLMTranslator,
    render_markdown as render_zh_markdown,
    translate_paragraphs,
)
from app.utils import setup_logging
from app.zh2en import LLMBackend, _safe_error, render_markdown, translate_document

setup_logging()
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Research Assistant Agent API", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20_000)
    thread_id: str = Field("default", max_length=200)
    user_id: str = Field("anonymous", max_length=200)
    attachment: str = Field("", max_length=20_000, description="已解析的附件文本；与 query 分开，避免污染路由")


class GeneralMessage(BaseModel):
    role: str = Field("user", max_length=20)
    content: str = Field(..., min_length=1, max_length=20_000)


class GeneralChatRequest(BaseModel):
    messages: list[GeneralMessage] = Field(..., min_length=1, max_length=40)
    system: str = Field("", max_length=4_000)
    temperature: float = Field(0.3, ge=0, le=2)
    thread_id: str = Field("", max_length=200, description="可选；传了就把问答写入会话历史")
    user_id: str = Field("anonymous", max_length=200)


class ImageGenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2_000)
    size: str = Field("1024*1024", max_length=20)
    n: int = Field(1, ge=1, le=4)
    negative_prompt: str = Field("", max_length=1_000)


class ResumeRequest(BaseModel):
    graph_thread: str = Field(..., description="run() 返回的图线程 ID（每轮对话一个）")
    decision: str = Field("approved", description="approved | rejected")
    feedback: str = Field("", max_length=20_000)


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200_000)
    direction: str = Field("zh2en", pattern="^(zh2en|en2zh)$")
    max_chars: int = Field(3000, ge=80, le=20_000)
    max_terms: int = Field(30, ge=0, le=100)
    glossary: dict[str, str] = Field(default_factory=dict)


class PaperQARequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    paper_ids: list[str] = Field(default_factory=list, max_length=50)
    top_k: int = Field(6, ge=1, le=20)


class PaperTranslateRequest(BaseModel):
    block_ids: list[str] = Field(default_factory=list, max_length=200)
    page: int | None = Field(None, ge=1)
    limit: int = Field(16, ge=1, le=40)
    force: bool = False


class SelectionTranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=6000)


class RenamePaperRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)


class ReplacePaperRequest(BaseModel):
    old_id: str = Field(..., min_length=1, max_length=200)


class ArxivDownloadRequest(BaseModel):
    arxiv_id: str = Field("", max_length=40)
    query: str = Field("", max_length=300)
    max_results: int = Field(5, ge=1, le=10)
    download: bool = True


class GraphRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list, max_length=100)
    topic: str = Field("", max_length=300)


class RecommendRequest(BaseModel):
    days: int = Field(14, ge=1, le=90)
    top: int = Field(5, ge=1, le=20)
    include_seen: bool = False
    use_github: bool = False


class FeedbackRequest(BaseModel):
    arxiv_id: str = Field(..., min_length=1, max_length=40)
    action: str = Field(..., pattern="^(like|dislike)$")


class FrameworkRequest(BaseModel):
    idea: Any
    paper_ids: list[str] = Field(default_factory=list, max_length=100)
    refs: list[Any] | None = None


class DiagramRequest(BaseModel):
    description: Any
    format: str = Field("mermaid", max_length=20)
    title: str = Field("", max_length=200)


@lru_cache(maxsize=1)
def _translator() -> LLMBackend:
    """复用 Provider 连接；MockLLM 会在构造时抛错，接口据此拒绝伪造译文。"""
    return LLMBackend()


@lru_cache(maxsize=1)
def _en2zh_translator() -> LLMTranslator:
    """英译中复用同一 Provider；Mock 模式下同样拒绝伪造。"""
    return LLMTranslator()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "agent-api"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    """前端只读配置摘要：**绝不返回密钥**。"""
    s = get_settings()
    tessdata = _tessdata_dir()
    langs = sorted(p.stem for p in Path(tessdata).glob("*.traineddata")) if tessdata else []
    return {
        "service": "agent-api",
        "version": app.version,
        "model": {
            "provider": s.LLM_PROVIDER,
            "name": s.LLM_MODEL,
            "mock": s.use_mock_llm,
            "key_set": bool(s.LLM_API_KEY),
            "base_url": s.LLM_BASE_URL,
            "vision_model": s.VISION_MODEL,
            "image_model": s.IMAGE_MODEL,
        },
        "ocr": {
            "lang": s.OCR_LANG,
            "tesseract": _tesseract_cmd(),
            "tessdata": tessdata,
            "languages": langs[:30],
        },
        "agent": {"hitl": s.AGENT_HITL, "max_revision": s.AGENT_MAX_REVISION},
        "history": {"db": str(history_db_path())},
        "library": library.library_stats(),
    }


# ══════════════════════════ 文献库 ══════════════════════════

def _lib_error(exc: Exception, status: int = 400) -> HTTPException:
    return HTTPException(status, detail=str(exc))


@app.on_event("startup")
def _startup_library() -> None:
    try:
        library.resume_incomplete()
    except Exception as exc:  # noqa: BLE001
        log.warning("library resume failed: %s", _safe_error(exc))
    hours = int(get_settings().RECOMMEND_INTERVAL_HOURS or 0)
    if hours > 0:
        threading.Thread(target=_recommend_loop, args=(hours,), daemon=True,
                         name="recommend-refresh").start()


def _recommend_loop(hours: int) -> None:
    while True:
        time.sleep(max(1, hours) * 3600)
        try:
            studio.recommend_refresh(days=14, top=5)
        except Exception as exc:  # noqa: BLE001
            log.warning("scheduled recommend refresh failed: %s", _safe_error(exc))


@app.get("/api/library/stats")
def library_stats_api() -> dict[str, Any]:
    return library.library_stats()


@app.get("/api/papers")
def papers_list(
    query: str = Query("", max_length=200),
    status: str = Query("", max_length=20),
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    return {"items": library.list_papers(query=query, status=status, limit=limit),
            "stats": library.library_stats()}


@app.post("/api/papers/upload")
def papers_upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(400, detail="没有收到文件")
    if len(files) > 20:
        raise HTTPException(400, detail="一次最多上传 20 个 PDF")
    items: list[dict[str, Any]] = []
    for f in files:
        name = Path(f.filename or "paper.pdf").name
        try:
            data = f.file.read(library.MAX_PDF_BYTES + 1)
            items.append({"filename": name, **library.create_paper(name, data)})
        except library.LibraryError as exc:
            items.append({"filename": name, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            log.error("paper upload failed: %s", _safe_error(exc))
            items.append({"filename": name, "error": f"上传失败：{_safe_error(exc)}"})
    return {"items": items}


@app.get("/api/papers/{paper_id}")
def paper_get(paper_id: str) -> dict[str, Any]:
    item = library.get_paper(paper_id)
    if item is None:
        raise HTTPException(404, detail="文献不存在")
    return item


@app.patch("/api/papers/{paper_id}")
def paper_rename(paper_id: str, req: RenamePaperRequest) -> dict[str, Any]:
    try:
        return library.rename_paper(paper_id, req.title)
    except library.LibraryError as exc:
        raise _lib_error(exc) from None


@app.delete("/api/papers/{paper_id}")
def paper_delete(paper_id: str) -> dict[str, Any]:
    if not library.delete_paper(paper_id):
        raise HTTPException(404, detail="文献不存在")
    return {"deleted": paper_id}


@app.post("/api/papers/{paper_id}/retry")
def paper_retry(paper_id: str) -> dict[str, Any]:
    try:
        return library.retry_paper(paper_id)
    except library.LibraryError as exc:
        raise _lib_error(exc) from None


@app.post("/api/papers/{paper_id}/replace")
def paper_replace(paper_id: str, req: ReplacePaperRequest) -> dict[str, Any]:
    try:
        return library.replace_with(paper_id, req.old_id)
    except library.LibraryError as exc:
        raise _lib_error(exc) from None


@app.get("/api/papers/{paper_id}/file")
def paper_file(paper_id: str) -> FileResponse:
    try:
        path = library.file_path(paper_id)
    except library.LibraryError as exc:
        raise HTTPException(404, detail=str(exc)) from None
    # inline：iframe 里直接渲染；默认 attachment 会让每次换页都弹下载
    return FileResponse(path, media_type="application/pdf", filename=path.name,
                        content_disposition_type="inline")


@app.get("/api/papers/{paper_id}/asset/{name}")
def paper_asset(paper_id: str, name: str) -> FileResponse:
    try:
        path = library.asset_path(paper_id, name)
    except library.LibraryError as exc:
        raise HTTPException(404, detail=str(exc)) from None
    return FileResponse(path)


@app.get("/api/papers/{paper_id}/reader")
def paper_reader(paper_id: str) -> dict[str, Any]:
    try:
        return library.reader(paper_id)
    except library.LibraryError as exc:
        raise _lib_error(exc, 409) from None


@app.post("/api/papers/{paper_id}/translate")
def paper_translate(paper_id: str, req: PaperTranslateRequest) -> dict[str, Any]:
    try:
        return library.translate_blocks(
            paper_id, block_ids=req.block_ids, page=req.page,
            limit=req.limit, force=req.force,
        )
    except library.LibraryError as exc:
        raise _lib_error(exc, 409) from None


@app.post("/api/papers/{paper_id}/translate-selection")
def paper_translate_selection(paper_id: str, req: SelectionTranslateRequest) -> dict[str, Any]:
    try:
        return library.translate_selection(paper_id, req.text)
    except library.LibraryError as exc:
        raise _lib_error(exc, 409) from None


@app.get("/api/papers/{paper_id}/compare/{other_id}")
def paper_compare(paper_id: str, other_id: str) -> dict[str, Any]:
    try:
        return library.compare_versions(paper_id, other_id)
    except library.LibraryError as exc:
        raise _lib_error(exc, 409) from None


@app.get("/api/papers/{paper_id}/markdown")
def paper_markdown(paper_id: str, lang: str = Query("en", pattern="^(en|zh|bi|bilingual)$")) -> dict[str, Any]:
    try:
        return {"markdown": library.markdown(paper_id, lang=lang), "lang": lang}
    except library.LibraryError as exc:
        raise _lib_error(exc, 409) from None


@app.post("/api/qa")
def paper_qa(req: PaperQARequest) -> dict[str, Any]:
    try:
        result = library.answer(req.query, paper_ids=req.paper_ids, top_k=req.top_k)
    except library.LibraryError as exc:
        raise _lib_error(exc, 409) from None
    except Exception as exc:  # noqa: BLE001
        log.error("paper qa failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"文献问答失败：{_safe_error(exc)}") from None
    return result


@app.post("/api/papers/{paper_id}/ask")
def paper_ask(paper_id: str, req: PaperQARequest) -> dict[str, Any]:
    try:
        return library.answer(req.query, paper_ids=[paper_id], top_k=req.top_k)
    except library.LibraryError as exc:
        raise _lib_error(exc, 409) from None


@app.post("/api/papers/download")
def paper_download(req: ArxivDownloadRequest) -> dict[str, Any]:
    try:
        if req.arxiv_id.strip():
            return studio.download_arxiv(req.arxiv_id.strip())
        if not req.query.strip():
            raise studio.StudioError("请提供 arxiv_id 或论文标题 query")
        return studio.search_arxiv(req.query, max_results=req.max_results, download=req.download)
    except studio.StudioError as exc:
        raise _lib_error(exc) from None


@app.post("/api/papers/{paper_id}/references/download")
def paper_refs_download(paper_id: str, limit: int = Query(5, ge=1, le=20)) -> dict[str, Any]:
    try:
        return studio.download_references(paper_id, limit=limit)
    except library.LibraryError as exc:
        raise _lib_error(exc, 409) from None
    except studio.StudioError as exc:
        raise _lib_error(exc) from None


# ── 图谱 / 综述 / Future Work ──

@app.post("/api/graph")
def graph_build(req: GraphRequest) -> dict[str, Any]:
    try:
        return studio.build_graph(req.paper_ids)
    except studio.StudioError as exc:
        raise _lib_error(exc) from None


@app.post("/api/graph/survey")
def graph_survey_api(req: GraphRequest) -> dict[str, Any]:
    try:
        return studio.survey(req.paper_ids, topic=req.topic)
    except studio.StudioError as exc:
        raise _lib_error(exc) from None


@app.post("/api/graph/future-work")
def graph_future_api(req: GraphRequest) -> dict[str, Any]:
    try:
        return studio.future_work(req.paper_ids)
    except studio.StudioError as exc:
        raise _lib_error(exc) from None


# ── 推荐 ──

@app.get("/api/recommend")
def recommend_latest() -> dict[str, Any]:
    path = studio.REC_STATE / "latest.json"
    try:
        return {"latest": json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return {"latest": None}


@app.post("/api/recommend/refresh")
def recommend_refresh(req: RecommendRequest) -> dict[str, Any]:
    try:
        return studio.recommend_refresh(
            days=req.days, top=req.top, include_seen=req.include_seen,
            use_github=req.use_github,
        )
    except studio.StudioError as exc:
        raise _lib_error(exc) from None


@app.post("/api/recommend/feedback")
def recommend_feedback(req: FeedbackRequest) -> dict[str, Any]:
    try:
        return studio.recommend_feedback(req.arxiv_id, req.action)
    except studio.StudioError as exc:
        raise _lib_error(exc) from None


# ── 写作助手 ──

@app.post("/api/write/framework")
def write_framework(req: FrameworkRequest, request: Request) -> dict[str, Any]:
    try:
        return studio.framework(
            req.idea, paper_ids=req.paper_ids, refs=req.refs,
            base_url=str(request.base_url),
        )
    except (studio.StudioError, ValueError) as exc:
        raise _lib_error(exc) from None


@app.post("/api/write/diagram")
def write_diagram(req: DiagramRequest) -> dict[str, Any]:
    try:
        return studio.diagram(req.description, req.format, req.title)
    except studio.StudioError as exc:
        raise _lib_error(exc) from None


@app.post("/api/write/chart")
def write_chart(
    file: UploadFile = File(...), title: str = Form(""), figure_no: int = Form(1)
) -> dict[str, Any]:
    data = file.file.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise HTTPException(400, detail="CSV 超过 2 MB 上限")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("gb18030", errors="replace")
    try:
        return studio.chart(text, title=title, figure_no=figure_no)
    except studio.StudioError as exc:
        raise _lib_error(exc) from None


@app.post("/api/parse")
def parse_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """上传 PDF / DOCX / 图片 / 文本，返回可送入对话或翻译的文本。

    同步读文件：FastAPI 会把整个函数放进线程池，OCR/PDF 解析不阻塞事件循环。
    """
    name = Path(file.filename or "upload").name
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    try:
        return extract_attachment(name, data)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        log.error("attachment parse failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"附件解析失败：{_safe_error(exc)}") from None


def _record_general(thread_id: str, user_id: str, query: str, result: dict[str, Any]) -> None:
    """通用问答留档。

    老页面/第三方调用没传 thread_id 时退回到 ``general-<user_id>``，
    避免“请求成功但历史里什么都没有”；新页面会带自己的会话 ID。

    写库失败只告警，不影响回答，也不在前端假装“已记录”。
    """
    thread_id = str(thread_id or "").strip() or f"general-{user_id or 'anonymous'}"[:200]
    try:
        append_chat_message(thread_id, "user", query, user_id=user_id, kind="general")
        append_chat_message(
            thread_id, "assistant", str(result.get("answer") or ""), kind="general",
            meta={"module": "general", "model": result.get("model")},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("general chat record failed: %s", _safe_error(exc))


@app.post("/api/general/chat")
def general_chat_api(req: GeneralChatRequest) -> dict[str, Any]:
    """通用大模型问答：独立于 Agent，不检索、不附引用。"""
    try:
        result = general_chat(
            [m.model_dump() for m in req.messages],
            system=req.system,
            temperature=req.temperature,
        )
    except GeneralError as exc:
        raise HTTPException(400, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        log.error("general chat failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"通用问答失败：{_safe_error(exc)}") from None
    query = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    _record_general(req.thread_id, req.user_id, query, result)
    return result


@app.post("/api/general/image")
def general_image_api(file: UploadFile = File(...)) -> dict[str, Any]:
    """图片转文字：只接受图片，逻辑与通用的 OCR 流水线一致。"""
    name = Path(file.filename or "image").name
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    try:
        return image_to_text(name, data)
    except GeneralError as exc:
        raise HTTPException(400, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        log.error("general image failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"图片识别失败：{_safe_error(exc)}") from None


@app.post("/api/general/image/chat")
def general_image_chat_api(
    file: UploadFile = File(...), question: str = Form(""), thread_id: str = Form("")
) -> dict[str, Any]:
    """图片理解：图片 + 问题交给多模态模型（默认 qwen3-vl-plus）。"""
    name = Path(file.filename or "image").name
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    try:
        result = image_chat(name, data, question=question)
    except GeneralError as exc:
        raise HTTPException(400, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        log.error("general vision failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"图片理解失败：{_safe_error(exc)}") from None
    _record_general(thread_id, "", str(question or "").strip() or "图片理解", result)
    return result


@app.post("/api/general/image/generate")
def general_image_generate_api(req: ImageGenRequest) -> dict[str, Any]:
    """文生图：通义万相异步任务，结果保存到 static/generated。"""
    started = time.perf_counter()
    try:
        result = generate_image(
            req.prompt,
            size=req.size,
            n=req.n,
            negative_prompt=req.negative_prompt,
        )
    except GeneralError as exc:
        raise HTTPException(400, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        log.error("general image generation failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"图片生成失败：{_safe_error(exc)}") from None
    try:
        save_generated_image(
            prompt=req.prompt,
            images=result.get("images") or [],
            model=str(result.get("model") or ""),
            size=req.size,
            negative_prompt=req.negative_prompt,
            actual_prompt=str(result.get("actual_prompt") or ""),
            task_id=str(result.get("task_id") or ""),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        # 记录失败不影响图片返回，只有日志；绝不伪造“已记录”。
        log.warning("generated image record failed: %s", _safe_error(exc))
    return result


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    try:
        return run(req.query, thread_id=req.thread_id, user_id=req.user_id,
                   attachment=req.attachment)
    except Exception as exc:  # noqa: BLE001
        log.error("chat failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"Agent 执行失败：{_safe_error(exc)}") from None


@app.get("/task/{thread_id}")
def task(thread_id: str) -> dict[str, Any]:
    """任务中间态查询：前端据此渲染 DAG 进度条。"""
    return get_task_state(thread_id)


@app.post("/resume")
def resume_task(req: ResumeRequest) -> dict[str, Any]:
    """HITL：人工确认/驳回后恢复被中断的执行。"""
    if req.decision not in ("approved", "rejected"):
        raise HTTPException(400, detail="decision 只能是 approved 或 rejected")
    try:
        return resume(req.graph_thread, decision=req.decision, feedback=req.feedback)
    except Exception as exc:  # noqa: BLE001
        log.error("resume failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"恢复执行失败：{_safe_error(exc)}") from None


@app.delete("/thread/{thread_id}")
def delete_thread(thread_id: str) -> dict[str, Any]:
    reset(thread_id)
    return {"deleted": thread_id}


@app.post("/api/translate")
def translate(req: TranslateRequest) -> dict[str, Any]:
    """学术翻译：zh2en 抽术语后逐段翻译；en2zh 先保公式再按句翻译。未配置时明确失败，不伪造。"""
    if len(req.glossary) > 200:
        raise HTTPException(400, detail="术语表最多 200 条")
    glossary = {
        src.strip()[:200]: dst.strip()[:200]
        for src, dst in req.glossary.items()
        if src.strip() and dst.strip()
    }
    started = time.perf_counter()
    try:
        if req.direction == "en2zh":
            # 术语表在英译中里是 {English: 中文}；片段上限按短句控制在模型输出长度内。
            raw = translate_paragraphs(
                req.text,
                _en2zh_translator(),
                glossary=glossary,
                max_unit_chars=min(max(80, req.max_chars), 600),
            )
            pairs = raw.get("pairs", [])
            translation = "\n\n".join(p.get("zh", "") for p in pairs).strip()
            result = {
                "translation": translation,
                "glossary": [
                    {"zh": zh, "en": en, "note": "自定义术语"}
                    for en, zh in glossary.items()
                ],
                "pairs": pairs,
                "issues": [],
                "stats": {
                    "paragraphs": len(pairs),
                    "terms": len(glossary),
                    "source_chars": len(req.text),
                    "translation_chars": len(translation),
                },
            }
            markdown = render_zh_markdown(result)
        else:
            translator = _translator()
            result = translate_document(
                req.text,
                translator,
                extractor=translator,
                glossary=glossary,
                max_chars=req.max_chars,
                max_terms=req.max_terms,
            )
            markdown = render_markdown(result)
    except Exception as exc:  # noqa: BLE001
        log.error("translate failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"翻译失败：{_safe_error(exc)}") from None

    latency_ms = round((time.perf_counter() - started) * 1000)
    history_id: int | None = None
    try:
        saved = save_translation(
            direction=req.direction,
            source_text=req.text,
            translation=result["translation"],
            glossary=result.get("glossary") or [],
            issues=result.get("issues") or [],
            stats=result.get("stats") or {},
            model=get_settings().LLM_MODEL,
            latency_ms=latency_ms,
        )
        history_id = saved.id
    except Exception as exc:  # noqa: BLE001
        # 落库失败不影响译文，只在校验面板提示，绝不伪造“已保存”。
        log.warning("history save failed: %s", _safe_error(exc))
        result["issues"] = [
            *(result.get("issues") or []),
            f"历史记录保存失败：{_safe_error(exc)}",
        ]
    return {
        **result,
        "markdown": markdown,
        "direction": req.direction,
        "model": get_settings().LLM_MODEL,
        "latency_ms": latency_ms,
        "history_id": history_id,
    }


@app.get("/api/history/threads")
def history_threads_api(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """会话列表：标题、消息数、执行轮数、最后一条预览，按最近活动倒序。"""
    try:
        items = list_threads(limit=limit, offset=offset)
    except Exception as exc:  # noqa: BLE001
        log.error("thread list failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"会话读取失败：{_safe_error(exc)}") from None
    return {"items": items, "limit": limit, "offset": offset}


@app.get("/api/history/threads/{thread_id}")
def history_thread_api(
    thread_id: str,
    message_limit: int = Query(200, ge=1, le=1000),
    run_limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """单个会话的完整记录：消息 + 每轮路由/计划/答案/校验/证据/子任务。"""
    try:
        item = get_chat_thread(thread_id, message_limit=message_limit, run_limit=run_limit)
    except Exception as exc:  # noqa: BLE001
        log.error("thread get failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"会话读取失败：{_safe_error(exc)}") from None
    if item is None:
        raise HTTPException(404, detail="会话不存在")
    return item


@app.get("/api/history/translations")
def history_list_api(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    direction: str = Query("", pattern="^(|zh2en|en2zh)$"),
) -> dict[str, Any]:
    """翻译历史摘要（长文只返回预览），按时间倒序。"""
    try:
        items = list_translations(limit=limit, offset=offset, direction=direction)
    except HistoryError as exc:
        raise HTTPException(400, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        log.error("history list failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"历史读取失败：{_safe_error(exc)}") from None
    return {"items": items, "limit": limit, "offset": offset}


@app.get("/api/history/translations/{record_id}")
def history_get_api(record_id: int) -> dict[str, Any]:
    """单条翻译记录全文 + 术语表。"""
    try:
        item = get_translation(record_id)
    except Exception as exc:  # noqa: BLE001
        log.error("history get failed: %s", _safe_error(exc))
        raise HTTPException(500, detail=f"历史读取失败：{_safe_error(exc)}") from None
    if item is None:
        raise HTTPException(404, detail="历史记录不存在")
    return item


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
