"""图谱 / 推荐 / 写作 / 图表 / arXiv 下载的薄服务层。

算法都在既有模块里，这里只负责把**真实文献库**的数据喂给它们：
- ``citation_graph``：库内参考文献匹配后建图，不引入虚构节点；
- ``graph_survey`` / ``future_work``：从库内摘要、原文 Future Work 抽材料；
- ``recommend``：用文献库标题/摘要生成 arXiv 查询，反馈与历史落 JSONL；
- ``paper_draft`` / ``diagram`` / ``chart``：参考文献白名单来自库内真实条目。
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

from app import library
from app.chart import analyze as chart_analyze
from app.citation_audit import guess_title
from app.citation_graph import build_result
from app.diagram import generate as diagram_generate
from app.future_work import analyze as future_analyze
from app.graph_survey import generate as survey_generate
from app.paper_draft import generate as draft_generate
from app import recommend as rec

log = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent.parent
REC_STATE = APP_DIR / "data" / "recommend"
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "using",
    "based", "paper", "method", "model", "approach", "results", "study", "learning",
}


class StudioError(ValueError):
    pass


def _ready(paper_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    wanted = set(paper_ids or [])
    rows = library.list_papers(status="ready", limit=1000)
    return [p for p in rows if not wanted or p["id"] in wanted]


def _reader(pid: str) -> dict[str, Any]:
    return library.reader(pid)


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text or "")
            if w.lower() not in _STOP}


def _match_ref(ref_text: str, papers: list[dict[str, Any]]) -> str:
    low = (ref_text or "").lower()
    base, _ver = library._arxiv_from(ref_text)
    for p in papers:
        if base and p.get("base_id") == base:
            return p["id"]
    norm = library._norm_title(ref_text)
    for p in papers:
        title = library._norm_title(p.get("title") or "")
        if len(title) >= 20 and title in norm:
            return p["id"]
    rt = _tokens(ref_text)
    best, best_score = "", 0.0
    for p in papers:
        pt = _tokens(p.get("title") or "")
        if not pt or not rt:
            continue
        score = len(pt & rt) / max(1, len(pt))
        if score > best_score:
            best, best_score = p["id"], score
    return best if best_score >= 0.75 else ""


def graph_input(paper_ids: Iterable[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    papers = _ready(paper_ids)
    if not papers:
        raise StudioError("没有可用的已解析文献")
    data_papers: list[dict[str, Any]] = []
    matched = unmatched = 0
    for p in papers:
        refs: list[str] = []
        try:
            reader = _reader(p["id"])
        except library.LibraryError:
            reader = {}
        for ref in reader.get("references") or []:
            target = _match_ref(ref.get("text") or "", papers)
            if target and target != p["id"]:
                refs.append(target)
                matched += 1
            elif ref.get("text"):
                unmatched += 1
        data_papers.append({
            "id": p["id"], "title": p["title"], "year": p.get("year"),
            "abstract": p.get("abstract") or "", "authors": p.get("authors") or [],
            "keywords": p.get("keywords") or [], "arxiv_id": p.get("arxiv_id") or "",
            "references": list(dict.fromkeys(refs)),
        })
    meta = {"papers": len(papers), "matched_refs": matched, "unmatched_refs": unmatched}
    return {"papers": data_papers}, meta


def build_graph(paper_ids: Iterable[str] | None = None) -> dict[str, Any]:
    data, meta = graph_input(paper_ids)
    if len(data["papers"]) < 2:
        raise StudioError("至少需要 2 篇已解析文献才能建图")
    result = build_result(data, keep_external=False)
    result["library"] = meta
    return result


def survey(paper_ids: Iterable[str] | None = None, *, topic: str = "") -> dict[str, Any]:
    data, meta = graph_input(paper_ids)
    if not data["papers"]:
        raise StudioError("没有可用文献")
    result = survey_generate(data, topic.strip() or None, include_quotes=True)
    result["library"] = meta
    return result


def _section_text(reader: dict[str, Any], pattern: str) -> str:
    blocks = reader.get("blocks") or []
    parts: list[str] = []
    active: int | None = None
    for b in blocks:
        if b.get("type") == "heading":
            if active is not None:
                break
            if re.search(pattern, b.get("text") or "", re.I):
                active = int(b.get("level") or 1)
            continue
        if active is not None and b.get("type") == "paragraph":
            parts.append((b.get("text") or "").strip())
    return "\n".join(x for x in parts if x).strip()


def future_work(paper_ids: Iterable[str] | None = None) -> dict[str, Any]:
    papers = _ready(paper_ids)
    items: list[dict[str, Any]] = []
    for p in papers:
        try:
            reader = _reader(p["id"])
        except library.LibraryError:
            continue
        future = _section_text(reader, r"future\s*(work|direction|research)|未来工作|展望|后续工作")
        conclusion = _section_text(reader, r"conclusion|conclusions|结论|总结")
        if not future and not conclusion:
            continue
        items.append({
            "id": p["id"], "title": p["title"], "year": p.get("year"),
            "date": str(p.get("year") or ""), "abstract": p.get("abstract") or "",
            "future_work": future, "conclusion": conclusion,
        })
    if not items:
        raise StudioError("这些文献中没有找到 Future Work / Conclusion 段落")
    result = future_analyze({"papers": items})
    result["stats"] = {"papers": len(items), "with_future": sum(1 for x in items if x["future_work"])}
    return result


# ────────────────────────── 推荐 ──────────────────────────

def _docs() -> list[rec.LibraryDoc]:
    return [
        rec.LibraryDoc(
            doc_id=p["id"], title=p["title"], abstract=p.get("abstract") or "",
            tags=list(p.get("keywords") or []), year=p.get("year"),
            arxiv_id=p.get("arxiv_id") or "",
        )
        for p in library.list_papers(status="ready", limit=1000)
    ]


def recommend_refresh(*, days: int = 14, top: int = 5, include_seen: bool = False,
                      use_github: bool = False) -> dict[str, Any]:
    docs = _docs()
    days = max(1, min(90, int(days)))
    try:
        query = rec.build_arxiv_query(docs, days=days)
        papers = rec.fetch_arxiv(query, max_results=max(20, top * 4), days=days)
    except rec.FetchError as exc:
        raise StudioError(str(exc)) from None
    state = rec.State(REC_STATE)
    result = rec.run_cycle(
        docs, papers, state, top=max(1, min(20, int(top))), include_seen=include_seen,
        use_github=use_github, code_top=3, source_label="arxiv", query_label=query,
    )
    result["library_docs"] = len(docs)
    (state.dir / "latest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def recommend_feedback(arxiv_id: str, action: str) -> dict[str, Any]:
    arxiv_id = str(arxiv_id or "").strip()
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5})(?:v\d+)?", arxiv_id):
        raise StudioError("arXiv ID 格式不正确")
    if action not in ("like", "dislike"):
        raise StudioError("action 只能是 like 或 dislike")
    state = rec.State(REC_STATE)
    recorded = rec.record_feedback_from_history(
        state, rec.load_history(state), [(arxiv_id, action)]
    )
    return {"recorded": recorded, "profile": rec.feedback_summary(rec.build_model(rec.load_events(state)))}


# ────────────────────────── arXiv 下载 ──────────────────────────

def _paper_from_arxiv(arxiv_id: str) -> rec.Paper:
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5})(?:v\d+)?", arxiv_id or ""):
        raise StudioError("arXiv ID 格式不正确（示例：1706.03762v7）")
    import urllib.parse
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": arxiv_id, "max_results": 1}
    )
    try:
        raw = rec._http_get(url, timeout=30)
    except Exception as exc:  # noqa: BLE001
        raise StudioError(f"arXiv 元数据拉取失败：{exc}") from None
    papers = rec.parse_arxiv_feed(raw)
    if not papers:
        raise StudioError(f"arXiv 上找不到 {arxiv_id}")
    return papers[0]


def download_arxiv(arxiv_id: str) -> dict[str, Any]:
    paper = _paper_from_arxiv(arxiv_id)
    existing = library.find_by_arxiv(paper.arxiv_id.split("v")[0])
    if existing:
        # 同主干但版本不同 → 照常下载，让两版并存以便对比
        want = re.search(r"[vV](\d+)$", paper.arxiv_id)
        have = re.search(r"[vV](\d+)$", existing.get("version") or existing.get("arxiv_id") or "")
        if not (want and have) or want.group(1) == have.group(1):
            return {"created": False, "paper": existing, "arxiv": paper.arxiv_id,
                    "note": "库中已有该 arXiv 论文（同版本）"}
    try:
        pdf = rec._http_get(f"https://arxiv.org/pdf/{paper.arxiv_id}", timeout=120)
    except Exception as exc:  # noqa: BLE001
        raise StudioError(f"PDF 下载失败：{exc}") from None
    made = library.create_paper(f"{paper.arxiv_id}.pdf", pdf)
    meta = {"arxiv_title": paper.title, "arxiv_authors": paper.authors,
            "arxiv_categories": paper.categories, "arxiv_url": paper.url}
    _merge_meta(made["paper"]["id"], meta, paper)
    return {**made, "arxiv": paper.arxiv_id, "meta": meta}


def _merge_meta(pid: str, extra: dict[str, Any], paper: rec.Paper | None = None) -> None:
    row = library._fetch(pid)
    if row is None:
        return
    meta = library._json(row["meta_json"], {})
    meta.update(extra)
    fields: dict[str, Any] = {"meta_json": json.dumps(meta, ensure_ascii=False)}
    if paper is not None:
        # PDF 元数据里的年份可能是文件生成年份；arXiv 的提交时间更可信
        if paper.title:
            fields["title"] = paper.title[:300]
        if paper.authors:
            fields["authors_json"] = json.dumps(paper.authors, ensure_ascii=False)
        year = re.match(r"(\d{4})", paper.published or "")
        if year:
            fields["year"] = int(year.group(1))
    library._write_row(pid, **fields)


def search_arxiv(title: str, *, max_results: int = 5, download: bool = False) -> dict[str, Any]:
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    if len(title) < 4:
        raise StudioError("论文标题太短")
    query = f'ti:"{title[:300]}"'
    try:
        papers = rec.fetch_arxiv(query, max_results=max(1, min(10, max_results)), days=0)
    except rec.FetchError as exc:
        raise StudioError(str(exc)) from None
    candidates = [{
        "arxiv_id": p.arxiv_id, "title": p.title, "authors": p.authors,
        "published": p.published, "abstract": p.abstract[:500], "url": p.url,
    } for p in papers]
    downloaded = None
    if download and candidates:
        downloaded = download_arxiv(candidates[0]["arxiv_id"])
    return {"query": query, "candidates": candidates, "downloaded": downloaded}


def download_references(paper_id: str, *, limit: int = 5) -> dict[str, Any]:
    """把参考文献按标题回查 arXiv；只下载标题相似度足够高的结果，低价值条目跳过。"""
    data = library.reader(paper_id)
    refs = data.get("references") or []
    limit = max(1, min(20, int(limit)))
    results: list[dict[str, Any]] = []
    skipped = 0
    for ref in refs:
        if len([r for r in results if r.get("created") or r.get("existing")]) >= limit:
            break
        text = (ref.get("text") or "").strip()
        title = guess_title(text)
        if len(text) < 30 or len(title) < 12 or len(_tokens(title)) < 2:
            skipped += 1
            continue
        try:
            papers = rec.fetch_arxiv(f'ti:"{title[:240]}"', max_results=3, days=0)
        except rec.FetchError as exc:
            results.append({"title": title, "status": "error", "error": str(exc)})
            break
        best = None
        for p in papers:
            score = library._title_sim(title, p.title)
            if score >= 0.82:
                best = (p, score)
                break
        if not best:
            skipped += 1
            results.append({"title": title, "status": "not_found"})
            continue
        try:
            made = download_arxiv(best[0].arxiv_id)
            results.append({
                "title": best[0].title, "arxiv_id": best[0].arxiv_id,
                "score": round(best[1], 3), "status": "downloaded" if made["created"] else "existing",
                "paper_id": made["paper"]["id"], "created": made["created"],
            })
        except StudioError as exc:
            results.append({"title": title, "arxiv_id": best[0].arxiv_id,
                            "status": "error", "error": str(exc)})
    return {"source": paper_id, "results": results, "skipped_low_value": skipped,
            "attempted": len(refs)}


# ────────────────────────── 写作 / 图表 / 拓扑 ──────────────────────────

def _refs_from_library(paper_ids: Iterable[str] | None, base_url: str) -> list[dict[str, Any]]:
    out = []
    for p in _ready(paper_ids):
        url = f"https://arxiv.org/abs/{p['arxiv_id']}" if p.get("arxiv_id") else \
            f"{base_url.rstrip('/')}/api/papers/{p['id']}/file"
        out.append({
            "title": p["title"], "authors": p.get("authors") or [],
            "year": p.get("year"), "abstract": p.get("abstract") or "",
            "arxiv_id": p.get("arxiv_id") or "", "url": url, "source": "library",
        })
    return out


def framework(idea: Any, *, paper_ids: Iterable[str] | None = None,
              refs: Any = None, base_url: str = "http://localhost:8000") -> dict[str, Any]:
    items = refs if refs is not None else _refs_from_library(paper_ids, base_url)
    result = draft_generate(idea, {"retrieved_refs": items})
    result["source"] = "library" if refs is None else "input"
    return result


def diagram(description: Any, fmt: str = "mermaid", title: str = "") -> dict[str, Any]:
    try:
        return diagram_generate(description, fmt, title.strip() or None)
    except ValueError as exc:
        raise StudioError(str(exc)) from None


def chart(csv_text: str, *, title: str = "", figure_no: int = 1) -> dict[str, Any]:
    if len(csv_text.encode("utf-8")) > 2 * 1024 * 1024:
        raise StudioError("CSV 超过 2 MB 上限")
    try:
        rows = list(csv.DictReader(io.StringIO(csv_text)))
    except Exception as exc:  # noqa: BLE001
        raise StudioError(f"CSV 解析失败：{exc}") from None
    rows = [r for r in rows if any((v or "").strip() for v in r.values())]
    if not rows:
        raise StudioError("CSV 没有数据行")
    if len(rows) > 5000:
        raise StudioError("CSV 行数超过 5000，请先聚合")
    try:
        result = chart_analyze(rows, None, title=title.strip() or None, figure_no=int(figure_no))
    except ValueError as exc:
        raise StudioError(str(exc)) from None
    result["image_url"] = _render_chart(result.get("script") or "")
    return result


def _render_chart(script: str) -> str:
    """只在本地跑由 ``app.chart`` 生成的脚本；失败时保留脚本，不伪造图片。"""
    out_dir = APP_DIR / "static" / "generated" / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex[:12]
    work = out_dir / name
    work.mkdir()
    (work / "figure.py").write_text(script, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "figure.py", "figure.png"],
            cwd=str(work), capture_output=True, text=True, timeout=90,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        image = work / "figure.png"
        if proc.returncode == 0 and image.is_file():
            return f"/static/generated/charts/{name}/figure.png"
        log.warning("chart render failed: %s", (proc.stderr or "")[-500:])
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("chart render failed: %s", exc)
    return ""
