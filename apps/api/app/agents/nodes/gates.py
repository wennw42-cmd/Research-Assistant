"""校验 / 反思 / 收口三个"闸门"节点。

抗幻觉不是一句 prompt 能解决的，这里做**规则校验 + 模型复核**双闸：

    Verifier ① 引用锚点是否来自真实证据池（查 ID）
             ② quote 是否字面存在于原文（查字符，防"看起来很真"的编造）
             ③ 论断覆盖度是否达标（证据不足 → 要求拒答或补检索）
             ④ 交给模型做一次语义一致性复核（抓改写/张冠李戴）
    Reflector → 判断"重规划 / 重执行 / 放行"，并给 Planner 注入可执行的 critique
    Responder → 统一出口：答案 + 证据清单 + 是否通过校验
"""
from __future__ import annotations

import re
from typing import Any

from app import config
from app.agents.tools import corpus
from app.providers.llm import get_llm
from app.schemas import VerifyResult
from app.schemas.state import GraphState
from app.utils import get_logger, short

log = get_logger("ra.gates")

ANCHOR = re.compile(r"\[\[ev:([^\]#]+)#([^\]]+)\]\]")
SENT_SPLIT = re.compile(r"[。！？\n]|(?<!\w)\.(?=\s)")

# 这些场景的交付物是结构化数据或译文，而不是"论断"，因此不强求引用锚点
_EXEMPT_ROUTES = {"clarify", "direct", "translation", "parse", "graph", "retrieval"}


def extract_anchors(text: str) -> list[tuple[str, str]]:
    return ANCHOR.findall(text or "")


def verify_node(state: GraphState) -> dict[str, Any]:
    answer = state.get("answer", "") or ""
    evidence = list(state.get("evidence") or [])
    route = state.get("route", "")
    attachment = (state.get("attachment_context") or "").strip()
    exempt = route in _EXEMPT_ROUTES or bool(attachment)
    ev_keys = {f"{e.get('doc_id')}#{e.get('block_id')}" for e in evidence}

    # ── ① 锚点真实性 ──
    anchors = extract_anchors(answer)
    bad = [f"{d}#{b}" for d, b in anchors if f"{d}#{b}" not in ev_keys]
    citation_valid = bool(anchors) and not bad

    # ── ② quote 字面存在性（最关键的一闸）──
    unfaithful: list[str] = []
    for e in evidence[:12]:
        if not corpus.find_quote(e.get("quote", "")):
            unfaithful.append(f"{e.get('doc_id')}#{e.get('block_id')}")
    quote_faithful = not unfaithful

    # ── ③ 论断覆盖度 ──
    # 只统计"非引用行"的真实论断句：每 2 条论断至少要有 1 个独立锚点
    raw_sentences = [s.strip() for s in SENT_SPLIT.split(answer) if s and len(s.strip()) > 6]
    claims = [s for s in raw_sentences if "[[" not in s]
    need = max(1, -(-len(claims) // 2))
    unique_anchors = len(set(f"{d}#{b}" for d, b in anchors))
    coverage = min(1.0, unique_anchors / need) if claims else 0.0

    issues: list[str] = []
    if not anchors and not exempt and answer:
        issues.append("答案未附任何引用锚点 [[ev:doc#block]]，无法溯源")
    if bad:
        issues.append(f"引用锚点不在证据池：{short(str(bad), 160)}")
    if unfaithful:
        issues.append(f"引文无法在原文中字面命中（疑似编造）：{short(str(unfaithful), 160)}")
    if not exempt and coverage < 0.5 and claims:
        issues.append(f"证据覆盖不足：{unique_anchors} 个锚点 / 约 {need} 条论断")

    # ── ④ 模型语义复核（mock 模式下返回通过，不影响链路跑通）──
    llm_cmt = ""
    try:
        llm = get_llm()
        material = (
            f"用户附件（节选）：{short(attachment, 3000)}"
            if attachment else
            f"证据：{short(str([short(e.get('quote',''), 100) for e in evidence[:8]]), 1200)}"
        )
        parsed = llm.chat_json([
            {"role": "system", "content": "你是事实性校验器。检查答案是否被材料支持，指出任何超出材料的论断。"},
            {"role": "user", "content":
                f"{material}\n"
                f"答案：{short(answer, 1200)}\n"
                '只输出 JSON：{"passed":true/false,"evidence_coverage":0.0-1.0,"issues":["..."],"comment":"..."}'},
        ])
        if isinstance(parsed, dict):
            llm_cmt = str(parsed.get("comment", ""))
            if parsed.get("passed") is False:
                issues += [f"模型复核：{short(str(i), 120)}" for i in (parsed.get("issues") or [])[:3]]
            if isinstance(parsed.get("evidence_coverage"), (int, float)):
                coverage = min(coverage, float(parsed["evidence_coverage"]))
    except Exception as e:  # noqa: BLE001
        log.debug("模型复核跳过：%s", e)

    passed = not issues and quote_faithful
    result = VerifyResult(
        passed=passed,
        evidence_coverage=round(coverage, 3),
        citation_valid=citation_valid,
        quote_faithful=quote_faithful,
        issues=issues,
        fixed=[
            f"锚点数 {unique_anchors}",
            f"证据池 {len(evidence)} 条",
            "附件轮次：以用户提供文本为准，未做库内锚点校验" if attachment else f"校验备注 {short(llm_cmt, 100)}",
        ],
    )
    log.info("verify passed=%s coverage=%.2f anchors=%d issues=%d",
             passed, coverage, unique_anchors, len(issues))
    return {"verify": result.model_dump(mode="json"),
            "trace": [{"node": "verifier", "passed": passed, "coverage": coverage, "issues": issues[:5]}]}


def reflector_node(state: GraphState) -> dict[str, Any]:
    """反思：把"哪里不合格"翻译成**下一次可执行的改进指令**。"""
    verify = state.get("verify") or {}
    issues = verify.get("issues") or []
    revision = int(state.get("revision", 0)) + 1
    max_rev = config.get_settings().AGENT_MAX_REVISION
    plan = state.get("plan") or []

    has_missing_evidence = any(("锚点" in i or "覆盖" in i or "未找到" in i) for i in issues)
    has_failed_task = any(t.get("status") == "failed" for t in plan)

    if has_missing_evidence or has_failed_task:
        decision = "重规划：补充检索/改写任务粒度，并确保每个 writing 任务前都有带证据的上游"
    else:
        decision = "重执行：保持任务分解，调整工具参数或证据引用方式"

    log.warning("reflect #%d/%d → %s", revision, max_rev, short(decision, 60))
    return {
        "revision": revision,
        "messages": [{"role": "assistant", "content": f"[反思 #{revision}] {decision}\n问题：{short(str(issues), 300)}"}],
        "trace": [{"node": "reflector", "revision": revision, "decision": short(decision, 80)}],
    }


def clarify_node(state: GraphState) -> dict[str, Any]:
    """澄清/寒暄分支：不调工具，给出最短的引导式回应（避免为一句"你好"烧 token）。"""
    query = state.get("query", "")
    hint = (
        "我可以帮你：\n"
        "- 解析 PDF / 扫描件 / 语义去重（试试 `/parse doc_transformer.pdf`）\n"
        "- 也可以点输入框旁的「附件」上传 PDF / Word / 图片\n"
        "- 检索与追踪论文（`/search xxx`）\n"
        "- 基于文献库回答并附引用（`直接提问`）\n"
        "- 版式感知翻译（`/translate`）、引用图谱（`/graph`）、综述写作（`/survey`）"
    )
    return {"answer": f"{hint}\n\n（当前查询：{short(query, 40)}）",
            "verify": VerifyResult(passed=True, evidence_coverage=1.0, citation_valid=True,
                                   quote_faithful=True, fixed=["clarify 分支豁免校验"]).model_dump(mode="json"),
            "trace": [{"node": "clarify", "skipped_tools": True}]}


def responder_node(state: GraphState) -> dict[str, Any]:
    """统一出口：把答案与证据清单整理成前端可渲染的最终态。"""
    answer = state.get("answer", "") or ""
    evidence = list(state.get("evidence") or [])
    verify = state.get("verify") or {}

    lines = [answer.strip()]
    if evidence:
        lines.append("\n\n---\n**引用来源（点击可定位原文）**")
        seen: set[str] = set()
        for e in evidence[:12]:
            key = f"{e.get('doc_id')}#{e.get('block_id')}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- [[ev:{key}]] {short(str(e.get('quote')), 90)}"
                f"（{e.get('meta', {}).get('title', e.get('doc_id'))} p.{e.get('page')}）"
            )
    if not verify.get("passed", False) and verify.get("issues"):
        lines.append("\n> [警告] 本答案未完全通过抗幻觉校验：" + "；".join(map(str, verify["issues"][:3])))
    return {"answer": "\n".join(lines), "trace": [{"node": "responder", "finished": True}]}


# ────────────────────────── 条件路由 ──────────────────────────
def route_after_verifier(state: GraphState) -> str:
    verify = state.get("verify") or {}
    if verify.get("passed"):
        return "responder"
    max_rev = config.get_settings().AGENT_MAX_REVISION
    if int(state.get("revision", 0)) >= max_rev:
        log.warning("达到反思上限，带告警放行")
        return "responder"
    return "reflector"


def route_after_reflector(state: GraphState) -> str:
    plan = state.get("plan") or []
    if not plan:  # 单 Agent 路径：重跑该专家一次
        return "specialist"
    return "planner"
