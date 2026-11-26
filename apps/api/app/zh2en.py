"""中文学术文本 → 国际期刊英文：术语统一、学术句式、公式/变量/引用逐字保留。

与 ``app.translate``（英译中）共用同一套占位符保护/校验/还原机制：
公式、代码、URL、DOI、数字引用、作者-年份引用先变成 ``⟦PHn⟧``，
模型只负责翻译自然语言；译文占位符丢失/重复/臆造会重试一次，仍失败则整段报错，
不会把半成品当译文输出。

生产路径：LLM 先抽领域术语（JSON），再按术语表逐片段翻译；没有 API Key 时
拒绝假装翻译，离线效果用 ``--demo``。

运行（在 apps/api 下）：
    python -m app.zh2en --demo
    python -m app.zh2en --file paper.md --out paper.en.md
    python -m app.zh2en --text "中文摘要……" --json
    python -m app.zh2en --selftest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from app.translate import (
    PlaceholderError,
    PlaceholderIssue,
    TranslationError,
    _PH_FIND_RE,
    _ph,
    _pid_from_match,
    _prepare_paragraph,
    check_placeholders,
    protect,
    restore,
    split_paragraphs,
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ZH_TERMINAL = "。！？!?"
_ZH_CLOSERS = "”’」』）)】]\"'"
_DEFAULT_NOTE = "按国际期刊通用译法处理；全篇统一。"
_SECRET_RE = re.compile(
    r"(?i)(?:\b(?:sk|pk|api|token|key)[-_][A-Za-z0-9_-]{8,}\b"
    r"|\bBearer\s+[A-Za-z0-9._-]{8,}\b"
    r"|\b(?:api[_-]?key|llm[_-]?api[_-]?key|access[_-]?token)\b\s*[=:]\s*[^\s,;]+)"
)


def _safe_error(exc: BaseException) -> str:
    """错误信息可能来自 Provider；统一抹掉疑似密钥，绝不把密钥写进日志/JSON。"""
    return _SECRET_RE.sub("***", str(exc))


# ══════════════════════════════ 数据结构 ══════════════════════════════
@dataclass(frozen=True)
class Term:
    zh: str
    en: str
    note: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"zh": self.zh, "en": self.en, "note": self.note}


@dataclass(frozen=True)
class Chunk:
    raw: str                         # 还原后的片段原文（演示 TM 的键）
    protected: str                   # 送模型的占位符文本
    placeholders: dict[int, str]     # 本片段实际出现的占位符
    context: str                     # 整段原文，供跨句消歧
    paragraph: int
    index: int


@dataclass(frozen=True)
class ZhParagraph:
    source: str
    chunks: tuple[Chunk, ...]
    paragraph: int


class TranslatorLike(Protocol):
    def translate(
        self,
        chunk: Chunk,
        glossary: Sequence[Term],
        repair: PlaceholderIssue | None = None,
    ) -> str:
        ...


class TermExtractor(Protocol):
    def extract(
        self,
        text: str,
        existing: Sequence[Term] = (),
        limit: int = 30,
    ) -> tuple[list[Term], list[str]]:
        ...


# ══════════════════════════════ 中文分段 / 切句 ══════════════════════════════
def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _flat(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _term_key(text: str) -> str:
    return _flat(text)


def split_zh_sentences(text: str) -> list[str]:
    """按中文句末标点切句，并把紧跟的引号/括号留在本句。"""
    text = str(text or "")
    out: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] in _ZH_TERMINAL:
            j = i + 1
            while j < n and (text[j] in _ZH_CLOSERS or text[j] in _ZH_TERMINAL):
                j += 1
            piece = text[start:j].strip()
            if piece:
                out.append(piece)
            start = j
            i = j
        else:
            i += 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def chunk_text(text: str, max_chars: int) -> list[str]:
    """只在句末切分；单句超长时宁可不切，也不把公式/术语从中间劈开。"""
    text = str(text or "").strip()
    if not text:
        return []
    limit = max(80, int(max_chars))
    if len(text) <= limit:
        return [text]
    pieces = split_zh_sentences(text)
    if len(pieces) <= 1:
        return [text]
    chunks: list[str] = []
    buf: list[str] = []
    for piece in pieces:
        candidate = "".join(buf) + piece
        if buf and len(candidate) > limit:
            chunks.append("".join(buf))
            buf = [piece]
        else:
            buf.append(piece)
    if buf:
        chunks.append("".join(buf))
    return chunks


def _has_translatable(protected: str) -> bool:
    plain = _PH_FIND_RE.sub(" ", protected).strip()
    return bool(re.search(r"[\u4e00-\u9fff]|[A-Za-z]{2,}", plain))


def _pids_in(text: str) -> list[int]:
    seen: list[int] = []
    for m in _PH_FIND_RE.finditer(text):
        pid = _pid_from_match(m)
        if pid not in seen:
            seen.append(pid)
    return seen


def _make_paragraphs(text: str, max_chars: int) -> list[ZhParagraph]:
    out: list[ZhParagraph] = []
    for p_idx, raw in enumerate(split_paragraphs(text), 1):
        clean, protected, mapping = _prepare_paragraph(raw)
        if not protected or not _has_translatable(protected):
            out.append(ZhParagraph(source=clean, chunks=(), paragraph=p_idx))
            continue
        chunks: list[Chunk] = []
        for c_idx, piece in enumerate(chunk_text(protected, max_chars), 1):
            pids = _pids_in(piece)
            ph = {pid: mapping[pid] for pid in pids}
            chunks.append(Chunk(
                raw=restore(piece, ph),
                protected=piece,
                placeholders=ph,
                context=clean,
                paragraph=p_idx,
                index=c_idx,
            ))
        out.append(ZhParagraph(source=clean, chunks=tuple(chunks), paragraph=p_idx))
    return out


# ══════════════════════════════ 术语表 ══════════════════════════════
def _coerce_terms(value: Any) -> list[Term]:
    """接受 [{"zh","en","note"}]、{"中文":"English"} 或 [["中文","English"]]。"""
    if value is None:
        return []
    if isinstance(value, Mapping):
        if "glossary" in value and isinstance(value["glossary"], (list, tuple, Mapping)):
            value = value["glossary"]
        elif (any(k in value for k in ("zh", "source", "中文"))
              and any(k in value for k in ("en", "target", "英文"))):
            value = [value]
    terms: list[Term] = []
    if isinstance(value, Mapping):
        for zh, en in value.items():
            note = ""
            if isinstance(en, Mapping):
                note = str(en.get("note") or "")
                en = en.get("en") or en.get("translation") or en.get("target")
            if not str(zh).strip() or not str(en or "").strip():
                raise ValueError(f"术语表条目缺少中文或英文：{zh!r}")
            terms.append(Term(str(zh).strip(), str(en).strip(), note.strip()))
        return terms
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for idx, item in enumerate(value, 1):
            if isinstance(item, Term):
                terms.append(item)
                continue
            if isinstance(item, Mapping):
                zh = item.get("zh") or item.get("source") or item.get("中文")
                en = item.get("en") or item.get("target") or item.get("英文")
                note = item.get("note") or item.get("说明") or ""
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                zh, en = item[0], item[1]
                note = item[2] if len(item) >= 3 else ""
            else:
                raise ValueError(f"术语表第 {idx} 条格式不支持：{item!r}")
            if not str(zh or "").strip() or not str(en or "").strip():
                raise ValueError(f"术语表第 {idx} 条缺少中文或英文")
            terms.append(Term(str(zh).strip(), str(en).strip(), str(note).strip()))
        return terms
    raise ValueError("术语表必须是 JSON 对象、数组或 {glossary: [...]}")


def _validate_terms(
    raw_terms: Sequence[Any],
    source_flat: str,
    *,
    existing: Sequence[Term] = (),
    limit: int = 30,
) -> tuple[list[Term], list[str]]:
    """只保留原文真实出现、非公式、非变量、非重复的候选术语。"""
    cap = max(0, int(limit))
    if cap == 0:
        return [], []
    terms: list[Term] = []
    issues: list[str] = []
    dropped = 0
    conflicts = 0
    by_key: dict[str, Term] = {_term_key(t.zh): t for t in existing}
    for item in raw_terms:
        if isinstance(item, Term):
            zh, en, note = item.zh, item.en, item.note
        elif isinstance(item, Mapping):
            zh = item.get("zh") or item.get("source") or item.get("中文") or ""
            en = item.get("en") or item.get("target") or item.get("英文") or ""
            note = item.get("note") or item.get("说明") or ""
        else:
            dropped += 1
            continue
        zh, en, note = _norm(zh), _norm(en), _norm(note)
        if not zh or not en:
            dropped += 1
            continue
        if len(zh) > 40 or len(en) > 100:
            dropped += 1
            continue
        if ("⟦" in zh or "⟧" in zh or "⟦" in en or "⟧" in en
                or re.search(r"\bPH\d+\b", zh + en)):
            dropped += 1
            continue
        if _has_cjk(en) or not re.search(r"[A-Za-z]", en):
            dropped += 1
            continue
        if not re.search(r"[\u4e00-\u9fffA-Za-z]", zh):
            dropped += 1
            continue
        if "$" in zh or "\\" in zh:
            dropped += 1
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", zh) and ("_" in zh or len(zh) <= 1):
            dropped += 1
            continue
        key = _term_key(zh)
        if key not in source_flat:
            dropped += 1
            continue
        prior = by_key.get(key)
        if prior is not None:
            if prior.en.strip().lower() != en.lower():
                conflicts += 1
            continue
        term = Term(zh, en, note[:200] or _DEFAULT_NOTE)
        by_key[key] = term
        terms.append(term)
        if len(terms) >= cap:
            break
    if dropped:
        issues.append(f"术语抽取丢弃 {dropped} 个无效/未在原文出现的候选")
    if conflicts:
        issues.append(f"术语译法冲突 {conflicts} 处：保留首次译法，未覆盖已有术语")
    return terms, issues


def _merge_terms(
    seed: Sequence[Term],
    extracted: Sequence[Term],
    source: str,
    max_terms: int,
) -> tuple[list[Term], list[str]]:
    merged: list[Term] = []
    by_key: dict[str, Term] = {}
    conflicts = 0
    empty = 0
    for term in [*seed, *extracted]:
        zh, en, note = _norm(term.zh), _norm(term.en), _norm(term.note)
        if not zh or not en:
            empty += 1
            continue
        key = _term_key(zh)
        if key in by_key:
            if by_key[key].en.lower() != en.lower():
                conflicts += 1
            continue
        clean = Term(zh, en, note[:200] or _DEFAULT_NOTE)
        by_key[key] = clean
        merged.append(clean)
    flat_source = _flat(source)
    merged.sort(key=lambda t: (
        flat_source.find(_term_key(t.zh)) if _term_key(t.zh) in flat_source else 10 ** 9,
        len(t.zh),
        t.zh,
    ))
    issues: list[str] = []
    if empty:
        issues.append(f"术语表丢弃 {empty} 个空白条目")
    if conflicts:
        issues.append(f"术语表存在 {conflicts} 处译法冲突：保留首次出现的译法")
    cap = max(0, int(max_terms))
    if len(merged) > cap:
        issues.append(f"术语数超过上限 {cap}，已截断 {len(merged) - cap} 条")
        merged = merged[:cap]
    return merged, issues


# ══════════════════════════════ 翻译片段 ══════════════════════════════
def _clean_model_output(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(?:译文|翻译|Translation|English)\s*[:：]\s*", "", text, flags=re.I)
    return text.strip()


def _translate_chunk(
    chunk: Chunk,
    translator: TranslatorLike,
    glossary: Sequence[Term],
) -> str:
    # 独立成句的公式/代码/纯引用不送模型，直接原样回填，避免“只含占位符”的片段被改写或漏掉。
    if not _has_translatable(chunk.protected):
        return chunk.raw
    repair: PlaceholderIssue | None = None
    last_error = "空译文"
    for _ in range(2):
        try:
            out = translator.translate(chunk, glossary, repair=repair)
        except Exception as exc:  # 网络抖动/临时限流也重试一次
            last_error = f"调用失败：{type(exc).__name__}: {_safe_error(exc)}"
            continue
        if not isinstance(out, str):
            last_error = "模型返回值不是字符串"
            continue
        out = out.strip()
        if not out:
            last_error = "空译文"
            continue
        issue = check_placeholders(out, chunk.placeholders)
        if issue.ok:
            try:
                return restore(out, chunk.placeholders)
            except PlaceholderError as exc:
                last_error = str(exc)
                continue
        last_error = issue.describe()
        repair = issue
    raise TranslationError(f"片段翻译失败（{last_error}）：{chunk.raw[:100]}")


def translate_document(
    text: str,
    translator: TranslatorLike,
    *,
    extractor: TermExtractor | None = None,
    glossary: Sequence[Term] | Mapping[str, str] | None = None,
    max_chars: int = 3000,
    max_terms: int = 30,
) -> dict[str, Any]:
    """中译英主入口；返回 translation/glossary/pairs/issues/stats。"""
    source = str(text or "").strip()
    max_chars = max(80, int(max_chars))
    max_terms = max(0, int(max_terms))
    seed_terms = _coerce_terms(glossary)
    if not source:
        return {
            "translation": "",
            "glossary": [],
            "pairs": [],
            "issues": [],
            "stats": {
                "paragraphs": 0, "chunks": 0, "terms": 0,
                "source_chars": 0, "translation_chars": 0, "protected_spans": 0,
            },
        }

    issues: list[str] = []
    extracted: list[Term] = []
    remaining = max(0, max_terms - len(seed_terms))
    if extractor is not None and remaining > 0:
        try:
            found, extract_issues = extractor.extract(
                source, existing=tuple(seed_terms), limit=remaining,
            )
        except Exception as exc:
            issues.append(f"术语抽取失败（仍继续翻译）：{type(exc).__name__}: {_safe_error(exc)}")
        else:
            if isinstance(found, (list, tuple)):
                extracted = [t for t in found if isinstance(t, Term)]
            if isinstance(extract_issues, (list, tuple)):
                issues.extend(str(x) for x in extract_issues)
    terms, term_issues = _merge_terms(seed_terms, extracted, source, max_terms)
    issues.extend(term_issues)

    paragraphs = _make_paragraphs(source, max_chars)
    pairs: list[dict[str, str]] = []
    for para in paragraphs:
        if not para.chunks:
            pairs.append({"zh": para.source, "en": para.source})
            continue
        translated = [
            _translate_chunk(chunk, translator, terms).strip()
            for chunk in para.chunks
        ]
        pairs.append({"zh": para.source, "en": " ".join(x for x in translated if x).strip()})

    translation = "\n\n".join(p["en"] for p in pairs).strip()
    return {
        "translation": translation,
        "glossary": [t.as_dict() for t in terms],
        "pairs": pairs,
        "issues": list(dict.fromkeys(issues)),
        "stats": {
            "paragraphs": len(paragraphs),
            "chunks": sum(len(p.chunks) for p in paragraphs),
            "terms": len(terms),
            "source_chars": len(source),
            "translation_chars": len(translation),
            "protected_spans": sum(
                len(chunk.placeholders) for p in paragraphs for chunk in p.chunks
            ),
        },
    }


# ══════════════════════════════ LLM 后端 ══════════════════════════════
TRANSLATE_SYSTEM = """你是学术英文润色与翻译专家，把中文学术论文片段译成可直接投稿的国际期刊英文。

硬性规则：
1. 术语严格遵循用户给出的术语表；同一术语全篇一致。方法名/模型名（如 Transformer、BERT）保留英文原名。
2. 文本中的 ⟦PH数字⟧ 是公式、变量、代码、URL 或文献标记的占位符：必须原样保留，包括编号与括号；不得翻译、改写、增删、调换顺序，也不得在译文中补写其内容。
3. 占位符外层的括号/引号属于原文结构：按英文标点规范保留或转换，不要额外再加一层括号。
4. 句式学术化：使用正式、客观、简洁的英文；可显式补出中文隐含的逻辑连接（however/therefore/whereas 等），但不得改变论证强度，不得添加原文没有的数据、结论或引用。
5. 只翻译给定片段，不解释、不总结、不扩写；不添加原文没有的信息。
6. 输出纯英文译文，不要引号、代码块、"Translation:" 等前后缀。
7. 【术语表】只约束译法，不是待译内容；【段落上下文】仅供理解，不要翻译。"""

EXTRACT_SYSTEM = """你是中文学术论文的术语抽取器。只输出 JSON：
{"terms":[{"zh":"中文术语","en":"English term","note":"一句中文译法说明"}]}

规则：
1. 只抽取原文中真实出现的领域术语、方法名、指标名；不要抽取公式、变量（如 x、d_k）、代码、文献作者。
2. zh 必须是原文中的连续子串；en 用国际期刊通行译法；note 说明固定搭配或与近义译法的区别。
3. 最多输出指定条数；不要重复“已确定术语”。
4. 没有合适术语时输出 {"terms":[]}。不要输出 JSON 以外的文字。"""


def _mask_for_terms(text: str) -> str:
    protected, _ = protect(text)
    plain = _PH_FIND_RE.sub(" ", protected)
    return re.sub(r"\s+", " ", plain).strip()


def _compose_translation_user(
    chunk: Chunk,
    glossary: Sequence[Term],
    repair: PlaceholderIssue | None,
) -> str:
    parts: list[str] = []
    if glossary:
        parts.append("【术语表（必须严格遵守）】")
        for term in glossary:
            note = f"；{term.note}" if term.note else ""
            parts.append(f"- {term.zh} → {term.en}{note}")
    if chunk.context and chunk.context != chunk.raw:
        parts.append("【段落上下文，仅供理解，不要翻译】")
        parts.append(chunk.context)
    parts.append("【待译中文】")
    parts.append(chunk.protected)
    if repair and not repair.ok:
        parts.append("【修正要求】")
        parts.append(
            f"上一版译文占位符有误：{repair.describe()}。"
            "请重新输出完整英文译文，所有占位符必须原样出现且只出现一次。"
        )
    return "\n".join(parts)


class LLMBackend:
    """生产后端：OpenAI 兼容 Provider；没有真实凭据时拒绝伪造译文。"""

    def __init__(self, llm: Any | None = None) -> None:
        if llm is None:
            from app.providers.llm import get_llm  # 延迟导入：离线自检不依赖 openai

            llm = get_llm()
        self.llm = llm
        if getattr(self.llm, "model_name", "") == "mock-llm":
            raise RuntimeError(
                "检测到 MockLLM，无法真实翻译。请配置 LLM_PROVIDER=openai_compatible 与 LLM_API_KEY；"
                "离线查看效果请用 --demo。"
            )

    def translate(
        self,
        chunk: Chunk,
        glossary: Sequence[Term] | None = None,
        repair: PlaceholderIssue | None = None,
    ) -> str:
        resp = self.llm.chat(
            [{"role": "system", "content": TRANSLATE_SYSTEM},
             {"role": "user", "content": _compose_translation_user(chunk, list(glossary or []), repair)}],
            temperature=0.1,
            max_tokens=4096,
        )
        return _clean_model_output(getattr(resp, "content", "") or "")

    def extract(
        self,
        text: str,
        existing: Sequence[Term] = (),
        limit: int = 30,
    ) -> tuple[list[Term], list[str]]:
        masked = _mask_for_terms(text)
        if not masked or limit <= 0:
            return [], []
        source_flat = _flat(masked)
        terms: list[Term] = []
        issues: list[str] = []
        for idx, piece in enumerate(chunk_text(masked, 6000), 1):
            if len(terms) >= limit:
                break
            prompt = [f"【原文片段 {idx}】", piece, f"最多抽取 {limit - len(terms)} 条术语。"]
            if existing or terms:
                prompt.append("【已确定术语（不要重复输出）】")
                prompt.extend(f"- {t.zh} → {t.en}" for t in [*existing, *terms])
            try:
                data = self.llm.chat_json(
                    [{"role": "system", "content": EXTRACT_SYSTEM},
                     {"role": "user", "content": "\n".join(prompt)}],
                )
            except Exception as exc:
                issues.append(f"术语抽取调用失败（第 {idx} 段）：{type(exc).__name__}: {_safe_error(exc)}")
                continue
            raw = data.get("terms") if isinstance(data, Mapping) else data
            if not isinstance(raw, list):
                issues.append(f"术语抽取返回格式不符（第 {idx} 段），已跳过")
                continue
            valid, sub_issues = _validate_terms(
                raw, source_flat,
                existing=[*existing, *terms],
                limit=limit - len(terms),
            )
            terms.extend(valid)
            issues.extend(sub_issues)
        return terms, issues


class CannedZh2En:
    """离线演示/回归：按完整片段精确匹配，再逐字回填受保护内容。"""

    def __init__(self, tm: Mapping[str, str]) -> None:
        self.tm = {_norm(k): v for k, v in tm.items()}

    def translate(
        self,
        chunk: Chunk,
        glossary: Sequence[Term] | None = None,
        repair: PlaceholderIssue | None = None,
    ) -> str:
        en = self.tm.get(_norm(chunk.raw))
        if en is None:
            raise TranslationError(f"演示翻译记忆缺少片段：{chunk.raw[:100]!r}")
        for pid, orig in chunk.placeholders.items():
            if orig not in en:
                raise TranslationError(f"演示译文未保留受保护内容 {orig!r}：{chunk.raw[:80]!r}")
            en = en.replace(orig, _ph(pid), 1)
        return en


# ══════════════════════════════ 输出 ══════════════════════════════
def _md_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = ["## English Translation / 英文译文", "", str(result.get("translation") or "").strip(), ""]
    lines.extend(["## Glossary / 术语表", ""])
    glossary = result.get("glossary") or []
    if glossary:
        lines.extend(["| 中文术语 | English | 译法说明 |", "|---|---|---|"])
        for term in glossary:
            lines.append(
                f"| {_md_cell(term.get('zh'))} | {_md_cell(term.get('en'))} | "
                f"{_md_cell(term.get('note'))} |"
            )
    else:
        lines.append("（未抽取到术语。）")
    issues = result.get("issues") or []
    if issues:
        lines.extend(["", "## Checks / 校验提示", ""])
        lines.extend(f"- {x}" for x in issues)
    stats = result.get("stats") or {}
    if stats:
        lines.extend([
            "",
            "---",
            f"段落 {stats.get('paragraphs', 0)} · 片段 {stats.get('chunks', 0)} · "
            f"术语 {stats.get('terms', 0)} · 受保护片段 {stats.get('protected_spans', 0)}",
        ])
    return "\n".join(lines).rstrip() + "\n"


def render_json(result: Mapping[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2) + "\n"


# ══════════════════════════════ 演示材料 ══════════════════════════════
DEMO_TEXT = (
    "本文提出一种面向长文档的稀疏注意力框架[1]。设序列长度为 $n$，键维度为 $d_k$，"
    "则标准自注意力的计算复杂度为 $O(n^2 d_k)$。已有研究（Zhang et al., 2020）表明，"
    "全局注意力在长序列上会带来显著的内存开销。我们在 WMT 2014 英德翻译任务上设置 "
    "`d_model = 512`，并采用 $h = 8$ 个注意力头。实验表明，该框架在保持翻译质量的同时"
    "显著降低了显存占用（见表 1）；然而，当文档长度超过 4096 个词元时，稀疏模式的选择"
    "仍会引入额外误差，因此需要进一步研究自适应稀疏度调度。"
)

DEMO_EN = (
    "This paper proposes a sparse attention framework for long documents [1]. "
    "Let the sequence length be $n$ and the key dimension be $d_k$; "
    "the computational complexity of standard self-attention is then $O(n^2 d_k)$. "
    "Prior work (Zhang et al., 2020) has shown that global attention incurs "
    "substantial memory overhead on long sequences. We set `d_model = 512` and use "
    "$h = 8$ attention heads on the WMT 2014 English–German translation task. "
    "Experiments show that the framework substantially reduces GPU memory usage "
    "while maintaining translation quality (see Table 1); however, when the document "
    "length exceeds 4096 tokens, the choice of sparse pattern still introduces "
    "additional error, which calls for further research on adaptive sparsity scheduling."
)
DEMO_TM = {DEMO_TEXT: DEMO_EN}

DEMO_GLOSSARY: tuple[Term, ...] = (
    Term("稀疏注意力", "sparse attention", "与标准自注意力对应；全篇统一使用 sparse attention。"),
    Term("标准自注意力", "standard self-attention", "指未加稀疏约束的 self-attention；不写作 vanilla attention。"),
    Term("长文档", "long document", "文档级长文本；复数按语境写 long documents。"),
    Term("计算复杂度", "computational complexity", "算法分析固定搭配。"),
    Term("键维度", "key dimension", "与 query/value dimension 并列。"),
    Term("全局注意力", "global attention", "指覆盖全序列的注意力。"),
    Term("内存开销", "memory overhead", "指显存/内存占用带来的成本。"),
    Term("显存占用", "GPU memory usage", "实验语境下指显存；与 memory overhead 区分。"),
    Term("翻译质量", "translation quality", "机器翻译评价固定搭配。"),
    Term("词元", "token", "NLP 语境统一译作 token，不译作 word；复数 tokens 按语法变化。"),
    Term("稀疏模式", "sparse pattern", "指稀疏注意力的连接模式。"),
    Term("自适应稀疏度调度", "adaptive sparsity scheduling", "指按输入动态调整稀疏度。"),
)


# ══════════════════════════════ 自检 ══════════════════════════════
def _selftest() -> int:
    failures: list[str] = []
    total = 0

    def check(name: str, ok: bool, detail: Any = "") -> None:
        nonlocal total
        total += 1
        print(("[OK] " if ok else "[FAIL] ") + name)
        if not ok:
            failures.append(name)
            if detail != "":
                print(f"     {detail!r}")

    check("中文切句保留句末标点",
          split_zh_sentences("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"],
          split_zh_sentences("第一句。第二句！第三句？"))
    check("切句保留后引号", split_zh_sentences("他说：“好。”然后离开。")[0].endswith("。”"),
          split_zh_sentences("他说：“好。”然后离开。"))

    pieces = chunk_text("第一句。第二句。第三句。", 6)
    check("只在句末切块", all(p.endswith("。") for p in pieces) and "".join(pieces) == "第一句。第二句。第三句。", pieces)

    paragraphs = _make_paragraphs("设 $x$ 为输入。", 300)
    chunk = paragraphs[0].chunks[0]
    check("公式整体受保护", "$x$" in chunk.placeholders.values() and "$x$" in chunk.raw, chunk.placeholders)

    chunk = _make_paragraphs("已有研究（Smith et al., 2020）表明。", 300)[0].chunks[0]
    check("作者-年份逐字保留", "Smith et al., 2020" in chunk.placeholders.values()
          and "（Smith et al., 2020）" in chunk.raw, chunk.raw)

    chunk = _make_paragraphs("结果见文献[1]。", 300)[0].chunks[0]
    check("数字引用受保护", "[1]" in chunk.placeholders.values(), chunk.placeholders)

    formula = translate_document("$$\nE=mc^2\n$$", CannedZh2En({}), glossary=())
    check("纯公式段原样返回", "E=mc^2" in formula["translation"] and formula["stats"]["chunks"] == 0,
          formula["translation"])

    source_flat = _flat("本文研究稀疏注意力与长文档。")
    valid, vissues = _validate_terms([
        {"zh": "稀疏注意力", "en": "sparse attention", "note": "固定译法"},
        {"zh": "不存在的术语", "en": "fabricated"},
        {"zh": "$x$", "en": "x"},
        {"zh": "稀疏注意力", "en": "sparse attention"},
        {"zh": "长文档", "en": "long document"},
    ], source_flat, limit=10)
    check("术语必须在原文出现且非公式",
          [t.zh for t in valid] == ["稀疏注意力", "长文档"] and any("丢弃" in x for x in vissues),
          (valid, vissues))

    valid2, vissues2 = _validate_terms(
        [{"zh": "稀疏注意力", "en": "sparse focus"}], source_flat,
        existing=[Term("稀疏注意力", "sparse attention")], limit=10,
    )
    check("术语冲突不静默覆盖", valid2 == [] and any("冲突" in x for x in vissues2), vissues2)

    merged, missues = _merge_terms(
        [Term("稀疏注意力", "sparse attention")],
        [Term("稀疏注意力", "sparse focus")],
        "本文研究稀疏注意力。", 10,
    )
    check("合并术语保留首次译法", len(merged) == 1 and merged[0].en == "sparse attention"
          and any("冲突" in x for x in missues), (merged, missues))

    merged2, _ = _merge_terms([], [Term("乙", "B"), Term("甲", "A")], "甲乙", 1)
    check("术语上限与源顺序", len(merged2) == 1 and merged2[0].zh == "甲", merged2)

    one = _coerce_terms({"zh": "甲", "en": "A", "note": "n"})
    check("单词条 JSON 可加载", len(one) == 1 and one[0].en == "A", one)

    class _StubExtractorLLM:
        model_name = "stub"

        def chat_json(self, messages: Any, retries: int = 1, tools: Any = None) -> Any:
            return {"terms": [
                {"zh": "稀疏注意力", "en": "sparse attention", "note": "固定译法"},
                {"zh": "原文没有的词", "en": "fabricated"},
            ]}

    extract_terms, extract_issues = LLMBackend(llm=_StubExtractorLLM()).extract(
        "本文研究稀疏注意力。", limit=10,
    )
    check("LLM 术语抽取经过校验",
          len(extract_terms) == 1 and extract_terms[0].en == "sparse attention"
          and any("丢弃" in x for x in extract_issues),
          (extract_terms, extract_issues))

    class _MockLLM:
        model_name = "mock-llm"

    try:
        LLMBackend(llm=_MockLLM())
        mock_refused = False
    except RuntimeError:
        mock_refused = True
    check("MockLLM 不冒充翻译", mock_refused)

    class _OneTermExtractor:
        def extract(self, text: str, existing: Sequence[Term] = (),
                    limit: int = 30) -> tuple[list[Term], list[str]]:
            return [Term("稀疏注意力", "sparse attention", "固定译法")], []

    class _TextTranslator:
        def translate(self, chunk: Chunk, glossary: Sequence[Term],
                      repair: PlaceholderIssue | None = None) -> str:
            return "This study investigates sparse attention."

    two_pass = translate_document(
        "本文研究稀疏注意力。", _TextTranslator(),
        extractor=_OneTermExtractor(), glossary=(),
    )
    check("两段式术语注入翻译流程",
          two_pass["glossary"] and two_pass["glossary"][0]["en"] == "sparse attention"
          and two_pass["translation"].startswith("This study")
          and two_pass["stats"]["terms"] == 1,
          two_pass)

    class _RetryTranslator:
        def __init__(self) -> None:
            self.calls = 0

        def translate(self, chunk: Chunk, glossary: Sequence[Term],
                      repair: PlaceholderIssue | None = None) -> str:
            self.calls += 1
            if self.calls == 1:
                return "Let x be the input."
            pid = sorted(chunk.placeholders)[0]
            return f"Let {_ph(pid)} be the input."

    retry = _RetryTranslator()
    out = _translate_chunk(_make_paragraphs("设 $x$ 为输入。", 300)[0].chunks[0], retry, [])
    check("占位符缺失后重试并还原", retry.calls == 2 and out == "Let $x$ be the input.", out)

    class _BadTranslator:
        def translate(self, chunk: Chunk, glossary: Sequence[Term],
                      repair: PlaceholderIssue | None = None) -> str:
            return "No formula."

    try:
        _translate_chunk(_make_paragraphs("设 $x$ 为输入。", 300)[0].chunks[0], _BadTranslator(), [])
        bad_ok = False
    except TranslationError:
        bad_ok = True
    check("占位符持续丢失则报错", bad_ok)

    class _FabricatedTranslator:
        def translate(self, chunk: Chunk, glossary: Sequence[Term],
                      repair: PlaceholderIssue | None = None) -> str:
            return f"Let {_ph(99)} be the input."

    try:
        _translate_chunk(_make_paragraphs("设 $x$ 为输入。", 300)[0].chunks[0], _FabricatedTranslator(), [])
        fab_ok = False
    except TranslationError:
        fab_ok = True
    check("臆造占位符则报错", fab_ok)

    formula_chunk = Chunk(
        raw="$$\nE=mc^2\n$$", protected="⟦PH0⟧",
        placeholders={0: "$$\nE=mc^2\n$$"}, context="", paragraph=1, index=1,
    )
    check("独立公式片段不送模型",
          _translate_chunk(formula_chunk, _FabricatedTranslator(), []) == formula_chunk.raw
          and "E=mc^2" in formula_chunk.raw, formula_chunk.raw)

    demo = translate_document(
        DEMO_TEXT,
        CannedZh2En(DEMO_TM),
        glossary=DEMO_GLOSSARY,
    )
    check("演示译文完整且无占位符",
          bool(demo["translation"]) and "⟦" not in demo["translation"]
          and "[1]" in demo["translation"]
          and "$O(n^2 d_k)$" in demo["translation"]
          and "(Zhang et al., 2020)" in demo["translation"]
          and not _has_cjk(demo["translation"]),
          demo["translation"])
    check("演示术语表齐全", len(demo["glossary"]) == len(DEMO_GLOSSARY)
          and all(t["zh"] and t["en"] and t["note"] for t in demo["glossary"]), demo["glossary"])
    check("演示统计一致", demo["stats"]["paragraphs"] == 1 and demo["stats"]["chunks"] == 1
          and demo["stats"]["protected_spans"] >= 6, demo["stats"])

    md = render_markdown(demo)
    check("Markdown 含译文与术语表",
          "## English Translation" in md and "## Glossary" in md
          and "sparse attention" in md and "⟦" not in md, md[:200])
    payload = json.loads(render_json(demo))
    check("JSON 结构完整",
          isinstance(payload, dict) and {"translation", "glossary", "pairs", "issues", "stats"} <= set(payload),
          list(payload))

    empty = translate_document("   ", CannedZh2En({}), glossary=())
    check("空输入返回空结构", empty["translation"] == "" and empty["pairs"] == []
          and empty["glossary"] == [] and empty["stats"]["paragraphs"] == 0, empty)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "glossary.json"
        path.write_text(json.dumps([{"zh": "甲", "en": "A"}], ensure_ascii=False), encoding="utf-8")
        loaded = _load_glossary(str(path))
        check("术语表文件可加载", len(loaded) == 1 and loaded[0].zh == "甲", loaded)
        path.write_text(json.dumps([{"zh": "甲"}], ensure_ascii=False), encoding="utf-8")
        try:
            _load_glossary(str(path))
            bad_glossary = False
        except ValueError:
            bad_glossary = True
        check("残缺术语表报错", bad_glossary)

    check("模型前缀清理", _clean_model_output("```\nTranslation: Hello\n```") == "Hello",
          _clean_model_output("```\nTranslation: Hello\n```"))

    # 拼接构造假密钥：既测得到脱敏分支，又避免仓库里出现像真 key 的字面量
    fake, token = "sk-" + "abcdefghijklmnop", "abcdefghijklmnop"
    redacted = _safe_error(RuntimeError(f"key {fake} / Bearer {token}"))
    check("错误信息脱敏", "abcdefghijklmnop" not in redacted and "***" in redacted, redacted)

    print(f"\nselftest: {total - len(failures)}/{total} 通过")
    return 0 if not failures else 1


# ══════════════════════════════ CLI ══════════════════════════════
def _load_glossary(path: str) -> list[Term]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"术语表不是合法 JSON：{_safe_error(exc)}") from exc
    terms = _coerce_terms(data)
    if not terms:
        raise ValueError("术语表为空")
    return terms


def _read_source(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    raw = Path(path).read_bytes()
    last: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError as exc:
            last = exc
    raise ValueError(f"无法解码文件 {path}：{last}")


def _debug_plan(text: str, max_chars: int) -> None:
    paragraphs = _make_paragraphs(text, max_chars)
    print(f"[debug] 段落 {len(paragraphs)}，max_chars={max_chars}", file=sys.stderr)
    for para in paragraphs:
        print(f"[段 {para.paragraph}] 片段 {len(para.chunks)}", file=sys.stderr)
        for chunk in para.chunks:
            print(f"  [{chunk.index}] PH={sorted(chunk.placeholders)} "
                  f"protected={chunk.protected[:180]}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description="中文学术文本 → 国际期刊英文（公式/变量/引用逐字保留，附术语表）",
    )
    ap.add_argument("--text", help="中文原文")
    ap.add_argument("--file", "-f", help="中文原文文件；- 表示 stdin")
    ap.add_argument("--glossary", help="术语表 JSON：[{zh,en,note}] 或 {中文: English}")
    ap.add_argument("--max-chars", type=int, default=3000, help="每个翻译片段的软上限，默认 3000")
    ap.add_argument("--max-terms", type=int, default=30, help="术语表上限，默认 30")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非 Markdown")
    ap.add_argument("--out", help="输出文件；.json 后缀自动按 JSON 输出")
    ap.add_argument("--demo", action="store_true", help="离线演示：内置中文摘要与人工译文")
    ap.add_argument("--selftest", action="store_true", help="运行自检")
    ap.add_argument("--debug", action="store_true", help="把分段/占位符计划打印到 stderr")
    args = ap.parse_args(argv)
    args.max_chars = max(80, args.max_chars)
    args.max_terms = max(0, args.max_terms)

    if args.selftest:
        return _selftest()
    if args.demo and (args.text or args.file or args.glossary):
        ap.error("--demo 使用内置文本与内置术语表，不能同时提供 --text/--file/--glossary")
    if args.file and args.text:
        ap.error("--text 与 --file 只能二选一")
    if not args.demo and not args.text and not args.file:
        ap.error("请用 --text 或 --file 提供中文原文")

    try:
        if args.demo:
            source = DEMO_TEXT
            translator: TranslatorLike = CannedZh2En(DEMO_TM)
            extractor: TermExtractor | None = None
            glossary: Sequence[Term] = DEMO_GLOSSARY
        else:
            source = _read_source(args.file) if args.file else (args.text or "")
            if not source.strip():
                print("错误：中文原文为空", file=sys.stderr)
                return 2
            glossary = _load_glossary(args.glossary) if args.glossary else []
            try:
                backend = LLMBackend()
            except Exception as exc:
                print(f"无法初始化翻译后端：{_safe_error(exc)}", file=sys.stderr)
                return 1
            translator = backend
            extractor = backend
    except (OSError, ValueError) as exc:
        print(f"错误：{_safe_error(exc)}", file=sys.stderr)
        return 2

    if args.debug:
        _debug_plan(source, args.max_chars)

    try:
        result = translate_document(
            source,
            translator,
            extractor=extractor,
            glossary=glossary,
            max_chars=args.max_chars,
            max_terms=args.max_terms,
        )
    except (TranslationError, PlaceholderError, ValueError) as exc:
        print(f"翻译失败：{_safe_error(exc)}", file=sys.stderr)
        return 1

    as_json = args.json or (bool(args.out) and Path(args.out).suffix.lower() == ".json")
    output = render_json(result) if as_json else render_markdown(result)
    if args.out:
        try:
            Path(args.out).write_text(output, encoding="utf-8")
        except OSError as exc:
            print(f"写入失败：{_safe_error(exc)}", file=sys.stderr)
            return 1
        print(f"已写入：{args.out}", file=sys.stderr)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
