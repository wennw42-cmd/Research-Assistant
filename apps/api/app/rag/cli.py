"""RAG 子系统 CLI：合成教材 → 流式入库 → 分级加载 → 内存报告。

    python -m app.rag.cli --selftest
    python -m app.rag.cli --make-sample book.pdf --pages 541
    python -m app.rag.cli --ingest book.pdf --out idx/ --page-batch 24 --memory-limit 512
    python -m app.rag.cli --query "反向传播 链式法则" --index idx/ --budget 3000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

try:
    import pymupdf as pm
except ImportError:  # pragma: no cover
    import fitz as pm  # type: ignore

from app.rag.ingest import ingest_pdf
from app.rag.memory import MemoryMonitor
from app.rag.params import PARAM_RATIONALE, ChunkParams, count_tokens
from app.rag.retrieval import TieredLoader
from app.rag.store import ChunkIndex, LRUCache

# ────────────────────────── 合成教材 ──────────────────────────
CHAPTERS = [
    ("线性代数基础", ["向量与矩阵", "特征值与特征向量", "矩阵分解"]),
    ("概率与统计", ["随机变量", "常见分布", "参数估计"]),
    ("优化方法", ["梯度下降", "约束优化", "凸优化初步"]),
    ("神经网络", ["感知机", "多层前馈网络", "激活函数"]),
    ("反向传播", ["计算图", "链式法则", "梯度检查"]),
    ("卷积网络", ["卷积运算", "池化与不变性", "经典架构"]),
    ("序列建模", ["循环神经网络", "长短期记忆", "注意力机制"]),
    ("正则化", ["过拟合", "权重衰减", "Dropout"]),
    ("无监督学习", ["聚类", "降维", "自编码器"]),
    ("图神经网络", ["图卷积", "消息传递", "图池化"]),
    ("强化学习", ["马尔可夫决策过程", "策略梯度", "值函数方法"]),
    ("模型压缩", ["剪枝", "量化", "知识蒸馏"]),
]

def _find_fonts() -> tuple[str, str] | None:
    """探测 CJK 字体，返回 (正文, 标题)。

    **关键**：base14（Helvetica 系）装不下汉字，会把汉字渲染成 notdef，
    抽出来全是 `·` —— 标题识别会因此全军覆没。合成样本必须嵌入真字体。
    """
    names = ["msyh.ttc", "msyhl.ttc", "simsun.ttc", "simhei.ttf", "deng.ttf"]
    roots = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
             Path("/usr/share/fonts"), Path("/System/Library/Fonts"), Path("/Library/Fonts")]
    found = [str(r / n) for r in roots for n in names if (r / n).exists()]
    if not found:
        return None
    body = next((f for f in found if any(k in f for k in ("msyh", "simsun", "uming"))), found[0])
    head = next((f for f in found if any(k in f for k in ("simhei", "deng", "msyh"))), body)
    return body, head


_EN_FILLER = [
    "Before giving the formal definition, we start with a concrete example that motivates the method.",
    "The proof of this statement is left as an exercise and follows from the previous subsection.",
    "In practice, the formulation above is usually combined with numerically stable implementations.",
    "Experiments on standard benchmarks show stable convergence and low sensitivity to hyper-parameters.",
    "The table below compares several settings, and the trend agrees with the theoretical analysis.",
    "Readers who are not interested in the derivation may skip directly to the application section.",
    "This property guarantees convergence under mild assumptions and underpins the later analysis.",
    "It is recommended to rely on mature numerical libraries rather than reimplementing the kernels.",
]

_FILLER = [
    "在给出形式化定义之前，先通过一个具体例子说明该方法的动机与直觉。",
    "该结论的证明留作练习，读者可结合前一小节的推导自行完成。",
    "需要注意的是，上述形式在实际实现中通常会加入数值稳定的处理。",
    "实验结果表明，在标准数据集上该方法能够稳定收敛，且对超参数不敏感。",
    "下表给出了不同设置下的对比结果，可以看到趋势与理论分析一致。",
    "若读者对推导细节不感兴趣，可直接跳到下一节的应用部分。",
    "这一性质保证了算法在合理假设下的收敛性，是后续分析的基础。",
    "实现时建议优先使用成熟的数值库，避免自行实现导致的精度问题。",
]


def make_textbook(path: str | Path, pages: int = 541, seed: int = 7) -> Path:
    """合成一本 N 页教材（章节/小节/段落齐全，单栏排版）。"""
    rng = random.Random(seed)
    fonts = _find_fonts()
    cjk = fonts is not None
    f_body, f_head = fonts or ("helv", "hebo")
    filler = _FILLER if cjk else _EN_FILLER
    width = 40 if cjk else 78

    def put(text: str, size: float, *, head: bool = False) -> None:
        # 用 insert_font 注册的别名（F0/F1）：直接传 fontfile 会被 base14 的编码表吃掉汉字
        page.insert_text((72, y), text, fontsize=size,
                         fontname=("F1" if head else "F0") if cjk else ("hebo" if head else "helv"))

    doc = pm.open()

    def fresh_page():
        pg = doc.new_page()
        if cjk:                       # insert_font 是页级作用域，每页都要注册
            pg.insert_font(fontfile=f_body, fontname="F0")
            pg.insert_font(fontfile=f_head, fontname="F1")
        return pg

    page = fresh_page()
    y = 60.0

    def need(h: float = 0.0) -> None:
        nonlocal page, y
        if y + h > 790:
            page = fresh_page()
            y = 60.0

    need(40)
    put("深度学习教材（合成样本）" if cjk else "Deep Learning Textbook (synthetic)", 20, head=True)
    y += 40
    for _ in range(3):
        need(20)
        put(rng.choice(filler), 10)
        y += 16

    ci = 0
    while doc.page_count < pages:
        ch_name, sections = CHAPTERS[ci % len(CHAPTERS)]
        round_no = ci // len(CHAPTERS) + 1
        ch_title = (f"第 {ci + 1} 章  {ch_name}" + (f"（进阶 {round_no}）" if round_no > 1 else "")
                    if cjk else f"Chapter {ci + 1}. {ch_name}")
        need(60)
        put(ch_title, 16, head=True)
        y += 26
        put(rng.choice(filler), 10)
        y += 18

        for si, sec in enumerate(sections, 1):
            need(46)
            sec_title = (f"{ci + 1}.{si}  {sec}" if cjk
                         else f"{ci + 1}.{si} {['Basics', 'Details', 'Applications'][(si - 1) % 3]} of {ch_name}")
            put(sec_title, 12, head=True)
            y += 20
            for _ in range(rng.randint(8, 12)):          # 每节 8-12 段，保证 L1 会切成多块（可验证重叠）
                need(46)
                text = rng.choice(filler) + rng.choice(filler)
                for line in _wrap(text, width):
                    need(14)
                    put(line, 10)
                    y += 14
                y += 6
        ci += 1
        if ci > pages:  # 兜底：防止极端参数下死循环
            break

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return Path(path)


def _wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= width:
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return lines


# ────────────────────────── 报告 ──────────────────────────
def print_report(report: dict, params: ChunkParams, mon: MemoryMonitor) -> None:
    print("── 切块参数 ───────────────────────────────────────────────")
    rows = params.table_rows()
    w = [max(len(str(r.get(k, ""))) for r in rows) for k in ("层级", "单位", "目标", "重叠", "上限", "用途")]
    print("  " + "  ".join(f"{k:<{wi}}" for k, wi in zip(("层级", "单位", "目标", "重叠", "上限", "用途"), w)))
    for r in rows:
        print("  " + "  ".join(f"{str(r[k]):<{wi}}" for k, wi in zip(("层级", "单位", "目标", "重叠", "上限", "用途"), w)))
    print("── 入库结果 ───────────────────────────────────────────────")
    print(f"  页数 {report['pages']}  正文块 {report['blocks']}  标题 {report['headings']}  小节 {report['sections']}")
    print(f"  chunk 总数 {report['chunks']}  分层 {report['by_level']}  总 token {report['tokens']}")
    print(f"  索引常驻 {report['index_bytes'] / 1024:.1f} KB   正文落盘 "
          f"{sum(report['store_bytes'].values()) / 1024:.1f} KB")
    print("── 内存 ───────────────────────────────────────────────────")
    print(mon.report())
    if report.get("memory_limit"):
        ml = report["memory_limit"]
        print(f"  限额 {ml['limit_mb']:.0f}MB：{'通过' if ml['ok'] else '超限'} —— {ml['msg']}")
    for w in report["warnings"]:
        print(f"  [!] {w}")


def strip_crumb(text: str) -> str:
    return re.sub(r"^【[^】]*】\s*", "", text or "")


# ────────────────────────── 自测 ──────────────────────────
def _selftest() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="rag_"))
    pdf = make_textbook(tmp / "textbook.pdf", pages=24, seed=3)
    params = ChunkParams()
    mon = MemoryMonitor()
    report = ingest_pdf(pdf, tmp / "idx", params=params, page_batch=8, monitor=mon)
    idx = ChunkIndex.load(tmp / "idx" / "index.json")
    loader = TieredLoader(index=idx, base_dir=tmp / "idx")
    checks: list[tuple[str, bool, str]] = []

    by = report["by_level"]
    checks.append(("四层都产出", all(by.get(lv, 0) > 0 for lv in ("T0", "L0", "L1", "L2")), str(by)))
    checks.append(("标题与小节识别", report["headings"] >= 5 and report["sections"] >= 3,
                   f"标题 {report['headings']} 小节 {report['sections']}"))

    bad = [c["id"] for c in idx.records if not idx.path_of(c) or c["page_start"] > c["page_end"]]
    checks.append(("每块带标题路径与页码区间", not bad, f"异常 {bad[:3]}"))

    l1 = idx.by_level("L1")
    over = [c["id"] for c in l1 if c["n_tokens"] > params.para_max + 40]
    checks.append(("L1 不超上限", not over, f"最大 {max((c['n_tokens'] for c in l1), default=0)} / 上限 {params.para_max}"))
    l2 = idx.by_level("L2")
    checks.append(("L2 不超上限", all(c["n_tokens"] <= params.sent_max + 30 for c in l2),
                   f"最大 {max((c['n_tokens'] for c in l2), default=0)} / 上限 {params.sent_max}"))

    # 重叠：相邻 L1 块的正文必须真的共享尾部内容
    shared = 0
    same_sec: dict[str, list[dict]] = {}
    for c in l1:
        same_sec.setdefault(str(c["section_id"]), []).append(c)
    for group in same_sec.values():
        group.sort(key=lambda c: c["offset"])
        for a, b in zip(group, group[1:]):
            ta, tb = strip_crumb(loader.text_of(a)), strip_crumb(loader.text_of(b))
            if ta and tb and ta[-24:] in tb:
                shared += 1
    checks.append(("相邻 L1 存在真实重叠", shared >= 1, f"{shared} 对相邻块共享尾部"))

    # 正文不常驻：索引必须小于正文总量（字典编码后）
    total_store = sum(report["store_bytes"].values())
    ratio = report["index_bytes"] / max(total_store, 1)
    checks.append(("索引小于正文（正文不常驻）", report["index_bytes"] < total_store,
                   f"索引 {report['index_bytes'] / 1024:.1f}KB vs 正文 {total_store / 1024:.1f}KB"
                   f"（比值 {ratio:.2f}）"))

    # 按需加载内容一致（seek 偏移正确）
    probe = idx.by_level("L1")[0]
    direct = json.loads(Path(tmp / "idx" / f"chunks_{probe['store']}.jsonl").read_text(
        encoding="utf-8").splitlines()[0])["text"]
    checks.append(("偏移寻址读取正确", loader.text_of(probe) == direct, probe["id"]))

    # LRU 淘汰
    cache = LRUCache(max_bytes=4096)
    for i in range(60):
        cache.put(f"k{i}", "x" * 200)
    checks.append(("LRU 按字节预算淘汰", cache.stats()["evictions"] > 0 and cache.stats()["bytes"] <= 4096,
                   f"{cache.stats()}"))

    # 分级加载
    res = loader.load("反向传播 链式法则", token_budget=1200, top_chapters=1, top_sections=2, top_paras=3)
    checks.append(("分级加载不超预算", res["used_tokens"] <= res["budget"],
                   f"used={res['used_tokens']}/{res['budget']} tiers={res['tiers']}"))
    checks.append(("四层都被触达", sum(res["tiers"].values()) >= 3 and res["tiers"]["T0"] >= 1,
                   str(res["tiers"])))
    hit = next((i for i in res["items"] if i["tier"] == "L1"), None)
    checks.append(("命中块带完整溯源路径", bool(hit and hit["title_path"] and hit["page_start"]),
                   f"{hit['title_path'][:40]} p{hit['page_start']}" if hit else "-"))
    checks.append(("分级加载只触达少量正文", res["io"]["loaded_bytes"] < total_store * 0.5,
                   f"加载 {res['io']['loaded_bytes'] / 1024:.1f}KB / 正文 {total_store / 1024:.1f}KB"
                   f"（{res['io']['loaded_bytes'] / max(total_store, 1) * 100:.1f}%）"))
    checks.append(("阶段内存计量已记录", len(mon.stages) >= 3 and mon.peak_rss_mb >= 0,
                   f"{len(mon.stages)} 个阶段，峰值 RSS {mon.peak_rss_mb:.1f}MB"))

    width = max(len(c[0]) for c in checks)
    passed = 0
    for name, ok, detail in checks:
        passed += ok
        print(f"[{'OK' if ok else 'XX'}] {name:<{width}}  {detail}")
    print("─" * 78)
    print(f"通过 {passed}/{len(checks)}   峰值 RSS {mon.peak_rss_mb:.1f}MB  峰值堆 {mon.peak_alloc_mb:.1f}MB")
    print(f"样例 PDF {pdf}   索引目录 {tmp / 'idx'}")
    return 0 if passed == len(checks) else 1


# ────────────────────────── CLI ──────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="541 页教材 RAG 入库与分级加载")
    ap.add_argument("--make-sample", default="", help="生成合成教材 PDF 到该路径")
    ap.add_argument("--pages", type=int, default=541, help="合成页数")
    ap.add_argument("--ingest", default="", help="待入库 PDF")
    ap.add_argument("--out", default="rag_index", help="索引输出目录")
    ap.add_argument("--index", default="", help="已入库的索引目录（查询用）")
    ap.add_argument("--query", default="", help="查询词（分级加载）")
    ap.add_argument("--budget", type=int, default=4000, help="上下文 token 预算")
    ap.add_argument("--page-batch", type=int, default=24, help="流式解析的页批大小")
    ap.add_argument("--memory-limit", type=float, default=0, help="内存限额 MB（0=不限）")
    ap.add_argument("--max-pages", type=int, default=0, help="页数上限（0=不限）")
    ap.add_argument("--para-target", type=int, default=512)
    ap.add_argument("--para-overlap", type=int, default=64)
    ap.add_argument("--sent-target", type=int, default=128)
    ap.add_argument("--sent-overlap", type=int, default=32)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())
    if args.make_sample:
        p = make_textbook(args.make_sample, pages=args.pages)
        print(f"已生成 {p}（{pm.open(str(p)).page_count} 页）")
        return

    params = ChunkParams(para_target=args.para_target, para_overlap=args.para_overlap,
                         sent_target=args.sent_target, sent_overlap=args.sent_overlap)

    if args.ingest:
        mon = MemoryMonitor()
        report = ingest_pdf(args.ingest, args.out, params=params, page_batch=args.page_batch,
                            memory_limit_mb=args.memory_limit or None,
                            max_pages=args.max_pages, monitor=mon)
        print_report(report, params, mon)
        idx = ChunkIndex.load(Path(args.out) / "index.json")
        sample = idx.by_level("L1")[0]
        print("── 样例 chunk 溯源 ───────────────────────────────────────")
        print(f"  id={sample['id']}  path={idx.path_str_of(sample)}")
        print(f"  pages={sample['page_start']}-{sample['page_end']}  tokens={sample['n_tokens']}  "
              f"parent={sample['parent_id']}  blocks={sample['block_ids'][:3]}")
        print(f"  正文：{strip_crumb(TieredLoader(idx, Path(args.out)).text_of(sample))[:90]}…")
        print("\n── 参数取值理由 ───────────────────────────────────────────")
        for k, v in PARAM_RATIONALE:
            print(f"  {k:<14} {v}")
        return

    if args.index and args.query:
        idx = ChunkIndex.load(Path(args.index) / "index.json")
        loader = TieredLoader(index=idx, base_dir=Path(args.index))
        res = loader.load(args.query, token_budget=args.budget)
        print(f"query={res['query']}  used={res['used_tokens']}/{res['budget']}  tiers={res['tiers']}")
        for it in res["items"]:
            print(f"  [{it['tier']}] p{it['page_start']}-{it['page_end']} {it['n_tokens']}tok "
                  f"score={it['score']:.3f}  {it['title_path'][:36]}")
        print(f"  IO: 加载 {res['io']['loaded_items']} 块 / {res['io']['loaded_bytes'] / 1024:.1f} KB"
              f"   缓存命中率 {res['cache']['hit_rate']}")
        return

    ap.print_help()


if __name__ == "__main__":
    sys.exit(main())
