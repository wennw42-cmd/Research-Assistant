"""可运行示例（零密钥，直接跑）。

    cd apps/api
    pip install -r requirements.txt
    python examples/run_demo.py                    # 跑全部内置示例
    python examples/run_demo.py --query "对比 Transformer 与 BERT 并生成综述"
    python examples/run_demo.py --real             # 使用真实模型（需 .env 配 LLM_API_KEY）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 保证以脚本方式运行时也能 import app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents import run  # noqa: E402
from app.utils import setup_logging  # noqa: E402

DEMOS = [
    ("① 单步问答（Supervisor → RAG 专家）", "Transformer 的注意力复杂度是多少？"),
    ("② 复合意图（Supervisor → Planner → DAG → 写作）", "对比 Transformer 与 BERT 并生成综述"),
    ("③ 路由到解析/去重 st模块（v1/v7 用例）", "帮我解析 sample.pdf 并对 arxiv:1706.03762v1 与 v7 去重"),
    ("④ 路由到引用图谱", "/graph 以 Transformer 与 BERT 为种子构建引用图谱"),
    ("⑤ 澄清分支（不烧 token）", "你好"),
]


def print_result(title: str, res: dict) -> None:
    print("\n" + "═" * 86)
    print(f"  {title}")
    print("═" * 86)
    print(f"  查询        : {res.get('query')}")
    print(f"  路由        : {res.get('route')}  (complexity={res.get('complexity')}, "
          f"confidence={res.get('confidence')}, model={res.get('model')})")
    print(f"  路由理由    : {str(res.get('route_reason'))[:110]}")

    plan = res.get("plan") or []
    if plan:
        print(f"  任务 DAG    : 共 {len(plan)} 个子任务")
        for t in plan:
            deps = ",".join(t.get("depends_on") or []) or "-"
            print(f"      {t['id']:<4} [{t['status']:<8}] {t['agent']:<11} deps={deps:<8} {t['title']}")

    trace = res.get("trace") or []
    steps = [t for item in trace if isinstance(item, dict) for t in (item.get("steps") or [])]
    if steps:
        print("  工具调用    :")
        for s in steps[:10]:
            print(f"      step{s['step']} {s['tool']:<22} ok={s['ok']} cited={s['cited']} {s['latency_ms']}ms")

    ev = res.get("evidence") or []
    print(f"  证据池      : {len(ev)} 条")
    for e in ev[:5]:
        print(f"      [[ev:{e['doc_id']}#{e['block_id']}]] p.{e.get('page')} · {str(e.get('quote'))[:52]}…")

    v = res.get("verify") or {}
    print(f"  抗幻觉校验  : passed={v.get('passed')} coverage={v.get('evidence_coverage')} "
          f"quote_faithful={v.get('quote_faithful')}")
    for issue in (v.get("issues") or [])[:3]:
        print(f"      [!] {issue}")

    ans = (res.get("answer") or "").replace("\n", "\n  ")
    print(f"  回答预览    :\n  {ans[:900]}")
    print(f"  总耗时      : {res.get('latency_ms')}ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Research Assistant Multi-Agent Demo")
    ap.add_argument("--query", "-q", default="", help="单条查询（指定后只跑这一条）")
    ap.add_argument("--thread", "-t", default="demo-thread", help="会话 thread_id（用于演示记忆）")
    ap.add_argument("--real", action="store_true", help="使用真实 LLM（需配置 .env）")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    if args.real:
        os.environ.setdefault("LLM_PROVIDER", "openai_compatible")
    else:
        os.environ.setdefault("LLM_PROVIDER", "mock")

    cases = [(args.query, args.query)] if args.query else DEMOS
    for title, query in cases:
        res = run(query, thread_id=args.thread)
        print_result(title, res)

    print("\nDemo 完成。设置 LLM_PROVIDER=openai_compatible 并在 .env 配置 LLM_API_KEY 即可切换真实模型。")


if __name__ == "__main__":
    main()
