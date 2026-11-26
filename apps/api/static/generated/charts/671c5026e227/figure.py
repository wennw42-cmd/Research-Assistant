#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 由 app.chart 生成；依赖：matplotlib>=3.5（pip install matplotlib）
# 运行：python figure.py figure.png
# 数据：脚本内置数据

import argparse
import csv
import math
import os
import statistics
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates
except ImportError:
    raise SystemExit("需要 matplotlib：pip install matplotlib")

PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000', '#F0E442']
CJK = True

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": (["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                         "Arial", "DejaVu Sans"] if CJK
                        else ["Arial", "Helvetica", "DejaVu Sans"]),
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CSV_PATH = ''
X_COL = 'year'
Y_COLS = ['accuracy']
SERIES_COL = 'model'
LABEL_COL = ''
LABELS = {'accuracy': 'accuracy', 'year': 'year', 'model': 'model'}
TITLE = '准确率'
X_LABEL = 'year'
Y_LABEL = 'accuracy'
TIME_AXIS = True
HORIZONTAL = False
POLAR = False
FIG_W = 7.2
FIG_H = 4.2
EMBEDDED_ROWS = [{'year': '2020', 'model': 'A', 'accuracy': '62.1'}, {'year': '2020', 'model': 'B', 'accuracy': '60.4'}, {'year': '2021', 'model': 'A', 'accuracy': '65.3'}, {'year': '2021', 'model': 'B', 'accuracy': '63.8'}, {'year': '2022', 'model': 'A', 'accuracy': '68.9'}, {'year': '2022', 'model': 'B', 'accuracy': '66.2'}]


def read_csv(path):
    last = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, newline="", encoding=enc) as fh:
                return list(csv.DictReader(fh))
        except UnicodeDecodeError as exc:
            last = exc
    raise SystemExit(f"无法解码 CSV：{path}（{last}）")


def load_rows(path):
    if path and os.path.exists(path):
        return read_csv(path)
    if EMBEDDED_ROWS:
        return EMBEDDED_ROWS
    raise SystemExit(f"找不到数据文件：{path}；请传入 CSV 路径")


def num(value):
    if value is None:
        return None
    s = str(value).strip().replace(",", "").replace("%", "")
    if not s or s.lower() in ("na", "n/a", "nan", "none", "-", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_time(value):
    s = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M", "%Y-%m", "%Y/%m", "%Y年%m月", "%Y年", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def mean(values):
    return statistics.fmean(values) if values else 0.0


def save(fig, out_path):
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"已生成：{out_path}")


def main():
    ap = argparse.ArgumentParser(description="由 app.chart 生成的学术图表")
    ap.add_argument("csv", nargs="?", default=None, help="CSV 路径")
    ap.add_argument("out", nargs="?", default=None, help="输出 PNG 路径")
    args = ap.parse_args()
    csv_path = args.csv or CSV_PATH
    out_path = args.out or "figure.png"
    if args.csv and not args.out and args.csv.lower().endswith((".png", ".pdf", ".svg")):
        out_path = args.csv
        csv_path = CSV_PATH
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit("数据为空")
    if POLAR:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H),
                               subplot_kw={"projection": "polar"})
    else:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    draw(ax, fig, rows)
    if TITLE:
        ax.set_title(TITLE)
    save(fig, out_path)


def draw(ax, fig, rows):
    data = {}
    for row in rows:
        xv = parse_time(row.get(X_COL)) if TIME_AXIS else num(row.get(X_COL))
        if xv is None:
            continue
        if SERIES_COL:
            yv = num(row.get(Y_COLS[0]))
            if yv is None:
                continue
            key = str(row.get(SERIES_COL) or "").strip() or "NA"
            data.setdefault(key, {}).setdefault(xv, []).append(yv)
        else:
            for ycol in Y_COLS:
                yv = num(row.get(ycol))
                if yv is None:
                    continue
                key = LABELS.get(ycol, ycol)
                data.setdefault(key, {}).setdefault(xv, []).append(yv)
    for i, key in enumerate(sorted(data)):
        xs = sorted(data[key])
        ys = [mean(data[key][x]) for x in xs]
        ax.plot(xs, ys, color=PALETTE[i % len(PALETTE)], marker="o", ms=3,
                linewidth=1.4, label=key if (SERIES_COL or len(Y_COLS) > 1) else None)
    if SERIES_COL or len(Y_COLS) > 1:
        ax.legend(frameon=False, loc="best")
    if TIME_AXIS:
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.autofmt_xdate()
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)


if __name__ == "__main__":
    main()
