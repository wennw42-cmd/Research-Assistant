"""CSV → 学术图表：推荐图型、生成 300 dpi Matplotlib 脚本、提炼中英 Figure Caption。

输入可选 CSV（字段说明见 `schema`）或 JSON 中的 `rows`；所有统计量都由实际数据
计算，Caption 只陈述数据支持的结论，不使用因果或夸大措辞。

`schema` 支持::

    {"year": {"type": "time", "role": "x"},
     "model": {"type": "category", "role": "series"},
     "accuracy": {"type": "numeric", "role": "y", "unit": "%",
                  "label": "准确率", "label_en": "Accuracy"}}

也可写成 `[{"name": "year", "type": "time", "role": "x"}, ...]`
或 `["year:time", "model:category", "accuracy:numeric"]`。

用法::

    python -m app.chart --demo
    python -m app.chart --csv data.csv --schema schema.json --out figure.py
    python -m app.chart -i input.json --json
    python -m app.chart --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

CHART_NAMES = {
    "line": "折线图",
    "bar": "柱状图",
    "grouped_bar": "分组柱状图",
    "scatter": "散点图",
    "histogram": "直方图",
    "radar": "雷达图",
    "heatmap": "热力图",
    "bar_counts": "类别频数柱状图",
}
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#000000", "#F0E442",
]
HYPE_ZH = ("显著", "大幅", "巨大", "极高", "完美", "证明", "必然", "彻底", "突破性", "最优")
HYPE_EN = (
    "significantly", "dramatic", "drastic", "prove", "confirm",
    "unprecedented", "state-of-the-art", "best-ever", "causal",
)
DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
    "%Y-%m", "%Y/%m", "%Y年%m月", "%Y年", "%Y",
)
DATE_HINT = re.compile(
    r"(^|_)(date|time|year|month|quarter|period|day)($|_)|日期|时间|年份|年度|月份|季度",
    re.I,
)
ID_HINT = re.compile(r"(^|_)(id|uuid|key|code)($|_)|编号|序号", re.I)
TYPE_ALIASES = {
    "int": "numeric", "integer": "numeric", "float": "numeric", "double": "numeric",
    "number": "numeric", "numeric": "numeric", "连续": "numeric", "数值": "numeric",
    "date": "time", "datetime": "time", "year": "time", "time": "time",
    "时间": "time", "日期": "time",
    "category": "category", "categorical": "category", "string": "category",
    "str": "category", "text": "category", "类别": "category",
    "id": "id", "identifier": "id", "编号": "id",
}


# ────────────────────────── 基础解析 ──────────────────────────


def _num(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(",", "").replace("%", "")
    if not s or s.lower() in ("na", "n/a", "nan", "none", "-", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_time(value: Any) -> datetime | None:
    s = str(value or "").strip()
    if not s:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _canon_type(value: Any) -> str:
    return TYPE_ALIASES.get(str(value or "").strip().lower(), "")


def _canon_role(value: Any) -> str:
    r = str(value or "").strip().lower()
    if r in ("x", "axis", "横轴"):
        return "x"
    if r in ("y", "metric", "value", "measure", "纵轴"):
        return "y"
    if r in ("series", "group", "color", "hue", "分组"):
        return "series"
    if r in ("id", "identifier"):
        return "id"
    if r in ("ignore", "drop"):
        return "ignore"
    return ""


def _has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _count_zh(n: int, label: str, measure: str = "个") -> str:
    """中文数量词；拉丁列名前后补空格，避免“3 个model的”。"""
    lead = " " if label and label[0].isascii() else ""
    trail = " " if label and label[-1].isascii() else ""
    return f"{n} {measure}{lead}{label}{trail}"


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.{digits}g}"


def _signed(value: float) -> str:
    return f"+{_fmt(value)}" if value > 0 else _fmt(value)


def _val(value: float, unit: str = "") -> str:
    s = _fmt(value)
    if unit == "%":
        return s + "%"
    return f"{s} {unit}" if unit else s


def _delta(value: float, unit: str, lang: str) -> str:
    s = _signed(value)
    if unit == "%":
        return f"{s} 个百分点" if lang == "zh" else f"{s} percentage points"
    return f"{s} {unit}" if unit else s


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def _quantile(values: Sequence[float], q: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _read_rows(path: str) -> list[dict[str, str]]:
    last: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, newline="", encoding=enc) as fh:
                rows = list(csv.DictReader(fh))
            return rows
        except UnicodeDecodeError as exc:
            last = exc
        except OSError as exc:
            raise ValueError(f"读取 CSV 失败：{path}：{exc}") from exc
    raise ValueError(f"无法解码 CSV：{path}（{last}）")


def _range_phrase(zh: bool, a: str, b: str) -> str:
    if a == b:
        return f"在 {a}" if zh else f"In {a}"
    return f"在 {a}–{b} 范围内" if zh else f"Between {a} and {b}"


# ────────────────────────── Schema 与列画像 ──────────────────────────


@dataclass
class Column:
    name: str
    type: str = "category"
    role: str = ""
    unit: str = ""
    label: str = ""
    label_en: str = ""
    values: list[str] = field(default_factory=list)
    nums: list[float] = field(default_factory=list)
    times: list[datetime] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    missing: int = 0
    invalid: int = 0
    year_like: bool = False
    note: str = ""

    @property
    def disp(self) -> str:
        return self.label or self.name

    @property
    def disp_en(self) -> str:
        return self.label_en or self.name

    @property
    def n_values(self) -> int:
        return len(self.values) - self.missing

    @property
    def unique(self) -> int:
        return len(self.counts)


def _normalize_schema(schema: Any) -> dict[str, dict[str, Any]]:
    if schema is None:
        return {}
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError as exc:
            raise ValueError(f"schema 不是合法 JSON：{exc.msg}") from exc
    out: dict[str, dict[str, Any]] = {}
    if isinstance(schema, dict):
        for name, meta in schema.items():
            if isinstance(meta, str):
                out[str(name)] = {"type": meta}
            elif isinstance(meta, dict):
                out[str(name)] = dict(meta)
            else:
                out[str(name)] = {}
    elif isinstance(schema, list):
        for item in schema:
            if isinstance(item, str):
                name, _, typ = item.partition(":")
                out[name.strip()] = {"type": typ.strip()} if typ.strip() else {}
            elif isinstance(item, dict):
                name = item.get("name") or item.get("column") or item.get("field")
                if name:
                    out[str(name)] = dict(item)
    else:
        raise ValueError("schema 必须是对象、数组或 JSON 字符串")
    return out


def profile(rows: Sequence[dict[str, Any]], schema: Any = None) -> dict[str, Column]:
    meta = _normalize_schema(schema)
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    for name in meta:
        if name not in names:
            names.append(name)

    cols: dict[str, Column] = {}
    for name in names:
        raw = [str(row.get(name, "") if row.get(name) is not None else "").strip() for row in rows]
        nonempty = [value for value in raw if value]
        m = meta.get(name, {})
        ctype = _canon_type(m.get("type") or m.get("dtype"))
        role = _canon_role(m.get("role"))
        unit = str(m.get("unit") or "").strip()
        label = str(m.get("label") or m.get("name_zh") or name)
        label_en = str(m.get("label_en") or m.get("name_en") or name)
        note = ""

        nums = [x for value in nonempty if (x := _num(value)) is not None]
        numeric_ratio = len(nums) / len(nonempty) if nonempty else 0.0
        times = [x for value in nonempty if (x := _parse_time(value)) is not None]

        if not ctype:
            if ID_HINT.search(name):
                ctype = "id"
            elif DATE_HINT.search(name):
                ctype = "time"
            elif numeric_ratio >= 0.9:
                ctype = "numeric"
            elif len(rows) > 12 and len(set(nonempty)) / max(1, len(nonempty)) > 0.95:
                ctype = "id"
            else:
                ctype = "category"

        if ctype == "time" and nonempty and len(times) / len(nonempty) < 0.5:
            note = f"列“{name}”声明为时间，但多数值无法解析，已按类别处理"
            ctype = "category"
        if ctype == "numeric" and nonempty and numeric_ratio < 0.5:
            note = f"列“{name}”声明为数值，但多数值无法解析，统计时已跳过"

        year_like = bool(nonempty) and all(
            re.fullmatch(r"(?:19|20)\d{2}", value) for value in nonempty
        )
        col = Column(
            name=name,
            type=ctype,
            role=role,
            unit=unit,
            label=label,
            label_en=label_en,
            values=raw,
            nums=nums if ctype == "numeric" else [],
            times=times if ctype == "time" else [],
            counts=Counter(nonempty),
            missing=len(raw) - len(nonempty),
            invalid=len(nonempty) - (len(nums) if ctype == "numeric" else len(times) if ctype == "time" else 0),
            year_like=year_like,
            note=note,
        )
        cols[name] = col
    return cols


# ────────────────────────── 图型选择 ──────────────────────────


@dataclass
class Plan:
    chart: str
    x: Column | None
    ys: list[Column]
    series: Column | None
    label_col: Column | None
    reasons: list[str]
    alternatives: list[str]
    horizontal: bool = False
    normalize: bool = False


def _same_units(cols: Sequence[Column]) -> bool:
    return len({col.unit for col in cols}) <= 1


def _axis_label(col: Column) -> str:
    return f"{col.disp} ({col.unit})" if col.unit else col.disp


def _choose_plan(rows: Sequence[dict[str, Any]], cols: dict[str, Column]) -> Plan:
    usable = [col for col in cols.values() if col.n_values > 0 and col.role != "ignore"]
    nums = [col for col in usable if col.type == "numeric"]
    times = [col for col in usable if col.type == "time"]
    cats = [col for col in usable if col.type == "category"]
    entities = [col for col in usable if col.type in ("category", "id") and 2 <= col.unique <= 20]

    explicit_x = [col for col in usable if col.role == "x"]
    explicit_y = [col for col in usable if col.role == "y"]
    explicit_series = [col for col in usable if col.role == "series"]

    x: Column | None = explicit_x[0] if explicit_x else None
    if x is None:
        if times:
            x = times[0]
        elif entities:
            x = entities[0]
        elif len(nums) >= 2:
            x = nums[0]
        elif len(nums) == 1:
            x = None
        elif cats:
            x = cats[0]
        else:
            raise ValueError("没有可用于作图的列：请提供数值列或类别列")

    ys = [col for col in explicit_y if col.type == "numeric"] if explicit_y else []
    if not ys:
        ys = [col for col in nums if x is None or col.name != x.name]
        if not ys and x is None and nums:
            ys = [nums[0]]
        if not ys and x is not None and x.type == "numeric":
            raise ValueError("x 为唯一数值列，缺少可作图的 y 数值列；请补充 y 或去掉 x 角色")

    series: Column | None = explicit_series[0] if explicit_series else None
    if series is None and x is not None and len(ys) == 1:
        candidates = [
            col for col in usable
            if col.type in ("category", "id") and col.name != x.name and 2 <= col.unique <= 8
        ]
        if candidates:
            if x.type == "time":
                series = candidates[0]
            elif x.type in ("category", "id") and len(rows) >= x.unique * 2:
                series = candidates[0]

    label_col: Column | None = None
    if ys:
        if x is not None and x.type == "time":
            chart = "line"
            if len(ys) > 1 and not _same_units(ys):
                ys = ys[:1]
            if len(ys) > 1:
                series = None
        elif x is not None and x.type in ("category", "id"):
            if len(ys) >= 3:
                if 2 <= x.unique <= 8 and len(ys) <= 6:
                    chart = "radar"
                    series = None
                else:
                    chart = "heatmap"
                    series = None
            elif len(ys) == 2:
                if _same_units(ys):
                    chart = "grouped_bar"
                    series = None
                else:
                    label_col = x
                    x = ys[1]
                    ys = ys[:1]
                    series = None
                    chart = "scatter"
            else:
                chart = "bar"
        elif x is not None and x.type == "numeric":
            chart = "scatter"
            if len(ys) > 1:
                ys = ys[:1]
            series = None
        elif len(ys) == 1:
            chart = "histogram"
        else:
            chart = "scatter"
    else:
        chart = "bar_counts"
        series = None

    if chart in ("line", "bar") and x is None:
        chart = "histogram"
    if chart not in ("line", "bar"):
        series = None

    reasons: list[str] = []
    alternatives: list[str] = []
    horizontal = False
    normalize = False

    if chart == "line":
        xr = _column_range_label(x)
        reasons.append(f"x 轴为时间字段“{x.disp}”（{xr}），y 轴为数值型；折线图保留时间顺序与变化方向。")
        if series:
            reasons.append(f"“{series.disp}”只有 {series.unique} 个取值，可拆成多条序列比较，避免把分组压成单线。")
        if len(ys) > 1:
            reasons.append(f"{len(ys)} 个数值指标共用同一单位，可并列展示。")
        alternatives.append("若不强调时间顺序，可按类别聚合为分组柱状图；箱线图更适合展示分布而非均值。")
    elif chart == "bar":
        reasons.append(f"“{x.disp}”是低基数类别（{x.unique} 类），y 为数值型；柱状图适合比较类别间均值。")
        if series:
            reasons.append(f"“{series.disp}”提供 {series.unique} 个分组，可用分组柱并列比较。")
        horizontal = x.unique > 8
        alternatives.append("若关注组内分布而非均值，可改用箱线图；类别过多时建议先合并或改横向柱状图。")
    elif chart == "grouped_bar":
        reasons.append(f"“{x.disp}”为类别轴（{x.unique} 类），有 {len(ys)} 个同单位指标；分组柱可在同一坐标系内比较。")
        horizontal = x.unique > 8
        alternatives.append("若实体数较多或指标超过 6 个，改用热力图或雷达图更易读。")
    elif chart == "bar_counts":
        reasons.append(f"数据中没有可用的数值指标；“{x.disp}”的频数是可检验的主要信息，用柱状图展示。")
        alternatives.append("若后续补充数值字段，可改为柱状图（均值）或散点图。")
        horizontal = x.unique > 8
    elif chart == "scatter":
        if label_col is not None:
            reasons.append(
                f"“{label_col.disp}”有 {label_col.unique} 个类别，两个数值指标单位不同；"
                "散点图直接展示两项指标的关系，避免把不同量纲压在同一纵轴。"
            )
        else:
            reasons.append("x、y 均为连续数值；散点图用于观察两个变量的关系与离群点。")
        alternatives.append("若关注分组差异，可按类别着色或分面；散点图不用于推断因果。")
    elif chart == "histogram":
        reasons.append(f"只有一个主要数值字段“{ys[0].disp}”；直方图展示分布范围、集中位置与偏态。")
        alternatives.append("若要比较多个组的分布，可改用箱线图或小提琴图。")
    elif chart == "radar":
        reasons.append(
            f"“{x.disp}”有 {x.unique} 个实体、{len(ys)} 项指标；雷达图可同时展示多维轮廓。"
            "各指标按列 min–max 归一化后绘制。"
        )
        alternatives.append("三维雷达会互相遮挡且难以读数，未采用；若需精确比较，分组柱状图或小倍数图更合适。")
        normalize = True
    elif chart == "heatmap":
        reasons.append(
            f"“{x.disp}”有 {x.unique} 个实体、{len(ys)} 项指标；实体或指标较多，"
            "热力图按列 z-score 标准化后可同屏比较，避免不同量纲共轴。"
        )
        alternatives.append("实体较少时可用雷达图或分组柱状图；热力图适合整体模式而非精确读数。")

    return Plan(
        chart=chart,
        x=x,
        ys=ys,
        series=series,
        label_col=label_col,
        reasons=reasons,
        alternatives=alternatives,
        horizontal=horizontal,
        normalize=normalize,
    )


def _column_range_label(col: Column | None) -> str:
    if col is None:
        return ""
    if col.type == "time" and col.times:
        a, b = min(col.times), max(col.times)
        if col.year_like:
            return f"{a.year}–{b.year}"
        return f"{a:%Y-%m}–{b:%Y-%m}"
    if col.type == "numeric" and col.nums:
        return f"{_fmt(min(col.nums))}–{_fmt(max(col.nums))}"
    return f"{col.unique} 个取值"


# ────────────────────────── 统计与 Caption ──────────────────────────


def _x_value(raw: Any, col: Column) -> Any:
    if col.type == "time":
        return _parse_time(raw)
    if col.type == "numeric":
        return _num(raw)
    return str(raw or "").strip() or None


def _time_text(value: datetime, col: Column) -> str:
    return f"{value.year}" if col.year_like else f"{value:%Y-%m}"


def _x_span(col: Column, values: Sequence[Any]) -> tuple[str, str]:
    if col.type == "time":
        a, b = min(values), max(values)
        return _time_text(a, col), _time_text(b, col)
    a, b = min(values), max(values)
    return _fmt(a), _fmt(b)


def _line_stats(rows: Sequence[dict[str, Any]], plan: Plan) -> dict[str, Any]:
    assert plan.x is not None
    data: dict[str, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    n = 0
    for row in rows:
        xv = _x_value(row.get(plan.x.name), plan.x)
        if xv is None:
            continue
        row_used = False
        if plan.series:
            yv = _num(row.get(plan.ys[0].name))
            if yv is not None:
                key = str(row.get(plan.series.name) or "").strip() or "NA"
                data[key][xv].append(yv)
                row_used = True
        else:
            for yc in plan.ys:
                yv = _num(row.get(yc.name))
                if yv is not None:
                    data[yc.disp][xv].append(yv)
                    row_used = True
        if row_used:
            n += 1

    if not data:
        raise ValueError("折线图没有有效数据：请检查 x/y 字段与数值格式")

    x_values: list[Any] = []
    y_values: list[float] = []
    aggregated = False
    infos: list[dict[str, Any]] = []
    for key, by_x in data.items():
        xs = sorted(by_x)
        for xv in xs:
            if len(by_x[xv]) > 1:
                aggregated = True
            x_values.append(xv)
            y_values.extend(by_x[xv])
        first_y = _mean(by_x[xs[0]]) if xs else None
        last_y = _mean(by_x[xs[-1]]) if xs else None
        delta = None if first_y is None or last_y is None or len(xs) < 2 else last_y - first_y
        key_en = next((col.disp_en for col in plan.ys if col.disp == key), key)
        infos.append({
            "key": key,
            "key_en": key_en,
            "first": first_y,
            "last": last_y,
            "delta": delta,
            "n_points": len(xs),
        })

    deltas = [info for info in infos if info["delta"] is not None]
    best = max(deltas, key=lambda item: abs(item["delta"])) if deltas else None
    unit = plan.ys[0].unit if plan.series or len(plan.ys) == 1 else (
        plan.ys[0].unit if _same_units(plan.ys) else ""
    )
    if plan.series:
        y_label, y_label_en = plan.ys[0].disp, plan.ys[0].disp_en
    elif len(plan.ys) == 1:
        y_label, y_label_en = plan.ys[0].disp, plan.ys[0].disp_en
    else:
        y_label = " / ".join(col.disp for col in plan.ys)
        y_label_en = " / ".join(col.disp_en for col in plan.ys)

    xa, xb = _x_span(plan.x, x_values)
    return {
        "infos": infos,
        "k": len(data),
        "aggregated": aggregated,
        "n": n,
        "x_a": xa,
        "x_b": xb,
        "x_points": len(set(x_values)),
        "y_min": min(y_values),
        "y_max": max(y_values),
        "unit": unit,
        "y_label": y_label,
        "y_label_en": y_label_en,
        "best": best,
        "series": plan.series,
    }


def _line_caption(stats: dict[str, Any], plan: Plan, figure_no: int) -> dict[str, str]:
    xr_zh = _range_phrase(True, stats["x_a"], stats["x_b"])
    xr_en = _range_phrase(False, stats["x_a"], stats["x_b"])
    unit = stats["unit"]
    yl, yl_en = stats["y_label"], stats["y_label_en"]
    n = stats["n"]
    infos = stats["infos"]
    deltas = [item for item in infos if item["delta"] is not None]
    best = stats["best"]
    aggregated = stats["aggregated"]

    if stats["k"] == 1 and deltas:
        info = deltas[0]
        if not plan.series and len(plan.ys) > 1:
            yl, yl_en = info["key"], info["key_en"]
        a, b = info["first"], info["last"]
        delta = info["delta"]
        direction = _direction(delta, a)
        subject_zh = f"{yl} 的均值" if aggregated else yl
        subject_en = f"the mean {yl_en}" if aggregated else yl_en
        if stats["x_points"] >= 3:
            suffix_zh = {
                "up": "，整体呈上升趋势",
                "down": "，整体呈下降趋势",
                "flat": "，整体基本持平",
            }[direction]
            suffix_en = {
                "up": ", showing an overall upward trend",
                "down": ", showing an overall downward trend",
                "flat": ", with little overall change",
            }[direction]
        else:
            suffix_zh = suffix_en = ""
        zh = (
            f"图 {figure_no}｜{xr_zh}，{subject_zh}由 {_val(a, unit)} 变为 "
            f"{_val(b, unit)}（{_delta(delta, unit, 'zh')}）"
            f"{suffix_zh}；共 {n} 条观测。"
        )
        en = (
            f"Figure {figure_no} | {xr_en}, {subject_en} changed from {_val(a, unit)} "
            f"to {_val(b, unit)} ({_delta(delta, unit, 'en')})"
            f"{suffix_en}; n = {n}."
        )
        return {"zh": zh, "en": en}

    if not deltas:
        zh = (
            f"图 {figure_no}｜{xr_zh}，{yl} 的取值覆盖 "
            f"{_val(stats['y_min'], unit)}–{_val(stats['y_max'], unit)}；共 {n} 条观测。"
        )
        en = (
            f"Figure {figure_no} | {xr_en}, {yl_en} ranged from "
            f"{_val(stats['y_min'], unit)} to {_val(stats['y_max'], unit)}; n = {n}."
        )
        return {"zh": zh, "en": en}

    up = sum(1 for item in deltas if _direction(item["delta"], item["first"]) == "up")
    down = sum(1 for item in deltas if _direction(item["delta"], item["first"]) == "down")
    flat = sum(1 for item in deltas if _direction(item["delta"], item["first"]) == "flat")
    single = len(infos) - len(deltas)
    if plan.series:
        group_zh = _count_zh(len(infos), plan.series.disp)
        group_en = f"{len(infos)} {plan.series.disp_en} groups"
    else:
        group_zh = f"{len(infos)} 条指标序列"
        group_en = f"{len(infos)} metric series"

    best_zh = ""
    best_en = ""
    if best and abs(best["delta"]) > 0:
        best_zh = f"，{best['key']} 的变化最大（{_delta(best['delta'], unit, 'zh')}）"
        best_en = f", with the largest change for {best['key']} ({_delta(best['delta'], unit, 'en')})"
    single_zh = f"，另有 {single} 条仅含单个时间点" if single else ""
    single_en = f", while {single} had a single time point" if single else ""

    zh = (
        f"图 {figure_no}｜{xr_zh}，{group_zh}的{yl}取值覆盖 "
        f"{_val(stats['y_min'], unit)}–{_val(stats['y_max'], unit)}；"
        f"其中 {up} 个上升、{down} 个下降、{flat} 个基本持平"
        f"{single_zh}{best_zh}；共 {n} 条观测。"
    )
    en = (
        f"Figure {figure_no} | {xr_en}, {group_en} of {yl_en} spanned "
        f"{_val(stats['y_min'], unit)}–{_val(stats['y_max'], unit)}; "
        f"{up} increased, {down} decreased, and {flat} were stable"
        f"{single_en}{best_en}; n = {n}."
    )
    return {"zh": zh, "en": en}


def _direction(delta: float, base: float | None) -> str:
    if base is None or delta == 0:
        return "flat"
    if base == 0:
        return "up" if delta > 0 else "down"
    rel = delta / abs(base)
    if abs(rel) < 0.01:
        return "flat"
    return "up" if rel > 0 else "down"


def _bar_stats(rows: Sequence[dict[str, Any]], plan: Plan) -> dict[str, Any]:
    assert plan.x is not None
    cats: list[str] = []
    groups: list[str] = []
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    n = 0
    for row in rows:
        cat = str(row.get(plan.x.name) or "").strip()
        if not cat:
            continue
        if cat not in cats:
            cats.append(cat)
        if plan.series:
            group = str(row.get(plan.series.name) or "").strip() or "NA"
            pairs = [(group, _num(row.get(plan.ys[0].name)))]
        elif len(plan.ys) > 1:
            pairs = [(yc.disp, _num(row.get(yc.name))) for yc in plan.ys]
        else:
            pairs = [(plan.ys[0].disp, _num(row.get(plan.ys[0].name)))]
        row_used = False
        for group, yv in pairs:
            if yv is None:
                continue
            values[(cat, group)].append(yv)
            if group not in groups:
                groups.append(group)
            row_used = True
        if row_used:
            n += 1

    cats = [cat for cat in cats if any(values.get((cat, group)) for group in groups)]
    if not cats or not values:
        raise ValueError("柱状图没有有效数据：请检查类别与数值字段")

    def overall(cat: str) -> float:
        vals = [v for group in groups for v in values.get((cat, group), [])]
        return _mean(vals)

    multi_metrics = bool(len(plan.ys) > 1 and not plan.series)
    primary_group = plan.ys[0].disp if multi_metrics else None

    def primary(cat: str) -> float:
        if primary_group is not None:
            return _mean(values.get((cat, primary_group), []))
        return overall(cat)

    ranked = sorted(cats, key=primary, reverse=True)
    top, bottom = ranked[0], ranked[-1]
    top_val, bottom_val = primary(top), primary(bottom)
    top_ties = sum(1 for cat in cats if abs(primary(cat) - top_val) < 1e-12)
    bottom_ties = sum(1 for cat in cats if abs(primary(cat) - bottom_val) < 1e-12)
    all_means = [overall(cat) for cat in cats]
    agg = any(len(vals) > 1 for vals in values.values())
    unit = plan.ys[0].unit if plan.series or len(plan.ys) == 1 else (
        plan.ys[0].unit if _same_units(plan.ys) else ""
    )
    return {
        "n_cat": len(cats),
        "n_groups": len(groups),
        "n": n,
        "top": top,
        "bottom": bottom,
        "top_val": top_val,
        "bottom_val": bottom_val,
        "range": top_val - bottom_val,
        "top_ties": top_ties,
        "bottom_ties": bottom_ties,
        "vmin": min(all_means),
        "vmax": max(all_means),
        "agg": agg,
        "unit": unit,
        "multi_metrics": multi_metrics,
        "groups": groups,
    }


def _bar_caption(stats: dict[str, Any], plan: Plan, figure_no: int) -> dict[str, str]:
    assert plan.x is not None
    cat = plan.x.disp
    cat_en = plan.x.disp_en
    unit = stats["unit"]
    agg_word = "均值" if stats["agg"] else "值"
    agg_word_en = "mean " if stats["agg"] else ""
    err_zh = "；误差线为 1 个标准差" if stats["agg"] else ""
    err_en = "; error bars show 1 SD" if stats["agg"] else ""
    n = stats["n"]
    all_equal = abs(stats["range"]) < 1e-12

    if stats["n_cat"] == 1:
        yl = plan.ys[0].disp
        yl_en = plan.ys[0].disp_en
        zh = (
            f"图 {figure_no}｜在 1 个{cat}中，{stats['top']} 的{yl}{agg_word}为 "
            f"{_val(stats['top_val'], unit)}{err_zh}；共 {n} 条观测。"
        )
        en = (
            f"Figure {figure_no} | For the single {cat_en} category {stats['top']}, "
            f"the {agg_word_en}{yl_en} was {_val(stats['top_val'], unit)}{err_en}; n = {n}."
        )
        return {"zh": zh, "en": en}

    primary_metric = plan.ys[0]
    top_v = _val(stats["top_val"], unit)
    bottom_v = _val(stats["bottom_val"], unit)
    top_ties = stats["top_ties"]
    bottom_ties = stats["bottom_ties"]

    if stats["multi_metrics"]:
        if all_equal:
            zh = (
                f"图 {figure_no}｜在 {_count_zh(stats['n_cat'], cat)}上比较了 "
                f"{len(plan.ys)} 项指标；以 {primary_metric.disp} 计，所有类别"
                f"{agg_word}相同（{top_v}）{err_zh}；共 {n} 条观测。"
            )
            en = (
                f"Figure {figure_no} | Across {stats['n_cat']} {cat_en} categories, "
                f"{len(plan.ys)} metrics were compared; for {primary_metric.disp_en}, "
                f"all categories had the same value ({top_v}){err_en}; n = {n}."
            )
            return {"zh": zh, "en": en}
        top_zh = (
            f"{stats['top']} 的{agg_word}最高（{top_v}）" if top_ties == 1
            else f"{top_ties} 个类别并列最高（{top_v}）"
        )
        bottom_zh = (
            f"{stats['bottom']} 最低（{bottom_v}）" if bottom_ties == 1
            else f"{bottom_ties} 个类别并列最低（{bottom_v}）"
        )
        top_en = (
            f"{stats['top']} had the highest value ({top_v})" if top_ties == 1
            else f"{top_ties} categories tied for the highest value ({top_v})"
        )
        bottom_en = (
            f"{stats['bottom']} the lowest ({bottom_v})" if bottom_ties == 1
            else f"{bottom_ties} categories tied for the lowest ({bottom_v})"
        )
        zh = (
            f"图 {figure_no}｜在 {_count_zh(stats['n_cat'], cat)}上比较了 "
            f"{len(plan.ys)} 项指标；以 {primary_metric.disp} 计，{top_zh}，"
            f"{bottom_zh}，极差 {_val(stats['range'], unit)}{err_zh}；共 {n} 条观测。"
        )
        en = (
            f"Figure {figure_no} | Across {stats['n_cat']} {cat_en} categories, "
            f"{len(plan.ys)} metrics were compared; for {primary_metric.disp_en}, "
            f"{top_en} and {bottom_en}, a range of {_val(stats['range'], unit)}"
            f"{err_en}; n = {n}."
        )
        return {"zh": zh, "en": en}

    yl = plan.ys[0].disp
    yl_en = plan.ys[0].disp_en
    if plan.series:
        prefix_zh = (
            f"在 {_count_zh(stats['n_cat'], cat)}和 "
            f"{_count_zh(stats['n_groups'], plan.series.disp)}上，{yl} 的{agg_word}为 "
            f"{_val(stats['vmin'], unit)}–{_val(stats['vmax'], unit)}；"
        )
        prefix_en = (
            f"Across {stats['n_cat']} {cat_en} categories and "
            f"{stats['n_groups']} {plan.series.disp_en} groups, {agg_word_en}{yl_en} ranged "
            f"from {_val(stats['vmin'], unit)} to {_val(stats['vmax'], unit)}; "
        )
        if all_equal:
            zh = (
                f"图 {figure_no}｜{prefix_zh}所有类别总体{agg_word}相同"
                f"（{top_v}）{err_zh}；共 {n} 条观测。"
            )
            en = (
                f"Figure {figure_no} | {prefix_en}all categories had the same overall "
                f"value ({top_v}){err_en}; n = {n}."
            )
            return {"zh": zh, "en": en}
        if top_ties == 1:
            zh = f"图 {figure_no}｜{prefix_zh}{stats['top']} 的总体{agg_word}最高（{top_v}）{err_zh}；共 {n} 条观测。"
            en = f"Figure {figure_no} | {prefix_en}{stats['top']} had the highest overall {agg_word_en}{yl_en} ({top_v}){err_en}; n = {n}."
        else:
            zh = f"图 {figure_no}｜{prefix_zh}{top_ties} 个类别总体{agg_word}并列最高（{top_v}）{err_zh}；共 {n} 条观测。"
            en = f"Figure {figure_no} | {prefix_en}{top_ties} categories tied for the highest overall {agg_word_en}{yl_en} ({top_v}){err_en}; n = {n}."
        return {"zh": zh, "en": en}

    if all_equal:
        zh = (
            f"图 {figure_no}｜在 {_count_zh(stats['n_cat'], cat)}中，所有类别"
            f"{yl}{agg_word}相同（{top_v}）{err_zh}；共 {n} 条观测。"
        )
        en = (
            f"Figure {figure_no} | Across {stats['n_cat']} {cat_en} categories, all "
            f"categories had the same {agg_word_en}{yl_en} ({top_v}){err_en}; n = {n}."
        )
        return {"zh": zh, "en": en}

    if top_ties == 1:
        top_zh = f"{stats['top']} 的{yl}{agg_word}最高（{top_v}）"
        top_en = f"{stats['top']} had the highest {agg_word_en}{yl_en} ({top_v})"
    else:
        top_zh = f"{top_ties} 个类别{yl}{agg_word}并列最高（{top_v}）"
        top_en = f"{top_ties} categories tied for the highest {agg_word_en}{yl_en} ({top_v})"
    if bottom_ties == 1:
        bottom_zh = f"{stats['bottom']} 最低（{bottom_v}）"
        bottom_en = f"{stats['bottom']} the lowest ({bottom_v})"
    else:
        bottom_zh = f"{bottom_ties} 个类别并列最低（{bottom_v}）"
        bottom_en = f"{bottom_ties} categories tied for the lowest ({bottom_v})"

    zh = (
        f"图 {figure_no}｜在 {_count_zh(stats['n_cat'], cat)}中，{top_zh}，"
        f"{bottom_zh}，极差 {_val(stats['range'], unit)}{err_zh}；共 {n} 条观测。"
    )
    en = (
        f"Figure {figure_no} | Across {stats['n_cat']} {cat_en} categories, "
        f"{top_en} and {bottom_en}, a range of {_val(stats['range'], unit)}"
        f"{err_en}; n = {n}."
    )
    return {"zh": zh, "en": en}


def _scatter_stats(rows: Sequence[dict[str, Any]], plan: Plan) -> dict[str, Any]:
    assert plan.x is not None and plan.ys
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        xv = _num(row.get(plan.x.name))
        yv = _num(row.get(plan.ys[0].name))
        if xv is None or yv is None:
            continue
        xs.append(xv)
        ys.append(yv)
    if not xs:
        raise ValueError("散点图没有有效数据：请检查两个数值字段")
    r = _pearson(xs, ys)
    return {
        "n": len(xs),
        "r": r,
        "x_min": min(xs), "x_max": max(xs),
        "y_min": min(ys), "y_max": max(ys),
        "x_col": plan.x,
        "y_col": plan.ys[0],
        "labels": plan.label_col,
    }


def _scatter_caption(stats: dict[str, Any], figure_no: int) -> dict[str, str]:
    xc, yc = stats["x_col"], stats["y_col"]
    xl, xl_en = xc.disp, xc.disp_en
    yl, yl_en = yc.disp, yc.disp_en
    xu, yu = xc.unit, yc.unit
    xr = f"{_val(stats['x_min'], xu)}–{_val(stats['x_max'], xu)}"
    yr = f"{_val(stats['y_min'], yu)}–{_val(stats['y_max'], yu)}"
    n = stats["n"]
    r = stats["r"]
    if r is None:
        zh = (
            f"图 {figure_no}｜{xl}（{xr}）与 {yl}（{yr}）共有 {n} 条有效观测；"
            "样本量或方差不足，未计算相关系数。"
        )
        en = (
            f"Figure {figure_no} | {xl_en} ({xr}) and {yl_en} ({yr}) had {n} valid "
            f"observations (n = {n}); the correlation was not computed because of limited "
            "sample size or zero variance."
        )
        return {"zh": zh, "en": en}

    strength_zh, strength_en, sign_zh, sign_en = _corr_words(r)
    if sign_zh:
        zh = (
            f"图 {figure_no}｜在 {xl}（{xr}）与 {yl}（{yr}）的共 {n} 条有效观测中，"
            f"二者呈{strength_zh}{sign_zh}线性相关（Pearson r = {r:.2f}）；未评估因果关系。"
        )
        en = (
            f"Figure {figure_no} | With n = {n} observations of {xl_en} ({xr}) and "
            f"{yl_en} ({yr}), the two variables showed a {strength_en} {sign_en} linear "
            f"correlation (Pearson r = {r:.2f}); causality was not assessed."
        )
    else:
        zh = (
            f"图 {figure_no}｜在 {xl}（{xr}）与 {yl}（{yr}）的共 {n} 条有效观测中，"
            f"二者{strength_zh}线性相关（Pearson r = {r:.2f}）；未评估因果关系。"
        )
        en = (
            f"Figure {figure_no} | With n = {n} observations of {xl_en} ({xr}) and "
            f"{yl_en} ({yr}), the two variables showed {strength_en} linear correlation "
            f"(Pearson r = {r:.2f}); causality was not assessed."
        )
    return {"zh": zh, "en": en}


def _corr_words(r: float) -> tuple[str, str, str, str]:
    a = abs(r)
    if a < 0.1:
        return "几乎不存在", "almost no", "", ""
    if a < 0.3:
        strength_zh, strength_en = "弱", "weak"
    elif a < 0.5:
        strength_zh, strength_en = "中等", "moderate"
    elif a < 0.7:
        strength_zh, strength_en = "较强", "moderately strong"
    else:
        strength_zh, strength_en = "强", "strong"
    if r > 0:
        return strength_zh, strength_en, "正", "positive"
    return strength_zh, strength_en, "负", "negative"


def _hist_stats(rows: Sequence[dict[str, Any]], plan: Plan) -> dict[str, Any]:
    col = plan.ys[0]
    vals = [v for row in rows if (v := _num(row.get(col.name))) is not None]
    if not vals:
        raise ValueError("直方图没有有效数值")
    q1 = _quantile(vals, 0.25) if len(vals) >= 4 else None
    q3 = _quantile(vals, 0.75) if len(vals) >= 4 else None
    return {
        "col": col,
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "median": statistics.median(vals),
        "q1": q1,
        "q3": q3,
    }


def _hist_caption(stats: dict[str, Any], figure_no: int) -> dict[str, str]:
    col = stats["col"]
    u = col.unit
    n = stats["n"]
    extra_zh = ""
    extra_en = ""
    if stats["q1"] is not None and stats["q3"] is not None:
        extra_zh = (
            f"，中间 50% 的观测位于 {_val(stats['q1'], u)}–{_val(stats['q3'], u)}"
        )
        extra_en = (
            f"; the middle 50% of observations lay between "
            f"{_val(stats['q1'], u)} and {_val(stats['q3'], u)}"
        )
    zh = (
        f"图 {figure_no}｜{col.disp} 共 {n} 条观测，范围为 "
        f"{_val(stats['min'], u)}–{_val(stats['max'], u)}，中位数为 "
        f"{_val(stats['median'], u)}{extra_zh}。"
    )
    en = (
        f"Figure {figure_no} | The {col.disp_en} values (n = {n}) ranged from "
        f"{_val(stats['min'], u)} to {_val(stats['max'], u)}, with a median of "
        f"{_val(stats['median'], u)}{extra_en}."
    )
    return {"zh": zh, "en": en}


def _radar_matrix(rows: Sequence[dict[str, Any]], plan: Plan) -> dict[str, Any]:
    assert plan.x is not None
    data: dict[str, dict[str, list[float]]] = {}
    entities: list[str] = []
    n = 0
    for row in rows:
        entity = str(row.get(plan.x.name) or "").strip()
        if not entity:
            continue
        if entity not in data:
            data[entity] = {}
            entities.append(entity)
        row_used = False
        for metric in plan.ys:
            v = _num(row.get(metric.name))
            if v is not None:
                data[entity].setdefault(metric.name, []).append(v)
                row_used = True
        if row_used:
            n += 1
    entities = [e for e in entities if any(m.name in data[e] for m in plan.ys)]
    if not entities:
        raise ValueError("雷达图没有有效数据")

    means: dict[str, dict[str, float | None]] = {}
    for entity in entities:
        means[entity] = {
            metric.name: (_mean(data[entity][metric.name]) if data[entity].get(metric.name) else None)
            for metric in plan.ys
        }

    ranges: dict[str, tuple[float, float]] = {}
    for metric in plan.ys:
        vals = [means[e][metric.name] for e in entities if means[e][metric.name] is not None]
        if not vals:
            raise ValueError(f"指标“{metric.disp}”没有可用数据；请移除该指标或补充数值")
        ranges[metric.name] = (min(vals), max(vals))

    normalized: dict[str, dict[str, float]] = {}
    for entity in entities:
        normalized[entity] = {}
        for metric in plan.ys:
            v = means[entity][metric.name]
            lo, hi = ranges[metric.name]
            if v is None:
                normalized[entity][metric.name] = 0.0
            elif hi == lo:
                normalized[entity][metric.name] = 0.5
            else:
                normalized[entity][metric.name] = (v - lo) / (hi - lo)

    # 缺失指标在图中按 0 绘制，得分也必须按全部指标平均，否则排名与图形不一致
    scores = {
        entity: _mean([normalized[entity][metric.name] for metric in plan.ys])
        for entity in entities
    }
    ranked = sorted(entities, key=lambda e: scores[e], reverse=True)
    constants = [metric.disp for metric in plan.ys if ranges[metric.name][0] == ranges[metric.name][1]]
    return {
        "entities": entities,
        "means": means,
        "ranges": ranges,
        "normalized": normalized,
        "scores": scores,
        "top": ranked[0],
        "bottom": ranked[-1],
        "n": n,
        "missing": sum(
            1 for entity in entities for metric in plan.ys
            if means[entity][metric.name] is None
        ),
        "constants": constants,
    }


def _radar_caption(stats: dict[str, Any], plan: Plan, figure_no: int) -> dict[str, str]:
    assert plan.x is not None
    entity = plan.x.disp
    entity_en = plan.x.disp_en
    n_e = len(stats["entities"])
    m = len(plan.ys)
    scores = stats["scores"]
    lo, hi = min(scores.values()), max(scores.values())
    n = stats["n"]
    top_entities = [e for e in stats["entities"] if abs(scores[e] - hi) < 1e-12]
    bottom_entities = [e for e in stats["entities"] if abs(scores[e] - lo) < 1e-12]
    if n_e == 1:
        zh = (
            f"图 {figure_no}｜在 1 个{entity}的 {m} 项指标上，按列 min–max 归一化后的"
            f"平均得分为 {_fmt(lo)}；共 {n} 条观测。"
        )
        en = (
            f"Figure {figure_no} | Across one {entity_en} entry and {m} metrics, the "
            f"column-wise min–max normalized average score was {_fmt(lo)}; n = {n}."
        )
        return {"zh": zh, "en": en}

    prefix_zh = (
        f"图 {figure_no}｜在 {_count_zh(n_e, entity)}的 {m} 项指标上，"
        f"按列 min–max 归一化后的平均得分为 {_fmt(lo)}–{_fmt(hi)}"
    )
    prefix_en = (
        f"Figure {figure_no} | Across {n_e} {entity_en} entries and {m} metrics, "
        f"column-wise min–max normalized scores averaged {_fmt(lo)}–{_fmt(hi)}"
    )
    if hi - lo < 1e-12:
        zh = prefix_zh + f"；各实体平均得分相同，未区分高低；共 {n} 条观测。"
        en = prefix_en + f"; all entities had the same average score; n = {n}."
        return {"zh": zh, "en": en}

    if len(top_entities) == 1:
        top_zh = f"{top_entities[0]} 最高（{_fmt(hi)}）"
        top_en = f"{top_entities[0]} had the highest average ({_fmt(hi)})"
    else:
        top_zh = f"{len(top_entities)} 个实体并列最高（{_fmt(hi)}）"
        top_en = f"{len(top_entities)} entities tied for the highest average ({_fmt(hi)})"
    if len(bottom_entities) == 1:
        bottom_zh = f"{bottom_entities[0]} 最低（{_fmt(lo)}）"
        bottom_en = f"{bottom_entities[0]} the lowest ({_fmt(lo)})"
    else:
        bottom_zh = f"{len(bottom_entities)} 个实体并列最低（{_fmt(lo)}）"
        bottom_en = f"{len(bottom_entities)} entities tied for the lowest ({_fmt(lo)})"
    zh = f"{prefix_zh}；{top_zh}，{bottom_zh}；共 {n} 条观测。"
    en = f"{prefix_en}; {top_en} and {bottom_en}; n = {n}."
    return {"zh": zh, "en": en}


def _heatmap_stats(rows: Sequence[dict[str, Any]], plan: Plan) -> dict[str, Any]:
    base = _radar_matrix(rows, plan)
    entities = base["entities"]
    means = base["means"]
    z: list[list[float]] = [[0.0] * len(plan.ys) for _ in entities]
    for j, metric in enumerate(plan.ys):
        col = [means[e][metric.name] for e in entities if means[e][metric.name] is not None]
        mu = _mean(col)
        sd = statistics.stdev(col) if len(col) > 1 else 0.0
        for i, entity in enumerate(entities):
            v = means[entity][metric.name]
            z[i][j] = 0.0 if v is None or sd == 0 else (v - mu) / sd

    flat = [
        (z[i][j], entities[i], plan.ys[j].name)
        for i in range(len(entities)) for j in range(len(plan.ys))
    ]
    lo = min(item[0] for item in flat)
    hi = max(item[0] for item in flat)
    return {
        "entities": entities,
        "z": z,
        "lo": lo,
        "hi": hi,
        "max_cell": max(flat, key=lambda item: item[0]),
        "min_cell": min(flat, key=lambda item: item[0]),
        "n": base["n"],
    }


def _heatmap_caption(stats: dict[str, Any], plan: Plan, figure_no: int) -> dict[str, str]:
    assert plan.x is not None
    entity = plan.x.disp
    entity_en = plan.x.disp_en
    n_e = len(stats["entities"])
    m = len(plan.ys)
    n = stats["n"]
    metric_label = {metric.name: metric.disp for metric in plan.ys}
    metric_label_en = {metric.name: metric.disp_en for metric in plan.ys}
    if abs(stats["hi"] - stats["lo"]) < 1e-12:
        zh = (
            f"图 {figure_no}｜在 {_count_zh(n_e, entity)} × {m} 项指标上，按列 z-score "
            f"标准化后的均值均为 {_fmt(stats['lo'])}，各单元格无差异；共 {n} 条观测。"
        )
        en = (
            f"Figure {figure_no} | Across {n_e} {entity_en} entries and {m} metrics, "
            f"column-wise z-scored means were all {_fmt(stats['lo'])}, with no "
            f"between-cell differences; n = {n}."
        )
        return {"zh": zh, "en": en}

    max_v, max_e, max_metric = stats["max_cell"]
    min_v, min_e, min_metric = stats["min_cell"]
    zh = (
        f"图 {figure_no}｜在 {_count_zh(n_e, entity)} × {m} 项指标上，按列 z-score 标准化后的"
        f"均值为 {_fmt(stats['lo'])}–{_fmt(stats['hi'])}；最高值出现在 {max_e} 的"
        f"{metric_label[max_metric]}（z = {_fmt(max_v)}），最低值出现在 {min_e} 的"
        f"{metric_label[min_metric]}（z = {_fmt(min_v)}）；共 {n} 条观测。"
    )
    en = (
        f"Figure {figure_no} | Across {n_e} {entity_en} entries and {m} metrics, "
        f"column-wise z-scored means ranged from {_fmt(stats['lo'])} to {_fmt(stats['hi'])}; "
        f"the highest value was {max_e} / {metric_label_en[max_metric]} "
        f"(z = {_fmt(max_v)}) and the lowest was {min_e} / {metric_label_en[min_metric]} "
        f"(z = {_fmt(min_v)}); n = {n}."
    )
    return {"zh": zh, "en": en}


def _bar_counts_stats(rows: Sequence[dict[str, Any]], x: Column) -> dict[str, Any]:
    counts: Counter = Counter()
    for row in rows:
        value = str(row.get(x.name) or "").strip()
        if value:
            counts[value] += 1
    if not counts:
        raise ValueError("类别频数图没有有效类别")
    ranked = counts.most_common()
    return {
        "counts": counts,
        "n_cat": len(counts),
        "n": sum(counts.values()),
        "top": ranked[0][0],
        "top_val": ranked[0][1],
        "bottom": ranked[-1][0],
        "bottom_val": ranked[-1][1],
    }


def _bar_counts_caption(stats: dict[str, Any], x: Column, figure_no: int) -> dict[str, str]:
    if stats["n_cat"] == 1:
        zh = (
            f"图 {figure_no}｜{x.disp} 仅有 1 个类别“{stats['top']}”，共 "
            f"{stats['n']} 条记录。"
        )
        en = (
            f"Figure {figure_no} | {x.disp_en} contained a single category "
            f"“{stats['top']}” with n = {stats['n']} records."
        )
        return {"zh": zh, "en": en}
    if stats["top_val"] == stats["bottom_val"]:
        zh = (
            f"图 {figure_no}｜{x.disp} 的 {stats['n_cat']} 个类别共 "
            f"{stats['n']} 条记录；各类别频数相同（{stats['top_val']} 条）。"
        )
        en = (
            f"Figure {figure_no} | The {stats['n_cat']} {x.disp_en} categories "
            f"comprised n = {stats['n']} records; all categories had the same "
            f"frequency ({stats['top_val']})."
        )
        return {"zh": zh, "en": en}
    zh = (
        f"图 {figure_no}｜{x.disp} 的 {stats['n_cat']} 个类别共 {stats['n']} 条记录；"
        f"{stats['top']} 最多（{stats['top_val']} 条），{stats['bottom']} 最少"
        f"（{stats['bottom_val']} 条）。"
    )
    en = (
        f"Figure {figure_no} | The {stats['n_cat']} {x.disp_en} categories comprised "
        f"n = {stats['n']} records; {stats['top']} was most frequent ({stats['top_val']}) "
        f"and {stats['bottom']} least frequent ({stats['bottom_val']})."
    )
    return {"zh": zh, "en": en}


# ────────────────────────── 脚本生成 ──────────────────────────


_SCRIPT_HEAD = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 由 app.chart 生成；依赖：matplotlib>=3.5（pip install matplotlib）
# 运行：__RUN__
# 数据：__SOURCE__

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

PALETTE = __PALETTE__
CJK = __CJK__

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

CSV_PATH = __CSV_PATH__
X_COL = __X_COL__
Y_COLS = __Y_COLS__
SERIES_COL = __SERIES_COL__
LABEL_COL = __LABEL_COL__
LABELS = __LABELS__
TITLE = __TITLE__
X_LABEL = __X_LABEL__
Y_LABEL = __Y_LABEL__
TIME_AXIS = __TIME_AXIS__
HORIZONTAL = __HORIZONTAL__
POLAR = __POLAR__
FIG_W = __FIG_W__
FIG_H = __FIG_H__
EMBEDDED_ROWS = __ROWS__


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
    ap = argparse.ArgumentParser(description="__DESC__")
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


if __name__ == "__main__":
    main()
'''


_LINE_BODY = '''
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
'''

_BAR_BODY = '''
def draw(ax, fig, rows):
    cats, groups = [], []
    values = {}
    for row in rows:
        cat = str(row.get(X_COL) or "").strip()
        if not cat:
            continue
        if cat not in cats:
            cats.append(cat)
        if SERIES_COL:
            group = str(row.get(SERIES_COL) or "").strip() or "NA"
            pairs = [(group, num(row.get(Y_COLS[0])))]
        elif len(Y_COLS) > 1:
            pairs = [(LABELS.get(yc, yc), num(row.get(yc))) for yc in Y_COLS]
        else:
            pairs = [(LABELS.get(Y_COLS[0], Y_COLS[0]), num(row.get(Y_COLS[0])))]
        for group, yv in pairs:
            if yv is None:
                continue
            values.setdefault((cat, group), []).append(yv)
            if group not in groups:
                groups.append(group)
    cats = [cat for cat in cats if any(values.get((cat, group)) for group in groups)]
    if not cats:
        raise SystemExit("没有可用类别")

    def cat_mean(cat):
        if not SERIES_COL and len(Y_COLS) > 1:
            vals = values.get((cat, LABELS.get(Y_COLS[0], Y_COLS[0])), [])
        else:
            vals = [v for group in groups for v in values.get((cat, group), [])]
        return mean(vals) if vals else float("-inf")

    cats = sorted(cats, key=cat_mean, reverse=True)
    n = max(1, len(groups))
    width = 0.8 / n
    for i, group in enumerate(groups):
        means, errs = [], []
        for cat in cats:
            vals = values.get((cat, group), [])
            means.append(mean(vals) if vals else float("nan"))
            errs.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        pos = list(range(len(cats))) if n == 1 else [
            k + (i - (n - 1) / 2) * width for k in range(len(cats))
        ]
        label = group if (SERIES_COL or len(Y_COLS) > 1) else None
        if HORIZONTAL:
            ax.barh(pos, means, height=width, color=PALETTE[i % len(PALETTE)],
                    label=label, xerr=errs if any(errs) else None, capsize=2)
        else:
            ax.bar(pos, means, width=width, color=PALETTE[i % len(PALETTE)],
                   label=label, yerr=errs if any(errs) else None, capsize=2)

    ticks = list(range(len(cats)))
    if HORIZONTAL:
        ax.set_yticks(ticks)
        ax.set_yticklabels(cats)
        ax.invert_yaxis()
        ax.set_xlabel(Y_LABEL)
        ax.set_ylabel(X_LABEL)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(cats, rotation=30 if len(cats) > 6 else 0,
                           ha="right" if len(cats) > 6 else "center")
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
    if SERIES_COL or len(Y_COLS) > 1:
        ax.legend(frameon=False, loc="best")
'''

_SCATTER_BODY = '''
def draw(ax, fig, rows):
    xs, ys, labels = [], [], []
    for row in rows:
        xv = num(row.get(X_COL))
        yv = num(row.get(Y_COLS[0]))
        if xv is None or yv is None:
            continue
        xs.append(xv)
        ys.append(yv)
        if LABEL_COL:
            labels.append(str(row.get(LABEL_COL) or "").strip())
    ax.scatter(xs, ys, s=18, alpha=0.75, color=PALETTE[0],
               edgecolors="white", linewidths=0.4, zorder=2)
    if LABEL_COL and len(xs) <= 20:
        for xv, yv, label in zip(xs, ys, labels):
            if label:
                ax.annotate(label, (xv, yv), xytext=(3, 3), textcoords="offset points",
                            fontsize=7, color="#4B5563")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
'''

_HIST_BODY = '''
def draw(ax, fig, rows):
    vals = [v for v in (num(row.get(Y_COLS[0])) for row in rows) if v is not None]
    if not vals:
        raise SystemExit("没有可用数值")
    ax.hist(vals, bins="auto", color=PALETTE[0], edgecolor="white",
            linewidth=0.6, alpha=0.9)
    med = statistics.median(vals)
    ax.axvline(med, color=PALETTE[1], linestyle="--", linewidth=1.0,
               label=f"Median = {med:.3g}")
    ax.set_xlabel(Y_LABEL)
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
'''

_RADAR_BODY = '''
def draw(ax, fig, rows):
    data = {}
    entities = []
    for row in rows:
        entity = str(row.get(X_COL) or "").strip()
        if not entity:
            continue
        if entity not in data:
            data[entity] = {}
            entities.append(entity)
        for metric in Y_COLS:
            v = num(row.get(metric))
            if v is not None:
                data[entity].setdefault(metric, []).append(v)
    entities = [e for e in entities if any(m in data[e] for m in Y_COLS)]
    if not entities:
        raise SystemExit("没有可用实体")

    def cell(entity, metric):
        vals = data[entity].get(metric, [])
        return mean(vals) if vals else None

    ranges = {}
    for metric in Y_COLS:
        vals = [cell(e, metric) for e in entities if cell(e, metric) is not None]
        if not vals:
            raise SystemExit(f"指标 {LABELS.get(metric, metric)} 没有可用数据")
        ranges[metric] = (min(vals), max(vals))

    def scaled(entity):
        out = []
        for metric in Y_COLS:
            value = cell(entity, metric)
            lo, hi = ranges[metric]
            if value is None:
                out.append(0.0)
            elif hi == lo:
                out.append(0.5)
            else:
                out.append((value - lo) / (hi - lo))
        return out

    angles = [2 * math.pi * i / len(Y_COLS) for i in range(len(Y_COLS))]
    angles += angles[:1]
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    for i, entity in enumerate(entities):
        values = scaled(entity)
        values += values[:1]
        ax.plot(angles, values, color=PALETTE[i % len(PALETTE)], linewidth=1.4, label=entity)
        ax.fill(angles, values, color=PALETTE[i % len(PALETTE)], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([LABELS.get(m, m) for m in Y_COLS])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.spines["polar"].set_color("#D1D5DB")
    ax.grid(color="#E5E7EB", linewidth=0.6)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.28, 1.12))
'''

_HEATMAP_BODY = '''
def draw(ax, fig, rows):
    data = {}
    entities = []
    for row in rows:
        entity = str(row.get(X_COL) or "").strip()
        if not entity:
            continue
        if entity not in data:
            data[entity] = {}
            entities.append(entity)
        for metric in Y_COLS:
            v = num(row.get(metric))
            if v is not None:
                data[entity].setdefault(metric, []).append(v)
    entities = [e for e in entities if any(m in data[e] for m in Y_COLS)]
    if not entities:
        raise SystemExit("没有可用实体")

    matrix = [
        [mean(data[e][m]) if data[e].get(m) else float("nan") for m in Y_COLS]
        for e in entities
    ]
    for j in range(len(Y_COLS)):
        col = [matrix[i][j] for i in range(len(entities)) if not math.isnan(matrix[i][j])]
        if not col:
            raise SystemExit(f"指标 {LABELS.get(Y_COLS[j], Y_COLS[j])} 没有可用数据")
        mu = mean(col)
        sd = statistics.stdev(col) if len(col) > 1 else 0.0
        for i in range(len(entities)):
            if math.isnan(matrix[i][j]):
                continue
            matrix[i][j] = (matrix[i][j] - mu) / sd if sd > 0 else 0.0

    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(Y_COLS)))
    ax.set_xticklabels([LABELS.get(m, m) for m in Y_COLS], rotation=30, ha="right")
    ax.set_yticks(range(len(entities)))
    ax.set_yticklabels(entities)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Column z-score")
'''

_BAR_COUNTS_BODY = '''
def draw(ax, fig, rows):
    counts = {}
    for row in rows:
        cat = str(row.get(X_COL) or "").strip()
        if cat:
            counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        raise SystemExit("没有可用类别")
    cats = sorted(counts, key=counts.get, reverse=True)
    vals = [counts[c] for c in cats]
    if HORIZONTAL:
        ax.barh(range(len(cats)), vals, color=PALETTE[0], height=0.7)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_ylabel(X_LABEL)
    else:
        ax.bar(range(len(cats)), vals, color=PALETTE[0], width=0.7)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=30 if len(cats) > 6 else 0,
                           ha="right" if len(cats) > 6 else "center")
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel("Count")
'''

_BODIES = {
    "line": _LINE_BODY,
    "bar": _BAR_BODY,
    "grouped_bar": _BAR_BODY,
    "scatter": _SCATTER_BODY,
    "histogram": _HIST_BODY,
    "radar": _RADAR_BODY,
    "heatmap": _HEATMAP_BODY,
    "bar_counts": _BAR_COUNTS_BODY,
}


def _axis_y_label(plan: Plan) -> str:
    if not plan.ys:
        return "Count"
    if len(plan.ys) == 1 or plan.series:
        return _axis_label(plan.ys[0])
    names = " / ".join(col.disp for col in plan.ys)
    if len(names) > 30:
        names = "指标值" if _has_cjk(names) else "Value"
    units = {col.unit for col in plan.ys}
    if len(units) == 1 and plan.ys[0].unit:
        names = f"{names} ({plan.ys[0].unit})"
    return names


def _figure_size(plan: Plan, rows: Sequence[dict[str, Any]]) -> tuple[float, float]:
    if plan.chart == "radar":
        return 5.6, 5.2
    if plan.chart == "heatmap":
        n_e = plan.x.unique if plan.x else 6
        return 7.0, max(3.6, min(10.0, 0.34 * n_e + 2.0))
    return 7.2, 4.2


def _make_script(plan: Plan, rows: Sequence[dict[str, Any]], csv_path: str | None,
                 embed_rows: bool, title: str | None = None) -> tuple[str, str]:
    x_name = plan.x.name if plan.x else ""
    y_names = [col.name for col in plan.ys]
    series_name = plan.series.name if plan.series else ""
    label_name = plan.label_col.name if plan.label_col else ""
    labels = {col.name: col.disp for col in plan.ys}
    if plan.x:
        labels.setdefault(plan.x.name, plan.x.disp)
    if plan.series:
        labels.setdefault(plan.series.name, plan.series.disp)
    if plan.label_col:
        labels.setdefault(plan.label_col.name, plan.label_col.disp)

    x_label = _axis_label(plan.x) if plan.x else (plan.ys[0].disp if plan.ys else "")
    y_label = _axis_y_label(plan)
    if plan.chart == "bar_counts":
        y_label = "Count"
    source = csv_path or ("脚本内置数据" if embed_rows else "CSV")
    csv_value = csv_path or ""
    title_text = str(title or "").strip()
    text_cols = [col for col in (plan.x, plan.series, plan.label_col) if col]
    sample_values = [
        str(row.get(col.name) or "") for col in text_cols for row in rows[:50]
    ]
    all_text = " ".join([title_text, x_label, y_label, *labels.values(), *sample_values])
    cjk = _has_cjk(all_text)
    fig_w, fig_h = _figure_size(plan, rows)
    run = (
        "python figure.py figure.png"
        if embed_rows and not csv_path
        else f"python figure.py {csv_value} figure.png"
    )

    script = _SCRIPT_HEAD
    replacements = {
        "__RUN__": run,
        "__SOURCE__": source,
        "__PALETTE__": repr(PALETTE),
        "__CSV_PATH__": repr(csv_value),
        "__X_COL__": repr(x_name),
        "__Y_COLS__": repr(y_names),
        "__SERIES_COL__": repr(series_name),
        "__LABEL_COL__": repr(label_name),
        "__LABELS__": repr(labels),
        "__TITLE__": repr(title_text),
        "__CJK__": repr(cjk),
        "__X_LABEL__": repr(x_label),
        "__Y_LABEL__": repr(y_label),
        "__TIME_AXIS__": repr(bool(plan.x and plan.x.type == "time")),
        "__HORIZONTAL__": repr(plan.horizontal),
        "__POLAR__": repr(plan.chart == "radar"),
        "__FIG_W__": repr(fig_w),
        "__FIG_H__": repr(fig_h),
        "__ROWS__": repr(list(rows)) if embed_rows else "[]",
        "__DESC__": "由 app.chart 生成的学术图表",
    }
    for token, value in replacements.items():
        script = script.replace(token, value)

    body = _BODIES[plan.chart]
    marker = "if __name__ == \"__main__\":"
    script = script.replace(marker, body.strip() + "\n\n\n" + marker, 1)
    return script, run


# ────────────────────────── 校验与主流程 ──────────────────────────


def verify_captions(captions: dict[str, str]) -> list[str]:
    zh, en = captions["zh"], captions["en"]
    issues: list[str] = []
    if not zh.endswith("。") or zh.count("。") != 1:
        issues.append("中文 Caption 必须是一句以“。”结尾的完整句")
    if not en.endswith(".") or ". " in en[:-1]:
        issues.append("English caption must be a single sentence ending with a period")
    for word in HYPE_ZH:
        if word in zh:
            issues.append(f"中文 Caption 含夸大/因果措辞：{word}")
    low = en.lower()
    for word in HYPE_EN:
        if re.search(rf"\b{re.escape(word)}\b", low):
            issues.append(f"English caption contains overstated wording: {word}")
    if "共" not in zh or "n =" not in en:
        issues.append("Caption 未标注观测数")
    return issues


def analyze(rows: Sequence[dict[str, Any]], schema: Any = None,
            title: str | None = None, csv_path: str | None = None,
            figure_no: int = 1, summary: Any = None) -> dict[str, Any]:
    if not rows:
        raise ValueError("没有数据行：请提供 CSV 或 JSON 中的 rows")

    cols = profile(rows, schema)
    plan = _choose_plan(rows, cols)
    issues: list[str] = []

    for col in cols.values():
        if col.note:
            issues.append(col.note)

    used_cols: list[Column] = []
    if plan.x:
        used_cols.append(plan.x)
    used_cols.extend(plan.ys)
    if plan.series:
        used_cols.append(plan.series)
    if plan.label_col:
        used_cols.append(plan.label_col)
    missing = sum(col.missing for col in used_cols)
    if missing:
        issues.append(f"参与绘图的字段共有 {missing} 个空值，统计与绘图时已跳过")

    if len(rows) < 3:
        issues.append(f"样本量较小（{len(rows)} 行），Caption 只陈述观测值，不作统计推断")

    if plan.chart in ("line", "grouped_bar") and len(plan.ys) > 1:
        if all(not col.unit for col in plan.ys):
            issues.append("多个指标未声明单位；图中按同一量纲处理，若量纲不同请补充 unit 或拆分小图")

    if isinstance(summary, dict):
        claimed = summary.get("n_rows")
        if isinstance(claimed, (int, float)) and int(claimed) != len(rows):
            issues.append(
                f"data_summary 给出的 n_rows={int(claimed)} 与实际 {len(rows)} 行不一致，以实际数据为准"
            )

    if plan.chart == "line":
        stats = _line_stats(rows, plan)
        captions = _line_caption(stats, plan, figure_no)
        if stats["aggregated"]:
            issues.append("同一时间点存在重复观测，图中使用均值")
        if len(plan.ys) < len([c for c in cols.values() if c.type == "numeric" and c.role != "ignore"]):
            dropped = [
                c.disp for c in cols.values()
                if c.type == "numeric" and c.role != "ignore" and c.name not in {y.name for y in plan.ys}
            ]
            if dropped:
                issues.append(
                    "折线图各指标单位不一致，仅绘制 " + "、".join(y.disp for y in plan.ys)
                    + "；未绘制：" + "、".join(dropped) + "（建议拆分小图）"
                )
        stats_out = {
            "n_used": stats["n"],
            "x_range": [stats["x_a"], stats["x_b"]],
            "y_range": [stats["y_min"], stats["y_max"]],
        }
    elif plan.chart in ("bar", "grouped_bar"):
        stats = _bar_stats(rows, plan)
        captions = _bar_caption(stats, plan, figure_no)
        if stats["agg"]:
            issues.append("同一类别存在重复观测，柱高为均值，误差线为 1 SD")
        stats_out = {
            "n_used": stats["n"],
            "n_categories": stats["n_cat"],
            "top": stats["top"],
            "bottom": stats["bottom"],
            "range": stats["range"],
        }
    elif plan.chart == "scatter":
        stats = _scatter_stats(rows, plan)
        captions = _scatter_caption(stats, figure_no)
        used_numeric = {plan.x.name, plan.ys[0].name}
        dropped = [
            col.disp for col in cols.values()
            if col.type == "numeric" and col.role != "ignore" and col.name not in used_numeric
        ]
        if dropped:
            issues.append("散点图一次展示两个变量；未绘制：" + "、".join(dropped))
        if stats["r"] is None:
            issues.append("样本量不足或某一变量方差为零，未计算 Pearson 相关系数")
        stats_out = {
            "n_used": stats["n"],
            "pearson_r": stats["r"],
            "x_range": [stats["x_min"], stats["x_max"]],
            "y_range": [stats["y_min"], stats["y_max"]],
        }
    elif plan.chart == "histogram":
        stats = _hist_stats(rows, plan)
        captions = _hist_caption(stats, figure_no)
        stats_out = {
            "n_used": stats["n"],
            "range": [stats["min"], stats["max"]],
            "median": stats["median"],
            "q1": stats["q1"],
            "q3": stats["q3"],
        }
    elif plan.chart == "radar":
        stats = _radar_matrix(rows, plan)
        captions = _radar_caption(stats, plan, figure_no)
        if stats["constants"]:
            issues.append("常数指标（" + "、".join(stats["constants"]) + "）归一化后记为 0.5")
        if stats["missing"]:
            issues.append("雷达图中缺失指标按归一化下限 0 绘制，平均得分按全部指标计算")
        stats_out = {
            "n_used": stats["n"],
            "n_entities": len(stats["entities"]),
            "n_metrics": len(plan.ys),
            "top": stats["top"],
            "bottom": stats["bottom"],
            "top_score": stats["scores"][stats["top"]],
        }
    elif plan.chart == "heatmap":
        stats = _heatmap_stats(rows, plan)
        captions = _heatmap_caption(stats, plan, figure_no)
        stats_out = {
            "n_used": stats["n"],
            "n_entities": len(stats["entities"]),
            "n_metrics": len(plan.ys),
            "z_range": [stats["lo"], stats["hi"]],
        }
    else:  # bar_counts
        if plan.x is None:
            raise ValueError("频数图缺少类别列")
        stats = _bar_counts_stats(rows, plan.x)
        captions = _bar_counts_caption(stats, plan.x, figure_no)
        stats_out = {
            "n_used": stats["n"],
            "n_categories": stats["n_cat"],
            "top": stats["top"],
            "bottom": stats["bottom"],
        }

    caption_issues = verify_captions(captions)
    issues.extend(caption_issues)

    script, run = _make_script(
        plan, rows, csv_path,
        embed_rows=bool(csv_path is None), title=title,
    )

    chart_name = CHART_NAMES[plan.chart]
    if plan.chart == "bar" and plan.series:
        chart_name = "分组柱状图"

    return {
        "chart": {
            "type": plan.chart,
            "name": chart_name,
            "x": plan.x.name if plan.x else None,
            "y": [col.name for col in plan.ys],
            "series": plan.series.name if plan.series else None,
            "label": plan.label_col.name if plan.label_col else None,
            "reasons": plan.reasons,
            "alternatives": plan.alternatives,
        },
        "filename": "figure.py",
        "script": script,
        "run": run,
        "caption": captions,
        "stats": stats_out,
        "issues": list(dict.fromkeys(issues)),
    }


# ────────────────────────── 演示与自检 ──────────────────────────


def demo_input() -> dict[str, Any]:
    rows = []
    data = {
        2020: {"Model A": 62.1, "Model B": 60.4, "Model C": 63.0},
        2021: {"Model A": 65.3, "Model B": 63.8, "Model C": 67.1},
        2022: {"Model A": 68.9, "Model B": 66.2, "Model C": 71.4},
        2023: {"Model A": 72.5, "Model B": 69.0, "Model C": 75.2},
        2024: {"Model A": 76.8, "Model B": 71.6, "Model C": 79.3},
    }
    for year, values in data.items():
        for model, accuracy in values.items():
            rows.append({"year": str(year), "model": model, "accuracy": str(accuracy)})
    return {
        "rows": rows,
        "title": "",
        "schema": {
            "year": {"type": "time", "role": "x", "label": "年份", "label_en": "Year"},
            "model": {"type": "category", "role": "series", "label": "模型", "label_en": "Model"},
            "accuracy": {"type": "numeric", "role": "y", "unit": "%",
                         "label": "准确率", "label_en": "Accuracy"},
        },
    }


def _selftest() -> int:
    checks = 0
    failures = 0

    def ok(name: str, cond: bool, detail: str = "") -> None:
        nonlocal checks, failures
        checks += 1
        if cond:
            print(f"[OK] {name}")
        else:
            failures += 1
            print(f"[XX] {name}{'  → ' + detail if detail else ''}")

    demo = demo_input()
    out = analyze(demo["rows"], demo["schema"])
    ok("演示识别为折线图", out["chart"]["type"] == "line", out["chart"]["type"])
    ok("识别出 series 列", out["chart"]["series"] == "model")
    ok("Captions 双语非空",
       bool(out["caption"]["zh"]) and bool(out["caption"]["en"]))
    ok("中文 Caption 为一句", out["caption"]["zh"].count("。") == 1)
    ok("英文 Caption 为一句", out["caption"]["en"].count(". ") == 0 and out["caption"]["en"].endswith("."))
    ok("Caption 含观测数", "共 15 条观测" in out["caption"]["zh"] and "n = 15" in out["caption"]["en"])
    ok("Caption 无夸大措辞",
       not any(word in out["caption"]["zh"] for word in ("显著", "大幅", "证明")))
    ok("Caption 使用真实数据范围与关键发现",
       all(token in out["caption"]["zh"] for token in ("60.4", "79.3", "16.3")),
       out["caption"]["zh"])
    ok("脚本可编译", _compiles(out["script"]))
    ok("脚本含 300 dpi", "dpi=300" in out["script"])
    ok("脚本含学术配色", "#0072B2" in out["script"] and "#D55E00" in out["script"])
    ok("脚本不依赖 pandas", "pandas" not in out["script"])
    ok("脚本可独立读取 CSV", "csv.DictReader" in out["script"])
    ok("脚本内置演示数据", "EMBEDDED_ROWS = [" in out["script"])
    ok("脚本含图例", "ax.legend" in out["script"])
    ok("运行说明可用", out["run"].startswith("python figure.py"))
    titled = analyze(demo["rows"], demo["schema"], title="准确率对比")
    ok("标题写入脚本", "TITLE = '准确率对比'" in titled["script"])
    ok("中文图优先使用中文字体", '"Microsoft YaHei", "SimHei"' in titled["script"])
    cjk_values = analyze(
        [{"group": "甲组", "score": "1"}, {"group": "乙组", "score": "2"}],
        {"group": "category", "score": "numeric"},
    )
    ok("中文类别值优先使用中文字体",
       '"Microsoft YaHei", "SimHei"' in cjk_values["script"])

    # 柱状图：类别 + 一个数值
    bar_rows = [
        {"method": "A", "score": "0.81"},
        {"method": "B", "score": "0.77"},
        {"method": "C", "score": "0.84"},
    ]
    bar = analyze(bar_rows, {"method": "category", "score": "numeric"})
    ok("类别+数值推荐柱状图", bar["chart"]["type"] == "bar", bar["chart"]["type"])
    ok("柱状图 Caption 给出极值",
       "C" in bar["caption"]["zh"] and "最高" in bar["caption"]["zh"]
       and "最低" in bar["caption"]["zh"], bar["caption"]["zh"])

    # 分组柱状图：两个同单位指标
    grouped_rows = [
        {"method": "A", "acc": "0.8", "f1": "0.75"},
        {"method": "B", "acc": "0.7", "f1": "0.72"},
    ]
    grouped = analyze(grouped_rows, {"method": "category", "acc": "numeric", "f1": "numeric"})
    ok("两个同单位指标推荐分组柱",
       grouped["chart"]["type"] == "grouped_bar", grouped["chart"]["type"])

    # 两指标不同单位 → 散点
    mixed = analyze([
        {"method": "A", "acc": "0.8", "latency": "12"},
        {"method": "B", "acc": "0.7", "latency": "20"},
        {"method": "C", "acc": "0.9", "latency": "15"},
    ], {
        "method": {"type": "category", "role": "x"},
        "acc": {"type": "numeric", "role": "y", "unit": "%"},
        "latency": {"type": "numeric", "role": "y", "unit": "ms"},
    })
    ok("不同单位的两指标推荐散点图",
       mixed["chart"]["type"] == "scatter", mixed["chart"]["type"])
    ok("散点 Caption 明确未评估因果",
       "未评估因果关系" in mixed["caption"]["zh"]
       and "causality was not assessed" in mixed["caption"]["en"])

    # 多指标多实体 → 雷达 / 热图
    radar_rows = []
    for model in ("A", "B", "C"):
        radar_rows.append({
            "model": model, "acc": "0.8", "f1": "0.7", "recall": "0.75",
        })
    radar = analyze(radar_rows, {
        "model": {"type": "category", "role": "x"},
        "acc": {"type": "numeric", "role": "y"},
        "f1": {"type": "numeric", "role": "y"},
        "recall": {"type": "numeric", "role": "y"},
    })
    ok("三指标三实体推荐雷达图", radar["chart"]["type"] == "radar", radar["chart"]["type"])
    ok("雷达 Caption 声明归一化", "min–max" in radar["caption"]["zh"])
    ok("雷达脚本使用极坐标", 'subplot_kw={"projection": "polar"}' in radar["script"])

    many_rows = []
    for i in range(10):
        many_rows.append({"model": f"M{i}", "acc": "0.7", "f1": "0.7", "recall": "0.7"})
    heat = analyze(many_rows, {
        "model": {"type": "category", "role": "x"},
        "acc": {"type": "numeric", "role": "y"},
        "f1": {"type": "numeric", "role": "y"},
        "recall": {"type": "numeric", "role": "y"},
    })
    ok("十实体三指标推荐热力图", heat["chart"]["type"] == "heatmap", heat["chart"]["type"])
    ok("热图 Caption 声明 z-score", "z-score" in heat["caption"]["zh"])

    # 单数值 → 直方图
    hist = analyze([{"score": str(v)} for v in (1, 2, 2, 3, 5, 8, 13)], {"score": "numeric"})
    ok("单数值推荐直方图", hist["chart"]["type"] == "histogram", hist["chart"]["type"])
    ok("直方图 Caption 含中位数", "中位数" in hist["caption"]["zh"])

    # 纯类别 → 频数柱
    counts = analyze([{"type": "A"}, {"type": "A"}, {"type": "B"}], {"type": "category"})
    ok("纯类别推荐频数柱状图", counts["chart"]["type"] == "bar_counts", counts["chart"]["type"])
    ok("频数 Caption 含总数", "共 3 条" in counts["caption"]["zh"])

    # 缺失值、空数据、单行
    missing = analyze([
        {"x": "2020", "y": "1"}, {"x": "2021", "y": ""}, {"x": "2022", "y": "3"},
    ], {"x": "time", "y": "numeric"})
    ok("缺失值被跳过并提示", any("空值" in x for x in missing["issues"]))
    ok("有效观测数正确", "共 2 条观测" in missing["caption"]["zh"],
       missing["caption"]["zh"])

    try:
        analyze([], {"x": "numeric"})
        empty_ok = False
    except ValueError:
        empty_ok = True
    ok("空数据被拒绝", empty_ok)

    # 显式 x 是唯一数值列，没有 y，明确报错而非静默画直方图
    try:
        analyze([{"x": "1"}, {"x": "2"}],
                {"x": {"type": "numeric", "role": "x"}})
        lonely_x_ok = False
    except ValueError:
        lonely_x_ok = True
    ok("显式 x 无数值 y 被拒绝", lonely_x_ok)

    # 单行不臆造趋势
    one = analyze([{"x": "2020", "y": "5"}], {"x": "time", "y": "numeric"})
    ok("单行不生成趋势", "上升" not in one["caption"]["zh"] and "下降" not in one["caption"]["zh"])
    ok("单行仍给出观测数", "共 1 条观测" in one["caption"]["zh"])

    # 数据摘要不一致
    mismatch = analyze(
        demo["rows"], demo["schema"],
        summary={"n_rows": 99},
    )
    ok("摘要行数不一致会提示", any("不一致" in x for x in mismatch["issues"]))

    # 文本摘要不用于编造数值
    text_summary = analyze(demo["rows"], demo["schema"], summary="准确率显著提升")
    ok("文本摘要不进入 Caption", "显著" not in text_summary["caption"]["zh"])

    # 三种图型脚本均可编译
    for name, result in (("bar", bar), ("scatter", mixed), ("radar", radar),
                         ("heatmap", heat), ("histogram", hist), ("counts", counts)):
        ok(f"{name} 脚本可编译", _compiles(result["script"]))

    print(f"\nchart: {checks - failures}/{checks} 通过")
    return 0 if failures == 0 else 1


def _compiles(script: str) -> bool:
    try:
        compile(script, "figure.py", "exec")
        return True
    except SyntaxError:
        return False


# ────────────────────────── CLI ──────────────────────────


def _load_json_arg(value: str | None, label: str) -> Any:
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{label} 文件不是合法 JSON：{path}：{exc.msg}") from exc
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} 既不是文件也不是合法 JSON：{exc.msg}") from exc


def _resolve_csv(path: str, base: Path | None) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    if base is not None:
        alt = (base / path).resolve()
        if alt.exists():
            return str(alt)
    return str(candidate)


def _load_input(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Any, str | None, Any, str | None]:
    data: Any = None
    base: Path | None = None
    if args.input:
        data = _load_json_arg(args.input, "输入")
        base = Path(args.input).resolve().parent
    elif args.file:
        data = _load_json_arg(args.file, "输入")
        base = Path(args.file).resolve().parent

    rows: list[dict[str, Any]] = []
    schema: Any = None
    summary: Any = None
    title: str | None = args.title

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("rows") or data.get("data") or []
        schema = data.get("schema")
        summary = data.get("data_summary")
        title = title or data.get("title")
        if not rows and isinstance(summary, dict):
            rows = summary.get("rows") or []
    elif data is not None:
        raise SystemExit("输入必须是 JSON 对象或数组")

    if args.csv:
        rows = _read_rows(args.csv)
    elif rows and not isinstance(rows, list):
        raise SystemExit("rows 必须是对象数组")

    if args.schema:
        schema = _load_json_arg(args.schema, "schema")
    if args.summary:
        summary = _load_json_arg(args.summary, "data_summary")

    csv_path = args.csv
    if not csv_path and isinstance(data, dict) and data.get("csv_path"):
        csv_path = _resolve_csv(str(data["csv_path"]), base)
        if not rows:
            rows = _read_rows(csv_path)
    if not csv_path and rows:
        csv_path = None
    if not rows:
        raise SystemExit("没有数据：请传 --csv，或在输入 JSON 中提供 rows / csv_path")

    rows = [dict(row) for row in rows if isinstance(row, dict)]
    if not rows:
        raise SystemExit("rows 中没有对象")
    return rows, schema, csv_path, summary, title


def _report(result: dict[str, Any], include_code: bool = True) -> str:
    chart = result["chart"]
    lines = [
        "## 推荐图表",
        "",
        f"- **类型**：{chart['name']}（`{chart['type']}`）",
        f"- **x**：`{chart['x']}`；**y**：{', '.join('`' + y + '`' for y in chart['y'])}"
        + (f"；**series**：`{chart['series']}`" if chart["series"] else ""),
        "",
        "**理由**",
        "",
    ]
    lines += [f"- {reason}" for reason in chart["reasons"]]
    lines += ["", "**备选**", ""]
    lines += [f"- {alt}" for alt in chart["alternatives"]]
    lines += ["", "## Figure Caption", "", f"**中文**：{result['caption']['zh']}", "",
              f"**English**: {result['caption']['en']}", "", "## 运行", "", "```bash",
              result["run"], "```"]
    if result["issues"]:
        lines += ["", "## 校验提示", ""]
        lines += [f"- {issue}" for issue in result["issues"]]
    if include_code:
        lines += ["", "## 图表代码", "", "```python", result["script"].rstrip(), "```"]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(
        prog="python -m app.chart",
        description="从 CSV 推荐图型，生成 300 dpi Matplotlib 脚本与中英 Figure Caption。",
    )
    ap.add_argument("file", nargs="?", help="输入 JSON（含 rows / schema / csv_path）")
    ap.add_argument("-i", "--input", help="输入 JSON 路径")
    ap.add_argument("--csv", help="CSV 文件路径")
    ap.add_argument("--schema", help="schema JSON 字符串或文件路径")
    ap.add_argument("--summary", help="data_summary JSON 字符串或文件路径")
    ap.add_argument("--title", help="图标题（默认留空）")
    ap.add_argument("--figure-no", type=int, default=1, help="Caption 编号（默认 1）")
    ap.add_argument("-o", "--out", help="把生成的 Python 脚本写入此路径")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    ap.add_argument("--demo", action="store_true", help="用内置 15 行演示数据")
    ap.add_argument("--selftest", action="store_true", help="运行内置自检")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.demo:
        demo = demo_input()
        rows, schema = demo["rows"], demo["schema"]
        csv_path, summary, title = None, None, args.title or demo.get("title")
    else:
        try:
            rows, schema, csv_path, summary, title = _load_input(args)
        except ValueError as exc:
            raise SystemExit(f"[chart] {exc}") from exc

    try:
        result = analyze(
            rows, schema, title=title, csv_path=csv_path,
            figure_no=max(1, args.figure_no), summary=summary,
        )
    except ValueError as exc:
        raise SystemExit(f"[chart] {exc}") from exc

    if args.out:
        try:
            Path(args.out).write_text(result["script"], encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"写入失败：{args.out}：{exc}") from exc
        print(f"[chart] 脚本已写入 {args.out}", file=sys.stderr)
        print(f"[chart] 运行：{result['run']}", file=sys.stderr)

    if args.json:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(_report(result, include_code=not args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
