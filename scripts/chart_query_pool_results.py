#!/usr/bin/env python3
"""Chart stored-versus-generated exact-NN query-pool Criterion sweeps."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FUNCTIONS = {
    "stored_eytzinger",
    "stored_donnelly",
    "generated_eytzinger",
    "generated_donnelly",
    "generated_control",
}


@dataclass(frozen=True)
class Point:
    axis: str
    point_count: int
    function: str
    pool_size: int
    query_ns: float
    query_ns_low: float
    query_ns_high: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("charts", "all"))
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-label", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--html-name",
        default="donnelly-vs-eytzinger-query-pool.html",
    )
    return parser.parse_args()


def load_points(path: Path) -> list[Point]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("results")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{path} contains no Criterion results")

    points: list[Point] = []
    seen: set[tuple[str, int, str, int]] = set()
    for entry in entries:
        metadata = entry["metadata"]
        function = metadata.get("function_id")
        if function not in FUNCTIONS:
            continue
        group_parts = metadata["group_id"].split("/")
        if len(group_parts) < 3:
            raise RuntimeError(f"unrecognized group id: {metadata['group_id']}")
        axis = group_parts[-2]
        point_count = int(group_parts[-1])
        pool_size = int(metadata["value_str"])
        throughput = int(metadata["throughput"]["Elements"])
        if throughput != pool_size:
            raise RuntimeError(
                f"throughput {throughput} does not equal pool size {pool_size}"
            )
        slope = entry["estimates"]["slope"]
        interval = slope["confidence_interval"]
        point = Point(
            axis=axis,
            point_count=point_count,
            function=function,
            pool_size=pool_size,
            query_ns=float(slope["point_estimate"]) / pool_size,
            query_ns_low=float(interval["lower_bound"]) / pool_size,
            query_ns_high=float(interval["upper_bound"]) / pool_size,
        )
        key = (axis, point_count, function, pool_size)
        if key in seen:
            raise RuntimeError(f"duplicate benchmark point: {key}")
        seen.add(key)
        points.append(point)

    if not points:
        raise RuntimeError(f"{path} contains no query-pool benchmark results")
    return sorted(points, key=lambda point: (point.axis, point.pool_size, point.function))


def lookup(
    points: list[Point], axis: str, point_count: int
) -> dict[tuple[str, int], Point]:
    return {
        (point.function, point.pool_size): point
        for point in points
        if point.axis == axis and point.point_count == point_count
    }


def pool_sizes(points: list[Point], axis: str, point_count: int) -> list[int]:
    return sorted(
        {
            point.pool_size
            for point in points
            if point.axis == axis and point.point_count == point_count
        }
    )


def speedup(
    table: dict[tuple[str, int], Point],
    mode: str,
    size: int,
    subtract_generation: bool = False,
) -> float:
    eytzinger = table[(f"{mode}_eytzinger", size)].query_ns
    donnelly = table[(f"{mode}_donnelly", size)].query_ns
    if subtract_generation:
        control = table[("generated_control", size)].query_ns
        eytzinger -= control
        donnelly -= control
    if eytzinger <= 0.0 or donnelly <= 0.0:
        return math.nan
    return (eytzinger / donnelly - 1.0) * 100.0


def render_chart(points: list[Point], axis: str, point_count: int, path: Path) -> None:
    matplotlib_config = Path(tempfile.gettempdir()) / "kiddo-matplotlib"
    matplotlib_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to generate charts") from error

    sizes = pool_sizes(points, axis, point_count)
    if not sizes:
        return
    table = lookup(points, axis, point_count)
    point_log2 = round(math.log2(point_count))
    if 2**point_log2 != point_count:
        raise RuntimeError(f"tree size is not a power of two: {point_count}")
    x = [math.log2(size) for size in sizes]

    figure, (time_axis, speedup_axis) = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.5),
        height_ratios=(2.2, 1.1),
        sharex=True,
        constrained_layout=True,
    )
    series = (
        ("stored_eytzinger", "Stored · Eytzinger", "#3264a8", "-", "o"),
        ("stored_donnelly", "Stored · Donnelly", "#d35400", "-", "o"),
        ("generated_eytzinger", "Generated · Eytzinger", "#6c8ebf", "--", "s"),
        ("generated_donnelly", "Generated · Donnelly", "#e58b52", "--", "s"),
    )
    for function, label, color, linestyle, marker in series:
        values = [table[(function, size)] for size in sizes]
        time_axis.plot(
            x,
            [value.query_ns for value in values],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.1,
        )
        time_axis.fill_between(
            x,
            [value.query_ns_low for value in values],
            [value.query_ns_high for value in values],
            color=color,
            alpha=0.12,
        )

    control = [table[("generated_control", size)].query_ns for size in sizes]
    time_axis.plot(
        x,
        control,
        color="#777",
        linestyle=":",
        linewidth=1.6,
        label="Generated-query control",
    )
    time_axis.set_title(
        f"Exact nearest-one, SQ Euclidean, 3D {axis}, "
        f"2^{point_log2} points — query-pool sweep"
    )
    time_axis.set_ylabel("Mean time/query (ns)\n(lower is better)")
    time_axis.grid(True, color="#dfe3e8", linewidth=0.8)
    time_axis.legend(ncol=2)

    stored_speedup = [speedup(table, "stored", size) for size in sizes]
    generated_speedup = [speedup(table, "generated", size) for size in sizes]
    adjusted_speedup = [
        speedup(table, "generated", size, subtract_generation=True) for size in sizes
    ]
    speedup_axis.axhline(0.0, color="#555", linewidth=1.0)
    speedup_axis.plot(
        x,
        stored_speedup,
        marker="o",
        linewidth=2.2,
        color="#d35400",
        label="Stored",
    )
    speedup_axis.plot(
        x,
        generated_speedup,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#7b4ab5",
        label="Generated · raw",
    )
    speedup_axis.plot(
        x,
        adjusted_speedup,
        linewidth=1.6,
        linestyle=":",
        color="#333",
        label="Generated · control-subtracted",
    )
    speedup_axis.set_ylabel("Donnelly throughput\nadvantage")
    speedup_axis.yaxis.set_major_formatter(lambda value, _: f"{value:+.0f}%")
    speedup_axis.set_xlabel("Distinct queries in repeatedly executed pool")
    speedup_axis.set_xticks(x, [f"{size:,}" for size in sizes])
    speedup_axis.grid(True, color="#dfe3e8", linewidth=0.8)
    speedup_axis.legend()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def summary_rows(points: list[Point]) -> str:
    rows: list[str] = []
    for axis in ("f64", "f32"):
        point_counts = sorted(
            {point.point_count for point in points if point.axis == axis}
        )
        for point_count in point_counts:
            table = lookup(points, axis, point_count)
            point_log2 = round(math.log2(point_count))
            for size in pool_sizes(points, axis, point_count):
                stored = speedup(table, "stored", size)
                generated = speedup(table, "generated", size)
                adjusted = speedup(table, "generated", size, subtract_generation=True)
                control = table[("generated_control", size)].query_ns
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(axis)}</td>"
                    f"<td>2^{point_log2}</td>"
                    f"<td>{size:,}</td>"
                    f"<td>{stored:+.2f}%</td>"
                    f"<td>{generated:+.2f}%</td>"
                    f"<td>{adjusted:+.2f}%</td>"
                    f"<td>{control:.2f} ns</td>"
                    "</tr>"
                )
    return "\n".join(rows)


def render_html(
    points: list[Point],
    result_label: str,
    chart_names: list[str],
    path: Path,
) -> None:
    images = "\n".join(
        f'<section><img src="{html.escape(name)}" alt="Query-pool crossover chart"></section>'
        for name in chart_names
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Donnelly vs Eytzinger query-pool crossover</title>
<style>
body {{ font: 16px/1.45 system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; color: #20242a; }}
.lede {{ max-width: 82ch; }}
img {{ display: block; width: 100%; height: auto; margin: 1rem 0 2.5rem; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ border-bottom: 1px solid #d9dde2; padding: .55rem .7rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
code {{ background: #f3f4f6; padding: .12rem .3rem; }}
</style>
</head>
<body>
<h1>Donnelly vs Eytzinger query-pool crossover</h1>
<p class="lede">
Result <code>{html.escape(result_label)}</code>. Each Criterion benchmark repeatedly
executes one finite exact-nearest-neighbour query pool against one active tree.
Stored and generated-by-index modes use identical SplitMix64 coordinates.
The generated control estimates query-construction cost; its subtraction is
informative rather than a formally independent timing measurement.
</p>
{images}
<h2>Speedup summary</h2>
<table>
<thead><tr><th>Axis</th><th>Points</th><th>Pool</th><th>Stored</th><th>Generated raw</th><th>Generated adjusted</th><th>Generation control</th></tr></thead>
<tbody>
{summary_rows(points)}
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    points = load_points(args.result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_names: list[str] = []
    for axis in ("f64", "f32"):
        point_counts = sorted(
            {point.point_count for point in points if point.axis == axis}
        )
        for point_count in point_counts:
            point_log2 = round(math.log2(point_count))
            if len(point_counts) == 1:
                name = f"donnelly-vs-eytzinger-query-pool-{axis}-{args.result_label}.png"
            else:
                name = (
                    f"donnelly-vs-eytzinger-query-pool-{axis}-2p{point_log2}-"
                    f"{args.result_label}.png"
                )
            render_chart(points, axis, point_count, args.output_dir / name)
            chart_names.append(name)
    if args.mode == "all":
        render_html(
            points,
            args.result_label,
            chart_names,
            args.output_dir / args.html_name,
        )


if __name__ == "__main__":
    main()
