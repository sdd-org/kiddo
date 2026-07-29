#!/usr/bin/env python3
"""Render the focused Donnelly-versus-Eytzinger exact-NN profile."""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STRATEGIES = {
    "donnelly": ("Donnelly", "#1565c0"),
    "eytzinger": ("Eytzinger", "#5f6368"),
}


@dataclass(frozen=True)
class Point:
    point_count: int
    throughput_mq_s: float
    low_mq_s: float
    high_mq_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--html-name", default="donnelly-vs-eytz-exact.html")
    return parser.parse_args()


def throughput(record: dict[str, Any]) -> Point:
    metadata = record["metadata"]
    elements = int(metadata["throughput"]["Elements"])
    point_count = int(metadata["value_str"])
    slope = record["estimates"]["slope"]
    estimate = float(slope["point_estimate"])
    interval = slope["confidence_interval"]
    lower_ns = float(interval["lower_bound"])
    upper_ns = float(interval["upper_bound"])

    return Point(
        point_count=point_count,
        throughput_mq_s=elements * 1000.0 / estimate,
        low_mq_s=elements * 1000.0 / upper_ns,
        high_mq_s=elements * 1000.0 / lower_ns,
    )


def collect(path: Path) -> dict[str, dict[str, list[Point]]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    series: dict[str, dict[str, list[Point]]] = {
        "f32": {strategy: [] for strategy in STRATEGIES},
        "f64": {strategy: [] for strategy in STRATEGIES},
    }
    prefix = "profile_v6_stem_strategies/"

    for record in payload["results"]:
        benchmark = record["benchmark"]
        for scalar in ("f32", "f64"):
            legacy_prefix = f"profile_v6_stem_strategies_{scalar}/"
            if benchmark.startswith(legacy_prefix):
                benchmark = (
                    f"{prefix}{scalar}/"
                    f"{benchmark.removeprefix(legacy_prefix)}"
                )
                break
        if not benchmark.startswith(prefix):
            continue
        parts = benchmark.split("/")
        if len(parts) != 4:
            continue
        _, scalar, strategy, _ = parts
        if scalar in series and strategy in STRATEGIES:
            series[scalar][strategy].append(throughput(record))

    for scalar, scalar_series in series.items():
        for strategy, points in scalar_series.items():
            points.sort(key=lambda point: point.point_count)
            if not points:
                raise RuntimeError(f"missing {scalar}/{strategy} results")

        donnelly_sizes = {point.point_count for point in scalar_series["donnelly"]}
        eytzinger_sizes = {point.point_count for point in scalar_series["eytzinger"]}
        if donnelly_sizes != eytzinger_sizes:
            raise RuntimeError(f"{scalar} strategies have different tree sizes")

    return series


def x_position(log2_points: int, minimum: int, maximum: int, width: int) -> float:
    if minimum == maximum:
        return width / 2
    return 76.0 + (log2_points - minimum) * (width - 112.0) / (maximum - minimum)


def y_position(value: float, maximum: float, height: int) -> float:
    return 30.0 + (maximum - value) * (height - 84.0) / maximum


def render_svg(
    scalar: str,
    series: dict[str, list[Point]],
    destination: Path,
) -> None:
    width = 1040
    height = 520
    logs = [
        point.point_count.bit_length() - 1
        for points in series.values()
        for point in points
    ]
    minimum = min(logs)
    maximum = max(logs)
    max_y = (
        max(point.high_mq_s for points in series.values() for point in points) * 1.08
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{scalar} exact nearest-neighbour throughput">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#202124}'
        ".axis{stroke:#9aa0a6;stroke-width:1}.grid{stroke:#e8eaed;stroke-width:1}"
        ".series{fill:none;stroke-width:3}.dot{stroke:#fff;stroke-width:1.5}</style>",
        f'<text x="76" y="21" font-size="18" font-weight="700">'
        f"{scalar} scalar exact 1-NN</text>",
    ]

    for tick in range(6):
        value = max_y * tick / 5
        y = y_position(value, max_y, height)
        lines.append(
            f'<line class="grid" x1="76" y1="{y:.2f}" '
            f'x2="{width - 36}" y2="{y:.2f}"/>'
        )
        lines.append(
            f'<text x="68" y="{y + 4:.2f}" font-size="12" '
            f'text-anchor="end">{value:.1f}</text>'
        )

    for log2_points in range(minimum, maximum + 1):
        x = x_position(log2_points, minimum, maximum, width)
        lines.append(
            f'<line class="grid" x1="{x:.2f}" y1="30" '
            f'x2="{x:.2f}" y2="{height - 54}"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{height - 34}" font-size="12" '
            f'text-anchor="middle">2^{log2_points}</text>'
        )

    lines.extend(
        [
            f'<line class="axis" x1="76" y1="{height - 54}" '
            f'x2="{width - 36}" y2="{height - 54}"/>',
            f'<line class="axis" x1="76" y1="30" x2="76" y2="{height - 54}"/>',
            f'<text x="18" y="{height / 2}" font-size="13" '
            f'transform="rotate(-90 18 {height / 2})" '
            f'text-anchor="middle">million queries/s</text>',
            f'<text x="{width / 2}" y="{height - 8}" font-size="13" '
            f'text-anchor="middle">point count</text>',
        ]
    )

    legend_x = width - 280
    for index, (strategy, (label, color)) in enumerate(STRATEGIES.items()):
        points = series[strategy]
        coordinates = [
            (
                x_position(
                    point.point_count.bit_length() - 1,
                    minimum,
                    maximum,
                    width,
                ),
                y_position(point.throughput_mq_s, max_y, height),
            )
            for point in points
        ]
        path = " ".join(
            ("M" if point_index == 0 else "L") + f" {x:.2f} {y:.2f}"
            for point_index, (x, y) in enumerate(coordinates)
        )
        lines.append(f'<path class="series" d="{path}" stroke="{color}"/>')
        for point, (x, y) in zip(points, coordinates, strict=True):
            lines.append(
                f'<circle class="dot" cx="{x:.2f}" cy="{y:.2f}" '
                f'r="4.5" fill="{color}"><title>'
                f"2^{point.point_count.bit_length() - 1}: "
                f"{point.throughput_mq_s:.4f} Mq/s</title></circle>"
            )

        legend_y = 20 + index * 22
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 36}" y="{legend_y + 4}" '
            f'font-size="12">{html.escape(label)}</text>'
        )

    lines.append("</svg>")
    destination.write_text("\n".join(lines), encoding="utf-8")


def render_table(scalar: str, series: dict[str, list[Point]]) -> str:
    donnelly = {point.point_count: point for point in series["donnelly"]}
    eytzinger = {point.point_count: point for point in series["eytzinger"]}
    rows = []
    ratios = []
    for point_count in sorted(donnelly):
        d_value = donnelly[point_count].throughput_mq_s
        e_value = eytzinger[point_count].throughput_mq_s
        ratio = d_value / e_value
        ratios.append(ratio)
        result_class = "win" if ratio >= 1 else "loss"
        rows.append(
            "<tr>"
            f"<td>2<sup>{point_count.bit_length() - 1}</sup></td>"
            f"<td>{e_value:.4f}</td><td>{d_value:.4f}</td>"
            f'<td class="{result_class}">{(ratio - 1) * 100:+.2f}%</td>'
            "</tr>"
        )

    geometric_mean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    return (
        f"<h3>{scalar} values</h3>"
        f"<p>Geometric-mean Donnelly advantage: "
        f"{(geometric_mean - 1) * 100:+.2f}%.</p>"
        "<table><thead><tr><th>Points</th><th>Eytzinger Mq/s</th>"
        "<th>Donnelly Mq/s</th><th>Donnelly / Eytzinger</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series = collect(args.result_json)

    chart_paths = {}
    for scalar in ("f32", "f64"):
        chart_path = args.output_dir / f"donnelly-vs-eytz-exact-{scalar}.svg"
        render_svg(scalar, series[scalar], chart_path)
        chart_paths[scalar] = chart_path

    sections = []
    for scalar in ("f32", "f64"):
        sections.append(
            f"<section><h2>{scalar}</h2>"
            f'<img src="{html.escape(chart_paths[scalar].name, quote=True)}" '
            f'alt="{scalar} throughput chart">'
            f"{render_table(scalar, series[scalar])}</section>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Donnelly versus Eytzinger exact-NN</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;color:#202124;margin:0;background:#f8f9fa}}
main{{max-width:1120px;margin:auto;padding:28px}}
section{{background:#fff;margin:24px 0;padding:20px;border:1px solid #dadce0;border-radius:10px}}
img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 12px;border-bottom:1px solid #e8eaed;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
.win{{color:#137333;font-weight:700}}.loss{{color:#b3261e;font-weight:700}}
code{{background:#f1f3f4;padding:2px 5px;border-radius:4px}}
</style>
</head>
<body><main>
<h1>Donnelly versus Eytzinger: scalar exact 1-NN</h1>
<p>Criterion slope estimates. Source:
<code>{html.escape(args.result_json.name)}</code>.
Eytzinger uses its configured software-prefetch policy.</p>
{''.join(sections)}
</main></body></html>
"""
    html_path = args.output_dir / args.html_name
    html_path.write_text(document, encoding="utf-8")
    print(html_path)


if __name__ == "__main__":
    main()
