#!/usr/bin/env python3
"""Chart exact-NN Donnelly variant screening results."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


STRATEGIES = {
    "eytzinger": "Eytzinger",
    "donnelly": "Donnelly scalar",
    "donnelly_unrolled": "Donnelly unrolled",
    "donnelly_unrolled_block_dim": "Donnelly unrolled/block-dim",
    "donnelly_simd_descent": "Donnelly SIMD descent",
    "donnelly_cyclic_simd_descent": "Donnelly cyclic SIMD descent",
    "donnelly_cyclic_simd_full": "Donnelly cyclic SIMD full",
    "donnelly_simd_initial_descent": "Donnelly initial-only SIMD",
    "donnelly_simd_full": "Donnelly full SIMD",
}
COLORS = {
    "eytzinger": "#3264a8",
    "donnelly": "#d35400",
    "donnelly_unrolled": "#b03a8f",
    "donnelly_unrolled_block_dim": "#298f75",
    "donnelly_simd_descent": "#8a6d1d",
    "donnelly_cyclic_simd_descent": "#16a085",
    "donnelly_cyclic_simd_full": "#c0392b",
    "donnelly_simd_initial_descent": "#7d5fff",
    "donnelly_simd_full": "#c0392b",
}


@dataclass(frozen=True)
class Point:
    axis: str
    point_count: int
    mode: str
    strategy: str
    pool_size: int
    query_ns: float
    low_ns: float
    high_ns: float


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("charts", "all"))
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--html-name", default="donnelly-variant-screen.html")
    return parser.parse_args()


def load(path: Path) -> tuple[list[Point], dict[tuple[str, int, int], float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points: list[Point] = []
    controls: dict[tuple[str, int, int], float] = {}
    for entry in payload.get("results", []):
        metadata = entry["metadata"]
        function = metadata.get("function_id", "")
        group = metadata["group_id"].split("/")
        axis = group[-2]
        point_count = int(group[-1])
        pool_size = int(metadata["value_str"])
        slope = entry["estimates"]["slope"]
        interval = slope["confidence_interval"]
        query_ns = float(slope["point_estimate"]) / pool_size
        if function == "generated_control":
            controls[(axis, point_count, pool_size)] = query_ns
            continue
        matched = next(
            (
                (mode, strategy)
                for mode in ("stored", "generated")
                for strategy in STRATEGIES
                if function == f"{mode}_{strategy}"
            ),
            None,
        )
        if matched is None:
            continue
        mode, strategy = matched
        points.append(
            Point(
                axis=axis,
                point_count=point_count,
                mode=mode,
                strategy=strategy,
                pool_size=pool_size,
                query_ns=query_ns,
                low_ns=float(interval["lower_bound"]) / pool_size,
                high_ns=float(interval["upper_bound"]) / pool_size,
            )
        )
    if not points:
        raise RuntimeError(f"{path} contains no Donnelly variant results")
    return points, controls


def render(
    points: list[Point],
    controls: dict[tuple[str, int, int], float],
    axis: str,
    point_count: int,
    output: Path,
) -> None:
    matplotlib_config = Path(tempfile.gettempdir()) / "kiddo-matplotlib"
    matplotlib_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    selected = [
        point
        for point in points
        if point.axis == axis and point.point_count == point_count
    ]
    modes = ("stored", "generated")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 9.0),
        height_ratios=(2.0, 1.0),
        sharex="col",
        constrained_layout=True,
    )
    point_log2 = round(math.log2(point_count))

    for column, mode in enumerate(modes):
        timing = axes[0][column]
        advantage = axes[1][column]
        mode_points = [point for point in selected if point.mode == mode]
        sizes = sorted({point.pool_size for point in mode_points})
        x = [math.log2(size) for size in sizes]
        strategies = [
            strategy
            for strategy in STRATEGIES
            if any(point.strategy == strategy for point in mode_points)
        ]
        table = {
            (point.strategy, point.pool_size): point for point in mode_points
        }
        for strategy in strategies:
            samples = [table[(strategy, size)] for size in sizes]
            timing.plot(
                x,
                [sample.query_ns for sample in samples],
                label=STRATEGIES[strategy],
                color=COLORS[strategy],
                marker="o",
                linewidth=2.0,
            )
            timing.fill_between(
                x,
                [sample.low_ns for sample in samples],
                [sample.high_ns for sample in samples],
                color=COLORS[strategy],
                alpha=0.10,
            )

        timing.set_title(f"{mode.capitalize()} query pool")
        timing.set_ylabel("Criterion slope (ns/query)")
        timing.grid(True, color="#dfe3e8", linewidth=0.8)
        timing.legend(fontsize=8.5)

        advantage.axhline(0.0, color="#555", linewidth=1.0)
        eytzinger = [table[("eytzinger", size)].query_ns for size in sizes]
        if mode == "generated":
            eytzinger = [
                value - controls.get((axis, point_count, size), 0.0)
                for value, size in zip(eytzinger, sizes)
            ]
        for strategy in strategies:
            if strategy == "eytzinger":
                continue
            values = [table[(strategy, size)].query_ns for size in sizes]
            if mode == "generated":
                values = [
                    value - controls.get((axis, point_count, size), 0.0)
                    for value, size in zip(values, sizes)
                ]
            speedups = [
                (baseline / value - 1.0) * 100.0
                if baseline > 0.0 and value > 0.0
                else math.nan
                for baseline, value in zip(eytzinger, values)
            ]
            advantage.plot(
                x,
                speedups,
                label=STRATEGIES[strategy],
                color=COLORS[strategy],
                marker="o",
                linewidth=2.0,
            )
        advantage.set_ylabel("Advantage over Eytzinger")
        advantage.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:+.0f}%")
        )
        advantage.set_xlabel("Distinct queries in pool")
        advantage.set_xticks(x, [f"{size:,}" for size in sizes])
        advantage.grid(True, color="#dfe3e8", linewidth=0.8)

    if "_k" in axis:
        scalar, dimension_text = axis.split("_k", 1)
        dimensions = int(dimension_text)
    else:
        scalar = axis
        dimensions = 3 if axis == "f64" else 4
    present = {point.strategy for point in selected}
    if (
        "donnelly_unrolled_block_dim" in present
        and "donnelly_cyclic_simd_descent" in present
    ):
        experiment = "balanced full-strategy screen"
    elif "donnelly_unrolled_block_dim" in present:
        experiment = "balanced UBD control"
    elif "donnelly_cyclic_simd_descent" in present:
        experiment = "cyclic-layout screen"
    else:
        experiment = "strategy screen"
    figure.suptitle(
        f"Exact nearest-one Donnelly variant screen — {dimensions}D {scalar}, "
        f"2^{point_log2} points — {experiment}"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = arguments()
    points, controls = load(args.result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_names: list[str] = []
    for axis in sorted({point.axis for point in points}):
        for point_count in sorted(
            {point.point_count for point in points if point.axis == axis}
        ):
            point_log2 = round(math.log2(point_count))
            name = f"donnelly-variant-screen-{axis}-2p{point_log2}.png"
            render(points, controls, axis, point_count, args.output_dir / name)
            chart_names.append(name)

    if args.mode == "all":
        images = "\n".join(
            f'<section><img src="{html.escape(name)}" alt="Donnelly variant chart"></section>'
            for name in chart_names
        )
        (args.output_dir / args.html_name).write_text(
            "<!doctype html><meta charset='utf-8'>"
            f"<title>{html.escape(args.result_label)}</title>"
            "<style>body{font:16px system-ui;margin:2rem;background:#f7f7f7}"
            "section{margin:1rem auto;max-width:1450px;background:white;padding:1rem}"
            "img{width:100%;height:auto}</style>"
            f"<h1>{html.escape(args.result_label)}</h1>{images}",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
