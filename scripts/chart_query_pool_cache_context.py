#!/usr/bin/env python3
"""Chart exact-NN query-pool results with cache-capacity context.

The timing panels are measurements.  The working-set panel is deliberately a
simple lower bound: expected distinct direct-descent stem lines plus one full
leaf per query.  It excludes exact-search backtracking, replacement,
associativity, prefetching, alignment effects, and sharing among leaf paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


FUNCTIONS = {
    "stored_eytzinger",
    "stored_donnelly",
    "generated_eytzinger",
    "generated_donnelly",
    "generated_control",
}
K = 3
LEAF_SIZE = 32
L1D_MIB = 48 / 1024
L2_MIB = 1.0
L3_MIB = 32.0


@dataclass(frozen=True)
class Point:
    axis: str
    point_count: int
    function: str
    pool_size: int
    query_ns: float
    low_ns: float
    high_ns: float


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_points(path: Path) -> list[Point]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points: list[Point] = []
    for entry in payload.get("results", []):
        metadata = entry["metadata"]
        function = metadata.get("function_id")
        if function not in FUNCTIONS:
            continue
        group = metadata["group_id"].split("/")
        axis = group[-2]
        point_count = int(group[-1])
        pool_size = int(metadata["value_str"])
        slope = entry["estimates"]["slope"]
        interval = slope["confidence_interval"]
        points.append(
            Point(
                axis=axis,
                point_count=point_count,
                function=function,
                pool_size=pool_size,
                query_ns=float(slope["point_estimate"]) / pool_size,
                low_ns=float(interval["lower_bound"]) / pool_size,
                high_ns=float(interval["upper_bound"]) / pool_size,
            )
        )
    if not points:
        raise RuntimeError(f"{path} contains no query-pool Criterion results")
    return points


def expected_occupied_bins(bins: int, draws: int) -> float:
    if bins <= 1:
        return 1.0
    return -bins * math.expm1(draws * math.log1p(-1.0 / bins))


def leaf_working_set_bytes(point_count: int, query_count: int, item_bytes: int) -> float:
    leaf_count = math.ceil(point_count / LEAF_SIZE)
    occupied = expected_occupied_bins(leaf_count, query_count)
    leaf_bytes = LEAF_SIZE * (K * item_bytes + 4)
    leaf_lines = math.ceil(leaf_bytes / 64)
    return occupied * leaf_lines * 64


def eytzinger_stem_bytes(point_count: int, query_count: int, item_bytes: int) -> float:
    leaf_count = math.ceil(point_count / LEAF_SIZE)
    depth = math.ceil(math.log2(leaf_count))
    values_per_line = 64 // item_bytes
    lines = 0.0
    for level in range(depth):
        nodes = 1 << level
        possible_lines = math.ceil(nodes / values_per_line)
        lines += expected_occupied_bins(possible_lines, query_count)
    return lines * 64


def donnelly_stem_bytes(point_count: int, query_count: int, item_bytes: int) -> float:
    leaf_count = math.ceil(point_count / LEAF_SIZE)
    depth = math.ceil(math.log2(leaf_count))
    block_height = 3 if item_bytes == 8 else 4
    lines = 0.0
    for first_level in range(0, depth, block_height):
        possible_blocks = 1 << first_level
        lines += expected_occupied_bins(possible_blocks, query_count)
    return lines * 64


def table(
    points: list[Point], axis: str, point_count: int
) -> dict[tuple[str, int], Point]:
    return {
        (point.function, point.pool_size): point
        for point in points
        if point.axis == axis and point.point_count == point_count
    }


def speedup(
    values: dict[tuple[str, int], Point],
    mode: str,
    pool_size: int,
    subtract_control: bool = False,
) -> float:
    eytzinger = values[(f"{mode}_eytzinger", pool_size)].query_ns
    donnelly = values[(f"{mode}_donnelly", pool_size)].query_ns
    if subtract_control:
        control = values[("generated_control", pool_size)].query_ns
        eytzinger -= control
        donnelly -= control
    return (eytzinger / donnelly - 1.0) * 100.0


def render(points: list[Point], axis: str, point_count: int, output: Path) -> None:
    matplotlib_config = Path(tempfile.gettempdir()) / "kiddo-matplotlib"
    matplotlib_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    values = table(points, axis, point_count)
    sizes = sorted(
        {
            point.pool_size
            for point in points
            if point.axis == axis and point.point_count == point_count
        }
    )
    point_log2 = round(math.log2(point_count))
    x = [math.log2(size) for size in sizes]

    figure, (timing, advantage, working_set) = plt.subplots(
        3,
        1,
        figsize=(12.4, 11.0),
        height_ratios=(2.1, 1.15, 1.55),
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
        samples = [values[(function, size)] for size in sizes]
        timing.plot(
            x,
            [sample.query_ns for sample in samples],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.1,
        )
        timing.fill_between(
            x,
            [sample.low_ns for sample in samples],
            [sample.high_ns for sample in samples],
            color=color,
            alpha=0.12,
        )
    timing.set_title(
        f"Serial exact nearest-one, SQ Euclidean, 3D {axis}, "
        f"2^{point_log2} points"
    )
    timing.set_ylabel("Criterion slope estimate\n(ns/query; lower is better)")
    timing.grid(True, color="#dfe3e8", linewidth=0.8)
    timing.legend(ncol=2)

    advantage.axhline(0.0, color="#555", linewidth=1.0)
    advantage.plot(
        x,
        [speedup(values, "stored", size) for size in sizes],
        marker="o",
        linewidth=2.2,
        color="#d35400",
        label="Stored",
    )
    advantage.plot(
        x,
        [speedup(values, "generated", size, True) for size in sizes],
        marker="s",
        linestyle="--",
        linewidth=2.0,
        color="#7b4ab5",
        label="Generated · control-subtracted",
    )
    advantage.set_ylabel("Donnelly throughput\nadvantage")
    advantage.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.0f}%"))
    advantage.grid(True, color="#dfe3e8", linewidth=0.8)
    advantage.legend()

    item_bytes = 8 if axis == "f64" else 4
    eytzinger_mib = []
    donnelly_mib = []
    for size in sizes:
        leaves = leaf_working_set_bytes(point_count, size, item_bytes)
        eytzinger_mib.append(
            (eytzinger_stem_bytes(point_count, size, item_bytes) + leaves) / 2**20
        )
        donnelly_mib.append(
            (donnelly_stem_bytes(point_count, size, item_bytes) + leaves) / 2**20
        )

    working_set.axhspan(0.001, L1D_MIB, color="#dff3df", alpha=0.72)
    working_set.axhspan(L1D_MIB, L2_MIB, color="#e8f0fa", alpha=0.72)
    working_set.axhspan(L2_MIB, L3_MIB, color="#fff0d8", alpha=0.72)
    working_set.axhspan(L3_MIB, 128, color="#f7dddd", alpha=0.55)
    for capacity, label, color in (
        (L1D_MIB, "L1D 48 KiB", "#3a8f3a"),
        (L2_MIB, "private L2 1 MiB", "#3264a8"),
        (L3_MIB, "CCD L3 32 MiB", "#a15c00"),
    ):
        working_set.axhline(capacity, color=color, linewidth=1.2, linestyle=":")
        working_set.text(
            x[-1] + 0.07,
            capacity,
            label,
            va="center",
            fontsize=8.5,
            color=color,
        )
    working_set.plot(
        x, eytzinger_mib, color="#3264a8", marker="o", linewidth=2.1, label="Eytzinger"
    )
    working_set.plot(
        x, donnelly_mib, color="#d35400", marker="o", linewidth=2.1, label="Donnelly"
    )
    working_set.set_yscale("log", base=2)
    working_set.set_ylim(0.03, 64)
    working_set.set_ylabel("Modeled minimum active set (MiB)\n"
                           "direct stem path + one full leaf/query")
    working_set.set_xlabel("Distinct queries in repeatedly executed pool")
    working_set.set_xticks(x, [f"{size:,}" for size in sizes])
    working_set.grid(True, which="both", color="#dfe3e8", linewidth=0.7)
    working_set.legend(loc="upper left")
    working_set.text(
        0.01,
        0.02,
        "Analytical lower bound only: excludes exact backtracking, replacement, "
        "associativity, prefetching and alignment effects.",
        transform=working_set.transAxes,
        fontsize=8.5,
        color="#444",
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = arguments()
    points = load_points(args.result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for axis in ("f64", "f32"):
        point_counts = sorted(
            {point.point_count for point in points if point.axis == axis}
        )
        for point_count in point_counts:
            point_log2 = round(math.log2(point_count))
            if len(point_counts) == 1:
                name = f"query-pool-cache-context-{axis}.png"
            else:
                name = f"query-pool-cache-context-{axis}-2p{point_log2}.png"
            render(points, axis, point_count, args.output_dir / name)


if __name__ == "__main__":
    main()
