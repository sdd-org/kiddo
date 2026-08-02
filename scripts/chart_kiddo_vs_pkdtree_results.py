#!/usr/bin/env python3
"""Chart the dedicated kiddo-vs-Pkd-tree single-query comparison (profile_kiddo_vs_pkdtree group).

Unlike chart_cpp_competitor_results.py, this needs no separate kiddo
baseline file: kiddo_vs_pkdtree_single in profile_cpp_competitors.rs
benchmarks both libraries directly (kiddo via its native API, no FFI) in one
pass, so a single result export already has both series. This exists
separately so it can be run at much larger tree sizes without nanoflann/
ALGLIB also needing to build and hold a tree that size in memory.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GROUP = "profile_kiddo_vs_pkdtree"
SCALARS = ("f32", "f64")
SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]*$")
LIBRARY_PREFIXES = (
    ("kiddo_", "kiddo"),
    ("pkdtree_", "Pkd-tree"),
)
LIBRARY_ORDER = ("kiddo", "Pkd-tree")
COLORS = {"kiddo": "#3264a8", "Pkd-tree": "#8e44ad"}


@dataclass(frozen=True)
class Point:
    tree_size: int
    duration_ns: float
    lower_ns: float
    upper_ns: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chart kiddo vs Pkd-tree, single-query, possibly at large N.")
    parser.add_argument("mode", choices=("charts", "all"))
    parser.add_argument("--result", type=Path, nargs="+", required=True)
    parser.add_argument("--result-label", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--html-name", default="kiddo-vs-pkdtree.html")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise RuntimeError(f"{path} is not a Criterion result export")
    return value


def finite_positive(value: Any, description: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid {description} in {path}: {value!r}") from error
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"invalid {description} in {path}: {value!r}")
    return number


def identity(function_id: str) -> tuple[str, str]:
    for prefix, library in LIBRARY_PREFIXES:
        if function_id.startswith(prefix):
            query = function_id[len(prefix) :]
            if query == "nearest_one" or re.fullmatch(r"nearest_n_k\d+", query):
                return library, query
            raise RuntimeError(f"unsupported query identity: {function_id}")
    raise RuntimeError(f"unexpected library in benchmark identity {function_id!r}")


def read_series(path: Path) -> dict[tuple[str, str], dict[str, list[Point]]]:
    charts: dict[tuple[str, str], dict[str, list[Point]]] = {}
    for result in load_json(path)["results"]:
        metadata = result.get("metadata")
        estimates = result.get("estimates")
        if not isinstance(metadata, dict) or not isinstance(estimates, dict):
            raise RuntimeError(f"{path} contains incomplete benchmark data")

        group_id = metadata.get("group_id")
        function_id = metadata.get("function_id")
        if not isinstance(group_id, str) or not isinstance(function_id, str):
            raise RuntimeError(f"{path} contains an invalid benchmark identity")
        if group_id.rsplit("/", 1)[0] != GROUP:
            raise RuntimeError(f"{path} contains group {group_id!r}, expected {GROUP!r}")
        scalar = group_id.rsplit("/", 1)[-1]
        if scalar not in SCALARS:
            raise RuntimeError(f"{path} contains unsupported scalar group {group_id!r}")

        library, query = identity(function_id)

        tree_size_value = finite_positive(metadata.get("value_str"), "tree size", path)
        tree_size = int(tree_size_value)
        if tree_size != tree_size_value or tree_size & (tree_size - 1):
            raise RuntimeError(f"tree size is not a power of two in {path}: {tree_size_value}")

        throughput = metadata.get("throughput")
        query_count = throughput.get("Elements") if isinstance(throughput, dict) else None
        query_count = finite_positive(query_count, "Elements throughput", path)
        mean = estimates.get("mean")
        interval = mean.get("confidence_interval") if isinstance(mean, dict) else None
        if not isinstance(mean, dict) or not isinstance(interval, dict):
            raise RuntimeError(f"{path} contains no mean confidence interval")

        point = Point(
            tree_size,
            finite_positive(mean.get("point_estimate"), "mean duration", path) / query_count,
            finite_positive(interval.get("lower_bound"), "lower duration bound", path) / query_count,
            finite_positive(interval.get("upper_bound"), "upper duration bound", path) / query_count,
        )
        chart = charts.setdefault((scalar, query), {})
        chart.setdefault(library, []).append(point)

    for chart in charts.values():
        for library, points in chart.items():
            points.sort(key=lambda point: point.tree_size)
            sizes = [point.tree_size for point in points]
            if len(sizes) != len(set(sizes)):
                raise RuntimeError(f"duplicate tree sizes for {library!r} in {path}")
    return charts


def merge_charts(paths: list[Path]) -> dict[tuple[str, str], dict[str, list[Point]]]:
    merged: dict[tuple[str, str], dict[str, list[Point]]] = {}
    for path in paths:
        for key, libraries in read_series(path).items():
            chart = merged.setdefault(key, {})
            for library, points in libraries.items():
                existing = chart.setdefault(library, [])
                existing_sizes = {point.tree_size for point in existing}
                for point in points:
                    if point.tree_size in existing_sizes:
                        raise RuntimeError(
                            f"duplicate tree size {point.tree_size} for {library!r}/{key!r} across --result inputs"
                        )
                existing.extend(points)
    return merged


def query_sort_key(query: str) -> tuple[int, int]:
    if query == "nearest_one":
        return (0, 0)
    match = re.fullmatch(r"nearest_n_k(\d+)", query)
    return (1, int(match.group(1))) if match else (2, 0)


def query_title(query: str) -> str:
    if query == "nearest_one":
        return "nearest_one"
    match = re.fullmatch(r"nearest_n_k(\d+)", query)
    return f"nearest_n (n={match.group(1)})" if match else query


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_") or "benchmark"


def render_charts(
    charts: dict[tuple[str, str], dict[str, list[Point]]], output_dir: Path, result_label: str
) -> list[tuple[str, Path]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to generate benchmark charts") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, Path]] = []
    ordered = sorted(charts.items(), key=lambda item: (SCALARS.index(item[0][0]), query_sort_key(item[0][1])))
    for (scalar, query), series in ordered:
        title = f"3D {query_title(query)} — kiddo vs Pkd-tree — {scalar}"
        figure, axis = plt.subplots(figsize=(10.5, 6.2))
        for library in LIBRARY_ORDER:
            points = series.get(library)
            if points is None:
                continue
            axis.errorbar(
                [point.tree_size for point in points],
                [point.duration_ns for point in points],
                yerr=[
                    [point.duration_ns - point.lower_ns for point in points],
                    [point.upper_ns - point.duration_ns for point in points],
                ],
                marker="o",
                markersize=4,
                linewidth=1.8,
                capsize=2,
                color=COLORS.get(library, "#555555"),
                label=library,
            )
        sizes = sorted({point.tree_size for points in series.values() for point in points})
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=10)
        axis.set_xticks(sizes)
        axis.set_xticklabels([f"2^{size.bit_length() - 1}" for size in sizes])
        axis.set_xlabel("Tree size (log₂ scale)")
        axis.set_ylabel("Mean query duration (ns/query, log₁₀ scale)")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(title="Library")
        figure.tight_layout()

        path = output_dir / f"bench_result-kiddo-vs-pkdtree-{slug(result_label)}-{scalar}-{slug(query)}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        rendered.append((title, path))
    return rendered


def write_html(path: Path, charts: list[tuple[str, Path]], result_label: str, sources: list[Path]) -> None:
    sections = "\n".join(
        (
            f"<section><h2>{html.escape(title)}</h2>"
            f'<img src="{html.escape(chart.name, quote=True)}" '
            f'alt="{html.escape(title, quote=True)}"></section>'
        )
        for title, chart in charts
    )
    source_items = "".join(f"<li><code>{html.escape(str(source))}</code></li>" for source in sources)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>kiddo vs Pkd-tree (single-query)</title>
  <style>
    body {{ font: 16px/1.45 system-ui, sans-serif; max-width: 1180px; margin: 0 auto; padding: 2rem; color: #18212b; }}
    section {{ margin: 2.5rem 0; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #d8dee4; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>kiddo vs Pkd-tree (single-query)</h1>
  <p>Result label: <code>{html.escape(result_label)}</code>. All benchmarks are 3D, single-threaded, sequential per-query.</p>
  <details><summary>Source result exports</summary><ul>{source_items}</ul></details>
  {sections}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not SAFE_LABEL.fullmatch(args.result_label):
        raise RuntimeError("result label must contain only letters, digits, '.', '_', '+', ':', or '-'")
    charts = merge_charts(args.result)
    rendered = render_charts(charts, args.output_dir, args.result_label)
    if args.mode == "all":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_html(args.output_dir / args.html_name, rendered, args.result_label, args.result)


if __name__ == "__main__":
    main()
