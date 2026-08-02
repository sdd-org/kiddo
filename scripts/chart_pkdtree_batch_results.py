#!/usr/bin/env python3
"""Chart batch-parallel throughput (profile_pkdtree_batch group).

This is deliberately a standalone chart, not a line on the
chart_cpp_competitor_results.py charts: batch-parallel (submit every query
at once, either via kiddo::batch::Executor::parallel() or Pkd-tree's
parlay::parallel_for, both across all cores) is a throughput metric, not a
per-query latency one, and isn't comparable to any of the sequential
single-query numbers charted elsewhere. Kiddo's serial executor is included
too, as the fairest same-API baseline for the parallel numbers. See
benches/profile_cpp_competitors.rs's module doc comment for the full
rationale.
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


GROUP = "profile_pkdtree_batch"
SCALARS = ("f32", "f64")
# Donnelly prefixes must precede the Eytzinger ones: "kiddo_batch_" is a
# prefix of nothing else, but matching is first-wins, and a future rename
# could easily make one stem's prefix shadow another's.
LIBRARY_PREFIXES = (
    ("kiddo_donnelly_batch_tuned_", "kiddo Donnelly (tuned)"),
    ("kiddo_donnelly_batch_parallel_", "kiddo Donnelly (parallel)"),
    ("kiddo_donnelly_batch_serial_", "kiddo Donnelly (serial)"),
    ("kiddo_batch_tuned_", "kiddo Eytzinger (tuned)"),
    ("kiddo_batch_parallel_", "kiddo Eytzinger (parallel)"),
    ("kiddo_batch_serial_", "kiddo Eytzinger (serial)"),
    ("pkdtree_batch_", "Pkd-tree"),
)
LIBRARY_ORDER = (
    "kiddo Donnelly (tuned)",
    "kiddo Donnelly (parallel)",
    "kiddo Eytzinger (tuned)",
    "kiddo Eytzinger (parallel)",
    "kiddo Donnelly (serial)",
    "kiddo Eytzinger (serial)",
    "Pkd-tree",
)
COLORS = {
    "kiddo Donnelly (tuned)": "#0f6b48",
    "kiddo Donnelly (parallel)": "#1a7f5a",
    "kiddo Eytzinger (tuned)": "#1f4e8c",
    "kiddo Eytzinger (parallel)": "#3264a8",
    "kiddo Donnelly (serial)": "#7fc4a8",
    "kiddo Eytzinger (serial)": "#7fa8d9",
    "Pkd-tree": "#8e44ad",
}
SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]*$")


@dataclass(frozen=True)
class Point:
    tree_size: int
    elements_per_sec: float
    lower: float
    upper: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chart batch-parallel throughput.")
    parser.add_argument("mode", choices=("charts", "all"))
    parser.add_argument("--result", type=Path, nargs="+", required=True)
    parser.add_argument("--result-label", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--html-name", default="pkdtree-batch-throughput.html")
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
            raise RuntimeError(f"unsupported batch query identity: {function_id}")
    raise RuntimeError(f"unexpected library in batch benchmark identity {function_id!r}")


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

        duration_s = finite_positive(mean.get("point_estimate"), "mean duration", path) / 1.0e9
        lower_s = finite_positive(interval.get("lower_bound"), "lower duration bound", path) / 1.0e9
        upper_s = finite_positive(interval.get("upper_bound"), "upper duration bound", path) / 1.0e9

        point = Point(
            tree_size,
            query_count / duration_s,
            query_count / upper_s,
            query_count / lower_s,
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


def merge_charts(
    paths: list[Path],
) -> dict[tuple[str, str], dict[str, list[Point]]]:
    merged: dict[tuple[str, str], dict[str, list[Point]]] = {}
    for path in paths:
        for key, libraries in read_series(path).items():
            chart = merged.setdefault(key, {})
            for library, points in libraries.items():
                if library in chart:
                    raise RuntimeError(f"duplicate {library} series for {key!r} across --result inputs")
                chart[library] = points
    return merged


def query_sort_key(query: str) -> int:
    """nearest_one sorts first; nearest_n sorts by k."""
    if query == "nearest_one":
        return 1
    match = re.fullmatch(r"nearest_n_k(\d+)", query)
    return int(match.group(1)) if match else 0


def query_title(query: str) -> str:
    """Pkd-tree has no single-nearest entry point, so its nearest_one series
    is its k-nearest call with k=1; kiddo's is the real nearest_one."""
    if query == "nearest_one":
        return "nearest_one (Pkd-tree: k=1)"
    return f"nearest_n (k={query_sort_key(query)})"


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
        title = f"Batch-parallel {query_title(query)} throughput — {scalar}"
        figure, axis = plt.subplots(figsize=(10.5, 6.2))
        for library in LIBRARY_ORDER:
            points = series.get(library)
            if points is None:
                continue
            axis.errorbar(
                [point.tree_size for point in points],
                [point.elements_per_sec for point in points],
                yerr=[
                    [point.elements_per_sec - point.lower for point in points],
                    [point.upper - point.elements_per_sec for point in points],
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
        axis.set_ylabel("Queries/sec, all cores (log₁₀ scale)")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(title="Library")
        figure.tight_layout()

        path = output_dir / f"bench_result-pkdtree-batch-{slug(result_label)}-{scalar}-{slug(query)}.png"
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
  <title>Batch-parallel throughput: kiddo vs Pkd-tree</title>
  <style>
    body {{ font: 16px/1.45 system-ui, sans-serif; max-width: 1180px; margin: 0 auto; padding: 2rem; color: #18212b; }}
    section {{ margin: 2.5rem 0; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #d8dee4; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Batch-parallel throughput: kiddo vs Pkd-tree</h1>
  <p>Result label: <code>{html.escape(result_label)}</code>. All queries submitted at once across every core (kiddo::batch::Executor::parallel() / Pkd-tree's parlay::parallel_for). kiddo's serial executor is included as the fairest same-API baseline. Not comparable to the sequential per-query charts.</p>
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
