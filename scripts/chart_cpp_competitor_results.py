#!/usr/bin/env python3
"""Chart non-Rust k-d tree/NN library results against separately benchmarked Kiddo results.

Mirrors chart_external_kd_tree_results.py's approach for the Rust
competitors. Accepts one or more --cpp result files sharing the
profile_cpp_competitors/{f32,f64} group schema: the FFI-driven Rust bench
export (nanoflann, ALGLIB, Pkd-tree single-query mode; see
benches/profile_cpp_competitors.rs's module doc comment for what was tried
and dropped) and the standalone Julia NearestNeighbors.jl export
(scripts/bench_nearestneighbors_jl.jl), which write the same schema from a
separate process since there's no practical way to embed a Julia runtime in
the Rust criterion harness.
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


CPP_GROUP = "profile_cpp_competitors"
KIDDO_GROUPS = {
    "nearest_one": "profile_v6_nearest_one_eytzinger",
    "nearest_n": "profile_v6_nearest_n_eytzinger",
    "within_radius": "profile_v6_query_family_eytzinger",
}
LIBRARY_PREFIXES = (
    ("nanoflann_", "nanoflann"),
    ("alglib_", "ALGLIB"),
    ("pkdtree_", "Pkd-tree"),
    ("nearestneighborsjl_", "NearestNeighbors.jl"),
)
LIBRARY_ORDER = ("kiddo", "nanoflann", "ALGLIB", "Pkd-tree", "NearestNeighbors.jl")
COLORS = {
    "kiddo": "#3264a8",
    "nanoflann": "#d35400",
    "ALGLIB": "#1f8f3a",
    "Pkd-tree": "#8e44ad",
    "NearestNeighbors.jl": "#c2185b",
}
SCALARS = ("f32", "f64")
DEFAULT_RADIUS = 0.05
SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]*$")


@dataclass(frozen=True)
class Point:
    tree_size: int
    duration_ns: float
    lower_ns: float
    upper_ns: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare C++ k-d tree/NN library Criterion exports with Kiddo's "
            "separately collected 3D Eytzinger benchmarks."
        )
    )
    parser.add_argument("mode", choices=("charts", "all"))
    parser.add_argument(
        "--cpp",
        type=Path,
        nargs="+",
        required=True,
        help="one or more result exports sharing the profile_cpp_competitors schema",
    )
    parser.add_argument("--kiddo-nearest-one", type=Path, required=True)
    parser.add_argument("--kiddo-nearest-n", type=Path, required=True)
    parser.add_argument("--kiddo-query-family", type=Path, required=True)
    parser.add_argument("--result-label", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--html-name", default="cpp-competitor-benchmarks.html")
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


def read_series(path: Path, expected_group: str) -> dict[tuple[str, str], list[Point]]:
    series: dict[tuple[str, str], list[Point]] = {}
    for result in load_json(path)["results"]:
        metadata = result.get("metadata")
        estimates = result.get("estimates")
        if not isinstance(metadata, dict) or not isinstance(estimates, dict):
            raise RuntimeError(f"{path} contains incomplete benchmark data")

        group_id = metadata.get("group_id")
        function_id = metadata.get("function_id")
        if not isinstance(group_id, str) or not isinstance(function_id, str):
            raise RuntimeError(f"{path} contains an invalid benchmark identity")
        if group_id.rsplit("/", 1)[0] != expected_group:
            raise RuntimeError(
                f"{path} contains group {group_id!r}, expected {expected_group!r}"
            )
        scalar = group_id.rsplit("/", 1)[-1]
        if scalar not in SCALARS:
            raise RuntimeError(
                f"{path} contains unsupported scalar/dimensional group {group_id!r}; "
                "this comparison is strictly 3D f32/f64"
            )

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
            finite_positive(interval.get("lower_bound"), "lower duration bound", path)
            / query_count,
            finite_positive(interval.get("upper_bound"), "upper duration bound", path)
            / query_count,
        )
        series.setdefault((scalar, function_id), []).append(point)

    for key, points in series.items():
        points.sort(key=lambda point: point.tree_size)
        sizes = [point.tree_size for point in points]
        if len(sizes) != len(set(sizes)):
            raise RuntimeError(f"duplicate tree sizes for {key!r} in {path}")
    return series


def cpp_identity(function_id: str) -> tuple[str, str]:
    for prefix, library in LIBRARY_PREFIXES:
        if function_id.startswith(prefix):
            query = function_id[len(prefix) :]
            if (
                query == "nearest_one"
                or re.fullmatch(r"nearest_n_k(?:5|20|50)", query)
                or re.fullmatch(r"within_radius_r[0-9.eE+-]+", query)
            ):
                return library, query
            raise RuntimeError(f"unsupported C++ competitor query identity: {function_id}")
    raise RuntimeError(
        f"unexpected library in C++ competitor benchmark identity {function_id!r}; "
        "Kiddo must come from its focused benchmark exports"
    )


def kiddo_points(
    scalar: str,
    query: str,
    nearest_one: dict[tuple[str, str], list[Point]],
    nearest_n: dict[tuple[str, str], list[Point]],
    query_family: dict[tuple[str, str], list[Point]],
) -> list[Point]:
    if query == "nearest_one":
        source = nearest_one
        function_id = query
    elif query.startswith("nearest_n_k"):
        source = nearest_n
        function_id = query
    else:
        radius_match = re.fullmatch(r"within_radius_r([0-9.eE+-]+)", query)
        if radius_match is None:
            raise RuntimeError(f"unsupported query {query!r}")
        radius = float(radius_match.group(1))
        if not math.isclose(radius, DEFAULT_RADIUS, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(
                f"C++ competitor radius {radius:g} has no equivalent focused Kiddo result; "
                f"profile_v6_query_family_eytzinger uses radius {DEFAULT_RADIUS:g}"
            )
        source = query_family
        function_id = "within_unsorted"

    try:
        return source[(scalar, function_id)]
    except KeyError as error:
        raise RuntimeError(
            f"missing Kiddo {scalar}/{function_id} results required for {query}"
        ) from error


def collect_charts(
    cpp_paths: list[Path],
    nearest_one_path: Path,
    nearest_n_path: Path,
    query_family_path: Path,
) -> dict[tuple[str, str], dict[str, list[Point]]]:
    cpp: dict[tuple[str, str], list[Point]] = {}
    for cpp_path in cpp_paths:
        for key, points in read_series(cpp_path, CPP_GROUP).items():
            if key in cpp:
                raise RuntimeError(
                    f"duplicate {key!r} series across --cpp inputs (last from {cpp_path})"
                )
            cpp[key] = points
    nearest_one = read_series(nearest_one_path, KIDDO_GROUPS["nearest_one"])
    nearest_n = read_series(nearest_n_path, KIDDO_GROUPS["nearest_n"])
    query_family = read_series(
        query_family_path, KIDDO_GROUPS["within_radius"]
    )

    charts: dict[tuple[str, str], dict[str, list[Point]]] = {}
    for (scalar, function_id), points in cpp.items():
        library, query = cpp_identity(function_id)
        chart = charts.setdefault((scalar, query), {})
        if library in chart:
            raise RuntimeError(f"duplicate {library} series for {scalar}/{query}")
        chart[library] = points

    # Library coverage is intentionally uneven (see module docstring), so
    # unlike a uniform matrix we only require that *some* competitor
    # reported each scalar/query combination, not that every library did.
    expected_queries = {
        "nearest_one",
        "nearest_n_k5",
        "nearest_n_k20",
        "nearest_n_k50",
        f"within_radius_r{DEFAULT_RADIUS}",
    }
    expected_keys = {(scalar, query) for scalar in SCALARS for query in expected_queries}
    missing = expected_keys - charts.keys()
    extra = charts.keys() - expected_keys
    if missing or extra:
        raise RuntimeError(
            "C++ competitor result matrix does not match the expected 3D query matrix; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )

    # Series are ragged rather than truncated to their shared tree sizes: a
    # library that cannot reach the largest sizes (Pkd-tree's f32 build
    # asserts above 2^21) would otherwise drag every other library's curve
    # down to its own ceiling, discarding data that was measured. Each
    # library is plotted over exactly the range it covers, and render_charts
    # marks any partial range in the legend so a short line is never mistaken
    # for a missing measurement.
    for (scalar, query), chart in charts.items():
        chart["kiddo"] = kiddo_points(
            scalar, query, nearest_one, nearest_n, query_family
        )
        for library, points in chart.items():
            if not points:
                raise RuntimeError(f"empty {library} series for {scalar}/{query}")
            chart[library] = sorted(points, key=lambda point: point.tree_size)
    return charts


def coverage_label(library: str, points: list[Point], full_sizes: list[int]) -> str:
    """Legend label, annotated with the tree-size range when it is partial."""
    sizes = {point.tree_size for point in points}
    if sizes == set(full_sizes):
        return library
    low = min(sizes).bit_length() - 1
    high = max(sizes).bit_length() - 1
    span = f"2^{low}" if low == high else f"2^{low}–2^{high}"
    return f"{library} ({span})"


def query_sort_key(query: str) -> tuple[int, int | float]:
    if query == "nearest_one":
        return (0, 0)
    nearest_n_match = re.fullmatch(r"nearest_n_k(\d+)", query)
    if nearest_n_match:
        return (1, int(nearest_n_match.group(1)))
    radius_match = re.fullmatch(r"within_radius_r(.+)", query)
    return (2, float(radius_match.group(1)) if radius_match else 0)


def query_title(query: str) -> str:
    if query == "nearest_one":
        return "nearest_one"
    nearest_n_match = re.fullmatch(r"nearest_n_k(\d+)", query)
    if nearest_n_match:
        return f"nearest_n (n={nearest_n_match.group(1)})"
    radius = query.removeprefix("within_radius_r")
    return f"within_radius (radius={radius})"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_") or "benchmark"


def render_charts(
    charts: dict[tuple[str, str], dict[str, list[Point]]],
    output_dir: Path,
    result_label: str,
) -> list[tuple[str, Path]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to generate benchmark charts") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, Path]] = []
    ordered = sorted(
        charts.items(),
        key=lambda item: (SCALARS.index(item[0][0]), query_sort_key(item[0][1])),
    )
    for (scalar, query), series in ordered:
        title = f"3D {query_title(query)} — {scalar}"
        figure, axis = plt.subplots(figsize=(10.5, 6.2))
        sizes = sorted(
            {point.tree_size for points in series.values() for point in points}
        )
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
                color=COLORS[library],
                label=coverage_label(library, points, sizes),
            )

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

        path = output_dir / (
            f"bench_result-cpp-competitors-{slug(result_label)}-"
            f"{scalar}-{slug(query)}.png"
        )
        figure.savefig(path, dpi=160)
        plt.close(figure)
        rendered.append((title, path))
    return rendered


def write_html(
    path: Path,
    charts: list[tuple[str, Path]],
    result_label: str,
    sources: list[Path],
) -> None:
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
  <title>Non-Rust competitor k-d tree benchmarks</title>
  <style>
    body {{ font: 16px/1.45 system-ui, sans-serif; max-width: 1180px; margin: 0 auto; padding: 2rem; color: #18212b; }}
    section {{ margin: 2.5rem 0; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #d8dee4; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Non-Rust competitor k-d tree benchmarks</h1>
  <p>Result label: <code>{html.escape(result_label)}</code>. All benchmarks are 3D. Each chart uses a base-2 logarithmic tree-size axis and a base-10 logarithmic per-query duration axis. Library coverage is uneven by design: ALGLIB is f64-only, Pkd-tree (single-query mode shown here) has no within_radius series and its f32 build asserts above 2^21 points, and NearestNeighbors.jl comes from a separate Julia process rather than the Rust criterion harness. Series are therefore ragged: each library is drawn over exactly the tree sizes it covers, and a legend entry carrying a range such as <code>2^16–2^21</code> marks a library that does not span the whole axis. Pkd-tree's batch-parallel throughput is a different metric and is charted separately.</p>
  <details><summary>Source result exports</summary><ul>{source_items}</ul></details>
  {sections}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not SAFE_LABEL.fullmatch(args.result_label):
        raise RuntimeError(
            "result label must contain only letters, digits, '.', '_', '+', ':', or '-'"
        )
    kiddo_sources = [args.kiddo_nearest_one, args.kiddo_nearest_n, args.kiddo_query_family]
    charts = collect_charts(args.cpp, *kiddo_sources)
    rendered = render_charts(charts, args.output_dir, args.result_label)
    if args.mode == "all":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_html(args.output_dir / args.html_name, rendered, args.result_label, args.cpp + kiddo_sources)


if __name__ == "__main__":
    main()
