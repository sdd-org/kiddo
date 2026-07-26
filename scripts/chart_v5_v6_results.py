#!/usr/bin/env python3
"""Generate direct Kiddo v5-versus-v6 benchmark charts and an HTML report."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCALARS = ("f32", "f64")
SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]*$")
SUITES = (
    (
        "nearest_one",
        "nearest_one",
        "profile_v5_nearest_one_eytzinger",
        "profile_v6_nearest_one_eytzinger",
    ),
    (
        "approx_nearest_one",
        "approx_nearest_one",
        "profile_v5_approx_nearest_one_eytzinger",
        "profile_v6_approx_nearest_one_eytzinger",
    ),
    (
        "nearest_n",
        "nearest_n",
        "profile_v5_nearest_n_eytzinger",
        "profile_v6_nearest_n_eytzinger",
    ),
    (
        "query-family",
        "query family",
        "profile_v5_query_family_eytzinger",
        "profile_v6_query_family_eytzinger",
    ),
)
COLORS = {"v5": "#7d3c98", "v6": "#3264a8"}


@dataclass(frozen=True)
class Point:
    tree_size: int
    duration_ns: float
    lower_ns: float
    upper_ns: float


@dataclass(frozen=True)
class Chart:
    scalar: str
    suite_label: str
    function_id: str
    v5: tuple[Point, ...]
    v6: tuple[Point, ...]

    @property
    def title(self) -> str:
        return f"{self.suite_label}: {display_query(self.function_id)} — {self.scalar}"

    @property
    def ratios(self) -> list[float]:
        return [
            v6.duration_ns / v5.duration_ns
            for v5, v6 in zip(self.v5, self.v6, strict=True)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chart matching Kiddo v5 and v6 Criterion result exports."
    )
    parser.add_argument("mode", choices=("charts", "all"))
    parser.add_argument("--v6-key", required=True)
    parser.add_argument("--v5-key", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--html-name", default="v5-v6-benchmarks.html")
    return parser.parse_args()


def result_path(results_dir: Path, version: str, suite: str, key: str) -> Path:
    return results_dir / f"bench_result-{version}-{suite}-eytzinger-{key}.json"


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


def read_series(
    path: Path, expected_group: str
) -> dict[tuple[str, str], tuple[Point, ...]]:
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
        group, separator, scalar = group_id.rpartition("/")
        if separator != "/" or group != expected_group or scalar not in SCALARS:
            raise RuntimeError(
                f"{path} contains group {group_id!r}; expected "
                f"{expected_group}/{{f32,f64}}"
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
            tree_size=tree_size,
            duration_ns=finite_positive(
                mean.get("point_estimate"), "mean duration", path
            )
            / query_count,
            lower_ns=finite_positive(
                interval.get("lower_bound"), "lower duration bound", path
            )
            / query_count,
            upper_ns=finite_positive(
                interval.get("upper_bound"), "upper duration bound", path
            )
            / query_count,
        )
        series.setdefault((scalar, function_id), []).append(point)

    finished: dict[tuple[str, str], tuple[Point, ...]] = {}
    for key, points in series.items():
        points.sort(key=lambda point: point.tree_size)
        sizes = [point.tree_size for point in points]
        if len(sizes) != len(set(sizes)):
            raise RuntimeError(f"duplicate tree sizes for {key!r} in {path}")
        finished[key] = tuple(points)
    return finished


def collect_charts(
    v6_key: str, v5_key: str, results_dir: Path
) -> tuple[list[Chart], list[Path]]:
    charts: list[Chart] = []
    sources: list[Path] = []
    for suite, suite_label, v5_group, v6_group in SUITES:
        v5_path = result_path(results_dir, "v5", suite, v5_key)
        v6_path = result_path(results_dir, "v6", suite, v6_key)
        sources.extend((v5_path, v6_path))
        v5_series = read_series(v5_path, v5_group)
        v6_series = read_series(v6_path, v6_group)
        common_keys = sorted(
            v5_series.keys() & v6_series.keys(),
            key=lambda key: (SCALARS.index(key[0]), query_sort_key(key[1])),
        )
        if not common_keys:
            raise RuntimeError(f"{v5_path} and {v6_path} have no common series")

        for scalar, function_id in common_keys:
            v5_by_size = {point.tree_size: point for point in v5_series[(scalar, function_id)]}
            v6_by_size = {point.tree_size: point for point in v6_series[(scalar, function_id)]}
            shared_sizes = sorted(v5_by_size.keys() & v6_by_size.keys())
            if not shared_sizes:
                raise RuntimeError(
                    f"no common tree sizes for {suite}/{scalar}/{function_id}"
                )
            charts.append(
                Chart(
                    scalar=scalar,
                    suite_label=suite_label,
                    function_id=function_id,
                    v5=tuple(v5_by_size[size] for size in shared_sizes),
                    v6=tuple(v6_by_size[size] for size in shared_sizes),
                )
            )
    return charts, sources


def query_sort_key(function_id: str) -> tuple[int, tuple[int, ...], str]:
    if function_id == "nearest_one":
        return (0, (), function_id)
    if function_id == "approx_nearest_one":
        return (1, (), function_id)
    numbers = tuple(int(value) for value in re.findall(r"\d+", function_id))
    if function_id.startswith("nearest_n_k"):
        category = 2
    elif function_id == "within":
        category = 3
    elif function_id == "within_unsorted":
        category = 4
    elif function_id.startswith("nearest_n_within_k"):
        category = 5
    elif function_id.startswith("nearest_n_within_unsorted_k"):
        category = 6
    elif function_id.startswith("best_n_within_k"):
        category = 7
    else:
        category = 8
    return (category, numbers, function_id)


def display_query(function_id: str) -> str:
    return re.sub(r"_k(\d+)$", r" (n=\1)", function_id)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_") or "benchmark"


def render_charts(
    charts: list[Chart], output_dir: Path, v6_key: str, v5_key: str
) -> list[tuple[Chart, Path]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to generate benchmark charts") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[Chart, Path]] = []
    for chart in charts:
        figure, axis = plt.subplots(figsize=(10.5, 6.2))
        for version, points in (("v5", chart.v5), ("v6", chart.v6)):
            axis.errorbar(
                [point.tree_size for point in points],
                [point.duration_ns for point in points],
                yerr=[
                    [point.duration_ns - point.lower_ns for point in points],
                    [point.upper_ns - point.duration_ns for point in points],
                ],
                marker="o",
                markersize=5,
                linewidth=2,
                capsize=2,
                color=COLORS[version],
                label=f"{version} ({v5_key if version == 'v5' else v6_key})",
            )

        sizes = [point.tree_size for point in chart.v5]
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=10)
        axis.set_xticks(sizes)
        axis.set_xticklabels([f"2^{size.bit_length() - 1}" for size in sizes])
        axis.set_xlabel("Tree size (log₂ scale)")
        axis.set_ylabel("Mean query duration (ns/query, log₁₀ scale)")
        axis.set_title(chart.title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        figure.tight_layout()

        path = output_dir / (
            f"bench_result-v5-v6-{slug(v5_key)}-{slug(v6_key)}-"
            f"{chart.scalar}-{slug(chart.function_id)}.png"
        )
        figure.savefig(path, dpi=160)
        plt.close(figure)
        rendered.append((chart, path))
    return rendered


def format_change(ratio: float) -> str:
    if ratio < 1:
        return f"v6 {1 / ratio:.2f}× faster"
    if ratio > 1:
        return f"v6 {ratio:.2f}× slower"
    return "equal"


def write_html(
    path: Path,
    rendered: list[tuple[Chart, Path]],
    v6_key: str,
    v5_key: str,
    sources: list[Path],
) -> None:
    all_ratios = [ratio for chart, _ in rendered for ratio in chart.ratios]
    summary_rows = "\n".join(
        (
            "<tr>"
            f"<td>{html.escape(chart.scalar)}</td>"
            f"<td>{html.escape(chart.suite_label)}</td>"
            f"<td><code>{html.escape(chart.function_id)}</code></td>"
            f"<td>{html.escape(format_change(statistics.geometric_mean(chart.ratios)))}</td>"
            f"<td>{min(chart.ratios):.3f}–{max(chart.ratios):.3f}×</td>"
            "</tr>"
        )
        for chart, _ in rendered
    )
    sections = "\n".join(
        (
            f"<section><h2>{html.escape(chart.title)}</h2>"
            f"<p>{html.escape(format_change(statistics.geometric_mean(chart.ratios)))}</p>"
            f'<img src="{html.escape(chart_path.name, quote=True)}" '
            f'alt="{html.escape(chart.title, quote=True)}"></section>'
        )
        for chart, chart_path in rendered
    )
    source_items = "".join(
        f"<li><code>{html.escape(str(source))}</code></li>" for source in sources
    )
    overall = format_change(statistics.geometric_mean(all_ratios))
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kiddo v5 versus v6 benchmarks</title>
  <style>
    body {{ font: 16px/1.45 system-ui, sans-serif; max-width: 1180px; margin: 0 auto; padding: 2rem; color: #18212b; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: .5rem; text-align: left; }}
    section {{ margin: 2.5rem 0; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #d8dee4; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Kiddo v5 versus v6 benchmarks</h1>
  <p>v5: <code>{html.escape(v5_key)}</code>; v6: <code>{html.escape(v6_key)}</code>. Overall geometric mean: <strong>{html.escape(overall)}</strong>.</p>
  <p>Each chart uses a base-2 logarithmic tree-size axis and a base-10 logarithmic per-query duration axis. Ratios are v6 duration divided by v5 duration, so values below 1 are improvements.</p>
  <details><summary>Source result exports</summary><ul>{source_items}</ul></details>
  <h2>Summary</h2>
  <table>
    <thead><tr><th>Scalar</th><th>Suite</th><th>Query</th><th>Geometric mean</th><th>v6/v5 range</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
  {sections}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    for label, key in (("v6 key", args.v6_key), ("v5 key", args.v5_key)):
        if not SAFE_KEY.fullmatch(key):
            raise RuntimeError(
                f"{label} must contain only letters, digits, '.', '_', '+', ':', or '-'"
            )
    charts, sources = collect_charts(args.v6_key, args.v5_key, args.results_dir)
    rendered = render_charts(charts, args.output_dir, args.v6_key, args.v5_key)
    if args.mode == "all":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_html(
            args.output_dir / args.html_name,
            rendered,
            args.v6_key,
            args.v5_key,
            sources,
        )


if __name__ == "__main__":
    main()
