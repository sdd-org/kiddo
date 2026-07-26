#!/usr/bin/env python3
"""Chart matching baseline and variant Criterion result exports."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULT_PREFIX = "bench_result-"
DEFAULT_BASELINE_KEY = "baseline"
SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]*$")
NS_PER_US = 1_000.0
DIST_METRIC_SUITE_PREFIX = "v6-dist-metrics-"
DIST_METRIC_ISA_ORDER = ("scalar", "avx2", "avx512", "neon")
DIST_METRIC_REQUIRED_ISAS = ("scalar", "avx2", "avx512")
DIST_METRIC_GROUP_ID = "profile_v6_dist_metrics/f64"
DIST_METRIC_ORDER = (
    "squared_euclidean",
    "manhattan",
    "chebyshev",
    "minkowski_p3",
)
MATRIX_COLORS = (
    "#3264a8",
    "#d35400",
    "#1f8f3a",
    "#8e44ad",
    "#7f5f00",
    "#00838f",
)
CONSTRUCTION_GROUP_PREFIX = "profile_v6_construction/"


@dataclass(frozen=True)
class SeriesKey:
    group_id: str
    function_id: str


@dataclass(frozen=True)
class Point:
    tree_log2: float
    duration_ns: float
    lower_ns: float
    upper_ns: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare every matching bench_result-*-baseline.json and "
            "bench_result-*-[VARIANT_KEY].json pair."
        )
    )
    parser.add_argument("variant_key", help="result key to compare with baseline")
    parser.add_argument(
        "--baseline-key",
        default=DEFAULT_BASELINE_KEY,
        help="result key for the v6 baseline (default: baseline)",
    )
    parser.add_argument(
        "--v5-baseline-key",
        help=(
            "optional result key for v5 nearest_one/nearest_n results to overlay "
            "on matching v6 charts"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path.cwd(),
        help="directory containing result JSON files (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="directory for charts and HTML (default: current directory)",
    )
    parser.add_argument("--scratch", action="store_true", help="chart scratch strategies as one comparison")
    parser.add_argument("--html-name", default="latest_benchmark_run.html")
    return parser.parse_args()


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return cleaned or "benchmark"


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


def duration_axis_label(group_id: str) -> str:
    operation = "build" if group_id.startswith(CONSTRUCTION_GROUP_PREFIX) else "query"
    return f"Mean duration per {operation} (us, log scale)"


def result_series(path: Path) -> dict[SeriesKey, list[Point]]:
    series: dict[SeriesKey, list[Point]] = {}
    for result in load_json(path)["results"]:
        metadata = result.get("metadata")
        estimates = result.get("estimates")
        if not isinstance(metadata, dict) or not isinstance(estimates, dict):
            raise RuntimeError(
                f"{path} lacks benchmark metadata; regenerate it with the current exporter"
            )

        group_id = metadata.get("group_id")
        function_id = metadata.get("function_id") or "benchmark"
        tree_size = finite_positive(metadata.get("value_str"), "tree size", path)
        if not isinstance(group_id, str) or not isinstance(function_id, str):
            raise RuntimeError(f"invalid benchmark identity in {path}: {metadata!r}")

        throughput = metadata.get("throughput")
        operation_count = throughput.get("Elements") if isinstance(throughput, dict) else None
        operation_count = finite_positive(
            operation_count, "operation count/Elements throughput", path
        )

        mean = estimates.get("mean")
        interval = mean.get("confidence_interval") if isinstance(mean, dict) else None
        if not isinstance(mean, dict) or not isinstance(interval, dict):
            raise RuntimeError(f"missing mean confidence interval in {path}")

        duration = finite_positive(mean.get("point_estimate"), "mean duration", path)
        lower = finite_positive(interval.get("lower_bound"), "lower duration bound", path)
        upper = finite_positive(interval.get("upper_bound"), "upper duration bound", path)
        key = SeriesKey(group_id, function_id)
        series.setdefault(key, []).append(
            Point(
                tree_log2=math.log2(tree_size),
                duration_ns=duration / operation_count,
                lower_ns=lower / operation_count,
                upper_ns=upper / operation_count,
            )
        )

    for key, points in series.items():
        points.sort(key=lambda point: point.tree_log2)
        x_values = [point.tree_log2 for point in points]
        if len(x_values) != len(set(x_values)):
            raise RuntimeError(f"duplicate tree sizes for {key} in {path}")
    return series


def cross_version_series(
    series: dict[SeriesKey, list[Point]], path: Path
) -> dict[tuple[str, str], list[Point]]:
    """Key series by scalar type and function, ignoring the versioned group prefix."""
    compatible: dict[tuple[str, str], list[Point]] = {}
    for key, points in series.items():
        scalar_type = key.group_id.rsplit("/", 1)[-1]
        comparable_key = (scalar_type, key.function_id)
        if comparable_key in compatible:
            raise RuntimeError(
                f"duplicate cross-version series {comparable_key!r} in {path}"
            )
        compatible[comparable_key] = points
    return compatible


def query_kind(suite: str) -> str | None:
    if "nearest_one" in suite:
        return "nearest_one"
    if "nearest_n" in suite:
        return "nearest_n"
    return None


def subbenchmark_name(key: SeriesKey) -> str:
    group_parts = key.group_id.split("/")
    parts = group_parts[1:] + [key.function_id]
    return slug("-".join(part for part in parts if part))


def unique_chart_path(
    output_dir: Path,
    suite: str,
    key: SeriesKey,
    variant_key: str,
    paths_seen: set[Path],
) -> Path:
    base = f"bench_result-{slug(suite)}-{subbenchmark_name(key)}-{slug(variant_key)}"
    path = output_dir / f"{base}.png"
    if path in paths_seen:
        identity = f"{key.group_id}\0{key.function_id}".encode()
        path = output_dir / f"{base}-{hashlib.sha256(identity).hexdigest()[:8]}.png"
    paths_seen.add(path)
    return path


def render_chart(
    plt: Any,
    suite: str,
    key: SeriesKey,
    baseline: list[Point],
    variant: list[Point],
    variant_key: str,
    baseline_key: str,
    v5_baseline: list[Point] | None,
    v5_baseline_key: str | None,
    output_path: Path,
) -> None:
    if len(baseline) != len(variant):
        raise RuntimeError(
            f"baseline/variant point count mismatch for {suite}: "
            f"{key.group_id} / {key.function_id}"
        )
    for baseline_point, variant_point in zip(baseline, variant):
        if baseline_point.tree_log2 != variant_point.tree_log2:
            raise RuntimeError(
                f"baseline/variant tree sizes differ for {suite}: "
                f"{key.group_id} / {key.function_id}"
            )

    figure, axis = plt.subplots(figsize=(10, 6))
    plotted_series = [
        (baseline_key, baseline, "#3264a8"),
        (variant_key, variant, "#d35400"),
    ]
    if v5_baseline is not None and v5_baseline_key is not None:
        plotted_series.append((v5_baseline_key, v5_baseline, "#7d3c98"))

    for label, points, color in plotted_series:
        x = [point.tree_log2 for point in points]
        y = [point.duration_ns / NS_PER_US for point in points]
        lower = [point.lower_ns / NS_PER_US for point in points]
        upper = [point.upper_ns / NS_PER_US for point in points]
        axis.plot(x, y, marker="o", linewidth=2, label=label, color=color)
        axis.fill_between(x, lower, upper, color=color, alpha=0.14)

    x_ticks = sorted(
        {point.tree_log2 for _, points, _ in plotted_series for point in points}
    )
    axis.set_xticks(x_ticks)
    axis.set_xticklabels([f"{value:g}" for value in x_ticks])
    axis.set_yscale("log")
    axis.set_xlabel("log2(tree size)")
    axis.set_ylabel(duration_axis_label(key.group_id))
    axis.set_title(f"{suite}: {key.group_id} / {key.function_id}")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()

    y_min, y_max = axis.get_ylim()
    upward_factor = 10**0.009
    downward_factor = 10**-0.0125
    lower_safe = y_min * (10**0.01)
    upper_safe = y_max * (10**-0.01)
    for baseline_point, variant_point in zip(baseline, variant):
        delta_fraction = (variant_point.duration_ns - baseline_point.duration_ns) / (
            baseline_point.duration_ns
        )
        delta_percent = round(delta_fraction * 100)
        label = f"{delta_percent:+.0f}%"
        baseline_y_us = baseline_point.duration_ns / NS_PER_US
        variant_y_us = variant_point.duration_ns / NS_PER_US
        if delta_fraction < 0:
            anchor_y = variant_y_us
            preferred_above = False
        else:
            anchor_y = baseline_y_us
            preferred_above = True

        preferred_y = anchor_y * (upward_factor if preferred_above else downward_factor)
        if lower_safe <= preferred_y <= upper_safe:
            place_above = preferred_above
            text_y = preferred_y
        else:
            place_above = not preferred_above
            text_y = anchor_y * (upward_factor if place_above else downward_factor)

        axis.text(
            variant_point.tree_log2,
            text_y,
            label,
            color="#1f8f3a" if delta_fraction < 0 else "#c0392b",
            fontsize=9,
            ha="center",
            va="bottom" if place_above else "top",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 0.25,
            },
        )

    if v5_baseline is not None:
        if len(v5_baseline) != len(baseline):
            raise RuntimeError(f"v5 baseline point count mismatch for {suite}: {key.group_id} / {key.function_id}")
        for baseline_point, v5_point in zip(baseline, v5_baseline):
            if baseline_point.tree_log2 != v5_point.tree_log2:
                raise RuntimeError(f"v5 baseline tree sizes differ for {suite}: {key.group_id} / {key.function_id}")
            # Report v6 relative to v5, matching the usual "current vs reference"
            # interpretation of the percentage marker.
            delta_fraction = (baseline_point.duration_ns - v5_point.duration_ns) / v5_point.duration_ns
            axis.annotate(
                f"{delta_fraction * 100:+.0f}%",
                (baseline_point.tree_log2, baseline_point.duration_ns / NS_PER_US),
                textcoords="offset points", xytext=(0, -14), ha="center", va="top",
                color="#3264a8", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.2},
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def validate_series_pair(
    baseline: list[Point],
    variant: list[Point],
    description: str,
) -> None:
    if len(baseline) != len(variant):
        raise RuntimeError(f"baseline/variant point count mismatch for {description}")
    for baseline_point, variant_point in zip(baseline, variant):
        if baseline_point.tree_log2 != variant_point.tree_log2:
            raise RuntimeError(f"baseline/variant tree sizes differ for {description}")


def render_matrix_chart(
    plt: Any,
    title: str,
    series: list[tuple[str, list[Point], list[Point]]],
    variant_key: str,
    baseline_key: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 7))
    x_ticks: set[float] = set()

    for index, (series_name, baseline, variant) in enumerate(series):
        validate_series_pair(baseline, variant, f"{title}: {series_name}")
        color = MATRIX_COLORS[index % len(MATRIX_COLORS)]
        label = series_name.replace("_", " ")
        x = [point.tree_log2 for point in baseline]
        x_ticks.update(x)

        for run_label, points, line_style, alpha in (
            (baseline_key, baseline, "--", 0.72),
            (variant_key, variant, "-", 1.0),
        ):
            y = [point.duration_ns / NS_PER_US for point in points]
            lower = [point.lower_ns / NS_PER_US for point in points]
            upper = [point.upper_ns / NS_PER_US for point in points]
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                linestyle=line_style,
                alpha=alpha,
                label=f"{label} / {run_label}",
                color=color,
            )
            axis.fill_between(x, lower, upper, color=color, alpha=0.045)

    sorted_ticks = sorted(x_ticks)
    axis.set_xticks(sorted_ticks)
    axis.set_xticklabels([f"{value:g}" for value in sorted_ticks])
    axis.set_yscale("log")
    axis.set_xlabel("log2(tree size)")
    axis.set_ylabel("Mean duration per query (us, log scale)")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def matrix_change_score(series: list[tuple[str, list[Point], list[Point]]]) -> float:
    return sum(change_score(baseline, variant) for _, baseline, variant in series)


def render_scratch_chart(plt: Any, suite: str, scalar: str,
                         series: dict[SeriesKey, list[Point]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = {"default": "#3264a8", "with_scratch": "#d35400", "local_scratch": "#7d3c98"}
    for function_id in ("default", "with_scratch", "local_scratch"):
        points = series.get(SeriesKey(f"{suite}/{scalar}", function_id))
        if points is None:
            continue
        x = [p.tree_log2 for p in points]
        axis.plot(x, [p.duration_ns / NS_PER_US for p in points], marker="o",
                  linewidth=2, label=function_id, color=colors[function_id])
    baseline = series.get(SeriesKey(f"{suite}/{scalar}", "default"))
    if baseline is not None:
        for function_id in ("with_scratch", "local_scratch"):
            points = series.get(SeriesKey(f"{suite}/{scalar}", function_id))
            if points is None or len(points) != len(baseline):
                continue
            for base_point, point in zip(baseline, points):
                if base_point.tree_log2 != point.tree_log2:
                    continue
                delta = (point.duration_ns - base_point.duration_ns) / base_point.duration_ns
                axis.annotate(
                    f"{delta * 100:+.0f}%",
                    (point.tree_log2, point.duration_ns / NS_PER_US),
                    textcoords="offset points", xytext=(0, 8 if function_id == "with_scratch" else -14),
                    ha="center", fontsize=8,
                    color="#1f8f3a" if delta < 0 else "#c0392b",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.2},
                )
    axis.set_xticks(sorted({p.tree_log2 for points in series.values() for p in points}))
    axis.set_xlabel("log2(tree size)"); axis.set_ylabel("Mean duration per query (us, log scale)")
    axis.set_yscale("log"); axis.set_title(f"{suite}: {scalar} scratch strategies"); axis.grid(True, which="both", alpha=0.25); axis.legend()
    figure.tight_layout(); figure.savefig(output_path, dpi=140); plt.close(figure)


def change_score(baseline: list[Point], variant: list[Point]) -> float:
    if len(baseline) != len(variant):
        raise RuntimeError("cannot score unmatched baseline/variant series")
    score = 0.0
    for baseline_point, variant_point in zip(baseline, variant):
        if baseline_point.tree_log2 != variant_point.tree_log2:
            raise RuntimeError("cannot score series with mismatched tree sizes")
        delta_fraction = (
            variant_point.duration_ns - baseline_point.duration_ns
        ) / baseline_point.duration_ns
        score += delta_fraction * delta_fraction
    return score


def write_html(
    output_path: Path,
    variant_key: str,
    baseline_key: str,
    v5_baseline_key: str | None,
    charts: list[tuple[str, Path]],
) -> None:
    sections = []
    for title, chart_path in charts:
        encoded = base64.b64encode(chart_path.read_bytes()).decode("ascii")
        sections.append(
            "<section>"
            f"<h2>{html.escape(title)}</h2>"
            f'<img src="data:image/png;base64,{encoded}" '
            f'alt="{html.escape(title, quote=True)}">'
            f"<p><code>{html.escape(chart_path.name)}</code></p>"
            "</section>"
        )

    comparison = f"{baseline_key} vs {variant_key}"
    if v5_baseline_key is not None:
        comparison += f" with {v5_baseline_key}"
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark run: {html.escape(comparison)}</title>
  <style>
    body {{ margin: 0 auto; max-width: 1100px; padding: 2rem; font-family: system-ui, sans-serif; background: #f6f7f9; color: #202124; }}
    section {{ margin: 2rem 0; padding: 1rem; border-radius: .5rem; background: white; box-shadow: 0 1px 5px #0002; }}
    img {{ display: block; width: 100%; height: auto; }}
    h1, h2 {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Benchmark run: {html.escape(comparison)}</h1>
  <p>{len(charts)} charts generated from matching result-file pairs.</p>
  {''.join(sections)}
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def import_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("charting requires Python's matplotlib package") from error
    return matplotlib, plt


def dist_metric_isa(suite: str) -> str | None:
    if not suite.startswith(DIST_METRIC_SUITE_PREFIX):
        return None
    isa = suite.removeprefix(DIST_METRIC_SUITE_PREFIX)
    return isa if isa in DIST_METRIC_ISA_ORDER else None


def generate_dist_metric_matrix_charts(
    plt: Any,
    variant_key: str,
    baseline_key: str,
    output_dir: Path,
    pairs: dict[str, tuple[Path, Path]],
    paths_seen: set[Path],
) -> tuple[list[tuple[str, Path]], list[dict[str, str | float]]]:
    if not pairs:
        return [], []

    missing_isas = set(DIST_METRIC_REQUIRED_ISAS) - pairs.keys()
    if missing_isas:
        missing = ", ".join(sorted(missing_isas))
        raise RuntimeError(f"distance metric result matrix is missing ISA modes: {missing}")

    loaded: dict[
        str,
        tuple[dict[SeriesKey, list[Point]], dict[SeriesKey, list[Point]]],
    ] = {}
    for isa in DIST_METRIC_ISA_ORDER:
        pair = pairs.get(isa)
        if pair is not None:
            loaded[isa] = (result_series(pair[0]), result_series(pair[1]))

    charts: list[tuple[str, Path]] = []
    metadata: list[dict[str, str | float]] = []
    expected_keys = [
        SeriesKey(DIST_METRIC_GROUP_ID, function_id)
        for function_id in DIST_METRIC_ORDER
    ]
    for isa, (baseline_series, variant_series) in loaded.items():
        for run_label, result in (
            ("baseline", baseline_series),
            (variant_key, variant_series),
        ):
            missing_metrics = [
                key.function_id for key in expected_keys if key not in result
            ]
            if missing_metrics:
                missing = ", ".join(missing_metrics)
                raise RuntimeError(
                    f"distance metric {isa}/{run_label} result is missing metrics: {missing}"
                )

    for key in expected_keys:
        series = [
            (
                isa,
                loaded[isa][0][key],
                loaded[isa][1][key],
            )
            for isa in DIST_METRIC_ISA_ORDER
            if isa in loaded and key in loaded[isa][0] and key in loaded[isa][1]
        ]
        chart_key = SeriesKey("v6-dist-metrics/metric", key.function_id)
        chart_path = unique_chart_path(
            output_dir,
            "v6-dist-metrics",
            chart_key,
            variant_key,
            paths_seen,
        )
        title = f"Distance metric {key.function_id}: ISA comparison"
        render_matrix_chart(
            plt,
            title,
            series,
            variant_key,
            baseline_key,
            chart_path,
        )
        charts.append((title, chart_path))
        metadata.append(
            {
                "title": title,
                "file_name": chart_path.name,
                "suite": "v6-dist-metrics",
                "group_id": "metric",
                "function_id": key.function_id,
                "change_score": matrix_change_score(series),
            }
        )

    for isa in DIST_METRIC_ISA_ORDER:
        if isa not in loaded:
            continue
        baseline_series, variant_series = loaded[isa]
        series = [
            (
                key.function_id,
                baseline_series[key],
                variant_series[key],
            )
            for key in expected_keys
        ]
        chart_key = SeriesKey("v6-dist-metrics/isa", isa)
        chart_path = unique_chart_path(
            output_dir,
            "v6-dist-metrics",
            chart_key,
            variant_key,
            paths_seen,
        )
        title = f"Distance metrics: {isa} comparison"
        render_matrix_chart(
            plt,
            title,
            series,
            variant_key,
            baseline_key,
            chart_path,
        )
        charts.append((title, chart_path))
        metadata.append(
            {
                "title": title,
                "file_name": chart_path.name,
                "suite": "v6-dist-metrics",
                "group_id": "isa",
                "function_id": isa,
                "change_score": matrix_change_score(series),
            }
        )

    return charts, metadata


def generate_charts(
    variant_key: str,
    results_dir: Path,
    output_dir: Path,
    baseline_key: str = DEFAULT_BASELINE_KEY,
    v5_baseline_key: str | None = None,
) -> list[dict[str, str | float]]:
    result_keys = [variant_key, baseline_key]
    if v5_baseline_key is not None:
        result_keys.append(v5_baseline_key)
    for result_key in result_keys:
        if not SAFE_KEY.fullmatch(result_key):
            raise RuntimeError(
                "result keys must contain only letters, digits, '.', '_', '+', ':', or '-'"
            )
    if variant_key in {baseline_key, v5_baseline_key}:
        raise RuntimeError("VARIANT_KEY must identify a non-baseline result")

    _, plt = import_matplotlib()
    results_dir = results_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_suffix = f"-{baseline_key}.json"
    # Older v6 runs were exported with the plain `baseline` key. Treat those
    # files as the v6 baseline when the requested v6-baseline files are absent.
    if not list(results_dir.glob(f"{RESULT_PREFIX}*{baseline_suffix}")) and baseline_key == "v6-baseline":
        baseline_suffix = "-baseline.json"
    pairs: list[tuple[str, Path, Path]] = []
    dist_metric_pairs: dict[str, tuple[Path, Path]] = {}
    for baseline_path in sorted(results_dir.glob(f"{RESULT_PREFIX}*{baseline_suffix}")):
        suite = baseline_path.name[len(RESULT_PREFIX) : -len(baseline_suffix)]
        variant_path = results_dir / f"{RESULT_PREFIX}{suite}-{variant_key}.json"
        if variant_path.is_file():
            isa = dist_metric_isa(suite)
            if isa is None:
                pairs.append((suite, baseline_path, variant_path))
            else:
                dist_metric_pairs[isa] = (baseline_path, variant_path)

    if not pairs and not dist_metric_pairs:
        raise RuntimeError(
            f"no {baseline_key!r}/result pairs found for variant {variant_key!r} "
            f"in {results_dir}"
        )

    charts: list[tuple[str, Path]] = []
    metadata: list[dict[str, str | float]] = []
    paths_seen: set[Path] = set()
    v5_cache: dict[Path, dict[tuple[str, str], list[Point]]] = {}
    used_v5_baseline = False
    for suite, baseline_path, variant_path in pairs:
        baseline_series = result_series(baseline_path)
        variant_series = result_series(variant_path)
        v5_series: dict[tuple[str, str], list[Point]] = {}
        kind = query_kind(suite)
        if v5_baseline_key is not None and kind is not None:
            v5_path = (
                results_dir
                / f"{RESULT_PREFIX}v5-{kind}-{v5_baseline_key}.json"
            )
            if v5_path.is_file():
                if v5_path not in v5_cache:
                    v5_cache[v5_path] = cross_version_series(
                        result_series(v5_path), v5_path
                    )
                v5_series = v5_cache[v5_path]
        common_keys = sorted(
            baseline_series.keys() & variant_series.keys(),
            key=lambda key: (key.group_id, key.function_id),
        )
        for key in common_keys:
            scalar_type = key.group_id.rsplit("/", 1)[-1]
            v5_points = v5_series.get((scalar_type, key.function_id))
            used_v5_baseline |= v5_points is not None
            chart_path = unique_chart_path(output_dir, suite, key, variant_key, paths_seen)
            render_chart(
                plt,
                suite,
                key,
                baseline_series[key],
                variant_series[key],
                variant_key,
                baseline_key,
                v5_points,
                v5_baseline_key,
                chart_path,
            )
            title = f"{suite}: {key.group_id} / {key.function_id}"
            charts.append((title, chart_path))
            metadata.append(
                {
                    "title": title,
                    "file_name": chart_path.name,
                    "suite": suite,
                    "group_id": key.group_id,
                    "function_id": key.function_id,
                    "change_score": change_score(baseline_series[key], variant_series[key]),
                }
            )

    matrix_charts, matrix_metadata = generate_dist_metric_matrix_charts(
        plt,
        variant_key,
        baseline_key,
        output_dir,
        dist_metric_pairs,
        paths_seen,
    )
    charts.extend(matrix_charts)
    metadata.extend(matrix_metadata)

    if not charts:
        raise RuntimeError("matching result files contained no common benchmark series")

    html_path = output_dir / "latest_benchmark_run.html"
    write_html(
        html_path,
        variant_key,
        baseline_key,
        v5_baseline_key if used_v5_baseline else None,
        charts,
    )
    return metadata


def generate_scratch_charts(result_key: str, results_dir: Path, output_dir: Path, html_name: str) -> None:
    path = results_dir / f"{RESULT_PREFIX}v6-nearest_one-scratch-{result_key}.json"
    if not path.is_file():
        raise RuntimeError(f"missing scratch result file: {path}")
    series = result_series(path)
    colors = {
        "default": "#3264a8",
        "with_scratch": "#d35400",
        "local_scratch": "#1f8f3a",
    }
    bands = {
        "default": "rgba(50, 100, 168, 0.26)",
        "with_scratch": "rgba(211, 84, 0, 0.26)",
        "local_scratch": "rgba(31, 143, 58, 0.26)",
    }
    chart_series = []
    for scalar in ("f32", "f64"):
        default_points = series.get(
            SeriesKey(f"profile_v6_nearest_one_scratch/{scalar}", "default")
        )
        default_by_tree = (
            {point.tree_log2: point.duration_ns for point in default_points}
            if default_points is not None
            else {}
        )
        for function_id, label in (("default", "default"), ("with_scratch", "with_scratch"), ("local_scratch", "local_scratch")):
            points = series.get(SeriesKey(f"profile_v6_nearest_one_scratch/{scalar}", function_id))
            if points is None:
                continue
            chart_series.append({
                "scalar": scalar,
                "strategy": function_id,
                "label": label,
                "color": colors[function_id],
                "band_color": bands[function_id],
                "points": [
                    {
                        "tree_log2": p.tree_log2,
                        "tree_size": round(2**p.tree_log2),
                        "duration_ns": p.duration_ns,
                        "lower_ns": p.lower_ns,
                        "upper_ns": p.upper_ns,
                        "change_from_default_percent": (
                            ((p.duration_ns - default_by_tree[p.tree_log2]) / default_by_tree[p.tree_log2]) * 100
                            if function_id != "default" and p.tree_log2 in default_by_tree
                            else None
                        ),
                    }
                    for p in points
                ],
            })
    if not chart_series:
        raise RuntimeError("scratch result contained no f32/f64 series")

    output_dir.mkdir(parents=True, exist_ok=True)

    data_name = f"{RESULT_PREFIX}v6-nearest_one-scratch-{result_key}.chart-data.json"
    data_path = output_dir / data_name
    data_path.write_text(
        json.dumps(
            {
                "result_key": result_key,
                "source_file": path.name,
                "suite": "profile_v6_nearest_one_scratch",
                "title": "v6 nearest-one scratch strategies",
                "series": chart_series,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    renderer_source = Path(__file__).with_name("d3_scratch_chart.js")
    renderer_name = renderer_source.name
    try:
        renderer_ref = renderer_source.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        renderer_ref = renderer_name
        renderer_target = output_dir / renderer_name
        shutil.copyfile(renderer_source, renderer_target)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v6 nearest-one scratch strategies</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <style>
    body {{ margin: 0 auto; max-width: 1100px; padding: 2rem; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #202124; }}
    header {{ margin-bottom: 1.5rem; }}
    h1 {{ font-size: 1.8rem; margin: .2rem 0 .5rem; }}
    p {{ color: #5f6368; }}
    section {{ margin: 2rem 0; padding: 1rem; border-radius: .5rem; background: white; box-shadow: 0 1px 5px #0002; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin-bottom: 1rem; }}
    label {{ font-weight: 600; color: #3c4043; }}
    select {{ margin-left: .35rem; padding: .4rem .6rem; border: 1px solid #dadce0; border-radius: .35rem; background: #fff; font: inherit; }}
    .chart-shell {{ position: relative; min-height: 680px; }}
    .chart-tooltip {{ position: absolute; pointer-events: none; opacity: 0; background: #e4e7eb; border: 2px solid #68717d; border-radius: 5px; padding: 8px 10px; color: #202124; font-size: 13px; box-shadow: 0 4px 14px #0002; }}
    .chart-tooltip table {{ border-collapse: collapse; }}
    .chart-tooltip th {{ padding: 0 14px 3px 0; text-align: right; color: #3c4043; font-weight: 700; white-space: nowrap; }}
    .chart-tooltip td {{ padding: 0 0 3px; white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .d3-chart svg {{ display: block; width: 100%; height: auto; cursor: default; }}
  </style>
</head>
<body>
  <header>
    <h1>v6 nearest-one scratch strategies</h1>
    <p>Result key: <code>{html.escape(result_key)}</code>. Compare query execution strategies across tree sizes.</p>
  </header>
  <section>
    <div class="controls">
      <label>Scalar:<select data-control="scalar"><option>f64</option><option>f32</option></select></label>
      <label>Y-axis:<select data-control="scale"><option value="log">Logarithmic</option><option value="linear">Linear</option></select></label>
    </div>
    <div class="chart-shell">
      <div class="d3-chart" data-scratch-chart data-data-url="{html.escape(data_name, quote=True)}"></div>
    </div>
  </section>
  <script src="{html.escape(renderer_ref, quote=True)}"></script>
</body>
</html>
"""
    (output_dir / html_name).write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.scratch:
        generate_scratch_charts(args.variant_key, args.results_dir, args.output_dir, args.html_name)
        return 0
    generate_charts(
        args.variant_key,
        args.results_dir,
        args.output_dir,
        baseline_key=args.baseline_key,
        v5_baseline_key=args.v5_baseline_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
