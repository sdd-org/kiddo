#!/usr/bin/env python3
"""Chart the focused within-radius projection benchmark."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_GROUP = "profile_v6_within_radius_projection"
SCALARS = ("f32", "f64")
SERIES = (
    ("kiddo_item_distance", "Kiddo: item + distance"),
    ("kiddo_item_only", "Kiddo: item only"),
    ("kiddo_point_item_distance", "Kiddo: point + item + distance"),
    ("kiddo_point_item", "Kiddo: point + item"),
    ("neighbourhood_item_only", "neighbourhood: item only"),
)
ITEM_ONLY_SERIES = (
    ("kiddo_item_only", "Kiddo: item only"),
    ("neighbourhood_item_only", "neighbourhood: item only"),
)
COLORS = {
    "kiddo_item_distance": "#3264a8",
    "kiddo_item_only": "#1f8f3a",
    "kiddo_point_item_distance": "#d35400",
    "kiddo_point_item": "#8e44ad",
    "neighbourhood_item_only": "#00838f",
}
SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]*$")


@dataclass(frozen=True)
class Point:
    tree_size: int
    duration_ns: float
    lower_ns: float
    upper_ns: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("charts", "all"))
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-label", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--html-name", default="within-radius-projection-benchmarks.html"
    )
    parser.add_argument(
        "--item-only",
        action="store_true",
        help="render only Kiddo item-only and neighbourhood item-only",
    )
    return parser.parse_args()


def finite_positive(value: Any, description: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid {description} in {path}: {value!r}") from error
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"invalid {description} in {path}: {value!r}")
    return number


def read_results(path: Path) -> dict[str, dict[str, list[Point]]]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError(f"{path} is not a Criterion result export")

    results: dict[str, dict[str, list[Point]]] = {
        scalar: {} for scalar in SCALARS
    }
    for result in payload["results"]:
        metadata = result.get("metadata")
        estimates = result.get("estimates")
        if not isinstance(metadata, dict) or not isinstance(estimates, dict):
            raise RuntimeError(f"{path} contains incomplete benchmark data")

        group_id = metadata.get("group_id")
        function_id = metadata.get("function_id")
        if not isinstance(group_id, str) or not isinstance(function_id, str):
            raise RuntimeError(f"{path} contains an invalid benchmark identity")
        prefix, separator, scalar = group_id.rpartition("/")
        if separator != "/" or prefix != EXPECTED_GROUP or scalar not in SCALARS:
            raise RuntimeError(f"unexpected benchmark group {group_id!r} in {path}")
        if function_id not in dict(SERIES):
            raise RuntimeError(f"unexpected benchmark function {function_id!r} in {path}")

        tree_size_value = finite_positive(
            metadata.get("value_str"), "tree size", path
        )
        tree_size = int(tree_size_value)
        if tree_size != tree_size_value or tree_size & (tree_size - 1):
            raise RuntimeError(f"tree size is not a power of two in {path}")

        throughput = metadata.get("throughput")
        query_count = (
            throughput.get("Elements") if isinstance(throughput, dict) else None
        )
        query_count = finite_positive(query_count, "Elements throughput", path)
        mean = estimates.get("mean")
        interval = mean.get("confidence_interval") if isinstance(mean, dict) else None
        if not isinstance(mean, dict) or not isinstance(interval, dict):
            raise RuntimeError(f"{path} contains no mean confidence interval")

        point = Point(
            tree_size,
            finite_positive(mean.get("point_estimate"), "mean duration", path)
            / query_count,
            finite_positive(interval.get("lower_bound"), "lower bound", path)
            / query_count,
            finite_positive(interval.get("upper_bound"), "upper bound", path)
            / query_count,
        )
        results[scalar].setdefault(function_id, []).append(point)

    expected_functions = {function_id for function_id, _ in SERIES}
    for scalar, scalar_results in results.items():
        missing = expected_functions - scalar_results.keys()
        extra = scalar_results.keys() - expected_functions
        if missing or extra:
            raise RuntimeError(
                f"{scalar} result matrix mismatch: "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
            )
        sizes = None
        for points in scalar_results.values():
            points.sort(key=lambda point: point.tree_size)
            point_sizes = [point.tree_size for point in points]
            if len(point_sizes) != len(set(point_sizes)):
                raise RuntimeError(f"duplicate {scalar} tree sizes in {path}")
            if sizes is None:
                sizes = point_sizes
            elif sizes != point_sizes:
                raise RuntimeError(f"{scalar} series use different tree sizes in {path}")
    return results


def render_charts(
    results: dict[str, dict[str, list[Point]]],
    output_dir: Path,
    result_label: str,
    series: tuple[tuple[str, str], ...],
    filename_suffix: str,
    item_only: bool,
) -> list[tuple[str, Path]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to generate charts") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for scalar in SCALARS:
        figure, axis = plt.subplots(figsize=(10.5, 6.2))
        for function_id, display_name in series:
            points = results[scalar][function_id]
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
                color=COLORS[function_id],
                label=display_name,
            )

        sizes = [
            point.tree_size
            for point in results[scalar]["kiddo_item_only"]
        ]
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=10)
        axis.set_xticks(sizes, [f"2^{size.bit_length() - 1}" for size in sizes])
        axis.set_xlabel("Tree size (log₂ scale)")
        axis.set_ylabel("Mean query duration (ns/query, log₁₀ scale)")
        chart_subject = (
            "item-only comparison" if item_only else "result projections"
        )
        axis.set_title(f"3D within-radius {chart_subject} — {scalar}")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(title="Implementation" if item_only else "Result projection")

        if item_only:
            lower_y, upper_y = axis.get_ylim()
            axis.set_ylim(lower_y / 1.2, upper_y)
            kiddo_points = results[scalar]["kiddo_item_only"]
            neighbourhood_points = results[scalar]["neighbourhood_item_only"]
            if len(kiddo_points) != len(neighbourhood_points):
                raise RuntimeError(
                    f"{scalar} item-only series use different point counts"
                )
            for index, (kiddo_point, neighbourhood_point) in enumerate(
                zip(kiddo_points, neighbourhood_points)
            ):
                if kiddo_point.tree_size != neighbourhood_point.tree_size:
                    raise RuntimeError(
                        f"{scalar} item-only series use different tree sizes"
                    )
                faster_percent = (
                    1.0
                    - kiddo_point.duration_ns / neighbourhood_point.duration_ns
                ) * 100.0
                horizontal_alignment = "center"
                if index == 0:
                    horizontal_alignment = "left"
                elif index == len(kiddo_points) - 1:
                    horizontal_alignment = "right"
                axis.annotate(
                    f"{faster_percent:.0f}%",
                    (kiddo_point.tree_size, kiddo_point.duration_ns),
                    textcoords="offset points",
                    xytext=(0, -13),
                    ha=horizontal_alignment,
                    va="top",
                    color="#176d2b",
                    fontsize=8,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 0.2,
                    },
                )
        figure.tight_layout()

        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", result_label).strip("-")
        path = output_dir / (
            f"bench_result-v6-within-radius-projection-{safe_label}"
            f"{filename_suffix}-{scalar}.png"
        )
        figure.savefig(path, dpi=160)
        plt.close(figure)
        rendered.append((scalar, path))
    return rendered


def render_html(
    charts: list[tuple[str, Path]],
    output_dir: Path,
    html_name: str,
    result_label: str,
    result_path: Path,
    item_only: bool,
) -> Path:
    sections = "\n".join(
        (
            f"<section><h2>{html.escape(scalar)}</h2>"
            f'<img src="{html.escape(path.name)}" alt="{html.escape(scalar)}"></section>'
        )
        for scalar, path in charts
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Within-radius projection benchmarks</title>
  <style>
    body {{ font: 16px/1.45 system-ui, sans-serif; max-width: 1180px; margin: 0 auto; padding: 2rem; color: #18212b; }}
    section {{ margin: 2.5rem 0; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #d8dee4; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Within-radius {"item-only comparison" if item_only else "projection benchmarks"}</h1>
  <p>Result label: <code>{html.escape(result_label)}</code>. All benchmarks use the same seeded 3D data and radius as the external comparison.</p>
  <p>Source result: <code>{html.escape(str(result_path))}</code></p>
  {sections}
</body>
</html>
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / html_name
    html_path.write_text(document, encoding="utf-8")
    return html_path


def main() -> None:
    args = parse_args()
    if not SAFE_LABEL.fullmatch(args.result_label):
        raise RuntimeError(
            "result label must contain only letters, digits, '.', '_', '+', ':', or '-'"
        )
    results = read_results(args.result)
    series = ITEM_ONLY_SERIES if args.item_only else SERIES
    filename_suffix = "-item-only" if args.item_only else ""
    charts = render_charts(
        results,
        args.output_dir,
        args.result_label,
        series,
        filename_suffix,
        args.item_only,
    )
    for _, path in charts:
        print(path)
    if args.mode == "all":
        print(
            render_html(
                charts,
                args.output_dir,
                args.html_name,
                args.result_label,
                args.result,
                args.item_only,
            )
        )


if __name__ == "__main__":
    main()
