#!/usr/bin/env python3
"""Publish benchmark runs into a gh-pages tree and render PR comments."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

from chart_benchmark_results import (
    NS_PER_US,
    Point,
    SeriesKey,
    SAFE_KEY,
    duration_axis_label,
    generate_charts,
    import_matplotlib,
    load_json,
    result_series,
    slug,
)


COMMENT_MARKER = "<!-- kiddo-benchmark-report -->"
RESULT_GLOB = "bench_result-*.json"
STEM_STRATEGY_SUITE_PREFIX = "v6-stem-strategies-"
STEM_STRATEGY_ISA_ORDER = ("scalar", "avx2", "avx512", "neon")


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    ref_name: str
    sha: str
    benchmark_variant: str
    baseline_ref_name: str
    variant_key: str
    branch_path_key: str
    is_baseline: bool
    created_at: str
    baseline_run_id: str | None
    result_files: list[str]
    chart_files: list[str]
    pr_numbers: list[int]
    run_page_path: str
    run_page_url: str | None
    featured_chart_file: str | None = None


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_timestamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%SZ")


def sanitize_branch_component(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "branch"


def derive_variant_key(ref_name: str, baseline_ref_name: str) -> str:
    if ref_name == baseline_ref_name:
        return "baseline"
    stripped = re.sub(r"^(feature|fix)/", "", ref_name, count=1)
    return sanitize_branch_component(stripped)


def derive_branch_path_key(ref_name: str) -> str:
    return sanitize_branch_component(ref_name)


def site_join(base_url: str | None, path: str) -> str | None:
    if base_url is None:
        return None
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_result_files(source_dir: Path, dest_dir: Path, result_key: str) -> list[str]:
    ensure_dir(dest_dir)
    files = sorted(source_dir.glob(f"bench_result-*-{result_key}.json"))
    if not files:
        raise RuntimeError(
            f"no benchmark result JSON files found in {source_dir} for key {result_key!r}"
        )
    copied: list[str] = []
    for source in files:
        destination = dest_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination.name)
    return copied


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    ensure_dir(path.parent)
    path.write_text(value, encoding="utf-8")


def load_run_summary(path: Path) -> RunSummary:
    value = read_json(path)
    value.setdefault("benchmark_variant", infer_benchmark_variant(value))
    value.setdefault("baseline_ref_name", "master")
    return RunSummary(**value)


def save_run_summary(path: Path, summary: RunSummary) -> None:
    write_json(path, asdict(summary))


def infer_benchmark_variant(value: dict[str, Any]) -> str:
    result_files = value.get("result_files")
    if not isinstance(result_files, list):
        return "basic"

    names = [name for name in result_files if isinstance(name, str)]
    if any(name.startswith("bench_result-v6-dist-metrics-") for name in names):
        return "dist"
    if any(name.startswith("bench_result-v5-dist-metrics-") for name in names):
        return "dist"
    if any(name.startswith("bench_result-v6-construction-") for name in names):
        return "construction"
    if any(name.startswith("bench_result-v6-leaf-strategies-") for name in names):
        return "leaf"
    if any(name.startswith("bench_result-v6-stem-strategies-") for name in names):
        return "stems"
    if any(name.startswith("bench_result-v6-query-family-") for name in names):
        return "extended"
    if any(name.startswith("bench_result-v5-query-family-") for name in names):
        return "extended"
    return "basic"


def run_dirs_for(summary: RunSummary) -> tuple[Path, Path]:
    if summary.is_baseline:
        history = Path("baseline/history") / summary.run_id
        latest = Path("baseline/latest")
    else:
        history = Path("branches") / summary.branch_path_key / "history" / summary.run_id
        latest = Path("branches") / summary.branch_path_key / "latest"
    return history, latest


def baseline_latest_dir(pages_root: Path) -> Path:
    return pages_root / "baseline" / "latest"


def branch_root_dir(pages_root: Path, branch_path_key: str) -> Path:
    return pages_root / "branches" / branch_path_key


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def gather_history_runs(history_root: Path) -> list[RunSummary]:
    if not history_root.is_dir():
        return []
    runs: list[RunSummary] = []
    for run_json in history_root.glob("*/run.json"):
        runs.append(load_run_summary(run_json))
    runs.sort(key=lambda summary: summary.created_at, reverse=True)
    return runs


def list_branch_keys(pages_root: Path) -> list[str]:
    branches_root = pages_root / "branches"
    if not branches_root.is_dir():
        return []
    return sorted(path.name for path in branches_root.iterdir() if path.is_dir())


def render_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin: 0 auto; max-width: 1100px; padding: 2rem; font-family: system-ui, sans-serif; background: #f6f7f9; color: #202124; }}
    a {{ color: #0b57d0; }}
    code {{ background: #eef2f8; padding: .1rem .25rem; border-radius: .25rem; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border-bottom: 1px solid #dde3ea; text-align: left; padding: .6rem; vertical-align: top; }}
    section {{ margin: 2rem 0; padding: 1rem; border-radius: .5rem; background: white; box-shadow: 0 1px 5px #0002; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    .meta {{ color: #5f6368; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def build_comparison_charts(
    pages_root: Path,
    result_dir: Path,
    variant_key: str,
    chart_dir: Path,
) -> tuple[list[str], str | None]:
    baseline_results_dir = baseline_latest_dir(pages_root) / "results"
    if not baseline_results_dir.is_dir():
        return [], None
    reset_dir(chart_dir)
    with TemporaryDirectory(prefix="kiddo-benchmark-compare-") as temp_dir:
        temp_root = Path(temp_dir)
        for source in baseline_results_dir.glob(RESULT_GLOB):
            shutil.copy2(source, temp_root / source.name)
        for source in result_dir.glob(RESULT_GLOB):
            shutil.copy2(source, temp_root / source.name)
        metadata = generate_charts(variant_key, temp_root, chart_dir)
    ranked = sorted(
        metadata,
        key=lambda entry: float(entry.get("change_score", 0.0)),
        reverse=True,
    )
    return [str(entry["file_name"]) for entry in metadata], (
        str(ranked[0]["file_name"]) if ranked else None
    )


def load_suite_series(result_dir: Path) -> dict[str, dict[SeriesKey, list[Point]]]:
    suites: dict[str, dict[SeriesKey, list[Point]]] = {}
    for result_path in sorted(result_dir.glob(RESULT_GLOB)):
        name = result_path.name
        if not name.startswith("bench_result-"):
            continue
        suite = name[len("bench_result-") : name.rfind("-")]
        suites[suite] = result_series(result_path)
    return suites


def render_run_page(
    pages_root: Path,
    summary: RunSummary,
    run_dir: Path,
) -> None:
    run_url = summary.run_page_url
    home_rel = Path(os.path.relpath(pages_root / "index.html", run_dir)).as_posix()
    chart_items = ""
    if summary.chart_files:
        chart_sections = []
        for file_name in summary.chart_files:
            rel = f"charts/{file_name}"
            chart_sections.append(
                f"<section><h2>{file_name}</h2><img src=\"{rel}\" alt=\"{file_name}\"></section>"
            )
        chart_items = "".join(chart_sections)
    else:
        chart_items = (
            "<section><p class=\"meta\">No comparison charts were generated for this run.</p></section>"
        )

    results_links = "".join(
        f"<li><a href=\"results/{name}\">{name}</a></li>" for name in summary.result_files
    )
    baseline_text = (
        f"<p>Compared against baseline run <code>{summary.baseline_run_id}</code>.</p>"
        if summary.baseline_run_id
        else "<p>No baseline comparison was available for this run.</p>"
    )
    run_link = f'<a href="{run_url}">{run_url}</a>' if run_url else ""
    body = f"""
<p><a href="{home_rel}">Home</a></p>
<h1>Benchmark run: {summary.ref_name}</h1>
<p class="meta"><code>{summary.sha}</code> · {summary.created_at}</p>
<p>Benchmark suite: <code>{summary.benchmark_variant}</code></p>
<p>{f'Baseline publication for {summary.baseline_ref_name}' if summary.is_baseline else 'Branch comparison run'}</p>
{baseline_text}
<p>{run_link}</p>
<section>
  <h2>Result exports</h2>
  <ul>{results_links}</ul>
</section>
{chart_items}
"""
    write_text(run_dir / "index.html", render_html(f"Benchmark run {summary.run_id}", body))


def render_branch_index(
    pages_root: Path,
    branch_path_key: str,
    runs: list[RunSummary],
    site_url_base: str | None,
) -> None:
    if not runs:
        return
    latest = runs[0]
    rows = []
    for run in runs:
        run_dir, _ = run_dirs_for(run)
        run_url = site_join(site_url_base, (run_dir / "index.html").as_posix())
        public_link = f'<a href="{run_url}">public</a>' if run_url else "—"
        rows.append(
            "<tr>"
            f"<td><a href=\"history/{run.run_id}/index.html\">{run.run_id}</a></td>"
            f"<td><code>{run.benchmark_variant}</code></td>"
            f"<td><code>{run.sha[:12]}</code></td>"
            f"<td>{run.created_at}</td>"
            f"<td>{len(run.chart_files)}</td>"
            f"<td>{run.baseline_run_id or '—'}</td>"
            f"<td>{public_link}</td>"
            "</tr>"
        )
    body = f"""
<p><a href="../../index.html">Home</a></p>
<h1>Branch benchmark history: {latest.ref_name}</h1>
<p class="meta">Path key: <code>{branch_path_key}</code></p>
<p>Latest run: <a href="latest/index.html">{latest.run_id}</a></p>
<section>
  <table>
    <thead>
      <tr><th>Run</th><th>Suite</th><th>SHA</th><th>Created</th><th>Charts</th><th>Baseline</th><th>Public URL</th></tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""
    write_text(
        branch_root_dir(pages_root, branch_path_key) / "index.html",
        render_html(f"Branch benchmark history {latest.ref_name}", body),
    )


def stem_strategy_isa(suite: str) -> str | None:
    if not suite.startswith(STEM_STRATEGY_SUITE_PREFIX):
        return None
    return suite[len(STEM_STRATEGY_SUITE_PREFIX) :]


def stem_suite_sort_key(suite: str) -> tuple[int, str]:
    isa = stem_strategy_isa(suite) or suite
    try:
        return STEM_STRATEGY_ISA_ORDER.index(isa), suite
    except ValueError:
        return len(STEM_STRATEGY_ISA_ORDER), suite


def stem_group_label(group_id: str) -> str:
    return group_id.rsplit("/", 1)[-1]


def stem_strategy_colors(
    plt: Any,
    strategies: list[str],
) -> dict[str, Any]:
    color_map = plt.get_cmap("tab20")
    return {
        strategy: color_map(index % color_map.N)
        for index, strategy in enumerate(strategies)
    }


def render_current_stem_strategy_chart(
    plt: Any,
    suite: str,
    run_id: str,
    run_dt: datetime,
    series_map: dict[SeriesKey, list[Point]],
    output_path: Path,
) -> None:
    group_ids = sorted({key.group_id for key in series_map})
    strategies = sorted({key.function_id for key in series_map})
    colors = stem_strategy_colors(plt, strategies)
    figure, axes_grid = plt.subplots(
        len(group_ids),
        1,
        figsize=(14, max(6, 5 * len(group_ids))),
        squeeze=False,
        sharex=True,
    )
    axes = [row[0] for row in axes_grid]
    legend_handles: dict[str, Any] = {}

    for axis, group_id in zip(axes, group_ids):
        x_ticks: set[float] = set()
        for strategy in strategies:
            points = series_map.get(SeriesKey(group_id, strategy))
            if not points:
                continue
            x_vals = [point.tree_log2 for point in points]
            y_vals = [point.duration_ns / NS_PER_US for point in points]
            lower = [point.lower_ns / NS_PER_US for point in points]
            upper = [point.upper_ns / NS_PER_US for point in points]
            (line,) = axis.plot(
                x_vals,
                y_vals,
                marker="o",
                linewidth=2,
                label=strategy,
                color=colors[strategy],
            )
            axis.fill_between(
                x_vals,
                lower,
                upper,
                color=colors[strategy],
                alpha=0.06,
            )
            legend_handles.setdefault(strategy, line)
            x_ticks.update(x_vals)
        axis.set_yscale("log")
        axis.set_ylabel("Mean duration/query (us, log)")
        axis.set_title(stem_group_label(group_id))
        axis.set_xticks(sorted(x_ticks))
        axis.set_xticklabels([f"2^{value:g}" for value in sorted(x_ticks)])
        axis.grid(True, which="both", alpha=0.25)

    axes[-1].set_xlabel("Tree size")
    isa = stem_strategy_isa(suite) or suite
    figure.suptitle(
        f"Current stem strategy comparison — {isa.upper()}\n"
        f"{run_id} · {format_timestamp(run_dt)}"
    )
    figure.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        fontsize="small",
    )
    figure.tight_layout(rect=(0, 0, 0.78, 0.94))
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def render_historical_stem_strategy_chart(
    plt: Any,
    suite: str,
    tree_size: float,
    entries: list[tuple[datetime, str, dict[SeriesKey, list[Point]]]],
    output_path: Path,
) -> None:
    group_ids = sorted(
        {
            key.group_id
            for _, _, series_map in entries
            for key in series_map
        }
    )
    strategies = sorted(
        {
            key.function_id
            for _, _, series_map in entries
            for key in series_map
        }
    )
    colors = stem_strategy_colors(plt, strategies)
    figure, axes_grid = plt.subplots(
        len(group_ids),
        1,
        figsize=(14, max(6, 5 * len(group_ids))),
        squeeze=False,
        sharex=True,
    )
    axes = [row[0] for row in axes_grid]
    legend_handles: dict[str, Any] = {}

    for axis, group_id in zip(axes, group_ids):
        for strategy in strategies:
            x_vals: list[datetime] = []
            y_vals: list[float] = []
            key = SeriesKey(group_id, strategy)
            for run_dt, _, series_map in entries:
                point = next(
                    (
                        point
                        for point in series_map.get(key, [])
                        if point.tree_log2 == tree_size
                    ),
                    None,
                )
                if point is None:
                    continue
                x_vals.append(run_dt)
                y_vals.append(point.duration_ns / NS_PER_US)
            if not x_vals:
                continue
            (line,) = axis.plot(
                x_vals,
                y_vals,
                marker="o",
                linewidth=2,
                label=strategy,
                color=colors[strategy],
            )
            legend_handles.setdefault(strategy, line)
        axis.set_yscale("log")
        axis.set_ylabel("Mean duration/query (us, log)")
        axis.set_title(stem_group_label(group_id))
        axis.grid(True, which="both", alpha=0.25)

    axes[-1].set_xlabel("Baseline run")
    isa = stem_strategy_isa(suite) or suite
    figure.suptitle(
        f"Historical stem strategy comparison — {isa.upper()} — "
        f"tree size=2^{tree_size:g}"
    )
    figure.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        fontsize="small",
    )
    figure.autofmt_xdate()
    figure.tight_layout(rect=(0, 0, 0.78, 0.94))
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def render_baseline_trends(pages_root: Path) -> list[str]:
    history_root = pages_root / "baseline" / "history"
    runs = gather_history_runs(history_root)
    charts_written: list[str] = []
    trend_dir = pages_root / "baseline" / "trends" / "charts"
    reset_dir(trend_dir)
    if not runs:
        return charts_written

    _, plt = import_matplotlib()
    trends: dict[tuple[str, str, str], list[tuple[datetime, str, list[Point]]]] = {}
    stem_runs: dict[
        str,
        list[tuple[datetime, str, dict[SeriesKey, list[Point]]]],
    ] = {}
    for run in reversed(runs):
        run_dt = datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        run_dir, _ = run_dirs_for(run)
        suites = load_suite_series(pages_root / run_dir / "results")
        for suite, series_map in suites.items():
            if stem_strategy_isa(suite) is not None:
                stem_runs.setdefault(suite, []).append((run_dt, run.run_id, series_map))
                continue
            for key, points in series_map.items():
                trends.setdefault((suite, key.group_id, key.function_id), []).append(
                    (run_dt, run.run_id, points)
                )

    current_stem_sections: list[str] = []
    historical_stem_sections: list[str] = []
    for suite in sorted(stem_runs, key=stem_suite_sort_key):
        entries = stem_runs[suite]
        run_dt, run_id, series_map = entries[-1]
        isa = stem_strategy_isa(suite) or suite
        chart_name = slug(f"current-stem-strategies-{isa}") + ".png"
        render_current_stem_strategy_chart(
            plt,
            suite,
            run_id,
            run_dt,
            series_map,
            trend_dir / chart_name,
        )
        charts_written.append(chart_name)
        current_stem_sections.append(
            f"<section><h3>{isa.upper()}</h3>"
            f"<p class=\"meta\">Latest stem benchmark: <code>{run_id}</code></p>"
            f"<img src=\"charts/{chart_name}\" alt=\"{chart_name}\"></section>"
        )

        if len(entries) < 2:
            continue
        tree_sizes = sorted(
            {
                point.tree_log2
                for _, _, entry_series in entries
                for points in entry_series.values()
                for point in points
            }
        )
        for tree_size in tree_sizes:
            chart_name = (
                slug(f"historical-stem-strategies-{isa}-log2-{tree_size:g}") + ".png"
            )
            render_historical_stem_strategy_chart(
                plt,
                suite,
                tree_size,
                entries,
                trend_dir / chart_name,
            )
            charts_written.append(chart_name)
            historical_stem_sections.append(
                f"<section><h3>{isa.upper()} · tree size=2<sup>{tree_size:g}</sup></h3>"
                f"<img src=\"charts/{chart_name}\" alt=\"{chart_name}\"></section>"
            )

    other_sections: list[str] = []
    for (suite, group_id, function_id), entries in sorted(trends.items()):
        if len(entries) < 2:
            continue
        figure, axis = plt.subplots(figsize=(10, 6))
        tree_sizes = sorted({point.tree_log2 for _, _, points in entries for point in points})
        for tree_size in tree_sizes:
            x_vals = []
            y_vals = []
            for run_dt, _, points in entries:
                match = next((point for point in points if point.tree_log2 == tree_size), None)
                if match is None:
                    continue
                x_vals.append(run_dt)
                y_vals.append(match.duration_ns / NS_PER_US)
            if x_vals:
                axis.plot(x_vals, y_vals, marker="o", linewidth=2, label=f"log2 n={tree_size:g}")
        axis.set_yscale("log")
        axis.set_xlabel("Baseline run")
        axis.set_ylabel(duration_axis_label(group_id))
        axis.set_title(f"{suite}: {group_id} / {function_id}")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        chart_name = slug(f"{suite}-{group_id}-{function_id}") + ".png"
        figure.tight_layout()
        figure.savefig(trend_dir / chart_name, dpi=140)
        plt.close(figure)
        charts_written.append(chart_name)
        other_sections.append(
            f"<section><h2>{suite}: {group_id} / {function_id}</h2>"
            f"<img src=\"charts/{chart_name}\" alt=\"{chart_name}\"></section>"
        )

    rows = "".join(
        "<tr>"
        f"<td><a href=\"../history/{run.run_id}/index.html\">{run.run_id}</a></td>"
        f"<td><code>{run.sha[:12]}</code></td>"
        f"<td>{run.created_at}</td>"
        f"<td>{len(run.result_files)}</td>"
        "</tr>"
        for run in runs
    )
    body = f"""
<p><a href="../../index.html">Home</a></p>
<h1>Baseline benchmark history</h1>
<section>
  <table>
    <thead><tr><th>Run</th><th>SHA</th><th>Created</th><th>Exports</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
<h2>Current stem strategy comparisons</h2>
<p class="meta">Latest available baseline stem results. Tree size is the x-axis and each series is a stem strategy; ISA variants are charted separately.</p>
{''.join(current_stem_sections) if current_stem_sections else '<section><p class="meta">No stem strategy baseline results are available.</p></section>'}
<h2>Historical stem strategy comparisons</h2>
<p class="meta">One chart per ISA and tree size. The x-axis is the baseline-run timeline and each series is a stem strategy.</p>
{''.join(historical_stem_sections) if historical_stem_sections else '<section><p class="meta">At least two stem strategy baseline runs are required before historical charts are available.</p></section>'}
<h2>Other benchmark history</h2>
{''.join(other_sections) if other_sections else '<section><p class="meta">At least two baseline runs are required before other trend charts are available.</p></section>'}
"""
    write_text(
        pages_root / "baseline" / "trends" / "index.html",
        render_html("Baseline benchmark history", body),
    )
    return charts_written


def render_root_index(pages_root: Path, site_url_base: str | None) -> None:
    baseline_runs = gather_history_runs(pages_root / "baseline" / "history")
    branch_keys = list_branch_keys(pages_root)
    baseline_latest = baseline_runs[0] if baseline_runs else None
    branch_rows = []
    branch_manifest_entries = []
    for branch_key in branch_keys:
        runs = gather_history_runs(branch_root_dir(pages_root, branch_key) / "history")
        if not runs:
            continue
        latest = runs[0]
        run_dir, _ = run_dirs_for(latest)
        branch_manifest_entries.append(
            {
                "branch_path_key": branch_key,
                "ref_name": latest.ref_name,
                "benchmark_variant": latest.benchmark_variant,
                "latest_run_id": latest.run_id,
                "latest_run_page_path": (run_dir / "index.html").as_posix(),
                "latest_run_page_url": site_join(site_url_base, (run_dir / "index.html").as_posix()),
                "run_count": len(runs),
            }
        )
        branch_rows.append(
            "<tr>"
            f"<td><a href=\"branches/{branch_key}/index.html\">{latest.ref_name}</a></td>"
            f"<td><code>{latest.benchmark_variant}</code></td>"
            f"<td>{len(runs)}</td>"
            f"<td><a href=\"branches/{branch_key}/latest/index.html\">{latest.run_id}</a></td>"
            f"<td><code>{latest.sha[:12]}</code></td>"
            f"<td>{latest.created_at}</td>"
            "</tr>"
        )

    baseline_history_json = [
        asdict(run) | {"run_page_url": run.run_page_url} for run in baseline_runs
    ]
    write_json(pages_root / "data" / "baseline-history.json", baseline_history_json)
    write_json(pages_root / "data" / "branches.json", branch_manifest_entries)
    for branch_key in branch_keys:
        runs = gather_history_runs(branch_root_dir(pages_root, branch_key) / "history")
        if runs:
            write_json(
                pages_root / "data" / "branch-runs" / f"{branch_key}.json",
                [asdict(run) for run in runs],
            )

    latest_baseline_text = (
        f'Latest baseline run: <a href="baseline/latest/index.html">{baseline_latest.run_id}</a> '
        f'· <code>{baseline_latest.sha[:12]}</code> · suite <code>{baseline_latest.benchmark_variant}</code>'
        if baseline_latest
        else "No baseline runs published yet."
    )
    body = f"""
<h1>Kiddo benchmark reports</h1>
<p>This site publishes Criterion exports, branch-vs-baseline comparisons, and baseline history from the repo benchmark workflow.</p>
<section>
  <h2>Baseline</h2>
  <p>{latest_baseline_text}</p>
  <p><a href="baseline/trends/index.html">View baseline history and trend charts</a></p>
</section>
<section>
  <h2>Branches</h2>
  <table>
    <thead><tr><th>Branch</th><th>Latest suite</th><th>Runs</th><th>Latest run</th><th>SHA</th><th>Created</th></tr></thead>
    <tbody>{''.join(branch_rows) if branch_rows else '<tr><td colspan="6">No branch runs published yet.</td></tr>'}</tbody>
  </table>
</section>
"""
    write_text(pages_root / "index.html", render_html("Kiddo benchmark reports", body))


def rebuild_site(pages_root: Path, site_url_base: str | None) -> None:
    baseline_runs = gather_history_runs(pages_root / "baseline" / "history")
    for run in baseline_runs:
        history_dir, _ = run_dirs_for(run)
        render_run_page(pages_root, run, pages_root / history_dir)
    latest_baseline = pages_root / "baseline" / "latest" / "run.json"
    if latest_baseline.is_file():
        latest_summary = load_run_summary(latest_baseline)
        _, latest_dir = run_dirs_for(latest_summary)
        render_run_page(pages_root, latest_summary, pages_root / latest_dir)
    for branch_key in list_branch_keys(pages_root):
        runs = gather_history_runs(branch_root_dir(pages_root, branch_key) / "history")
        for run in runs:
            history_dir, _ = run_dirs_for(run)
            render_run_page(pages_root, run, pages_root / history_dir)
        latest_run_json = branch_root_dir(pages_root, branch_key) / "latest" / "run.json"
        if latest_run_json.is_file():
            latest_summary = load_run_summary(latest_run_json)
            _, latest_dir = run_dirs_for(latest_summary)
            render_run_page(pages_root, latest_summary, pages_root / latest_dir)
        render_branch_index(pages_root, branch_key, runs, site_url_base)
    render_baseline_trends(pages_root)
    render_root_index(pages_root, site_url_base)


def publish_run(
    ref_name: str,
    sha: str,
    benchmark_variant: str,
    baseline_ref_name: str,
    results_dir: Path,
    pages_root: Path,
    site_url_base: str | None,
    pr_numbers: list[int],
    created_at: datetime | None = None,
) -> RunSummary:
    created_at = created_at or now_utc()
    variant_key = derive_variant_key(ref_name, baseline_ref_name)
    branch_path_key = derive_branch_path_key(ref_name)
    if not SAFE_KEY.fullmatch(variant_key):
        raise RuntimeError(f"derived variant key is invalid: {variant_key}")
    run_id = f"{compact_timestamp(created_at)}-{sha[:12]}"
    is_baseline = variant_key == "baseline"
    result_files_source = results_dir.resolve()
    pages_root = pages_root.resolve()
    ensure_dir(pages_root)

    summary = RunSummary(
        run_id=run_id,
        ref_name=ref_name,
        sha=sha,
        benchmark_variant=benchmark_variant,
        baseline_ref_name=baseline_ref_name,
        variant_key=variant_key,
        branch_path_key=branch_path_key,
        is_baseline=is_baseline,
        created_at=format_timestamp(created_at),
        baseline_run_id=None,
        result_files=[],
        chart_files=[],
        featured_chart_file=None,
        pr_numbers=sorted(set(pr_numbers)),
        run_page_path="",
        run_page_url=None,
    )
    history_rel, latest_rel = run_dirs_for(summary)
    history_dir = pages_root / history_rel
    latest_dir = pages_root / latest_rel
    reset_dir(history_dir / "results")
    summary = RunSummary(
        **{
            **asdict(summary),
            "result_files": copy_result_files(
                result_files_source, history_dir / "results", variant_key
            ),
        }
    )
    chart_files: list[str] = []
    featured_chart_file: str | None = None
    baseline_run_id: str | None = None
    if not is_baseline:
        latest_baseline_path = baseline_latest_dir(pages_root) / "run.json"
        if latest_baseline_path.is_file():
            baseline_summary = load_run_summary(latest_baseline_path)
            baseline_run_id = baseline_summary.run_id
            chart_files, featured_chart_file = build_comparison_charts(
                pages_root, history_dir / "results", variant_key, history_dir / "charts"
            )
        else:
            ensure_dir(history_dir / "charts")
    summary = RunSummary(
        **{
            **asdict(summary),
            "baseline_run_id": baseline_run_id,
            "chart_files": chart_files,
            "featured_chart_file": featured_chart_file,
            "run_page_path": relative_posix(history_dir / "index.html", pages_root),
            "run_page_url": site_join(site_url_base, relative_posix(history_dir / "index.html", pages_root)),
        }
    )
    save_run_summary(history_dir / "run.json", summary)
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(history_dir, latest_dir)
    rebuild_site(pages_root, site_url_base)
    return summary


def render_pr_comment(summary: RunSummary, site_url_base: str | None) -> str:
    run_url = summary.run_page_url or site_join(site_url_base, summary.run_page_path)
    lines = [
        COMMENT_MARKER,
        "## Benchmark report",
        "",
        f"- Branch: `{summary.ref_name}`",
        f"- Benchmark suite: `{summary.benchmark_variant}`",
        f"- Commit: `{summary.sha[:12]}`",
        f"- Published run: `{summary.run_id}`",
    ]
    if summary.baseline_run_id:
        lines.append(f"- Compared against baseline: `{summary.baseline_run_id}`")
    else:
        lines.append("- Compared against baseline: not available yet")
    if run_url:
        lines.append(f"- Full report: {run_url}")
    lines.append("")
    if summary.chart_files:
        if summary.featured_chart_file and run_url:
            chart_url = run_url.rsplit("/", 1)[0] + f"/charts/{summary.featured_chart_file}"
            lines.append("Most changed chart vs baseline:")
            lines.append("")
            lines.append(f"![{summary.featured_chart_file}]({chart_url})")
            lines.append("")
        lines.append(f"- Total charts generated: {len(summary.chart_files)}")
        lines.append("- See the full report page for the full chart set.")
        lines.append("")
    else:
        lines.extend(
            [
                "No comparison charts were generated for this run.",
                "",
                f"This usually means there is no published `{summary.baseline_ref_name}` baseline yet.",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def github_token() -> str:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        return subprocess.run(
            ["gh", "auth", "token"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("no GitHub token available via GITHUB_TOKEN, GH_TOKEN, or gh auth") from error


def github_request(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
) -> Any:
    token = github_token()
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "kiddo-benchmark-site",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {url} failed: {error.code} {details}") from error
    if not payload:
        return None
    return json.loads(payload)


def update_pr_comment(repo: str, pr_number: int, body: str) -> dict[str, Any]:
    owner, name = repo.split("/", 1)
    comments_url = (
        f"https://api.github.com/repos/{owner}/{name}/issues/{pr_number}/comments?per_page=100"
    )
    comments = github_request("GET", comments_url)
    existing = next(
        (comment for comment in comments if COMMENT_MARKER in comment.get("body", "")),
        None,
    )
    if existing is None:
        return github_request(
            "POST",
            f"https://api.github.com/repos/{owner}/{name}/issues/{pr_number}/comments",
            body={"body": body},
        )
    return github_request(
        "PATCH",
        existing["url"],
        body={"body": body},
    )


def find_pr_numbers(repo: str, ref_name: str) -> list[int]:
    owner, name = repo.split("/", 1)
    query = urllib.parse.urlencode({"state": "open", "head": f"{owner}:{ref_name}", "per_page": 100})
    url = f"https://api.github.com/repos/{owner}/{name}/pulls?{query}"
    pulls = github_request("GET", url)
    return [int(pull["number"]) for pull in pulls]


def parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    derive = subparsers.add_parser("derive-key", help="derive the benchmark variant key")
    derive.add_argument("--ref-name", required=True)
    derive.add_argument("--baseline-ref-name", default="master")

    derive_path = subparsers.add_parser("derive-path-key", help="derive the published branch path key")
    derive_path.add_argument("--ref-name", required=True)

    publish = subparsers.add_parser("publish-run", help="publish one benchmark run into a pages tree")
    publish.add_argument("--ref-name", required=True)
    publish.add_argument("--sha", required=True)
    publish.add_argument("--benchmark-variant", required=True)
    publish.add_argument("--baseline-ref-name", default="master")
    publish.add_argument("--results-dir", type=Path, required=True)
    publish.add_argument("--pages-root", type=Path, required=True)
    publish.add_argument("--site-url-base")
    publish.add_argument("--created-at")
    publish.add_argument("--summary-path", type=Path)
    publish.add_argument("--pr-number", action="append", type=int, default=[])

    render = subparsers.add_parser("render-pr-comment", help="render the sticky PR comment body")
    render.add_argument("--summary-path", type=Path, required=True)
    render.add_argument("--site-url-base")
    render.add_argument("--output", type=Path)

    update = subparsers.add_parser("update-pr-comment", help="upsert the sticky PR comment")
    update.add_argument("--repo", required=True)
    update.add_argument("--pr-number", type=int, required=True)
    update.add_argument("--summary-path", type=Path, required=True)
    update.add_argument("--site-url-base")

    prs = subparsers.add_parser("find-prs", help="find open PRs for a same-repo branch")
    prs.add_argument("--repo", required=True)
    prs.add_argument("--ref-name", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = args_parser()
    args = parser.parse_args(argv)
    if args.command == "derive-key":
        print(derive_variant_key(args.ref_name, args.baseline_ref_name))
        return 0
    if args.command == "derive-path-key":
        print(derive_branch_path_key(args.ref_name))
        return 0
    if args.command == "find-prs":
        print(json.dumps(find_pr_numbers(args.repo, args.ref_name)))
        return 0
    if args.command == "publish-run":
        summary = publish_run(
            ref_name=args.ref_name,
            sha=args.sha,
            benchmark_variant=args.benchmark_variant,
            baseline_ref_name=args.baseline_ref_name,
            results_dir=args.results_dir,
            pages_root=args.pages_root,
            site_url_base=args.site_url_base,
            pr_numbers=args.pr_number,
            created_at=parse_timestamp(args.created_at),
        )
        payload = json.dumps(asdict(summary), indent=2, sort_keys=True)
        if args.summary_path is not None:
            write_text(args.summary_path, payload + "\n")
        print(payload)
        return 0
    if args.command == "render-pr-comment":
        summary = RunSummary(**read_json(args.summary_path))
        body = render_pr_comment(summary, args.site_url_base)
        if args.output is not None:
            write_text(args.output, body)
        else:
            sys.stdout.write(body)
        return 0
    if args.command == "update-pr-comment":
        summary = RunSummary(**read_json(args.summary_path))
        body = render_pr_comment(summary, args.site_url_base)
        response = update_pr_comment(args.repo, args.pr_number, body)
        print(json.dumps({"id": response["id"], "html_url": response["html_url"]}, indent=2))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
