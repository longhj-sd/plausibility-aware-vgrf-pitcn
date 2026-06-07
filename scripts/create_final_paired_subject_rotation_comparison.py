"""Create paired PI-TCN minus TCN subject-rotation comparisons."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


ROOT = Path("outputs") / "revision_evidence"
OUT = ROOT / "final_audit"
METRIC_MAP = {
    "normalized_rmse": "normalized_rmse",
    "normalized_mae": "normalized_mae",
    "normalized_r2": "normalized_r2",
    "original_rmse": "original_rmse",
    "original_mae": "original_mae",
    "original_r2": "original_r2",
    "negative_vgrf_ratio": "negative_vgrf_ratio",
    "negative_vgrf_magnitude": "negative_vgrf_mean_magnitude",
    "swing_false_force_ratio": "swing_phase_false_force_ratio",
    "swing_false_force_magnitude": "swing_phase_false_force_mean",
    "stance_false_zero_ratio": "stance_phase_false_zero_ratio",
    "smoothness_index": "smoothness_index",
    "peak_error": "peak_error",
}
HIGHER_IS_BETTER = {"normalized_r2", "original_r2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["metric"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_csv(ROOT / "tables" / "subject_rotation_all_runs.csv")
    paired_rows: list[dict[str, Any]] = []
    for subject in sorted({row["test_subject"] for row in rows}):
        tcn = next(row for row in rows if row["test_subject"] == subject and row["model"] == "tcn")
        pi = next(row for row in rows if row["test_subject"] == subject and row["model"] == "pi_tcn_full")
        for out_metric, source_metric in METRIC_MAP.items():
            tcn_value = num(tcn[source_metric])
            pi_value = num(pi[source_metric])
            diff = pi_value - tcn_value
            better = diff > 0 if out_metric in HIGHER_IS_BETTER else diff < 0
            pct = (diff / abs(tcn_value) * 100.0) if tcn_value != 0 else float("nan")
            paired_rows.append(
                {
                    "test_subject": subject,
                    "metric": out_metric,
                    "source_metric": source_metric,
                    "tcn": tcn_value,
                    "pi_tcn_full": pi_value,
                    "pi_minus_tcn": diff,
                    "pi_better": better,
                    "percent_change_pi_vs_tcn": pct,
                    "direction": "higher_is_better" if out_metric in HIGHER_IS_BETTER else "lower_is_better",
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for out_metric in METRIC_MAP:
        diffs = [row["pi_minus_tcn"] for row in paired_rows if row["metric"] == out_metric]
        pcts = [row["percent_change_pi_vs_tcn"] for row in paired_rows if row["metric"] == out_metric and math.isfinite(row["percent_change_pi_vs_tcn"])]
        wins = sum(bool(row["pi_better"]) for row in paired_rows if row["metric"] == out_metric)
        summary_rows.append(
            {
                "metric": out_metric,
                "direction": "higher_is_better" if out_metric in HIGHER_IS_BETTER else "lower_is_better",
                "n_folds": len(diffs),
                "pi_better_folds": wins,
                "mean_paired_difference": mean(diffs),
                "median_paired_difference": median(diffs),
                "sd_paired_difference": stdev(diffs) if len(diffs) > 1 else 0.0,
                "mean_percent_change_pi_vs_tcn": mean(pcts) if pcts else float("nan"),
            }
        )

    print(f"Paired comparisons: {len(paired_rows)} rows; summary metrics: {len(summary_rows)}")
    if args.dry_run:
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "paired_subject_rotation_differences.csv", paired_rows)
    write_csv(OUT / "paired_subject_rotation_summary.csv", summary_rows)
    lines = [
        "# Paired Subject-Rotation Summary",
        "",
        "| Metric | Direction | PI Better Folds | Mean Difference | Median Difference | SD Difference | Mean % Change |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['metric']} | {row['direction']} | {row['pi_better_folds']}/{row['n_folds']} | "
            f"{row['mean_paired_difference']:.6g} | {row['median_paired_difference']:.6g} | "
            f"{row['sd_paired_difference']:.6g} | {row['mean_percent_change_pi_vs_tcn']:.3f}% |"
        )
    (OUT / "paired_subject_rotation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
