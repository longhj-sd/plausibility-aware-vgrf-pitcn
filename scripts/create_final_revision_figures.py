"""Create final revision audit figures from completed summary tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path("outputs") / "revision_evidence"
OUT = ROOT / "final_audit"
FIG = OUT / "figures"
DATA = FIG / "data"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def grouped_subject_plot(rows: list[dict[str, str]], metric: str, ylabel: str, output_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_rows = [
        {"test_subject": row["test_subject"], "model": row["model"], metric: num(row[metric])}
        for row in rows
        if row["model"] in {"tcn", "pi_tcn_full"}
    ]
    write_csv(DATA / output_name.replace(".png", ".csv"), plot_rows)
    subjects = sorted({row["test_subject"] for row in plot_rows})
    models = ["tcn", "pi_tcn_full"]
    x = np.arange(len(subjects))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, model in zip([-width / 2, width / 2], models):
        values = [
            next((row[metric] for row in plot_rows if row["test_subject"] == subject and row["model"] == model), float("nan"))
            for subject in subjects
        ]
        ax.bar(x + offset, values, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / output_name, dpi=300)
    plt.close(fig)


def multiseed_plot(rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = [
        {
            "model": row["model"],
            "original_rmse_mean": num(row["original_rmse_mean"]),
            "original_rmse_sd": num(row["original_rmse_sd"]),
        }
        for row in rows
    ]
    write_csv(DATA / "multiseed_original_rmse_mean_sd.csv", plot_rows)
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.bar(
        [row["model"] for row in plot_rows],
        [row["original_rmse_mean"] for row in plot_rows],
        yerr=[row["original_rmse_sd"] for row in plot_rows],
        capsize=5,
    )
    ax.set_ylabel("Original-scale RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "multiseed_original_rmse_mean_sd.png", dpi=300)
    plt.close(fig)


def tradeoff_plot(rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = [
        {
            "test_subject": row["test_subject"],
            "model": row["model"],
            "original_rmse": num(row["original_rmse"]),
            "swing_phase_false_force_ratio": num(row["swing_phase_false_force_ratio"]),
        }
        for row in rows
        if row["model"] in {"tcn", "pi_tcn_full"}
    ]
    write_csv(DATA / "accuracy_plausibility_tradeoff_subject_rotation.csv", plot_rows)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for model in ["tcn", "pi_tcn_full"]:
        xs = [row["original_rmse"] for row in plot_rows if row["model"] == model]
        ys = [row["swing_phase_false_force_ratio"] for row in plot_rows if row["model"] == model]
        ax.scatter(xs, ys, label=model, s=55)
    ax.set_xlabel("Original-scale RMSE")
    ax.set_ylabel("Swing-phase false-force ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_plausibility_tradeoff_subject_rotation.png", dpi=300)
    plt.close(fig)


def write_final_report() -> None:
    multi = read_csv(ROOT / "tables" / "multiseed_all_runs.csv")
    subject = read_csv(ROOT / "tables" / "subject_rotation_all_runs.csv")
    consistency = read_csv(OUT / "summary_consistency_audit.csv")
    paired = read_csv(OUT / "paired_subject_rotation_summary.csv")
    inventory = read_csv(OUT / "artifact_inventory.csv")
    prediction_count = sum(row["likely_artifact_type"] == "prediction_array" for row in inventory)
    checkpoint_count = sum(row["likely_artifact_type"] == "checkpoint" for row in inventory)
    split_count = sum(row["likely_artifact_type"] == "split_dataset" for row in inventory)
    failures = [row for row in consistency if row.get("status") != "pass"]
    smoke = [row for row in failures if "suspicious" in row.get("check", "")]
    figure_files = sorted(path.name for path in FIG.glob("*.png"))
    original_rmse = next((row for row in paired if row["metric"] == "original_rmse"), None)
    swing = next((row for row in paired if row["metric"] == "swing_false_force_ratio"), None)
    lines = [
        "# Final No-Retraining Audit Report",
        "",
        f"- Completed multi-seed runs: {len(multi)}",
        f"- Completed subject-rotation runs: {len(subject)}",
        f"- Summary consistency failures: {len(failures)}",
        f"- Smoke-test contamination flags: {len(smoke)}",
        f"- Prediction arrays exist: {'yes' if prediction_count else 'no'}",
        f"- Checkpoints available: {'yes' if checkpoint_count else 'no'} ({checkpoint_count})",
        f"- Split datasets available: {'yes' if split_count else 'no'} ({split_count})",
        f"- Inference-only export needed for downstream waveform metrics: {'yes' if not prediction_count and checkpoint_count and split_count else 'no'}",
        "",
        "## Downstream Metric Feasibility",
        "",
        "See `downstream_metrics_feasibility.md`. No retraining is required if inference-only export is approved.",
        "",
        "## Paired Fold-Level Summary",
        "",
    ]
    if original_rmse:
        lines.append(f"- Original RMSE: PI-TCN better in {original_rmse['pi_better_folds']}/{original_rmse['n_folds']} folds; mean paired difference {float(original_rmse['mean_paired_difference']):.6g}.")
    if swing:
        lines.append(f"- Swing false-force ratio: PI-TCN better in {swing['pi_better_folds']}/{swing['n_folds']} folds; mean paired difference {float(swing['mean_paired_difference']):.6g}.")
    lines.extend(
        [
            "",
            "## Figure List",
            "",
            *[f"- `figures/{name}`" for name in figure_files],
            "",
            "## Recommended Manuscript Result Statements",
            "",
            "- The final revision evidence includes 6 submitted-split multi-seed runs and 14 subject-rotation runs.",
            "- Subject-rotation results should be reported as paired fold-level comparisons between TCN and PI-TCN-full.",
            "- Downstream waveform metrics should be described only after inference-only prediction export and metric computation are approved.",
            "",
            "## Recommended Supplementary Files",
            "",
            "- `artifact_inventory.csv` and `summary_consistency_audit.csv` for reproducibility auditing.",
            "- `paired_subject_rotation_differences.csv` and `paired_subject_rotation_summary.csv` for fold-level robustness.",
            "- `figures/data/*.csv` as source data for final figures.",
        ]
    )
    (OUT / "final_no_retraining_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subject = read_csv(ROOT / "tables" / "subject_rotation_all_runs.csv")
    multiseed_summary = read_csv(ROOT / "tables" / "multiseed_summary_mean_sd.csv")
    print("Creating final audit figures from completed tables")
    if args.dry_run:
        return 0
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    grouped_subject_plot(subject, "original_rmse", "Original-scale RMSE", "subject_rotation_original_rmse_by_subject.png")
    grouped_subject_plot(subject, "original_r2", "Original-scale R2", "subject_rotation_original_r2_by_subject.png")
    grouped_subject_plot(subject, "swing_phase_false_force_ratio", "Swing-phase false-force ratio", "subject_rotation_swing_false_force_by_subject.png")
    grouped_subject_plot(subject, "negative_vgrf_ratio", "Negative vGRF ratio", "subject_rotation_negative_vgrf_by_subject.png")
    tradeoff_plot(subject)
    multiseed_plot(multiseed_summary)
    write_final_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
