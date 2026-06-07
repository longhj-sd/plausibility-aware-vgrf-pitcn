"""Create downstream metric figures from computed downstream summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path("outputs") / "revision_evidence" / "final_audit" / "downstream_metrics"
FIG = ROOT / "figures"
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


def grouped_subject_plot(rows: list[dict[str, str]], metric_mean: str, ylabel: str, filename: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_rows = [
        {
            "test_subject": row["test_subject"],
            "model": row["model"],
            metric_mean: num(row[metric_mean]),
        }
        for row in rows
        if row["model"] in {"tcn", "pi_tcn_full"} and row.get(metric_mean, "") != ""
    ]
    if not plot_rows:
        return
    write_csv(DATA / filename.replace(".png", ".csv"), plot_rows)
    subjects = sorted({row["test_subject"] for row in plot_rows})
    models = ["tcn", "pi_tcn_full"]
    x = np.arange(len(subjects))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, model in zip([-width / 2, width / 2], models):
        values = [
            next((row[metric_mean] for row in plot_rows if row["test_subject"] == subject and row["model"] == model), float("nan"))
            for subject in subjects
        ]
        ax.bar(x + offset, values, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=300)
    plt.close(fig)


def summary_plot(rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("contact_time_absolute_error_mean_mean", "Contact time abs. error"),
        ("peak_absolute_error_mean_mean", "Peak vGRF abs. error"),
        ("impulse_absolute_error_mean_mean", "Impulse abs. error"),
    ]
    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        for metric, label in metrics:
            if row.get(metric, "") != "":
                plot_rows.append({"model": row["model"], "metric": label, "value": num(row[metric])})
    if not plot_rows:
        return
    write_csv(DATA / "downstream_metric_summary_mean_sd.csv", plot_rows)
    labels = sorted({row["metric"] for row in plot_rows})
    models = sorted({row["model"] for row in plot_rows})
    x = np.arange(len(labels))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for idx, model in enumerate(models):
        values = [
            next((row["value"] for row in plot_rows if row["metric"] == label and row["model"] == model), float("nan"))
            for label in labels
        ]
        ax.bar(x + (idx - (len(models) - 1) / 2) * width, values, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean absolute error")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "downstream_metric_summary_mean_sd.png", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()
    by_run = read_csv(ROOT / "downstream_metrics_by_run.csv")
    by_model = read_csv(ROOT / "downstream_metrics_summary_by_model.csv")
    print(f"Downstream by-run rows: {len(by_run)}")
    if args.dry_run:
        return 0
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    grouped_subject_plot(by_run, "peak_absolute_error_mean", "Peak vGRF absolute error", "subject_rotation_peak_vgrf_error_by_subject.png")
    grouped_subject_plot(by_run, "contact_time_absolute_error_mean", "Contact time absolute error", "subject_rotation_contact_time_error_by_subject.png")
    grouped_subject_plot(by_run, "impulse_absolute_error_mean", "Impulse absolute error", "subject_rotation_impulse_error_by_subject.png")
    summary_plot(by_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
