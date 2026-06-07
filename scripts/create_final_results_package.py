"""Create manuscript-ready result tables and workflow figure."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


COMPARISON_CSV = Path("outputs") / "tables" / "physical_plausibility_comparison.csv"
IMPROVEMENT_CSV = Path("outputs") / "tables" / "pi_tcn_relative_improvement.csv"
FINAL_ACCURACY_CSV = Path("outputs") / "tables" / "final_accuracy_table.csv"
FINAL_ACCURACY_MD = Path("outputs") / "tables" / "final_accuracy_table.md"
FINAL_PHYSICAL_CSV = Path("outputs") / "tables" / "final_physical_plausibility_table.csv"
FINAL_PHYSICAL_MD = Path("outputs") / "tables" / "final_physical_plausibility_table.md"
FINAL_SUMMARY_TXT = Path("outputs") / "tables" / "final_results_summary.txt"
WORKFLOW_FIGURE = Path("outputs") / "figures" / "workflow_pipeline.png"

ACCURACY_COLUMNS = [
    ("model", "Model"),
    ("normalized_rmse", "Normalized RMSE"),
    ("normalized_mae", "Normalized MAE"),
    ("normalized_r2", "Normalized R\u00b2"),
    ("original_rmse", "Original-scale RMSE"),
    ("original_mae", "Original-scale MAE"),
    ("original_r2", "Original-scale R\u00b2"),
]

PHYSICAL_LABELS = {
    "negative_vgrf_ratio": "Negative vGRF ratio",
    "negative_vgrf_mean_magnitude": "Negative vGRF magnitude",
    "swing_phase_false_force_ratio": "Swing-phase false-force ratio",
    "swing_phase_false_force_mean": "Swing-phase false-force magnitude",
    "stance_phase_false_zero_ratio": "Stance-phase false-zero ratio",
    "smoothness_index": "Smoothness index",
    "spike_count_per_window": "Spike count per window",
    "peak_error": "Peak error",
}

PHYSICAL_METRICS = [
    "negative_vgrf_ratio",
    "negative_vgrf_mean_magnitude",
    "swing_phase_false_force_ratio",
    "swing_phase_false_force_mean",
    "stance_phase_false_zero_ratio",
    "smoothness_index",
    "spike_count_per_window",
    "peak_error",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: str | float, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def load_comparison_by_model(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["model"]: row for row in rows}


def create_final_accuracy_table(comparison_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_model = load_comparison_by_model(comparison_rows)
    final_rows: list[dict[str, str]] = []
    for model in ("TCN", "PI-TCN"):
        source = by_model[model]
        final_row: dict[str, str] = {}
        for source_key, label in ACCURACY_COLUMNS:
            if source_key == "model":
                final_row[label] = model
            else:
                final_row[label] = format_float(source[source_key], 3)
        final_rows.append(final_row)
    return final_rows


def load_improvements(path: Path) -> dict[str, float]:
    rows = read_csv_rows(path)
    return {row["metric"]: float(row["improvement"]) for row in rows}


def physical_value_decimals(metric: str) -> int:
    if metric in {"negative_vgrf_mean_magnitude", "smoothness_index"}:
        return 6
    return 4


def create_final_physical_table(
    comparison_rows: list[dict[str, str]],
    improvements: dict[str, float],
) -> list[dict[str, str]]:
    by_model = load_comparison_by_model(comparison_rows)
    tcn = by_model["TCN"]
    pi_tcn = by_model["PI-TCN"]
    final_rows: list[dict[str, str]] = []

    for metric in PHYSICAL_METRICS:
        tcn_value = float(tcn[metric])
        pi_value = float(pi_tcn[metric])
        if metric == "spike_count_per_window" and tcn_value == 0 and pi_value == 0:
            continue

        decimals = physical_value_decimals(metric)
        improvement = improvements.get(metric)
        improvement_text = "" if improvement is None else f"{improvement:+.1f}%"
        final_rows.append(
            {
                "Metric": PHYSICAL_LABELS[metric],
                "Direction": "Lower is better",
                "TCN": f"{tcn_value:.{decimals}f}",
                "PI-TCN": f"{pi_value:.{decimals}f}",
                "Relative change of PI-TCN vs TCN": improvement_text,
            }
        )

    return final_rows


def create_summary_text(path: Path) -> None:
    text = (
        "PI-TCN improved normalized RMSE, normalized MAE, normalized R\u00b2, "
        "original-scale RMSE, original-scale MAE, and original-scale R\u00b2 compared "
        "with the supervised TCN baseline.\n\n"
        "PI-TCN reduced negative vGRF ratio, negative vGRF magnitude, stance-phase "
        "false-zero ratio, and smoothness index, but swing-phase false-force ratio "
        "and swing-phase false-force magnitude worsened. Peak error changed only "
        "minimally.\n\n"
        "Overall, PI-TCN achieved modest but consistent improvements in cross-subject "
        "vGRF prediction accuracy and reduced several key biomechanical plausibility "
        "violations, although swing-phase false-force metrics indicated a trade-off.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_workflow_figure(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    boxes = [
        ("GroundLink public dataset", "SMPL/MoSh++ kinematics\nForce/GRF files"),
        ("Data matching and preprocessing", "Match NPZ and Force files\nExtract poses + trans\nExtract bilateral vGRF"),
        ("Window construction", "120-frame windows\nSubject-wise split\nTrain: s001-s005\nValidation: s006\nTest: s007"),
        ("TCN baseline", "Supervised MSE loss"),
        ("PI-TCN", "MSE loss\nNon-negative vGRF loss\nSwing/contact loss\nSmoothness loss"),
        ("Evaluation", "Accuracy metrics\nPhysical plausibility metrics\nRepresentative prediction overlay"),
    ]

    fig, ax = plt.subplots(figsize=(18, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_w = 0.145
    box_h = 0.56
    y = 0.24
    xs = [0.02, 0.19, 0.36, 0.53, 0.68, 0.835]
    colors = ["#e8f1fb", "#eef7ee", "#fff4df", "#f3edf9", "#fdeeee", "#eef2f5"]

    for idx, ((title, body), x) in enumerate(zip(boxes, xs)):
        patch = FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.2,
            edgecolor="#3a3a3a",
            facecolor=colors[idx],
        )
        ax.add_patch(patch)
        ax.text(x + box_w / 2, y + box_h - 0.09, title, ha="center", va="top", fontsize=10.5, weight="bold")
        ax.text(x + box_w / 2, y + box_h - 0.19, body, ha="center", va="top", fontsize=9.2, linespacing=1.35)

        if idx < len(xs) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.01, y + box_h / 2),
                xytext=(x + box_w + 0.01, y + box_h / 2),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#333333"},
            )

    ax.set_title("GroundLink vGRF estimation workflow", fontsize=15, weight="bold", pad=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    comparison_rows = read_csv_rows(COMPARISON_CSV)
    improvements = load_improvements(IMPROVEMENT_CSV)

    accuracy_rows = create_final_accuracy_table(comparison_rows)
    accuracy_fields = [label for _, label in ACCURACY_COLUMNS]
    write_csv_rows(FINAL_ACCURACY_CSV, accuracy_rows, accuracy_fields)
    write_markdown_table(FINAL_ACCURACY_MD, accuracy_rows, accuracy_fields)

    physical_rows = create_final_physical_table(comparison_rows, improvements)
    physical_fields = ["Metric", "Direction", "TCN", "PI-TCN", "Relative change of PI-TCN vs TCN"]
    write_csv_rows(FINAL_PHYSICAL_CSV, physical_rows, physical_fields)
    write_markdown_table(FINAL_PHYSICAL_MD, physical_rows, physical_fields)

    create_summary_text(FINAL_SUMMARY_TXT)
    create_workflow_figure(WORKFLOW_FIGURE)

    generated = [
        FINAL_ACCURACY_CSV,
        FINAL_ACCURACY_MD,
        FINAL_PHYSICAL_CSV,
        FINAL_PHYSICAL_MD,
        FINAL_SUMMARY_TXT,
        WORKFLOW_FIGURE,
    ]
    print("Generated final results package:")
    for path in generated:
        print(f"  [x] {path}")
    print("\nThese outputs are intended for manuscript Results and Methods visualization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
