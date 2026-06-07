"""Compare TCN and PI-TCN accuracy and physical plausibility on GroundLink test data."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_tcn_baseline import TCNModel, count_parameters  # noqa: E402


DEFAULT_DATASET_PATH = Path("data") / "processed" / "groundlink_vgrf_windows_uncompressed.npz"
DEFAULT_TCN_MODEL_PATH = Path("outputs") / "models" / "tcn_baseline_best.pt"
DEFAULT_PI_MODEL_PATH = Path("outputs") / "models" / "pi_tcn_best.pt"
DEFAULT_COMPARISON_CSV = Path("outputs") / "tables" / "physical_plausibility_comparison.csv"
DEFAULT_IMPROVEMENT_CSV = Path("outputs") / "tables" / "pi_tcn_relative_improvement.csv"
DEFAULT_ERROR_FIGURE = Path("outputs") / "figures" / "model_error_metrics_comparison.png"
DEFAULT_R2_FIGURE = Path("outputs") / "figures" / "model_r2_metrics_comparison.png"
DEFAULT_ACCURACY_RELATIVE_FIGURE = Path("outputs") / "figures" / "model_accuracy_relative_improvement.png"
DEFAULT_PHYSICAL_RELATIVE_FIGURE = Path("outputs") / "figures" / "physical_plausibility_relative_improvement.png"
DEFAULT_OVERLAY_FIGURE = Path("outputs") / "figures" / "tcn_vs_pi_tcn_prediction_overlay.png"
MODEL_NAMES = ("TCN", "PI-TCN")
ACCURACY_LABELS = {
    "normalized_rmse": "Normalized RMSE",
    "normalized_mae": "Normalized MAE",
    "original_rmse": "Original-scale RMSE",
    "original_mae": "Original-scale MAE",
    "normalized_r2": "Normalized R\u00b2",
    "original_r2": "Original-scale R\u00b2",
}
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        required = ("X_test", "y_test", "y_mean", "y_std")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Dataset is missing required arrays: {missing}")
        return {
            "X_test": torch.as_tensor(data["X_test"].astype(np.float32)),
            "y_test": torch.as_tensor(data["y_test"].astype(np.float32)),
            "y_mean": torch.as_tensor(data["y_mean"].astype(np.float32)),
            "y_std": torch.as_tensor(data["y_std"].astype(np.float32)),
        }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    # These checkpoints were generated locally by this project, so weights_only=False is acceptable here.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys for {path}: {list(checkpoint.keys())}")
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(f"Could not find a model state_dict in checkpoint: {path}")


def build_model(checkpoint_path: Path, device: torch.device) -> TCNModel:
    model = TCNModel(
        input_channels=168,
        output_channels=2,
        hidden_channels=128,
        num_blocks=5,
        dropout=0.1,
    ).to(device)
    print(f"Model parameter count for {checkpoint_path.name}: {count_parameters(model):,}")
    state_dict = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        predictions.append(model(X_batch).cpu())
    return torch.cat(predictions, dim=0)


def compute_accuracy_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    error = pred_flat - target_flat
    mse = torch.mean(error**2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(error))
    ss_res = torch.sum(error**2)
    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    r2 = 1.0 - ss_res / torch.clamp(ss_tot, min=1e-12)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def compute_physical_metrics(
    pred_original: torch.Tensor,
    target_original: torch.Tensor,
    contact_threshold: float,
    spike_threshold: float,
) -> dict[str, float]:
    negative = pred_original < 0
    swing_mask = target_original < contact_threshold
    stance_mask = target_original >= contact_threshold
    diff_abs = torch.abs(pred_original[:, 1:, :] - pred_original[:, :-1, :])
    spike_counts = (diff_abs > spike_threshold).sum(dim=(1, 2)).float()
    peak_error = torch.mean(torch.abs(pred_original.max(dim=1).values - target_original.max(dim=1).values))

    if swing_mask.any():
        swing_false_force_ratio = torch.mean((pred_original[swing_mask] > contact_threshold).float())
        swing_false_force_mean = torch.mean(torch.abs(pred_original[swing_mask]))
    else:
        swing_false_force_ratio = pred_original.new_tensor(0.0)
        swing_false_force_mean = pred_original.new_tensor(0.0)

    if stance_mask.any():
        stance_false_zero_ratio = torch.mean((pred_original[stance_mask] < contact_threshold).float())
    else:
        stance_false_zero_ratio = pred_original.new_tensor(0.0)

    return {
        "negative_vgrf_ratio": float(torch.mean(negative.float())),
        "negative_vgrf_mean_magnitude": float(torch.mean(torch.relu(-pred_original))),
        "swing_phase_false_force_ratio": float(swing_false_force_ratio),
        "swing_phase_false_force_mean": float(swing_false_force_mean),
        "stance_phase_false_zero_ratio": float(stance_false_zero_ratio),
        "smoothness_index": float(torch.mean(diff_abs**2)),
        "spike_count_per_window": float(torch.mean(spike_counts)),
        "peak_error": float(peak_error),
    }


def compute_relative_improvement(tcn: dict[str, float], pi_tcn: dict[str, float]) -> list[dict[str, Any]]:
    r2_metrics = {"normalized_r2", "original_r2"}
    rows: list[dict[str, Any]] = []
    for metric, tcn_value in tcn.items():
        pi_value = pi_tcn[metric]
        if metric in r2_metrics:
            improvement = pi_value - tcn_value
            improvement_type = "absolute"
        else:
            improvement = ((tcn_value - pi_value) / tcn_value * 100.0) if tcn_value != 0 else float("nan")
            improvement_type = "percent"
        rows.append(
            {
                "metric": metric,
                "tcn": tcn_value,
                "pi_tcn": pi_value,
                "improvement_type": improvement_type,
                "improvement": improvement,
            }
        )
    return rows


def save_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model"] + [key for key in rows[0] if key != "model"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_improvement_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "tcn", "pi_tcn", "improvement_type", "improvement"])
        writer.writeheader()
        writer.writerows(rows)


def add_vertical_bar_labels(ax: Any, bars: Any, decimals: int = 3) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )


def add_horizontal_bar_labels(ax: Any, bars: Any, decimals: int = 1, suffix: str = "%") -> None:
    widths = [bar.get_width() for bar in bars]
    x_min = min(0.0, min(widths))
    x_max = max(0.0, max(widths))
    pad = max((x_max - x_min) * 0.18, 1.0)
    ax.set_xlim(x_min - pad, x_max + pad)
    x_min, x_max = ax.get_xlim()
    offset = (x_max - x_min) * 0.01
    for bar in bars:
        value = bar.get_width()
        x = value + offset if value >= 0 else value - offset
        ha = "left" if value >= 0 else "right"
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.{decimals}f}{suffix}",
            va="center",
            ha=ha,
            fontsize=9,
        )


def plot_error_metrics_comparison(results: dict[str, dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["normalized_rmse", "normalized_mae", "original_rmse", "original_mae"]
    x = np.arange(len(metrics))
    width = 0.38
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    tcn_bars = ax.bar(x - width / 2, [results["TCN"][metric] for metric in metrics], width, label="TCN")
    pi_bars = ax.bar(x + width / 2, [results["PI-TCN"][metric] for metric in metrics], width, label="PI-TCN")
    ax.set_xticks(x)
    ax.set_xticklabels([ACCURACY_LABELS[metric] for metric in metrics], fontsize=10)
    ax.set_title("Prediction error comparison on the test set", fontsize=14)
    ax.set_ylabel("Error value (lower is better)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    add_vertical_bar_labels(ax, tcn_bars, decimals=3)
    add_vertical_bar_labels(ax, pi_bars, decimals=3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_r2_metrics_comparison(results: dict[str, dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["normalized_r2", "original_r2"]
    x = np.arange(len(metrics))
    width = 0.38
    values = [results[model][metric] for model in MODEL_NAMES for metric in metrics]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    tcn_bars = ax.bar(x - width / 2, [results["TCN"][metric] for metric in metrics], width, label="TCN")
    pi_bars = ax.bar(x + width / 2, [results["PI-TCN"][metric] for metric in metrics], width, label="PI-TCN")
    ax.set_xticks(x)
    ax.set_xticklabels([ACCURACY_LABELS[metric] for metric in metrics], fontsize=10)
    ax.set_title("R\u00b2 comparison on the test set", fontsize=14)
    ax.set_ylabel("R\u00b2 value (higher is better)", fontsize=11)
    if min(values) >= 0 and max(values) <= 1:
        ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    add_vertical_bar_labels(ax, tcn_bars, decimals=3)
    add_vertical_bar_labels(ax, pi_bars, decimals=3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def read_improvement_table(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["metric"]: {
                "tcn": float(row["tcn"]),
                "pi_tcn": float(row["pi_tcn"]),
                "improvement": float(row["improvement"]),
            }
            for row in csv.DictReader(handle)
        }


def plot_physical_relative_improvement(improvement_csv: Path, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    improvements = read_improvement_table(improvement_csv)
    metrics = [
        "negative_vgrf_ratio",
        "negative_vgrf_mean_magnitude",
        "swing_phase_false_force_ratio",
        "swing_phase_false_force_mean",
        "stance_phase_false_zero_ratio",
        "smoothness_index",
        "spike_count_per_window",
        "peak_error",
    ]
    plotted_metrics: list[str] = []
    values: list[float] = []
    for metric in metrics:
        row = improvements[metric]
        value = row["improvement"]
        if not np.isfinite(value):
            continue
        if metric == "spike_count_per_window" and row["tcn"] == 0 and row["pi_tcn"] == 0:
            continue
        plotted_metrics.append(metric)
        values.append(value)
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in values]
    y = np.arange(len(plotted_metrics))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.barh(y, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([PHYSICAL_LABELS[metric] for metric in plotted_metrics], fontsize=10)
    ax.invert_yaxis()
    ax.set_title("Relative change in physical plausibility metrics: PI-TCN vs TCN", fontsize=14)
    ax.set_xlabel(
        "Relative improvement of PI-TCN over TCN (%)\n"
        "Positive values indicate improvement; negative values indicate worsening.",
        fontsize=11,
    )
    ax.grid(True, axis="x", alpha=0.3)
    add_horizontal_bar_labels(ax, bars, decimals=1)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_accuracy_relative_improvement(results: dict[str, dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["normalized_rmse", "normalized_mae", "original_rmse", "original_mae", "normalized_r2", "original_r2"]
    values: list[float] = []
    for metric in metrics:
        tcn_value = results["TCN"][metric]
        pi_value = results["PI-TCN"][metric]
        if metric.endswith("_r2"):
            value = ((pi_value - tcn_value) / abs(tcn_value) * 100.0) if tcn_value != 0 else float("nan")
        else:
            value = ((tcn_value - pi_value) / tcn_value * 100.0) if tcn_value != 0 else float("nan")
        values.append(value)

    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in values]
    y = np.arange(len(metrics))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(y, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([ACCURACY_LABELS[metric] for metric in metrics], fontsize=10)
    ax.invert_yaxis()
    ax.set_title("Relative change in prediction accuracy: PI-TCN vs TCN", fontsize=14)
    ax.set_xlabel("Relative change where positive values mean PI-TCN is better (%)", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    add_horizontal_bar_labels(ax, bars, decimals=1)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def select_representative_window(
    target_original: torch.Tensor,
    tcn_original: torch.Tensor,
    pi_original: torch.Tensor,
) -> tuple[int, float, float, float]:
    tcn_rmse = torch.sqrt(torch.mean((tcn_original - target_original) ** 2, dim=(1, 2)))
    pi_rmse = torch.sqrt(torch.mean((pi_original - target_original) ** 2, dim=(1, 2)))
    improvement = tcn_rmse - pi_rmse
    positive_indices = torch.nonzero(improvement > 0, as_tuple=False).reshape(-1)

    if positive_indices.numel() > 0:
        positive_improvement = improvement[positive_indices].cpu().numpy()
        target_improvement = np.percentile(positive_improvement, 75)
        selected_position = int(np.argmin(np.abs(positive_improvement - target_improvement)))
        selected_index = int(positive_indices[selected_position])
    else:
        selected_index = int(torch.argmin(pi_rmse))

    return (
        selected_index,
        float(tcn_rmse[selected_index]),
        float(pi_rmse[selected_index]),
        float(improvement[selected_index]),
    )


def plot_prediction_overlay(
    target_original: torch.Tensor,
    tcn_original: torch.Tensor,
    pi_original: torch.Tensor,
    path: Path,
    window_index: int,
    tcn_rmse: float,
    pi_rmse: float,
    improvement: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    labels = ("Left vertical GRF", "Right vertical GRF")
    fig.suptitle("Representative test-window prediction: TCN vs PI-TCN", fontsize=14)
    for side, ax in enumerate(axes):
        ax.plot(target_original[window_index, :, side].numpy(), label="Target", linewidth=2.2, color="black")
        ax.plot(tcn_original[window_index, :, side].numpy(), label="TCN", linewidth=1.6)
        ax.plot(pi_original[window_index, :, side].numpy(), label="PI-TCN", linewidth=1.6)
        ax.set_ylabel("Vertical GRF")
        ax.set_title(labels[side], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[0].text(
        0.02,
        0.92,
        "Selected from test windows where PI-TCN improves over TCN;\naggregate metrics are reported separately.",
        transform=axes[0].transAxes,
        fontsize=10,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    axes[1].text(
        0.02,
        0.92,
        (
            f"Window index: {window_index}\n"
            f"TCN RMSE: {tcn_rmse:.3f}\n"
            f"PI-TCN RMSE: {pi_rmse:.3f}\n"
            f"Improvement: {improvement:.3f}"
        ),
        transform=axes[1].transAxes,
        fontsize=10,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )
    axes[-1].set_xlabel("Frame in window")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--spike-threshold", type=float, default=0.5)
    parser.add_argument("--tcn-model-path", type=Path, default=DEFAULT_TCN_MODEL_PATH)
    parser.add_argument("--pi-model-path", type=Path, default=DEFAULT_PI_MODEL_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_dataset(args.dataset_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_mean = data["y_mean"].reshape(1, 1, -1)
    y_std = data["y_std"].reshape(1, 1, -1)
    print(f"X_test shape: {tuple(X_test.shape)}")
    print(f"y_test shape: {tuple(y_test.shape)}")

    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    models = {
        "TCN": build_model(args.tcn_model_path, device),
        "PI-TCN": build_model(args.pi_model_path, device),
    }

    target_original = y_test * y_std + y_mean
    predictions_normalized: dict[str, torch.Tensor] = {}
    predictions_original: dict[str, torch.Tensor] = {}
    results: dict[str, dict[str, float]] = {}

    for model_name, model in models.items():
        pred = predict_model(model, loader, device)
        pred_original = pred * y_std + y_mean
        predictions_normalized[model_name] = pred
        predictions_original[model_name] = pred_original

        normalized = compute_accuracy_metrics(pred, y_test)
        original = compute_accuracy_metrics(pred_original, target_original)
        physical = compute_physical_metrics(
            pred_original=pred_original,
            target_original=target_original,
            contact_threshold=args.contact_threshold,
            spike_threshold=args.spike_threshold,
        )
        results[model_name] = {
            "normalized_rmse": normalized["rmse"],
            "normalized_mae": normalized["mae"],
            "normalized_r2": normalized["r2"],
            "original_rmse": original["rmse"],
            "original_mae": original["mae"],
            "original_r2": original["r2"],
            **physical,
        }

    comparison_rows = [{"model": model_name, **results[model_name]} for model_name in MODEL_NAMES]
    save_comparison_csv(comparison_rows, DEFAULT_COMPARISON_CSV)
    improvement_rows = compute_relative_improvement(results["TCN"], results["PI-TCN"])
    save_improvement_csv(improvement_rows, DEFAULT_IMPROVEMENT_CSV)
    plot_error_metrics_comparison(results, DEFAULT_ERROR_FIGURE)
    plot_r2_metrics_comparison(results, DEFAULT_R2_FIGURE)
    plot_accuracy_relative_improvement(results, DEFAULT_ACCURACY_RELATIVE_FIGURE)
    plot_physical_relative_improvement(DEFAULT_IMPROVEMENT_CSV, DEFAULT_PHYSICAL_RELATIVE_FIGURE)
    overlay_index, overlay_tcn_rmse, overlay_pi_rmse, overlay_improvement = select_representative_window(
        target_original=target_original,
        tcn_original=predictions_original["TCN"],
        pi_original=predictions_original["PI-TCN"],
    )
    plot_prediction_overlay(
        target_original=target_original,
        tcn_original=predictions_original["TCN"],
        pi_original=predictions_original["PI-TCN"],
        path=DEFAULT_OVERLAY_FIGURE,
        window_index=overlay_index,
        tcn_rmse=overlay_tcn_rmse,
        pi_rmse=overlay_pi_rmse,
        improvement=overlay_improvement,
    )

    for model_name in MODEL_NAMES:
        metrics = results[model_name]
        label = "TCN baseline" if model_name == "TCN" else "PI-TCN"
        print(f"\n{label}:")
        print(f"  normalized RMSE: {metrics['normalized_rmse']:.6f}")
        print(f"  original RMSE: {metrics['original_rmse']:.6f}")
        print(f"  original MAE: {metrics['original_mae']:.6f}")
        print(f"  original R2: {metrics['original_r2']:.6f}")
        print(f"  negative_vgrf_ratio: {metrics['negative_vgrf_ratio']:.6f}")
        print(f"  swing_phase_false_force_ratio: {metrics['swing_phase_false_force_ratio']:.6f}")
        print(f"  smoothness_index: {metrics['smoothness_index']:.6f}")

    accuracy_improved = results["PI-TCN"]["original_rmse"] < results["TCN"]["original_rmse"]
    physical_metrics = [
        "negative_vgrf_ratio",
        "negative_vgrf_mean_magnitude",
        "swing_phase_false_force_ratio",
        "swing_phase_false_force_mean",
        "stance_phase_false_zero_ratio",
        "smoothness_index",
        "spike_count_per_window",
        "peak_error",
    ]
    physical_wins = sum(results["PI-TCN"][metric] < results["TCN"][metric] for metric in physical_metrics)
    physical_improved = physical_wins > len(physical_metrics) / 2

    print("\nConclusion:")
    print(f"  PI-TCN improves accuracy: {'yes' if accuracy_improved else 'no'}")
    print(f"  PI-TCN improves physical plausibility: {'yes' if physical_improved else 'no'} ({physical_wins}/{len(physical_metrics)} metrics)")
    if accuracy_improved and physical_improved:
        manuscript = "supports"
    elif physical_improved:
        manuscript = "partially supports"
    else:
        manuscript = "does not support"
    print(f"  Result {manuscript} the manuscript claim based on this test-set comparison.")
    print("\nSelected overlay window:")
    print(f"  selected overlay window index: {overlay_index}")
    print(f"  TCN window RMSE: {overlay_tcn_rmse:.6f}")
    print(f"  PI-TCN window RMSE: {overlay_pi_rmse:.6f}")
    print(f"  RMSE improvement: {overlay_improvement:.6f}")
    print(f"\nSaved comparison table: {DEFAULT_COMPARISON_CSV}")
    print(f"Saved relative improvement table: {DEFAULT_IMPROVEMENT_CSV}")
    print("\nGenerated figures:")
    print(f"  {DEFAULT_ERROR_FIGURE}")
    print(f"  {DEFAULT_R2_FIGURE}")
    print(f"  {DEFAULT_ACCURACY_RELATIVE_FIGURE}")
    print(f"  {DEFAULT_PHYSICAL_RELATIVE_FIGURE}")
    print(f"  {DEFAULT_OVERLAY_FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
