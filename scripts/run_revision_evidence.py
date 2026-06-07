"""Run revision-evidence experiments for the GroundLink PI-TCN manuscript.

All generated artifacts are written under outputs/revision_evidence by default.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.subject_splits import (  # noqa: E402
    get_available_subjects,
    make_submitted_split,
    make_test_subject_rotation_splits,
)


MODEL_SPECS: dict[str, dict[str, float | str]] = {
    "tcn": {"lambda_nonneg": 0.0, "lambda_contact": 0.0, "lambda_smooth": 0.0, "label": "TCN"},
    "pi_tcn_full": {
        "lambda_nonneg": 0.1,
        "lambda_contact": 0.1,
        "lambda_smooth": 0.01,
        "label": "PI-TCN-full",
    },
    "pi_tcn_neg": {
        "lambda_nonneg": 0.1,
        "lambda_contact": 0.0,
        "lambda_smooth": 0.0,
        "label": "PI-TCN-neg",
    },
    "pi_tcn_neg_smooth": {
        "lambda_nonneg": 0.1,
        "lambda_contact": 0.0,
        "lambda_smooth": 0.01,
        "label": "PI-TCN-neg-smooth",
    },
    "pi_tcn_contact": {
        "lambda_nonneg": 0.0,
        "lambda_contact": 0.1,
        "lambda_smooth": 0.0,
        "label": "PI-TCN-contact",
    },
    "pi_tcn_contact0p2": {
        "lambda_nonneg": 0.1,
        "lambda_contact": 0.2,
        "lambda_smooth": 0.01,
        "label": "PI-TCN-contact0p2",
    },
}

ACCURACY_FIELDS = [
    "normalized_rmse",
    "normalized_mae",
    "normalized_r2",
    "original_rmse",
    "original_mae",
    "original_r2",
]
PHYSICAL_FIELDS = [
    "negative_vgrf_ratio",
    "negative_vgrf_mean_magnitude",
    "swing_phase_false_force_ratio",
    "swing_phase_false_force_mean",
    "stance_phase_false_zero_ratio",
    "smoothness_index",
    "peak_error",
]


@dataclass(frozen=True)
class RunSpec:
    analysis: str
    fold: dict[str, Any]
    model_name: str
    seed: int

    @property
    def fold_name(self) -> str:
        return str(self.fold["fold_name"])

    @property
    def test_subject(self) -> str:
        return ",".join(self.fold["test_subjects"])


class Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        choices=["subject_rotation", "multiseed_submitted_split", "loss_ablation", "all"],
        required=True,
    )
    parser.add_argument("--models", nargs="+", default=["tcn", "pi_tcn_full"], choices=sorted(MODEL_SPECS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs") / "revision_evidence")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--only-fold", type=str, default=None)
    parser.add_argument("--only-model", type=str, choices=sorted(MODEL_SPECS), default=None)
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--matched-csv", type=Path, default=Path("outputs") / "tables" / "matched_groundlink_files.csv")
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--spike-threshold", type=float, default=0.5)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(name)


def split_output_dir(output_root: Path, fold: dict[str, Any]) -> Path:
    return output_root / "split_datasets" / str(fold["fold_name"])


def run_output_dir(output_root: Path, spec: RunSpec) -> Path:
    return output_root / spec.analysis / spec.fold_name / spec.model_name / f"seed_{spec.seed}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_single_row_csv(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else None


def build_split_dataset(
    fold: dict[str, Any],
    dataset_path: Path,
    metadata_path: Path,
    args: argparse.Namespace,
) -> None:
    import numpy as np

    from scripts import build_dataset

    if dataset_path.exists() and metadata_path.exists():
        return

    split_subjects = {
        "train": set(fold["train_subjects"]),
        "val": set(fold["val_subjects"]),
        "test": set(fold["test_subjects"]),
    }
    split_X: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_y: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    all_metadata: list[dict[str, Any]] = []
    base_dir = Path.cwd()

    with args.matched_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject, trial_name, _ = build_dataset.parse_subject_and_motion(row)
            if subject in split_subjects["train"]:
                split = "train"
            elif subject in split_subjects["val"]:
                split = "val"
            elif subject in split_subjects["test"]:
                split = "test"
            else:
                continue

            trial = build_dataset.load_one_trial(row, base_dir)
            if trial is None:
                continue
            windowed = build_dataset.make_windows(trial, args.window_size, args.stride)
            if windowed is None:
                continue

            split_X[split].append(windowed.X)
            split_y[split].append(windowed.y)
            for item in windowed.metadata:
                item["split"] = split
            all_metadata.extend(windowed.metadata)

    X_train = build_dataset.stack_or_empty(split_X["train"], args.window_size, 168)
    y_train = build_dataset.stack_or_empty(split_y["train"], args.window_size, 2)
    X_val = build_dataset.stack_or_empty(split_X["val"], args.window_size, 168)
    y_val = build_dataset.stack_or_empty(split_y["val"], args.window_size, 2)
    X_test = build_dataset.stack_or_empty(split_X["test"], args.window_size, 168)
    y_test = build_dataset.stack_or_empty(split_y["test"], args.window_size, 2)

    x_mean, x_std = build_dataset.compute_mean_std(X_train, 168, "X")
    y_mean, y_std = build_dataset.compute_mean_std(y_train, 2, "y")
    X_train = build_dataset.normalize_windows(X_train, x_mean, x_std)
    X_val = build_dataset.normalize_windows(X_val, x_mean, x_std)
    X_test = build_dataset.normalize_windows(X_test, x_mean, x_std)
    y_train = build_dataset.normalize_windows(y_train, y_mean, y_std)
    y_val = build_dataset.normalize_windows(y_val, y_mean, y_std)
    y_test = build_dataset.normalize_windows(y_test, y_mean, y_std)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dataset_path,
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
    )
    build_dataset.write_metadata_csv(all_metadata, metadata_path)


def load_split_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    import numpy as np

    with np.load(dataset_path, allow_pickle=False) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def evaluate_model(
    model: Any,
    loader: Any,
    device: Any,
    y_mean: Any,
    y_std: Any,
    contact_threshold: float,
    spike_threshold: float,
) -> tuple[dict[str, float], dict[str, float]]:
    import torch

    from scripts.evaluate_physical_plausibility import compute_physical_metrics
    from src.train_tcn_baseline import compute_metrics

    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for X, y in loader:
            pred = model(X.to(device, non_blocking=True)).cpu()
            preds.append(pred)
            targets.append(y.cpu())

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    normalized = compute_metrics(pred_all, target_all)
    pred_original = pred_all * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
    target_original = target_all * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
    original = compute_metrics(pred_original, target_original)
    physical = compute_physical_metrics(
        pred_original=pred_original,
        target_original=target_original,
        contact_threshold=contact_threshold,
        spike_threshold=spike_threshold,
    )
    accuracy = {
        "normalized_rmse": normalized["rmse"],
        "normalized_mae": normalized["mae"],
        "normalized_r2": normalized["r2"],
        "original_rmse": original["rmse"],
        "original_mae": original["mae"],
        "original_r2": original["r2"],
    }
    physical = {key: value for key, value in physical.items() if key in PHYSICAL_FIELDS}
    return accuracy, physical


def train_one_run(spec: RunSpec, args: argparse.Namespace, device: torch.device) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from src.train_pi_tcn import physics_informed_loss
    from src.train_tcn_baseline import GroundLinkWindowDataset, TCNModel, count_parameters, set_seed

    out_dir = run_output_dir(args.output_root, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_combined = out_dir / "final_metrics_combined.csv"
    if args.skip_existing and final_combined.exists():
        print(f"SKIP existing completed run: {out_dir}")
        return

    dataset_dir = split_output_dir(args.output_root, spec.fold)
    dataset_path = dataset_dir / "dataset.npz"
    metadata_path = dataset_dir / "window_metadata.csv"
    build_split_dataset(spec.fold, dataset_path, metadata_path, args)

    config = {
        "analysis": spec.analysis,
        "fold_name": spec.fold_name,
        "train_subjects": spec.fold["train_subjects"],
        "val_subjects": spec.fold["val_subjects"],
        "test_subjects": spec.fold["test_subjects"],
        "model_name": spec.model_name,
        "model_spec": MODEL_SPECS[spec.model_name],
        "seed": spec.seed,
        "window_size": args.window_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "dataset_path": str(dataset_path),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    set_seed(spec.seed)
    data = load_split_dataset(dataset_path)
    train_dataset = GroundLinkWindowDataset(data["X_train"], data["y_train"])
    val_dataset = GroundLinkWindowDataset(data["X_val"], data["y_val"])
    test_dataset = GroundLinkWindowDataset(data["X_test"], data["y_test"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TCNModel(input_channels=168, output_channels=2, hidden_channels=128, num_blocks=5, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    y_mean = torch.as_tensor(data["y_mean"], dtype=torch.float32)
    y_std = torch.as_tensor(data["y_std"], dtype=torch.float32)
    y_mean_device = y_mean.to(device)
    y_std_device = y_std.to(device)
    model_spec = MODEL_SPECS[spec.model_name]

    print(f"Run: {spec.analysis}/{spec.fold_name}/{spec.model_name}/seed_{spec.seed}")
    print(f"Device: {device}; parameters: {count_parameters(model):,}")
    print(f"Train/val/test windows: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = out_dir / "checkpoint_best.pt"

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        totals = {"total_loss": 0.0, "mse_loss": 0.0, "nonneg_loss": 0.0, "contact_loss": 0.0, "smooth_loss": 0.0}
        total_samples = 0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(X)
            if spec.model_name == "tcn":
                mse_loss = criterion(pred, y)
                loss_parts = {
                    "total_loss": mse_loss,
                    "mse_loss": mse_loss,
                    "nonneg_loss": pred.new_tensor(0.0),
                    "contact_loss": pred.new_tensor(0.0),
                    "smooth_loss": pred.new_tensor(0.0),
                }
            else:
                loss_parts = physics_informed_loss(
                    pred=pred,
                    target=y,
                    y_mean=y_mean_device,
                    y_std=y_std_device,
                    lambda_nonneg=float(model_spec["lambda_nonneg"]),
                    lambda_contact=float(model_spec["lambda_contact"]),
                    lambda_smooth=float(model_spec["lambda_smooth"]),
                    contact_threshold=args.contact_threshold,
                )
            loss_parts["total_loss"].backward()
            optimizer.step()

            batch_size = int(X.shape[0])
            total_samples += batch_size
            for key in totals:
                totals[key] += float(loss_parts[key].detach().cpu()) * batch_size

        train_losses = {key: value / max(total_samples, 1) for key, value in totals.items()}
        val_accuracy, _ = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            contact_threshold=args.contact_threshold,
            spike_threshold=args.spike_threshold,
        )
        row = {
            "epoch": epoch,
            "train_total_loss": train_losses["total_loss"],
            "train_mse_loss": train_losses["mse_loss"],
            "train_nonneg_loss": train_losses["nonneg_loss"],
            "train_contact_loss": train_losses["contact_loss"],
            "train_smooth_loss": train_losses["smooth_loss"],
            "val_rmse": val_accuracy["normalized_rmse"],
            "val_mae": val_accuracy["normalized_mae"],
            "val_r2": val_accuracy["normalized_r2"],
            "val_orig_rmse": val_accuracy["original_rmse"],
            "val_orig_mae": val_accuracy["original_mae"],
            "val_orig_r2": val_accuracy["original_r2"],
        }
        history.append(row)

        if val_accuracy["normalized_rmse"] < best_val_rmse:
            best_val_rmse = val_accuracy["normalized_rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_rmse": best_val_rmse,
                    "run_config": config,
                },
                checkpoint_path,
            )
            status = "improved"
        else:
            epochs_without_improvement += 1
            status = f"no improvement {epochs_without_improvement}/{args.patience}"
        print(
            f"Epoch {epoch:03d} | train_total_loss={train_losses['total_loss']:.6f} | "
            f"val_rmse={val_accuracy['normalized_rmse']:.6f} | {status}"
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    write_csv(out_dir / "training_history.csv", history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    accuracy, physical = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        contact_threshold=args.contact_threshold,
        spike_threshold=args.spike_threshold,
    )
    accuracy_row = {
        "analysis": spec.analysis,
        "fold_name": spec.fold_name,
        "test_subject": spec.test_subject,
        "model": spec.model_name,
        "seed": spec.seed,
        **accuracy,
    }
    physical_row = {
        "analysis": spec.analysis,
        "fold_name": spec.fold_name,
        "test_subject": spec.test_subject,
        "model": spec.model_name,
        "seed": spec.seed,
        **physical,
    }
    combined_row = {**accuracy_row, **physical}
    write_csv(out_dir / "final_accuracy_metrics.csv", [accuracy_row])
    write_csv(out_dir / "final_physical_plausibility_metrics.csv", [physical_row])
    write_csv(out_dir / "final_metrics_combined.csv", [combined_row])
    print(f"Saved final metrics: {final_combined}")


def run_with_logs(spec: RunSpec, args: argparse.Namespace, device: torch.device) -> bool:
    out_dir = run_output_dir(args.output_root, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"
    err_path = out_dir / "error_log.txt"
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            tee = Tee(sys.stdout, log_handle)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                train_one_run(spec, args, device)
        if err_path.exists():
            err_path.unlink()
        return True
    except Exception:  # noqa: BLE001
        error_text = traceback.format_exc()
        err_path.write_text(error_text, encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write("\nERROR\n")
            log_handle.write(error_text)
        print(f"FAILED: {out_dir}\n{error_text}")
        return False


def analyses_to_run(name: str) -> list[str]:
    if name == "all":
        return ["subject_rotation", "multiseed_submitted_split", "loss_ablation"]
    return [name]


def default_seeds_for_analysis(analysis: str, args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        return args.seeds
    if analysis == "multiseed_submitted_split":
        return [7, 17, 42]
    return [7]


def make_run_specs(args: argparse.Namespace) -> list[RunSpec]:
    subjects = get_available_subjects()
    rotation_folds = make_test_subject_rotation_splits(subjects)
    submitted = make_submitted_split()
    specs: list[RunSpec] = []
    for analysis in analyses_to_run(args.analysis):
        if analysis == "subject_rotation":
            folds = rotation_folds
        else:
            folds = [submitted]
        for fold in folds:
            if args.only_fold and fold["fold_name"] != args.only_fold:
                continue
            for model_name in args.models:
                if args.only_model and model_name != args.only_model:
                    continue
                for seed in default_seeds_for_analysis(analysis, args):
                    if args.only_seed is not None and seed != args.only_seed:
                        continue
                    specs.append(RunSpec(analysis=analysis, fold=fold, model_name=model_name, seed=seed))
    if args.max_runs is not None:
        specs = specs[: args.max_runs]
    return specs


def completed_rows(output_root: Path, analysis: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = output_root / analysis
    if not base.exists():
        return rows
    for path in base.glob("*/*/seed_*/final_metrics_combined.csv"):
        row = read_single_row_csv(path)
        if row is not None:
            rows.append(row)
    return rows


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_mean_sd(rows: list[dict[str, Any]], group_fields: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        groups.setdefault(key, []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        out = {field: value for field, value in zip(group_fields, key)}
        out["n"] = len(group)
        for metric in metrics:
            values = [numeric(row.get(metric)) for row in group]
            values = [value for value in values if math.isfinite(value)]
            out[f"{metric}_mean"] = mean(values) if values else float("nan")
            out[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0 if values else float("nan")
        summary_rows.append(out)
    return summary_rows


def generate_tables(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    tables_dir = output_root / "tables"
    tables: dict[str, list[dict[str, Any]]] = {}
    subject_rows = completed_rows(output_root, "subject_rotation")
    multiseed_rows = completed_rows(output_root, "multiseed_submitted_split")
    ablation_rows = completed_rows(output_root, "loss_ablation")

    if subject_rows:
        write_csv(tables_dir / "subject_rotation_all_runs.csv", subject_rows)
        by_subject = summarize_mean_sd(subject_rows, ["test_subject", "model"], ACCURACY_FIELDS + PHYSICAL_FIELDS)
        summary = summarize_mean_sd(subject_rows, ["model"], ACCURACY_FIELDS + PHYSICAL_FIELDS)
        write_csv(tables_dir / "subject_rotation_by_test_subject.csv", by_subject)
        write_csv(tables_dir / "subject_rotation_summary_mean_sd.csv", summary)
        tables["subject_rotation"] = subject_rows
    if multiseed_rows:
        write_csv(tables_dir / "multiseed_all_runs.csv", multiseed_rows)
        summary = summarize_mean_sd(multiseed_rows, ["model"], ACCURACY_FIELDS + PHYSICAL_FIELDS)
        write_csv(tables_dir / "multiseed_summary_mean_sd.csv", summary)
        tables["multiseed"] = multiseed_rows
    if ablation_rows:
        write_csv(tables_dir / "loss_ablation_all_runs.csv", ablation_rows)
        summary = summarize_mean_sd(ablation_rows, ["model"], ACCURACY_FIELDS + PHYSICAL_FIELDS)
        write_csv(tables_dir / "loss_ablation_summary.csv", summary)
        tables["loss_ablation"] = ablation_rows
    return tables


def save_figure_data(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        write_csv(path, rows)


def metric_by_subject(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    wanted = [row for row in rows if row.get("model") in {"tcn", "pi_tcn_full"}]
    return sorted(
        [
            {
                "test_subject": row["test_subject"],
                "model": row["model"],
                metric: numeric(row[metric]),
            }
            for row in wanted
        ],
        key=lambda row: (row["test_subject"], row["model"]),
    )


def generate_figures(output_root: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = output_root / "figures"
    data_dir = figures_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    subject_rows = tables.get("subject_rotation", [])
    if subject_rows:
        for metric, filename, ylabel in [
            ("original_rmse", "subject_rotation_original_rmse_by_test_subject.png", "Original-scale RMSE"),
            ("original_r2", "subject_rotation_original_r2_by_test_subject.png", "Original-scale R2"),
            (
                "swing_phase_false_force_ratio",
                "subject_rotation_swing_false_force_by_test_subject.png",
                "Swing-phase false-force ratio",
            ),
        ]:
            plot_rows = metric_by_subject(subject_rows, metric)
            save_figure_data(data_dir / filename.replace(".png", ".csv"), plot_rows)
            subjects = sorted({row["test_subject"] for row in plot_rows})
            models = ["tcn", "pi_tcn_full"]
            x = np.arange(len(subjects))
            width = 0.38
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for offset, model in zip([-width / 2, width / 2], models):
                values = [
                    next((row[metric] for row in plot_rows if row["test_subject"] == subject and row["model"] == model), np.nan)
                    for subject in subjects
                ]
                ax.bar(x + offset, values, width, label=model)
            ax.set_xticks(x)
            ax.set_xticklabels(subjects)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(figures_dir / filename, dpi=300)
            plt.close(fig)

        trade_rows = [
            {
                "test_subject": row["test_subject"],
                "model": row["model"],
                "original_rmse": numeric(row["original_rmse"]),
                "swing_phase_false_force_ratio": numeric(row["swing_phase_false_force_ratio"]),
            }
            for row in subject_rows
            if row.get("model") in {"tcn", "pi_tcn_full"}
        ]
        save_figure_data(data_dir / "accuracy_plausibility_tradeoff_subject_rotation.csv", trade_rows)
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        for model in ["tcn", "pi_tcn_full"]:
            xs = [row["original_rmse"] for row in trade_rows if row["model"] == model]
            ys = [row["swing_phase_false_force_ratio"] for row in trade_rows if row["model"] == model]
            ax.scatter(xs, ys, label=model, s=55)
        ax.set_xlabel("Original-scale RMSE")
        ax.set_ylabel("Swing-phase false-force ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "accuracy_plausibility_tradeoff_subject_rotation.png", dpi=300)
        plt.close(fig)

    multiseed_rows = tables.get("multiseed", [])
    if multiseed_rows:
        summary = summarize_mean_sd(multiseed_rows, ["model"], ["original_rmse"])
        save_figure_data(data_dir / "multiseed_original_rmse_mean_sd.csv", summary)
        fig, ax = plt.subplots(figsize=(5.8, 4.5))
        models = [row["model"] for row in summary]
        means = [numeric(row["original_rmse_mean"]) for row in summary]
        sds = [numeric(row["original_rmse_sd"]) for row in summary]
        ax.bar(models, means, yerr=sds, capsize=5)
        ax.set_ylabel("Original-scale RMSE")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "multiseed_original_rmse_mean_sd.png", dpi=300)
        plt.close(fig)

    ablation_rows = tables.get("loss_ablation", [])
    if ablation_rows:
        metrics = [
            "original_rmse",
            "negative_vgrf_ratio",
            "swing_phase_false_force_ratio",
            "stance_phase_false_zero_ratio",
        ]
        summary = summarize_mean_sd(ablation_rows, ["model"], metrics)
        save_figure_data(data_dir / "loss_ablation_key_metrics.csv", summary)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        models = [row["model"] for row in summary]
        for ax, metric in zip(axes.reshape(-1), metrics):
            values = [numeric(row[f"{metric}_mean"]) for row in summary]
            ax.bar(models, values)
            ax.set_title(metric)
            ax.tick_params(axis="x", labelrotation=25, labelsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "loss_ablation_key_metrics.png", dpi=300)
        plt.close(fig)


def fmt(value: float, decimals: int = 3) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{decimals}f}"


def describe_model_mean(rows: list[dict[str, Any]], model: str, metric: str) -> str:
    values = [numeric(row.get(metric)) for row in rows if row.get("model") == model]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return "NA"
    sd = stdev(values) if len(values) > 1 else 0.0
    return f"{mean(values):.3f} +/- {sd:.3f}"


def generate_summary_md(output_root: Path, tables: dict[str, list[dict[str, Any]]], specs: list[RunSpec]) -> None:
    completed = sum(1 for spec in specs if (run_output_dir(output_root, spec) / "final_metrics_combined.csv").exists())
    failed = sum(1 for spec in specs if (run_output_dir(output_root, spec) / "error_log.txt").exists())
    subject_rows = tables.get("subject_rotation", [])
    multiseed_rows = tables.get("multiseed", [])
    ablation_rows = tables.get("loss_ablation", [])

    lines = [
        "# Revision Evidence Summary",
        "",
        "## Subject Split Protocol",
        "",
        "Submitted split: train s001-s005, validation s006, test s007. Subject rotation uses a 5/1/1 design: each subject is test once, the next sorted subject is validation, and all remaining subjects are training subjects. Normalization statistics are computed from training windows only for each split.",
        "",
        "## Run Status",
        "",
        f"- Planned runs in this invocation: {len(specs)}",
        f"- Completed runs with final metrics: {completed}",
        f"- Failed runs with error logs: {failed}",
        "",
        "## Mean +/- SD by Model",
        "",
        f"- Subject rotation original RMSE: TCN {describe_model_mean(subject_rows, 'tcn', 'original_rmse')}; PI-TCN-full {describe_model_mean(subject_rows, 'pi_tcn_full', 'original_rmse')}",
        f"- Multi-seed original RMSE: TCN {describe_model_mean(multiseed_rows, 'tcn', 'original_rmse')}; PI-TCN-full {describe_model_mean(multiseed_rows, 'pi_tcn_full', 'original_rmse')}",
        "",
        "## Per-Test-Subject Comparison",
        "",
    ]
    if subject_rows:
        for subject in sorted({row["test_subject"] for row in subject_rows}):
            tcn = next((row for row in subject_rows if row["test_subject"] == subject and row["model"] == "tcn"), None)
            pi = next((row for row in subject_rows if row["test_subject"] == subject and row["model"] == "pi_tcn_full"), None)
            if tcn and pi:
                delta = numeric(tcn["original_rmse"]) - numeric(pi["original_rmse"])
                lines.append(f"- {subject}: original RMSE TCN {numeric(tcn['original_rmse']):.3f}, PI-TCN-full {numeric(pi['original_rmse']):.3f}, delta {delta:+.3f}")
    else:
        lines.append("- Pending subject-rotation runs.")
    lines.extend(["", "## Loss Ablation", ""])
    if ablation_rows:
        for row in sorted(ablation_rows, key=lambda item: item["model"]):
            lines.append(
                f"- {row['model']}: original RMSE {numeric(row['original_rmse']):.3f}, "
                f"negative vGRF ratio {numeric(row['negative_vgrf_ratio']):.4f}, "
                f"swing false-force ratio {numeric(row['swing_phase_false_force_ratio']):.4f}, "
                f"stance false-zero ratio {numeric(row['stance_phase_false_zero_ratio']):.4f}"
            )
    else:
        lines.append("- Pending loss-ablation runs.")

    pi_rmse = [numeric(row["original_rmse"]) for row in subject_rows if row.get("model") == "pi_tcn_full"]
    tcn_rmse = [numeric(row["original_rmse"]) for row in subject_rows if row.get("model") == "tcn"]
    pi_neg = [numeric(row["negative_vgrf_ratio"]) for row in subject_rows if row.get("model") == "pi_tcn_full"]
    tcn_neg = [numeric(row["negative_vgrf_ratio"]) for row in subject_rows if row.get("model") == "tcn"]
    pi_stance = [numeric(row["stance_phase_false_zero_ratio"]) for row in subject_rows if row.get("model") == "pi_tcn_full"]
    tcn_stance = [numeric(row["stance_phase_false_zero_ratio"]) for row in subject_rows if row.get("model") == "tcn"]
    pi_swing = [numeric(row["swing_phase_false_force_ratio"]) for row in subject_rows if row.get("model") == "pi_tcn_full"]
    tcn_swing = [numeric(row["swing_phase_false_force_ratio"]) for row in subject_rows if row.get("model") == "tcn"]
    lines.extend(["", "## Concise Interpretation", ""])
    if pi_rmse and tcn_rmse:
        lines.append(f"1. PI-TCN-full {'improves' if mean(pi_rmse) < mean(tcn_rmse) else 'does not improve'} original-scale RMSE on average in completed subject-rotation runs.")
        lines.append(f"2. PI-TCN-full {'reduces' if mean(pi_neg) < mean(tcn_neg) and mean(pi_stance) < mean(tcn_stance) else 'does not consistently reduce'} negative vGRF and stance false-zero violations in completed subject-rotation runs.")
        lines.append(f"3. Swing-phase false-force {'remains a trade-off' if mean(pi_swing) > mean(tcn_swing) else 'does not worsen on average'} in completed subject-rotation runs.")
        if mean(pi_rmse) < mean(tcn_rmse):
            conclusion = "supports the submitted conclusion with more conservative wording"
        else:
            conclusion = "weakens or changes the submitted conclusion"
        lines.append(f"4. The revised evidence currently {conclusion}.")
    else:
        lines.extend(
            [
                "1. PI-TCN-full accuracy robustness is pending completed subject-rotation runs.",
                "2. Negative vGRF and stance false-zero robustness is pending completed subject-rotation runs.",
                "3. Swing-phase false-force trade-off is pending completed subject-rotation runs.",
                "4. The revised conclusion should be finalized after full analyses complete.",
            ]
        )
    (output_root / "revision_evidence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_snippets(output_root: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    snippets_dir = output_root / "manuscript_snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    subject_rows = tables.get("subject_rotation", [])
    multiseed_rows = tables.get("multiseed", [])
    ablation_rows = tables.get("loss_ablation", [])
    results_text = (
        "Revision robustness analyses evaluated subject-wise test rotation, repeated seeds on the submitted split, "
        "and physics-loss ablations. In completed subject-rotation runs, original-scale RMSE was "
        f"{describe_model_mean(subject_rows, 'tcn', 'original_rmse')} for TCN and "
        f"{describe_model_mean(subject_rows, 'pi_tcn_full', 'original_rmse')} for PI-TCN-full. "
        "For multi-seed submitted-split runs, original-scale RMSE was "
        f"{describe_model_mean(multiseed_rows, 'tcn', 'original_rmse')} for TCN and "
        f"{describe_model_mean(multiseed_rows, 'pi_tcn_full', 'original_rmse')} for PI-TCN-full. "
        "Loss-ablation results are summarized in the revision evidence tables; placeholders should be replaced "
        "with exact values for any incomplete runs before manuscript insertion."
    )
    discussion_text = (
        "The additional experiments support a more conservative interpretation of the physics-regularized TCN. "
        "The revision should emphasize robustness across subject splits and seeds only where completed results show it, "
        "and should explicitly acknowledge that reducing negative vGRF or stance false-zero violations may not eliminate "
        "the swing-phase false-force trade-off. These analyses strengthen the evidence base without changing the core "
        "research question."
    )
    cover_points = "\n".join(
        [
            "- Added subject-wise test rotation across s001-s007 to assess split robustness.",
            "- Added repeated-seed evaluation on the submitted s001-s005/s006/s007 split.",
            "- Added loss-ablation conditions isolating non-negativity, contact consistency, and smoothness terms.",
            "- Clarified terminology toward physics-regularized or lightweight physics-informed regularization.",
            "- Tempered claims to focus on observed robustness and remaining plausibility trade-offs.",
        ]
    )
    (snippets_dir / "results_revision_insert.txt").write_text(results_text + "\n", encoding="utf-8")
    (snippets_dir / "discussion_revision_insert.txt").write_text(discussion_text + "\n", encoding="utf-8")
    (snippets_dir / "cover_letter_resubmission_points.txt").write_text(cover_points + "\n", encoding="utf-8")
    _ = ablation_rows


def generate_terminology_recommendations(output_root: Path) -> None:
    text = """# Terminology Recommendations

Recommendation: consider changing "Physics-Informed Temporal Convolutional Network" to either "Physics-Regularized Temporal Convolutional Network" or "Lightweight Physics-Informed Regularization for Bilateral vGRF Estimation".

Classical PINNs usually imply governing-equation or physics-equation constraints embedded in the learning objective. This manuscript instead uses heuristic but interpretable regularization terms: non-negative vertical force, target-defined contact consistency, and temporal smoothness. Therefore, "physics-regularized" or "lightweight physics-informed" is more accurate and less vulnerable to reviewer criticism while preserving the scientific meaning of the method.
"""
    (output_root / "terminology_recommendations.md").write_text(text, encoding="utf-8")


def print_inspection_summary() -> None:
    print("Repository inspection summary")
    print("- preprocessing / dataset building: scripts/build_dataset.py")
    print("- TCN training script: src/train_tcn_baseline.py")
    print("- PI-TCN training script and loss terms: src/train_pi_tcn.py")
    print("- evaluation / physical plausibility: scripts/evaluate_physical_plausibility.py")
    print("- contact sensitivity evaluator: scripts/evaluate_contact_sensitivity.py")
    print("- existing manuscript outputs: outputs/manuscript/")
    print("- existing baseline outputs: outputs/models/, outputs/tables/, outputs/figures/")
    print("Reusable functions")
    print("- build_dataset.load_one_trial, make_windows, compute_mean_std, normalize_windows")
    print("- TCNModel, GroundLinkWindowDataset, physics_informed_loss, compute_physical_metrics")
    print("Hard-coded assumptions")
    print("- scripts/build_dataset.py defaults to train s001-s005, val s006, test s007.")
    print("- legacy training scripts default to outputs/models, outputs/tables, and outputs/figures.")
    print("Recommended minimal changes")
    print("- keep legacy defaults intact; build per-split datasets and run artifacts under outputs/revision_evidence/.")


def dry_run(specs: list[RunSpec], args: argparse.Namespace) -> None:
    print_inspection_summary()
    print("\nDry-run planned runs")
    for spec in specs:
        print(
            f"- {spec.analysis}/{spec.fold_name}/{spec.model_name}/seed_{spec.seed}: "
            f"train={spec.fold['train_subjects']} val={spec.fold['val_subjects']} test={spec.fold['test_subjects']} "
            f"-> {run_output_dir(args.output_root, spec)}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    generate_terminology_recommendations(args.output_root)
    generate_snippets(args.output_root, {})
    generate_summary_md(args.output_root, {}, specs)


def main() -> int:
    args = parse_args()
    specs = make_run_specs(args)
    if args.dry_run:
        dry_run(specs, args)
        return 0

    print_inspection_summary()
    device = resolve_device(args.device)
    success_count = 0
    for spec in specs:
        if run_with_logs(spec, args, device):
            success_count += 1

    tables = generate_tables(args.output_root)
    generate_figures(args.output_root, tables)
    generate_summary_md(args.output_root, tables, specs)
    generate_snippets(args.output_root, tables)
    generate_terminology_recommendations(args.output_root)
    print(f"\nCompleted {success_count}/{len(specs)} planned runs.")
    print(f"Revision evidence root: {args.output_root}")
    return 0 if success_count == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
