"""Train a TCN baseline for GroundLink bilateral vertical GRF estimation."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET_PATH = Path("data") / "processed" / "groundlink_vgrf_windows_uncompressed.npz"
DEFAULT_MODEL_PATH = Path("outputs") / "models" / "tcn_baseline_best.pt"
DEFAULT_METRICS_CSV = Path("outputs") / "tables" / "tcn_baseline_metrics.csv"
DEFAULT_TEST_METRICS_CSV = Path("outputs") / "tables" / "tcn_baseline_test_metrics.csv"
DEFAULT_CURVE_PATH = Path("outputs") / "figures" / "tcn_baseline_training_curve.png"
DEFAULT_EXAMPLE_PATH = Path("outputs") / "figures" / "tcn_baseline_prediction_example.png"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GroundLinkWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 3:
            raise ValueError(f"Expected X shape [windows, time, features], got {X.shape}")
        if y.ndim != 3:
            raise ValueError(f"Expected y shape [windows, time, targets], got {y.shape}")
        if X.shape[0] != y.shape[0] or X.shape[1] != y.shape[1]:
            raise ValueError(f"X/y window or time dimensions do not match: {X.shape} vs {y.shape}")

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 168,
        output_channels: int = 2,
        hidden_channels: int = 128,
        num_blocks: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TCNBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2**idx,
                    dropout=dropout,
                )
                for idx in range(num_blocks)
            ]
        )
        self.output_proj = nn.Conv1d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, time, features] -> [batch, features, time]
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        # [batch, targets, time] -> [batch, time, targets]
        return x.transpose(1, 2)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
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
        "rmse": float(rmse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "r2": float(r2.detach().cpu()),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        batch_size = int(X.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    y_mean: torch.Tensor | None = None,
    y_std: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(X)
        loss = criterion(pred, y)

        batch_size = int(X.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        preds.append(pred.detach().cpu())
        targets.append(y.detach().cpu())

    if not preds:
        return {
            "loss": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "orig_rmse": float("nan"),
            "orig_mae": float("nan"),
        }

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    metrics = compute_metrics(pred_all, target_all)
    metrics["loss"] = total_loss / max(total_samples, 1)

    if y_mean is not None and y_std is not None:
        pred_orig = pred_all * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
        target_orig = target_all * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
        orig_metrics = compute_metrics(pred_orig, target_orig)
        metrics["orig_rmse"] = orig_metrics["rmse"]
        metrics["orig_mae"] = orig_metrics["mae"]
    else:
        metrics["orig_rmse"] = float("nan")
        metrics["orig_mae"] = float("nan")

    return metrics


def plot_training_curve(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: Could not import matplotlib, skipping training curve: {exc}")
        return

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_rmse = [float(row["val_rmse"]) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(epochs, train_loss, label="Train loss", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train MSE loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_rmse, label="Val RMSE", color="tab:orange")
    ax2.set_ylabel("Validation RMSE", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_prediction_example(
    model: nn.Module,
    dataset: GroundLinkWindowDataset,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    output_path: Path,
) -> None:
    if len(dataset) == 0:
        print("WARNING: Test dataset is empty, skipping prediction example plot")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: Could not import matplotlib, skipping prediction plot: {exc}")
        return

    model.eval()
    X, y = dataset[0]
    pred = model(X.unsqueeze(0).to(device)).cpu().squeeze(0)
    pred_orig = pred * y_std.reshape(1, -1) + y_mean.reshape(1, -1)
    y_orig = y * y_std.reshape(1, -1) + y_mean.reshape(1, -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    labels = ("Left vertical GRF", "Right vertical GRF")
    for idx, ax in enumerate(axes):
        ax.plot(y_orig[:, idx].numpy(), label="Target")
        ax.plot(pred_orig[:, idx].numpy(), label="Prediction")
        ax.set_ylabel(labels[idx])
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Frame in window")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_rmse",
        "val_mae",
        "val_r2",
        "val_orig_rmse",
        "val_orig_mae",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_test_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["loss", "rmse", "mae", "r2", "orig_rmse", "orig_mae"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: metrics[key] for key in fieldnames})


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--test-metrics-csv", type=Path, default=DEFAULT_TEST_METRICS_CSV)
    parser.add_argument("--curve-path", type=Path, default=DEFAULT_CURVE_PATH)
    parser.add_argument("--example-path", type=Path, default=DEFAULT_EXAMPLE_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-only", action="store_true")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    set_seed(args.seed)

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {args.dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with np.load(args.dataset_path, allow_pickle=False) as data:
        X_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.float32)
        X_val = data["X_val"].astype(np.float32)
        y_val = data["y_val"].astype(np.float32)
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.float32)
        y_mean_np = data["y_mean"].astype(np.float32)
        y_std_np = data["y_std"].astype(np.float32)

    print("Dataset shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    train_dataset = GroundLinkWindowDataset(X_train, y_train)
    val_dataset = GroundLinkWindowDataset(X_val, y_val)
    test_dataset = GroundLinkWindowDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TCNModel(
        input_channels=168,
        output_channels=2,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    y_mean = torch.as_tensor(y_mean_np, dtype=torch.float32)
    y_std = torch.as_tensor(y_std_np, dtype=torch.float32)

    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    rows: list[dict[str, Any]] = []
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        print(f"Eval-only mode: loading best model from {args.model_path}")
    else:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device, y_mean=y_mean, y_std=y_std)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "val_orig_rmse": val_metrics["orig_rmse"],
                "val_orig_mae": val_metrics["orig_mae"],
            }
            rows.append(row)

            improved = val_metrics["rmse"] < best_val_rmse
            if improved:
                best_val_rmse = val_metrics["rmse"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_rmse": best_val_rmse,
                        "args": vars(args),
                    },
                    args.model_path,
                )
                status = "improved"
            else:
                epochs_without_improvement += 1
                status = f"no improvement {epochs_without_improvement}/{args.patience}"

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
                f"val_rmse={val_metrics['rmse']:.6f} | early_stopping={status}"
            )

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
                break

        write_epoch_metrics(args.metrics_csv, rows)
        plot_training_curve(rows, args.curve_path)

    # This checkpoint was generated locally by this script, so weights_only=False is acceptable here.
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, y_mean=y_mean, y_std=y_std)
    write_test_metrics(args.test_metrics_csv, test_metrics)
    plot_prediction_example(model, test_dataset, device, y_mean, y_std, args.example_path)

    print("\nFinal test metrics")
    print(f"  normalized RMSE: {test_metrics['rmse']:.6f}")
    print(f"  normalized MAE: {test_metrics['mae']:.6f}")
    print(f"  normalized R2: {test_metrics['r2']:.6f}")
    print(f"  original-scale RMSE: {test_metrics['orig_rmse']:.6f}")
    print(f"  original-scale MAE: {test_metrics['orig_mae']:.6f}")
    print(f"  saved best model: {args.model_path}")
    print(f"  saved epoch metrics: {args.metrics_csv}")
    print(f"  saved test metrics: {args.test_metrics_csv}")
    print(f"  saved training curve: {args.curve_path}")
    print(f"  saved prediction example: {args.example_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
