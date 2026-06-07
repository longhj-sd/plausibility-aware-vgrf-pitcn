"""Build windowed GroundLink vertical GRF learning data.

Input features are MoSh++ pose and translation per frame:
    X = concat([poses, trans], axis=1), shape [frames, 168]

Targets are bilateral vertical ground reaction forces:
    y = GRF[:, :, 2], shape [frames, 2]
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MATCHED_CSV = Path("outputs") / "tables" / "matched_groundlink_files.csv"
DEFAULT_OUTPUT_NPZ = Path("data") / "processed" / "groundlink_vgrf_windows_uncompressed.npz"
DEFAULT_METADATA_CSV = Path("outputs") / "tables" / "groundlink_window_metadata.csv"
TRAIN_SUBJECTS = ("s001", "s002", "s003", "s004", "s005")
VAL_SUBJECTS = ("s006",)
TEST_SUBJECTS = ("s007",)


@dataclass(frozen=True)
class TrialData:
    X: np.ndarray
    y: np.ndarray
    subject: str
    trial_name: str
    motion_type: str
    original_npz_frames: int
    original_grf_frames: int
    trimmed_frames: int


@dataclass(frozen=True)
class WindowedTrial:
    X: np.ndarray
    y: np.ndarray
    metadata: list[dict[str, Any]]


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def tensor_to_numpy(value: Any) -> np.ndarray | None:
    """Convert numpy arrays, torch tensors, and simple sequences to numpy."""
    if isinstance(value, np.ndarray):
        if value.dtype == object and value.size == 1:
            try:
                return tensor_to_numpy(value.item())
            except Exception as exc:  # noqa: BLE001
                warn(f"Could not unwrap numpy object array: {exc}")
                return None
        return value

    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        try:
            return value.detach().cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            warn(f"Could not convert torch.Tensor to numpy: {exc}")
            return None

    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except Exception as exc:  # noqa: BLE001
            warn(f"Could not convert sequence to numpy: {exc}")
            return None

    return None


def load_force_dict(force_path: Path) -> dict[str, Any] | None:
    """Load a GroundLink force .npy file as a dictionary."""
    try:
        loaded = np.load(force_path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not load force file {force_path}: {exc}")
        return None

    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        try:
            loaded = loaded.item()
        except Exception as exc:  # noqa: BLE001
            warn(f"Could not convert force object array with .item() for {force_path}: {exc}")
            return None

    if not isinstance(loaded, dict):
        warn(f"Force file is not a dictionary after loading: {force_path} ({type(loaded)})")
        return None

    return loaded


def parse_subject_and_motion(row: dict[str, str]) -> tuple[str, str, str]:
    """Return subject, trial name, and motion type from CSV row data."""
    trial_name = row.get("canonical_stem") or row.get("stem") or Path(row.get("force_path", "")).stem
    parts = trial_name.split("_")
    subject = row.get("subject") or (parts[0] if parts else "")

    motion_type = row.get("motion_type", "")
    if not motion_type:
        if len(parts) >= 4:
            motion_type = "_".join(parts[2:-1])
        elif len(parts) >= 3:
            motion_type = parts[2]

    return subject, trial_name, motion_type


def resolve_csv_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def load_one_trial(row: dict[str, str], base_dir: Path) -> TrialData | None:
    """Load one matched trial and return aligned per-frame X/y arrays."""
    subject, trial_name, motion_type = parse_subject_and_motion(row)
    npz_path = resolve_csv_path(row.get("npz_path", ""), base_dir)
    force_path = resolve_csv_path(row.get("force_path", ""), base_dir)

    try:
        with np.load(npz_path, allow_pickle=True) as npz:
            if "poses" not in npz or "trans" not in npz:
                warn(f"Skipping {trial_name}: NPZ missing poses or trans: {npz_path}")
                return None
            poses = np.asarray(npz["poses"])
            trans = np.asarray(npz["trans"])
    except Exception as exc:  # noqa: BLE001
        warn(f"Skipping {trial_name}: could not load NPZ {npz_path}: {exc}")
        return None

    if poses.ndim != 2 or poses.shape[1] != 165:
        warn(f"Skipping {trial_name}: expected poses shape [frames, 165], got {poses.shape}")
        return None
    if trans.ndim != 2 or trans.shape[1] != 3:
        warn(f"Skipping {trial_name}: expected trans shape [frames, 3], got {trans.shape}")
        return None

    force_dict = load_force_dict(force_path)
    if force_dict is None:
        warn(f"Skipping {trial_name}: could not load force dictionary")
        return None
    if "GRF" not in force_dict:
        warn(f"Skipping {trial_name}: force dictionary missing GRF key")
        return None

    grf = tensor_to_numpy(force_dict["GRF"])
    if grf is None:
        warn(f"Skipping {trial_name}: could not convert GRF to numpy")
        return None
    if grf.ndim != 3 or grf.shape[1:] != (2, 3):
        warn(f"Skipping {trial_name}: expected GRF shape [frames, 2, 3], got {grf.shape}")
        return None

    X = np.concatenate([poses, trans], axis=1)
    y = grf[:, :, 2]

    original_npz_frames = int(X.shape[0])
    original_grf_frames = int(y.shape[0])
    trimmed_frames = min(original_npz_frames, original_grf_frames)
    if trimmed_frames <= 0:
        warn(f"Skipping {trial_name}: no overlapping frames")
        return None

    X = X[:trimmed_frames].astype(np.float32, copy=False)
    y = y[:trimmed_frames].astype(np.float32, copy=False)

    if X.shape[1] != 168:
        warn(f"Skipping {trial_name}: expected X feature dimension 168, got {X.shape[1]}")
        return None
    if y.shape[1] != 2:
        warn(f"Skipping {trial_name}: expected y target dimension 2, got {y.shape[1]}")
        return None

    return TrialData(
        X=X,
        y=y,
        subject=subject,
        trial_name=trial_name,
        motion_type=motion_type,
        original_npz_frames=original_npz_frames,
        original_grf_frames=original_grf_frames,
        trimmed_frames=trimmed_frames,
    )


def make_windows(trial: TrialData, window_size: int, stride: int) -> WindowedTrial | None:
    """Create fixed-length windows and per-window metadata for one trial."""
    if trial.trimmed_frames < window_size:
        warn(
            f"Skipping {trial.trial_name}: trimmed frames {trial.trimmed_frames} "
            f"is shorter than window_size {window_size}"
        )
        return None

    X_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for start in range(0, trial.trimmed_frames - window_size + 1, stride):
        end = start + window_size
        X_windows.append(trial.X[start:end])
        y_windows.append(trial.y[start:end])
        metadata.append(
            {
                "subject": trial.subject,
                "trial_name": trial.trial_name,
                "motion_type": trial.motion_type,
                "window_start": start,
                "window_end": end,
                "original_npz_frames": trial.original_npz_frames,
                "original_grf_frames": trial.original_grf_frames,
                "trimmed_frames": trial.trimmed_frames,
            }
        )

    if not X_windows:
        warn(f"Skipping {trial.trial_name}: no windows created")
        return None

    return WindowedTrial(
        X=np.stack(X_windows).astype(np.float32, copy=False),
        y=np.stack(y_windows).astype(np.float32, copy=False),
        metadata=metadata,
    )


def stack_or_empty(arrays: list[np.ndarray], window_size: int, dim: int) -> np.ndarray:
    if not arrays:
        return np.empty((0, window_size, dim), dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def compute_mean_std(array: np.ndarray, dim: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    if array.shape[0] == 0:
        raise ValueError(f"Cannot compute {name} normalization: no training windows")
    flat = array.reshape(-1, dim)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def normalize_windows(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if array.shape[0] == 0:
        return array.astype(np.float32, copy=False)
    return ((array - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def array_size_mb(array: np.ndarray) -> float:
    return array.nbytes / (1024 * 1024)


def print_array_report(name: str, array: np.ndarray) -> None:
    print(f"  {name}: shape={array.shape}, dtype={array.dtype}, size={array_size_mb(array):.1f} MB")


def write_metadata_csv(metadata: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "subject",
        "trial_name",
        "motion_type",
        "window_start",
        "window_end",
        "original_npz_frames",
        "original_grf_frames",
        "trimmed_frames",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-csv", type=Path, default=DEFAULT_MATCHED_CSV)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--stride", type=int, default=30)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    base_dir = Path.cwd()

    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if not args.matched_csv.exists():
        raise FileNotFoundError(f"Matched CSV does not exist: {args.matched_csv}")

    split_subjects = {
        "train": set(TRAIN_SUBJECTS),
        "val": set(VAL_SUBJECTS),
        "test": set(TEST_SUBJECTS),
    }
    split_X: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_y: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    all_metadata: list[dict[str, Any]] = []
    subjects_seen: set[str] = set()
    motion_types: set[str] = set()
    used_trials = 0
    skipped_trials = 0

    with args.matched_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject, trial_name, motion_type = parse_subject_and_motion(row)
            subjects_seen.add(subject)

            if subject in split_subjects["train"]:
                split = "train"
            elif subject in split_subjects["val"]:
                split = "val"
            elif subject in split_subjects["test"]:
                split = "test"
            else:
                warn(f"Skipping {trial_name}: subject {subject} is not in train/val/test split")
                skipped_trials += 1
                continue

            try:
                trial = load_one_trial(row, base_dir)
                if trial is None:
                    skipped_trials += 1
                    continue

                windowed = make_windows(trial, args.window_size, args.stride)
                if windowed is None:
                    skipped_trials += 1
                    continue
            except Exception as exc:  # noqa: BLE001
                warn(f"Skipping {trial_name}: unexpected error: {exc}")
                skipped_trials += 1
                continue

            used_trials += 1
            motion_types.add(trial.motion_type)
            split_X[split].append(windowed.X)
            split_y[split].append(windowed.y)
            for item in windowed.metadata:
                item["split"] = split
            all_metadata.extend(windowed.metadata)

    missing_subjects = (set(TRAIN_SUBJECTS) | set(VAL_SUBJECTS) | set(TEST_SUBJECTS)) - subjects_seen
    if missing_subjects:
        warn(f"Expected split subjects missing from matched CSV: {sorted(missing_subjects)}")

    X_train = stack_or_empty(split_X["train"], args.window_size, 168)
    y_train = stack_or_empty(split_y["train"], args.window_size, 2)
    X_val = stack_or_empty(split_X["val"], args.window_size, 168)
    y_val = stack_or_empty(split_y["val"], args.window_size, 2)
    X_test = stack_or_empty(split_X["test"], args.window_size, 168)
    y_test = stack_or_empty(split_y["test"], args.window_size, 2)

    x_mean, x_std = compute_mean_std(X_train, 168, "X")
    y_mean, y_std = compute_mean_std(y_train, 2, "y")
    X_train = normalize_windows(X_train, x_mean, x_std)
    X_val = normalize_windows(X_val, x_mean, x_std)
    X_test = normalize_windows(X_test, x_mean, x_std)
    y_train = normalize_windows(y_train, y_mean, y_std)
    y_val = normalize_windows(y_val, y_mean, y_std)
    y_test = normalize_windows(y_test, y_mean, y_std)

    X_train = X_train.astype(np.float32, copy=False)
    y_train = y_train.astype(np.float32, copy=False)
    X_val = X_val.astype(np.float32, copy=False)
    y_val = y_val.astype(np.float32, copy=False)
    X_test = X_test.astype(np.float32, copy=False)
    y_test = y_test.astype(np.float32, copy=False)
    x_mean = x_mean.astype(np.float32, copy=False)
    x_std = x_std.astype(np.float32, copy=False)
    y_mean = y_mean.astype(np.float32, copy=False)
    y_std = y_std.astype(np.float32, copy=False)

    print("\nArray memory report before saving")
    print_array_report("X_train", X_train)
    print_array_report("y_train", y_train)
    print_array_report("X_val", X_val)
    print_array_report("y_val", y_val)
    print_array_report("X_test", X_test)
    print_array_report("y_test", y_test)
    print("\nSaving uncompressed NPZ. This may create a large file but should be faster than compressed saving.")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )
    write_metadata_csv(all_metadata, args.metadata_csv)

    split_metadata_subjects = {
        split: sorted({item["subject"] for item in all_metadata if item["split"] == split})
        for split in ("train", "val", "test")
    }

    print("\nDataset build summary")
    print(f"  matched trials used: {used_trials}")
    print(f"  skipped trials: {skipped_trials}")
    print(f"  train windows: {X_train.shape[0]}")
    print(f"  val windows: {X_val.shape[0]}")
    print(f"  test windows: {X_test.shape[0]}")
    print(f"  X feature dimension: {X_train.shape[-1] if X_train.ndim == 3 else 168}")
    print(f"  y target dimension: {y_train.shape[-1] if y_train.ndim == 3 else 2}")
    print(f"  train subjects: {split_metadata_subjects['train']}")
    print(f"  val subjects: {split_metadata_subjects['val']}")
    print(f"  test subjects: {split_metadata_subjects['test']}")
    print(f"  motion types included: {sorted(motion_types)}")
    print(f"  saved dataset: {args.output_npz}")
    print(f"  saved metadata: {args.metadata_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
