"""Shared helpers for the numbered reproducibility wrappers."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - dependency-free fallback for dry-run inspection
    yaml = None


def parse_config_arg(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("configs") / "default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print the underlying command without running it.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        if path.as_posix().replace("\\", "/") not in {"configs/default.yaml", "default.yaml"}:
            raise ModuleNotFoundError("PyYAML is required to read custom config files. Install requirements.txt first.")
        return {
            "paths": {
                "force_dir": "data/raw/force",
                "moshpp_dir": "data/raw/moshpp",
                "matched_csv": "outputs/tables/matched_groundlink_files.csv",
                "processed_dataset": "data/processed/groundlink_vgrf_windows_uncompressed.npz",
                "window_metadata_csv": "outputs/tables/groundlink_window_metadata.csv",
                "output_root": "outputs/revision_evidence",
                "prediction_dir": "outputs/revision_evidence/final_audit/predictions",
                "downstream_dir": "outputs/revision_evidence/final_audit/downstream_metrics",
            },
            "windowing": {"window_size": 120, "stride": 30},
            "training": {
                "epochs": 50,
                "patience": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "hidden_channels": 128,
                "num_blocks": 5,
                "dropout": 0.1,
                "seed": 7,
            },
            "physics_regularization": {
                "lambda_nonneg": 0.1,
                "lambda_contact": 0.1,
                "lambda_smooth": 0.01,
                "contact_threshold": 0.05,
            },
            "evaluation": {"batch_size": 64, "spike_threshold": 0.5},
        }
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_python(args: list[str], dry_run: bool = False) -> int:
    command = [sys.executable, *args]
    print("Underlying command:")
    print("  " + " ".join(command))
    if dry_run:
        return 0
    return subprocess.call(command)


def value(config: dict[str, Any], *keys: str) -> Any:
    current: Any = config
    for key in keys:
        current = current[key]
    return current
