"""Compute accuracy and physical-plausibility metrics for trained models."""

from __future__ import annotations

from _release_workflow import load_config, parse_config_arg, run_python, value


def main() -> int:
    args = parse_config_arg(__doc__ or "")
    config = load_config(args.config)
    eval_cfg = config["evaluation"]
    physics = config["physics_regularization"]
    return run_python(
        [
            "scripts/evaluate_physical_plausibility.py",
            "--dataset-path",
            value(config, "paths", "processed_dataset"),
            "--batch-size",
            str(eval_cfg["batch_size"]),
            "--contact-threshold",
            str(physics["contact_threshold"]),
            "--spike-threshold",
            str(eval_cfg["spike_threshold"]),
            "--seed",
            str(value(config, "training", "seed")),
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
