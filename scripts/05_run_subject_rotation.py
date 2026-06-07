"""Run the seven-fold subject-rotation experiment.

This wrapper calls scripts/run_revision_evidence.py. Full execution requires
local GroundLink data and performs long-running training.
"""

from __future__ import annotations

from _release_workflow import load_config, parse_config_arg, run_python, value


def main() -> int:
    args = parse_config_arg(__doc__ or "")
    config = load_config(args.config)
    training = config["training"]
    return run_python(
        [
            "scripts/run_revision_evidence.py",
            "--analysis",
            "subject_rotation",
            "--models",
            "tcn,pi_tcn_full",
            "--seeds",
            str(training["seed"]),
            "--epochs",
            str(training["epochs"]),
            "--patience",
            str(training["patience"]),
            "--batch-size",
            str(training["batch_size"]),
            "--output-root",
            value(config, "paths", "output_root"),
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
