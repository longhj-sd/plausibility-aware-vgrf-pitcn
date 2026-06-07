"""Compute downstream vGRF diagnostics from local prediction exports.

The prediction exports are not redistributed. Generate them locally before
running this wrapper.
"""

from __future__ import annotations

from _release_workflow import load_config, parse_config_arg, run_python, value


def main() -> int:
    args = parse_config_arg(__doc__ or "")
    config = load_config(args.config)
    return run_python(
        [
            "scripts/compute_final_downstream_vgrf_metrics.py",
            "--input-dir",
            value(config, "paths", "prediction_dir"),
            "--output-dir",
            value(config, "paths", "downstream_dir"),
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
