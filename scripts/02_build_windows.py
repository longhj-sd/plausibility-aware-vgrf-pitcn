"""Build windowed bilateral vGRF arrays from matched local GroundLink trials."""

from __future__ import annotations

from _release_workflow import load_config, parse_config_arg, run_python, value


def main() -> int:
    args = parse_config_arg(__doc__ or "")
    config = load_config(args.config)
    return run_python(
        [
            "scripts/build_dataset.py",
            "--matched-csv",
            value(config, "paths", "matched_csv"),
            "--output-npz",
            value(config, "paths", "processed_dataset"),
            "--metadata-csv",
            value(config, "paths", "window_metadata_csv"),
            "--window-size",
            str(value(config, "windowing", "window_size")),
            "--stride",
            str(value(config, "windowing", "stride")),
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
