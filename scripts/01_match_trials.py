"""Match local GroundLink force and MoSh++ trial files."""

from __future__ import annotations

from _release_workflow import load_config, parse_config_arg, run_python, value


def main() -> int:
    args = parse_config_arg(__doc__ or "")
    config = load_config(args.config)
    return run_python(
        [
            "scripts/inspect_groundlink.py",
            "--force-dir",
            value(config, "paths", "force_dir"),
            "--moshpp-dir",
            value(config, "paths", "moshpp_dir"),
            "--csv-path",
            value(config, "paths", "matched_csv"),
            "--figure-path",
            "outputs/figures/check_vgrf.png",
            "--seed",
            str(value(config, "training", "seed")),
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
