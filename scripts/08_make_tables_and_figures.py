"""Regenerate manuscript-oriented tables and figures from local result CSVs."""

from __future__ import annotations

from _release_workflow import parse_config_arg, run_python


def main() -> int:
    args = parse_config_arg(__doc__ or "")
    return run_python(
        [
            "scripts/create_final_results_package.py",
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
