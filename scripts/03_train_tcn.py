"""Train the supervised architecture-matched TCN baseline."""

from __future__ import annotations

from _release_workflow import load_config, parse_config_arg, run_python, value


def main() -> int:
    args = parse_config_arg(__doc__ or "")
    config = load_config(args.config)
    training = config["training"]
    return run_python(
        [
            "src/train_tcn_baseline.py",
            "--dataset-path",
            value(config, "paths", "processed_dataset"),
            "--epochs",
            str(training["epochs"]),
            "--patience",
            str(training["patience"]),
            "--batch-size",
            str(training["batch_size"]),
            "--learning-rate",
            str(training["learning_rate"]),
            "--hidden-channels",
            str(training["hidden_channels"]),
            "--num-blocks",
            str(training["num_blocks"]),
            "--dropout",
            str(training["dropout"]),
            "--seed",
            str(training["seed"]),
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
