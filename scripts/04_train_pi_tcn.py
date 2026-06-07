"""Train the physics-regularized PI-TCN model."""

from __future__ import annotations

from _release_workflow import load_config, parse_config_arg, run_python, value


def main() -> int:
    args = parse_config_arg(__doc__ or "")
    config = load_config(args.config)
    training = config["training"]
    physics = config["physics_regularization"]
    return run_python(
        [
            "src/train_pi_tcn.py",
            "--dataset-path",
            value(config, "paths", "processed_dataset"),
            "--run-name",
            "pi_tcn_full",
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
            "--lambda-nonneg",
            str(physics["lambda_nonneg"]),
            "--lambda-contact",
            str(physics["lambda_contact"]),
            "--lambda-smooth",
            str(physics["lambda_smooth"]),
            "--contact-threshold",
            str(physics["contact_threshold"]),
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
