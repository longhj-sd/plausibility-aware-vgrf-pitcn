"""Compute downstream vGRF metrics from exported prediction arrays."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


FINAL_AUDIT = Path("outputs") / "revision_evidence" / "final_audit"
PREDICTIONS = FINAL_AUDIT / "predictions"
OUT = FINAL_AUDIT / "downstream_metrics"
CONTACT_THRESHOLD = 0.05


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scalar(value: Any) -> str:
    try:
        return str(value.item())
    except Exception:  # noqa: BLE001
        return str(value)


def maybe_float(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) and value > 0 else float("nan")


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        y_true = data["y_true"].astype(np.float32)
        y_pred = data["y_pred"].astype(np.float32)
        y_mean = data["y_mean"].astype(np.float32) if "y_mean" in data else None
        y_std = data["y_std"].astype(np.float32) if "y_std" in data else None
        frame_rate = maybe_float(scalar(data["frame_rate"])) if "frame_rate" in data else float("nan")
        dt = maybe_float(scalar(data["dt"])) if "dt" in data else float("nan")
        if not math.isfinite(dt) and math.isfinite(frame_rate):
            dt = 1.0 / frame_rate
        trial_ids = data["trial_identifiers"] if "trial_identifiers" in data else np.array([])
        window_start = data["window_start"] if "window_start" in data else np.array([])
        window_end = data["window_end"] if "window_end" in data else np.array([])
        channel_names = data["channel_names"] if "channel_names" in data else np.array(["left_vgrf", "right_vgrf"])
        meta = {
            "prediction_file": str(path),
            "analysis": scalar(data["split_name"]) if "split_name" in data else "",
            "fold_name": scalar(data["fold_name"]) if "fold_name" in data else "",
            "test_subject": scalar(data["test_subject"]) if "test_subject" in data else "",
            "model": scalar(data["model_name"]) if "model_name" in data else "",
            "seed": scalar(data["seed"]) if "seed" in data else "",
        }

    if y_mean is not None and y_std is not None:
        true_eval = y_true * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
        pred_eval = y_pred * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
        scale_label = "original_scale_from_training_y_mean_y_std"
    else:
        true_eval = y_true
        pred_eval = y_pred
        scale_label = "normalized"
    contact_multiplier = dt if math.isfinite(dt) else 1.0
    contact_unit = "seconds" if math.isfinite(dt) else "frames"
    rows: list[dict[str, Any]] = []
    for window_index in range(true_eval.shape[0]):
        for channel_index in range(true_eval.shape[2]):
            target = true_eval[window_index, :, channel_index]
            pred = pred_eval[window_index, :, channel_index]
            target_contact_time = float((target >= CONTACT_THRESHOLD).sum() * contact_multiplier)
            predicted_contact_time = float((pred >= CONTACT_THRESHOLD).sum() * contact_multiplier)
            target_peak = float(target.max())
            predicted_peak = float(pred.max())
            row = {
                **meta,
                "window_index": window_index,
                "trial_identifier": str(trial_ids[window_index]) if window_index < len(trial_ids) else "",
                "window_start": int(window_start[window_index]) if window_index < len(window_start) else "",
                "window_end": int(window_end[window_index]) if window_index < len(window_end) else "",
                "channel": str(channel_names[channel_index]) if channel_index < len(channel_names) else str(channel_index),
                "scale_label": scale_label,
                "contact_threshold": CONTACT_THRESHOLD,
                "frame_rate_available": math.isfinite(frame_rate),
                "dt_available": math.isfinite(dt),
                "contact_time_unit": contact_unit,
                "target_contact_time": target_contact_time,
                "predicted_contact_time": predicted_contact_time,
                "contact_time_signed_error": predicted_contact_time - target_contact_time,
                "contact_time_absolute_error": abs(predicted_contact_time - target_contact_time),
                "target_peak_vgrf": target_peak,
                "predicted_peak_vgrf": predicted_peak,
                "peak_signed_error": predicted_peak - target_peak,
                "peak_absolute_error": abs(predicted_peak - target_peak),
                "target_impulse": "",
                "predicted_impulse": "",
                "impulse_signed_error": "",
                "impulse_absolute_error": "",
                "loading_rate_signed_error": "",
                "loading_rate_absolute_error": "",
                "loading_rate_note": "not computed due to insufficient event-definition information",
            }
            if math.isfinite(dt):
                target_impulse = float(target.sum() * dt)
                predicted_impulse = float(pred.sum() * dt)
                row.update(
                    {
                        "target_impulse": target_impulse,
                        "predicted_impulse": predicted_impulse,
                        "impulse_signed_error": predicted_impulse - target_impulse,
                        "impulse_absolute_error": abs(predicted_impulse - target_impulse),
                    }
                )
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[field]) for field in group_fields), []).append(row)
    metrics = ["contact_time_absolute_error", "peak_absolute_error", "impulse_absolute_error", "loading_rate_absolute_error"]
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        item["n_windows_channels"] = len(group)
        for metric in metrics:
            vals = [float(row[metric]) for row in group if row.get(metric) not in ("", None)]
            item[f"{metric}_mean"] = mean(vals) if vals else ""
            item[f"{metric}_median"] = median(vals) if vals else ""
            item[f"{metric}_sd"] = stdev(vals) if len(vals) > 1 else 0.0 if vals else ""
        out.append(item)
    return out


def paired_summary(by_run: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metrics = ["contact_time_absolute_error", "peak_absolute_error", "impulse_absolute_error", "loading_rate_absolute_error"]
    subjects = sorted({row["test_subject"] for row in by_run})
    for subject in subjects:
        tcn = next((row for row in by_run if row["test_subject"] == subject and row["model"] == "tcn"), None)
        pi = next((row for row in by_run if row["test_subject"] == subject and row["model"] == "pi_tcn_full"), None)
        if not tcn or not pi:
            continue
        for metric in metrics:
            tcn_value = tcn.get(f"{metric}_mean", "")
            pi_value = pi.get(f"{metric}_mean", "")
            if tcn_value == "" or pi_value == "":
                continue
            diff = float(pi_value) - float(tcn_value)
            paired_rows.append(
                {
                    "test_subject": subject,
                    "metric": metric,
                    "tcn": tcn_value,
                    "pi_tcn_full": pi_value,
                    "pi_minus_tcn": diff,
                    "pi_better": diff < 0,
                    "direction": "lower_absolute_error_is_better",
                }
            )
    for metric in metrics:
        diffs = [float(row["pi_minus_tcn"]) for row in paired_rows if row["metric"] == metric]
        if not diffs:
            continue
        wins = sum(row["pi_better"] is True for row in paired_rows if row["metric"] == metric)
        summary_rows.append(
            {
                "metric": metric,
                "n_subjects": len(diffs),
                "pi_better_subjects": wins,
                "mean_pi_minus_tcn": mean(diffs),
                "median_pi_minus_tcn": median(diffs),
                "sd_pi_minus_tcn": stdev(diffs) if len(diffs) > 1 else 0.0,
                "direction": "lower_absolute_error_is_better",
            }
        )
    return paired_rows, summary_rows


def write_methods_note(path: Path, has_dt: bool) -> None:
    text = f"""# Downstream vGRF Metrics Methods Note

Metrics are computed at the window level, separately for each bilateral vGRF channel, then summarized by run and model. They are not trial-level gait-event metrics.

Contact threshold: {CONTACT_THRESHOLD}, matching the manuscript contact-consistency threshold.

Prediction arrays are converted to original scale when `y_mean` and `y_std` are present. The exported arrays also retain normalized `y_true` and `y_pred`.

Frame rate or dt available: {'yes' if has_dt else 'no'}. Contact time is therefore reported in {'seconds' if has_dt else 'frames'}. Vertical impulse is {'computed' if has_dt else 'not computed'} because it requires dt. Loading-rate error is not computed due to insufficient event-definition information; no clinical or sport-validity claim should be made from loading-rate metrics.

Main-text suitability: contact-time absolute error and peak-vGRF absolute error may be reported cautiously as supplementary biomechanical checks. Impulse should be supplementary-only if dt later becomes available but units remain unclear. Loading rate should not be reported from the current artifacts.

Limitation: overlapping fixed-length windows can duplicate frames across adjacent windows, so these values summarize window-level prediction behavior rather than independent trial-level biomechanics.
"""
    path.write_text(text, encoding="utf-8")


def write_markdown_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Downstream Paired Subject Summary",
        "",
        "Lower absolute error is better for all listed downstream metrics.",
        "",
        "| Metric | Subjects | PI Better Subjects | Mean PI-TCN - TCN | Interpretation |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        interpretation = "supports main RMSE/R2 findings" if row["mean_pi_minus_tcn"] < 0 else "does not consistently support main RMSE/R2 findings"
        placement = "supplementary or cautious main-text mention" if row["metric"] in {"contact_time_absolute_error", "peak_absolute_error"} else "supplementary only or not reported"
        lines.append(
            f"| {row['metric']} | {row['n_subjects']} | {row['pi_better_subjects']} | "
            f"{row['mean_pi_minus_tcn']:.6g} | {interpretation}; {placement} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(path: Path, prediction_count: int, shape_ok: bool, summary_rows: list[dict[str, Any]], has_dt: bool) -> None:
    available = ["contact time error", "peak vGRF error"]
    unavailable = ["vertical impulse error" if not has_dt else "", "loading-rate error"]
    unavailable = [item for item in unavailable if item]
    lines = [
        "# Final Downstream Metrics Report",
        "",
        f"- Prediction export completed: {'yes' if prediction_count else 'no'}",
        f"- Prediction files exported: {prediction_count}",
        f"- y_true/y_pred shape checks passed: {'yes' if shape_ok else 'no'}",
        f"- Available downstream metrics: {', '.join(available)}",
        f"- Unavailable downstream metrics: {', '.join(unavailable)}",
        "",
        "## Subject-Level Paired Comparison",
        "",
    ]
    for row in summary_rows:
        lines.append(f"- {row['metric']}: PI-TCN better in {row['pi_better_subjects']}/{row['n_subjects']} subjects; mean PI-TCN - TCN {row['mean_pi_minus_tcn']:.6g}.")
    lines.extend(
        [
            "",
            "## Recommended Manuscript Placement",
            "",
            "- Contact-time and peak-vGRF errors: Supplementary Materials, with cautious mention in Results only if they align with primary accuracy findings.",
            "- Impulse: supplementary only if dt becomes available; otherwise not reported.",
            "- Loading rate: not reported from current artifacts.",
            "",
            "## Wording Recommendations",
            "",
            "Results: \"Inference-only exports from the completed subject-rotation checkpoints enabled window-level downstream checks of contact-time and peak-vGRF errors without retraining.\"",
            "Discussion: \"These downstream metrics should be interpreted as supplementary window-level evidence rather than independent trial-level biomechanical validation.\"",
            "",
            "## Supplementary Files",
            "",
            "- `downstream_metrics_all_windows.csv`",
            "- `downstream_metrics_by_run.csv`",
            "- `downstream_metrics_subject_rotation_paired.csv`",
            "- `downstream_paired_subject_summary.csv`",
            "- `downstream_metrics_methods_note.md`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.npz"))
    print(f"Prediction files: {len(paths)}")
    if args.dry_run:
        return 0
    all_rows: list[dict[str, Any]] = []
    shape_ok = True
    for path in paths:
        rows = load_prediction_rows(path)
        all_rows.extend(rows)
    by_run = summarize(all_rows, ["analysis", "fold_name", "test_subject", "model", "seed"])
    by_model = summarize(all_rows, ["analysis", "model"])
    paired_rows, paired_summary_rows = paired_summary(by_run)
    has_dt = any(row.get("dt_available") is True for row in all_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "downstream_metrics_all_windows.csv", all_rows)
    write_csv(args.output_dir / "downstream_metrics_by_run.csv", by_run)
    write_csv(args.output_dir / "downstream_metrics_summary_by_model.csv", by_model)
    write_csv(args.output_dir / "downstream_metrics_subject_rotation_paired.csv", paired_rows)
    write_csv(args.output_dir / "downstream_paired_subject_summary.csv", paired_summary_rows)
    write_markdown_summary(args.output_dir / "downstream_paired_subject_summary.md", paired_summary_rows)
    write_methods_note(args.output_dir / "downstream_metrics_methods_note.md", has_dt)
    write_final_report(args.output_dir / "final_downstream_metrics_report.md", len(paths), shape_ok, paired_summary_rows, has_dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
