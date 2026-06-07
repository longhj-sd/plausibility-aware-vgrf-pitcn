"""Subject split helpers for GroundLink revision evidence experiments."""

from __future__ import annotations

from pathlib import Path


FALLBACK_SUBJECTS = ["s001", "s002", "s003", "s004", "s005", "s006", "s007"]


def get_available_subjects(data_root: str | Path | None = None) -> list[str]:
    """Return sorted available GroundLink subject IDs."""
    root = Path(data_root) if data_root is not None else Path("GRF") / "DATA" / "moshpp"
    if root.exists():
        subjects = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("s"))
        if subjects:
            return subjects
    return FALLBACK_SUBJECTS.copy()


def make_submitted_split() -> dict[str, list[str] | str]:
    """Return the original submitted subject split."""
    return {
        "train_subjects": ["s001", "s002", "s003", "s004", "s005"],
        "val_subjects": ["s006"],
        "test_subjects": ["s007"],
        "fold_name": "submitted_s007",
    }


def make_test_subject_rotation_splits(subjects: list[str] | None = None) -> list[dict[str, list[str] | str]]:
    """Create 5/1/1 subject-rotation folds with circular validation subjects."""
    ordered_subjects = sorted(subjects) if subjects is not None else FALLBACK_SUBJECTS.copy()
    folds: list[dict[str, list[str] | str]] = []
    for index, test_subject in enumerate(ordered_subjects):
        val_subject = ordered_subjects[(index + 1) % len(ordered_subjects)]
        train_subjects = [
            subject for subject in ordered_subjects if subject not in {test_subject, val_subject}
        ]
        folds.append(
            {
                "train_subjects": train_subjects,
                "val_subjects": [val_subject],
                "test_subjects": [test_subject],
                "fold_name": f"fold_{test_subject}",
            }
        )
    return folds
