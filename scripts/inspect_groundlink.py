"""Inspect GroundLink force and MoSh++ data files before preprocessing.

This script intentionally avoids assuming the internal structure of the .npy
and .npz files. It first reports file matching statistics, then samples a few
matched pairs and prints the arrays it can discover.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


DEFAULT_FORCE_DIR = Path("GRF") / "DATA" / "force"
DEFAULT_MOSHPP_DIR = Path("GRF") / "DATA" / "moshpp"
DEFAULT_FIGURE_PATH = Path("outputs") / "figures" / "check_vgrf.png"
DEFAULT_CSV_PATH = Path("outputs") / "tables" / "matched_groundlink_files.csv"
GRF_CANDIDATE_TERMS = ("grf", "ground_reaction", "reaction_force", "force")


@dataclass(frozen=True)
class MatchedPair:
    canonical_stem: str
    force_stem: str
    npz_stem: str
    force_path: Path
    npz_path: Path


@dataclass(frozen=True)
class IndexedFile:
    canonical_stem: str
    raw_stem: str
    path: Path


@dataclass(frozen=True)
class FileIndex:
    files: dict[str, IndexedFile]
    total_files: int


@dataclass(frozen=True)
class ForceSummary:
    force_frames: int | None
    grf_shape: str
    cop_shape: str
    grf_candidate: tuple[str, np.ndarray] | None = None


def warn(message: str) -> None:
    """Print warnings in a consistent format without stopping the script."""
    print(f"WARNING: {message}", file=sys.stderr)


def canonical_npz_stem(stem: str) -> str:
    """Remove the MoSh++ stage suffix used in NPZ filenames."""
    suffix = "_stageii"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def path_contains_part(path: Path, part_name: str) -> bool:
    return any(part.lower() == part_name.lower() for part in path.parts)


def choose_duplicate_file(existing: IndexedFile, candidate: IndexedFile, prefer_non_soccerkick: bool) -> IndexedFile:
    """Choose which duplicate file to keep for a canonical stem."""
    if not prefer_non_soccerkick:
        return existing

    existing_in_soccerkick = path_contains_part(existing.path, "soccerkick_full")
    candidate_in_soccerkick = path_contains_part(candidate.path, "soccerkick_full")
    if existing_in_soccerkick and not candidate_in_soccerkick:
        return candidate
    return existing


def collect_files(
    root: Path,
    suffix: str,
    *,
    canonicalizer: Callable[[str], str] | None = None,
    prefer_non_soccerkick: bool = False,
) -> FileIndex:
    """Return files below root keyed by canonical stem.

    If duplicate canonical stems are present, the selected path is kept and
    later duplicates are reported. For force files, duplicates outside any
    soccerkick_full subfolder are preferred.
    """
    if not root.exists():
        warn(f"Folder does not exist: {root}")
        return FileIndex(files={}, total_files=0)

    total_files = 0
    make_canonical = canonicalizer or (lambda stem: stem)
    indexed_files: dict[str, IndexedFile] = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        total_files += 1
        raw_stem = path.stem
        canonical_stem = make_canonical(raw_stem)
        candidate = IndexedFile(canonical_stem=canonical_stem, raw_stem=raw_stem, path=path)

        if canonical_stem in indexed_files:
            existing = indexed_files[canonical_stem]
            selected = choose_duplicate_file(existing, candidate, prefer_non_soccerkick)
            skipped = candidate if selected is existing else existing
            indexed_files[canonical_stem] = selected
            warn(
                "Duplicate canonical stem found; keeping selected match: "
                f"{canonical_stem} -> {selected.path} and skipping {skipped.path}"
            )
            continue
        indexed_files[canonical_stem] = candidate
    return FileIndex(files=indexed_files, total_files=total_files)


def load_numpy_file(path: Path) -> Any | None:
    """Load a .npy or .npz file with robust error handling."""
    try:
        return np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001 - inspection should survive bad files.
        warn(f"Could not load {path}: {exc}")
        return None


def normalize_loaded_npy(value: Any) -> Any:
    """Unwrap common np.save(dict) forms while preserving unknown structures."""
    if isinstance(value, np.ndarray) and value.dtype == object:
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
        if value.size == 1:
            try:
                item = value.reshape(-1)[0]
            except Exception:  # noqa: BLE001
                return value
            if isinstance(item, dict):
                return item
    return value


def is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return isinstance(value, torch.Tensor)


def to_numpy_array(x: Any) -> np.ndarray | None:
    """Convert common Force value containers to numpy arrays."""
    if isinstance(x, np.ndarray):
        if x.dtype == object and x.size == 1:
            try:
                return to_numpy_array(x.item())
            except Exception:  # noqa: BLE001
                return None
        return x

    if is_torch_tensor(x):
        try:
            return x.detach().cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            warn(f"Could not convert torch.Tensor to numpy: {exc}")
            return None

    if isinstance(x, (list, tuple)):
        try:
            return np.asarray(x)
        except Exception as exc:  # noqa: BLE001
            warn(f"Could not convert list/tuple to numpy array: {exc}")
            return None

    return None


def iter_named_arrays(value: Any, prefix: str = "") -> Iterable[tuple[str, np.ndarray]]:
    """Yield arrays from dictionaries, npz files, arrays, and simple objects."""
    if isinstance(value, np.lib.npyio.NpzFile):
        for key in value.files:
            try:
                array = value[key]
            except Exception as exc:  # noqa: BLE001
                warn(f"Could not read NPZ key {key}: {exc}")
                continue
            yield from iter_named_arrays(array, key)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            array = to_numpy_array(item)
            if array is not None:
                yield name, array
                continue
            yield from iter_named_arrays(item, name)
        return

    converted = to_numpy_array(value)
    if converted is not None and not isinstance(value, np.ndarray):
        yield prefix or "<array>", converted
        return

    if isinstance(value, np.ndarray):
        name = prefix or "<array>"
        yield name, value
        if value.dtype == object and value.size <= 10:
            for index, item in enumerate(value.reshape(-1)):
                if isinstance(item, dict):
                    nested_prefix = f"{name}[{index}]"
                    yield from iter_named_arrays(item, nested_prefix)
        return

    # Some saved objects expose array-like attributes rather than dictionaries.
    for attr in dir(value):
        if attr.startswith("_"):
            continue
        try:
            item = getattr(value, attr)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(item, np.ndarray):
            name = f"{prefix}.{attr}" if prefix else attr
            yield name, item


def print_array_descriptions(named_arrays: Iterable[tuple[str, np.ndarray]]) -> None:
    for name, array in named_arrays:
        print(f"    {name}: shape={array.shape}, dtype={array.dtype}")


def print_force_keys(force_value: Any) -> None:
    """Print keys when force data is dictionary-like."""
    if isinstance(force_value, dict):
        print(f"  Force keys: {list(force_value.keys())}")
        return

    if isinstance(force_value, np.ndarray) and force_value.dtype == object:
        dict_items = [item for item in force_value.reshape(-1) if isinstance(item, dict)]
        if dict_items:
            print(f"  Force object-array dict keys: {list(dict_items[0].keys())}")


def format_shape(value: Any) -> str:
    shape = getattr(value, "shape", None)
    return "" if shape is None else str(tuple(shape))


def format_dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return "" if dtype is None else str(dtype)


def print_force_value_details(force_value: Any) -> None:
    """Print dictionary values with container-aware shape and dtype details."""
    if not isinstance(force_value, dict):
        return

    print("  Force dictionary values:")
    for key, value in force_value.items():
        print(f"    {key}: type={type(value)}")

        shape = format_shape(value)
        dtype = format_dtype(value)
        if shape:
            print(f"      shape={shape}")
        if dtype:
            print(f"      dtype={dtype}")

        if is_torch_tensor(value):
            print(f"      tensor shape={tuple(value.shape)}, tensor dtype={value.dtype}")

        if isinstance(value, (list, tuple)):
            print(f"      length={len(value)}")
            if value:
                first = value[0]
                first_shape = format_shape(first)
                print(f"      first element type={type(first)}")
                if first_shape:
                    print(f"      first element shape={first_shape}")


def is_numeric_array(array: np.ndarray) -> bool:
    try:
        return bool(np.issubdtype(array.dtype, np.number))
    except TypeError:
        return False


def candidate_score(name: str, array: np.ndarray) -> int:
    lowered = name.lower()
    score = 0
    for term in GRF_CANDIDATE_TERMS:
        if term in lowered:
            score += 10
    if array.ndim == 3:
        score += 5
    if array.ndim >= 2 and array.shape[-1] == 3:
        score += 3
    if array.ndim == 3 and array.shape[1:] == (2, 3):
        score += 20
    return score


def locate_grf(force_value: Any) -> tuple[str, np.ndarray] | None:
    """Find the most plausible GRF array after inspecting available arrays."""
    if isinstance(force_value, dict):
        for key, item in force_value.items():
            if str(key).lower() == "grf":
                array = to_numpy_array(item)
                if array is not None:
                    return str(key), array

    candidates: list[tuple[int, str, np.ndarray]] = []
    for name, array in iter_named_arrays(force_value):
        if not is_numeric_array(array):
            continue
        score = candidate_score(name, array)
        if score > 0:
            candidates.append((score, name, array))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, name, array = candidates[0]
    return name, array


def locate_force_array(force_value: Any, target_key: str) -> tuple[str, np.ndarray] | None:
    if isinstance(force_value, dict):
        for key, item in force_value.items():
            if str(key).lower() == target_key.lower():
                array = to_numpy_array(item)
                if array is not None:
                    return str(key), array
    return None


def extract_vertical_grf(grf: np.ndarray) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Extract left/right vertical GRF from supported GroundLink layouts."""
    if not is_numeric_array(grf):
        return None

    if grf.ndim == 3 and grf.shape[1:] == (2, 3):
        return grf[:, 0, 2], grf[:, 1, 2], "[frames, 2, 3]"

    if grf.ndim == 2 and grf.shape[1] == 6:
        return grf[:, 2], grf[:, 5], "[frames, 6]"

    if grf.ndim == 3 and grf.shape[0] == 2 and grf.shape[2] == 3:
        transposed = np.transpose(grf, (1, 0, 2))
        return transposed[:, 0, 2], transposed[:, 1, 2], "[2, frames, 3] transposed to [frames, 2, 3]"

    if grf.ndim == 3 and grf.shape[1:] and grf.shape[1:] == (3, 2):
        transposed = np.transpose(grf, (0, 2, 1))
        return transposed[:, 0, 2], transposed[:, 1, 2], "[frames, 3, 2] transposed to [frames, 2, 3]"

    return None


def shape_string(array: np.ndarray | None) -> str:
    return "" if array is None else str(tuple(array.shape))


def frame_count_from_grf(grf: np.ndarray | None) -> int | None:
    if grf is None or grf.ndim == 0:
        return None
    if grf.ndim == 3 and grf.shape[0] == 2 and grf.shape[2] == 3:
        return int(grf.shape[1])
    return int(grf.shape[0])


def frame_count_from_arrays(named_arrays: Iterable[tuple[str, np.ndarray]]) -> int | None:
    """Infer frame count from discovered arrays without assuming exact keys."""
    counts: list[int] = []
    for _, array in named_arrays:
        if array.ndim > 0 and is_numeric_array(array):
            counts.append(int(array.shape[0]))
    if not counts:
        return None

    # Prefer the most common leading dimension, which usually represents frames.
    return max(set(counts), key=counts.count)


def parse_stem(stem: str) -> tuple[str, str]:
    """Parse subject and motion type from stems like s001_20220513_chair_1."""
    parts = stem.split("_")
    subject = parts[0] if parts else ""
    if len(parts) >= 4:
        motion_type = "_".join(parts[2:-1])
    elif len(parts) >= 3:
        motion_type = parts[2]
    else:
        motion_type = ""
    return subject, motion_type


def inspect_pair(pair: MatchedPair) -> tuple[ForceSummary, int | None]:
    print("\n" + "=" * 80)
    print(f"Inspecting matched pair: {pair.canonical_stem}")
    print(f"  Force stem: {pair.force_stem}")
    print(f"  NPZ stem: {pair.npz_stem}")
    print(f"  NPZ file path: {pair.npz_path}")

    npz = load_numpy_file(pair.npz_path)
    npz_frames: int | None = None
    if isinstance(npz, np.lib.npyio.NpzFile):
        print(f"  NPZ keys: {npz.files}")
        npz_arrays = list(iter_named_arrays(npz))
        print("  NPZ arrays:")
        print_array_descriptions(npz_arrays)
        npz_frames = frame_count_from_arrays(npz_arrays)
    else:
        warn(f"Expected NPZ file but could not inspect keys: {pair.npz_path}")

    print(f"  Force file path: {pair.force_path}")
    force_loaded = load_numpy_file(pair.force_path)
    force_summary = ForceSummary(force_frames=None, grf_shape="", cop_shape="")
    if force_loaded is not None:
        force_value = normalize_loaded_npy(force_loaded)
        print(f"  Type of loaded Force object: {type(force_value)}")
        print_force_keys(force_value)
        print_force_value_details(force_value)

        force_arrays = list(iter_named_arrays(force_value))
        print("  Force values:")
        print_array_descriptions(force_arrays)

        grf_candidate = locate_grf(force_value)
        cop_candidate = locate_force_array(force_value, "CoP")
        cop = cop_candidate[1] if cop_candidate is not None else None
        if cop is not None:
            print(f"  CoP array: {cop_candidate[0]}, shape={cop.shape}, dtype={cop.dtype}")

        if grf_candidate is None:
            warn(f"No GRF array found in {pair.force_path}")
            force_frames = frame_count_from_arrays(force_arrays)
            force_summary = ForceSummary(
                force_frames=force_frames,
                grf_shape="",
                cop_shape=shape_string(cop),
            )
        else:
            grf_name, grf = grf_candidate
            print(f"  GRF array: {grf_name}, shape={grf.shape}, dtype={grf.dtype}")
            if extract_vertical_grf(grf) is None:
                warn(f"GRF array has unsupported shape for VGRF plotting: {grf.shape}")

            force_summary = ForceSummary(
                force_frames=frame_count_from_grf(grf),
                grf_shape=shape_string(grf),
                cop_shape=shape_string(cop),
                grf_candidate=grf_candidate,
            )

    if isinstance(npz, np.lib.npyio.NpzFile):
        npz.close()

    return force_summary, npz_frames


def inspect_force_for_summary(pair: MatchedPair) -> ForceSummary:
    """Load force data and summarize GRF/CoP fields for CSV output."""
    force_loaded = load_numpy_file(pair.force_path)
    if force_loaded is None:
        return ForceSummary(force_frames=None, grf_shape="", cop_shape="")

    force_value = normalize_loaded_npy(force_loaded)
    grf_candidate = locate_grf(force_value)
    cop_candidate = locate_force_array(force_value, "CoP")

    grf = grf_candidate[1] if grf_candidate is not None else None
    cop = cop_candidate[1] if cop_candidate is not None else None
    force_frames = frame_count_from_grf(grf)
    if force_frames is None:
        force_frames = frame_count_from_arrays(iter_named_arrays(force_value))

    return ForceSummary(
        force_frames=force_frames,
        grf_shape=shape_string(grf),
        cop_shape=shape_string(cop),
        grf_candidate=grf_candidate,
    )


def inspect_for_summary(pair: MatchedPair) -> tuple[ForceSummary, int | None]:
    """Load only enough data for CSV frame-count summary."""
    npz_frames: int | None = None

    npz = load_numpy_file(pair.npz_path)
    if isinstance(npz, np.lib.npyio.NpzFile):
        npz_frames = frame_count_from_arrays(iter_named_arrays(npz))
        npz.close()

    force_summary = inspect_force_for_summary(pair)
    return force_summary, npz_frames


def save_matched_csv(pairs: list[MatchedPair], csv_path: Path) -> tuple[int, int]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    rows_with_errors = 0

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stem",
                "force_stem",
                "npz_stem",
                "canonical_stem",
                "subject",
                "motion_type",
                "force_path",
                "npz_path",
                "force_frames",
                "npz_frames",
                "grf_shape",
                "cop_shape",
            ],
        )
        writer.writeheader()
        for pair in pairs:
            try:
                force_summary, npz_frames = inspect_for_summary(pair)
            except Exception as exc:  # noqa: BLE001
                warn(f"Could not summarize {pair.canonical_stem}: {exc}")
                force_summary = ForceSummary(force_frames=None, grf_shape="", cop_shape="")
                npz_frames = None
                rows_with_errors += 1

            subject, motion_type = parse_stem(pair.canonical_stem)
            writer.writerow(
                {
                    "stem": pair.canonical_stem,
                    "force_stem": pair.force_stem,
                    "npz_stem": pair.npz_stem,
                    "canonical_stem": pair.canonical_stem,
                    "subject": subject,
                    "motion_type": motion_type,
                    "force_path": str(pair.force_path),
                    "npz_path": str(pair.npz_path),
                    "force_frames": "" if force_summary.force_frames is None else force_summary.force_frames,
                    "npz_frames": "" if npz_frames is None else npz_frames,
                    "grf_shape": force_summary.grf_shape,
                    "cop_shape": force_summary.cop_shape,
                }
            )
            rows_written += 1

    return rows_written, rows_with_errors


def save_vgrf_plot(pair: MatchedPair, grf_name: str, grf: np.ndarray, figure_path: Path) -> bool:
    print(f"\nGRF shape selected for plotting: {grf.shape}")
    vertical = extract_vertical_grf(grf)
    if vertical is None:
        warn(
            f"Skipping VGRF plot for {pair.canonical_stem}; unsupported GRF shape: {grf.shape}. "
            "Supported shapes are [frames, 2, 3], [frames, 6], [2, frames, 3], and [frames, 3, 2]."
        )
        return False
    left_vertical, right_vertical, layout_note = vertical
    print(f"Using GRF layout: {layout_note}")

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not import matplotlib, so no plot was saved: {exc}")
        return False

    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(left_vertical, label="Left vertical GRF")
        ax.plot(right_vertical, label="Right vertical GRF")
        ax.set_title(f"Vertical GRF check: {pair.canonical_stem} ({grf_name})")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Vertical GRF")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not save VGRF plot for {pair.canonical_stem}: {exc}")
        return False

    print(f"\nSaved VGRF plot: {figure_path}")
    return True


def find_first_valid_grf(pairs: Iterable[MatchedPair]) -> tuple[MatchedPair, str, np.ndarray] | None:
    """Return the first matched sample with a supported GRF layout."""
    for pair in pairs:
        force_loaded = load_numpy_file(pair.force_path)
        if force_loaded is None:
            continue

        force_value = normalize_loaded_npy(force_loaded)
        grf_candidate = locate_grf(force_value)
        if grf_candidate is None:
            continue

        grf_name, grf = grf_candidate
        if extract_vertical_grf(grf) is not None:
            return pair, grf_name, grf

    return None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-dir", type=Path, default=DEFAULT_FORCE_DIR)
    parser.add_argument("--moshpp-dir", type=Path, default=DEFAULT_MOSHPP_DIR)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    force_dir = args.force_dir
    moshpp_dir = args.moshpp_dir
    print(f"Force folder: {force_dir.resolve()}")
    print(f"MoSh++ folder: {moshpp_dir.resolve()}")

    force_index = collect_files(force_dir, ".npy", prefer_non_soccerkick=True)
    npz_index = collect_files(moshpp_dir, ".npz", canonicalizer=canonical_npz_stem)
    force_files = force_index.files
    npz_files = npz_index.files

    force_stems = set(force_files)
    npz_stems = set(npz_files)
    matched_stems = sorted(force_stems & npz_stems)
    force_only = sorted(force_stems - npz_stems)
    npz_only = sorted(npz_stems - force_stems)
    pairs = [
        MatchedPair(
            canonical_stem=canonical_stem,
            force_stem=force_files[canonical_stem].raw_stem,
            npz_stem=npz_files[canonical_stem].raw_stem,
            force_path=force_files[canonical_stem].path,
            npz_path=npz_files[canonical_stem].path,
        )
        for canonical_stem in matched_stems
    ]

    print("\nDataset matching summary")
    print(f"  number of force files: {force_index.total_files}")
    print(f"  number of npz files: {npz_index.total_files}")
    print(f"  number of matched pairs: {len(pairs)}")
    print(f"  number of force-only files: {len(force_only)}")
    print(f"  number of npz-only files: {len(npz_only)}")

    if force_only[:5]:
        print(f"  first force-only stems: {force_only[:5]}")
    if npz_only[:5]:
        print(f"  first npz-only stems: {npz_only[:5]}")

    rng = random.Random(args.seed)
    sample_count = min(args.sample_count, len(pairs))
    sampled_pairs = rng.sample(pairs, sample_count) if sample_count else []

    first_valid_grf: tuple[MatchedPair, str, np.ndarray] | None = None
    sampled_with_load_errors = 0
    for pair in sampled_pairs:
        try:
            force_summary, _ = inspect_pair(pair)
        except Exception as exc:  # noqa: BLE001
            warn(f"Could not inspect sampled pair {pair.canonical_stem}: {exc}")
            sampled_with_load_errors += 1
            continue

        if first_valid_grf is None and force_summary.grf_candidate is not None:
            grf_name, grf = force_summary.grf_candidate
            if extract_vertical_grf(grf) is not None:
                first_valid_grf = (pair, grf_name, grf)

    if first_valid_grf is None:
        sampled_stems = {pair.canonical_stem for pair in sampled_pairs}
        remaining_pairs = [pair for pair in pairs if pair.canonical_stem not in sampled_stems]
        first_valid_grf = find_first_valid_grf(remaining_pairs)

    if first_valid_grf is not None:
        save_vgrf_plot(*first_valid_grf, figure_path=args.figure_path)
    else:
        warn("No force file contained GRF with a supported plotting shape; no plot saved.")

    rows_written, csv_errors = save_matched_csv(pairs, args.csv_path)
    print(f"\nSaved matched-file CSV: {args.csv_path} ({rows_written} rows)")

    print("\nRecommendation")
    if not pairs:
        print("  Dataset is not ready for preprocessing: no matched force/NPZ pairs were found.")
    elif first_valid_grf is None:
        print(
            "  Dataset is not ready for GRF preprocessing: matched pairs exist, but no "
            "force file exposed GRF with a supported shape."
        )
    else:
        print(
            "  Dataset appears ready for preprocessing: matched pairs exist and at least "
            "one force file contains a valid GRF array."
        )
        if force_only or npz_only:
            print(
                "  Note: unmatched files remain, so preprocessing should operate on the "
                "matched-file CSV or clean those extras separately."
            )
        if sampled_with_load_errors or csv_errors:
            print("  Note: some files could not be inspected or summarized and were skipped.")

    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        raise SystemExit(main())
