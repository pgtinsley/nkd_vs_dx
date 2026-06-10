#!/usr/bin/env python
"""Flag pose .npy arrays with physiologically implausible movement statistics.

The QC pass reads only pose arrays for flag decisions. If metadata is supplied,
only ``video_stem`` and ``final_code_for_ai_str`` are copied into the output for
review context; existing metadata flag columns are intentionally ignored.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


COCO17_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

DISTAL_KEYPOINTS = (
    COCO17_NAMES.index("left_wrist"),
    COCO17_NAMES.index("right_wrist"),
    COCO17_NAMES.index("left_ankle"),
    COCO17_NAMES.index("right_ankle"),
)
LIMB_SEGMENTS = (
    ("left_upper_arm", "left_shoulder", "left_elbow"),
    ("left_forearm", "left_elbow", "left_wrist"),
    ("right_upper_arm", "right_shoulder", "right_elbow"),
    ("right_forearm", "right_elbow", "right_wrist"),
    ("left_thigh", "left_hip", "left_knee"),
    ("left_shin", "left_knee", "left_ankle"),
    ("right_thigh", "right_hip", "right_knee"),
    ("right_shin", "right_knee", "right_ankle"),
)
METRIC_FIELDS = (
    "n_frames",
    "finite_fraction",
    "max_abs_coord",
    "p99_abs_coord",
    "out_of_bounds_fraction",
    "max_distal_speed",
    "p99_distal_speed",
    "max_all_keypoint_speed",
    "p99_all_keypoint_speed",
    "large_distal_speed_frame_fraction",
    "max_limb_length",
    "p99_limb_length",
    "max_limb_length_change",
    "p99_limb_length_change",
    "zero_frame_fraction",
    "duplicate_frame_fraction",
)
STAT_OUTLIER_METRICS = (
    "p99_abs_coord",
    "out_of_bounds_fraction",
    "p99_distal_speed",
    "max_distal_speed",
    "large_distal_speed_frame_fraction",
    "p99_limb_length",
    "p99_limb_length_change",
    "zero_frame_fraction",
    "duplicate_frame_fraction",
)


@dataclass(frozen=True)
class MetadataRow:
    final_code_for_ai_str: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute array-only pose QC metrics and flag robust statistical "
            "outliers for physiological infeasibility."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("pose_estimate_data_npy"),
        help="Directory containing one COCO-17 pose array per video.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("qc_pose_npy_statistical_outliers.csv"),
        help="CSV path for per-file metrics and flags.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("df_meta.csv"),
        help=(
            "Optional metadata CSV. Only video_stem and final_code_for_ai_str "
            "are read; any flag columns are ignored. Use --metadata-csv '' to skip."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate used to convert frame-to-frame displacement into speed.",
    )
    parser.add_argument(
        "--max-abs-coord",
        type=float,
        default=8.0,
        help="Absolute coordinate value beyond which a keypoint is out of bounds.",
    )
    parser.add_argument(
        "--max-distal-speed",
        type=float,
        default=100.0,
        help="Absolute distal-keypoint speed threshold in coordinate units per second.",
    )
    parser.add_argument(
        "--max-limb-length",
        type=float,
        default=8.0,
        help="Absolute limb segment length threshold in coordinate units.",
    )
    parser.add_argument(
        "--large-speed-threshold",
        type=float,
        default=30.0,
        help="Distal speed used to count suspicious high-speed frames.",
    )
    parser.add_argument(
        "--robust-z-threshold",
        type=float,
        default=6.0,
        help="Flag metrics whose robust z-score exceeds this threshold.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit the number of sorted .npy files processed. Use 0 for all.",
    )
    return parser.parse_args()


def finite_or_nan(values: np.ndarray, reducer, default: float = math.nan) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(reducer(finite))


def percentile(values: np.ndarray, q: float) -> float:
    return finite_or_nan(values, lambda x: np.percentile(x, q))


def fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return math.nan
    return float(np.count_nonzero(mask) / mask.size)


def load_metadata(metadata_csv: Path | None) -> dict[str, MetadataRow]:
    if metadata_csv is None or not metadata_csv.exists():
        return {}

    metadata: dict[str, MetadataRow] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_stem = row.get("video_stem", "")
            if not video_stem:
                continue
            metadata[video_stem] = MetadataRow(
                final_code_for_ai_str=row.get("final_code_for_ai_str", "")
            )
    return metadata


def load_pose_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[1:] != (17, 2):
        raise ValueError(f"expected shape (n_frames, 17, 2), got {arr.shape}")
    return arr.astype(float, copy=False)


def limb_lengths(arr: np.ndarray) -> np.ndarray:
    lengths = []
    for _, start_name, end_name in LIMB_SEGMENTS:
        start = COCO17_NAMES.index(start_name)
        end = COCO17_NAMES.index(end_name)
        lengths.append(np.linalg.norm(arr[:, start, :] - arr[:, end, :], axis=1))
    return np.stack(lengths, axis=1) if lengths else np.empty((arr.shape[0], 0))


def frame_duplicate_fraction(arr: np.ndarray) -> float:
    if arr.shape[0] < 2:
        return 0.0
    diffs = np.linalg.norm(np.diff(arr, axis=0), axis=2)
    duplicate_steps = np.nanmax(diffs, axis=1) <= 1e-8
    return fraction(duplicate_steps)


def compute_metrics(path: Path, fps: float, max_abs_coord: float, large_speed_threshold: float) -> dict[str, float | str]:
    arr = load_pose_array(path)
    finite_keypoint = np.isfinite(arr).all(axis=2)
    finite_frame = np.isfinite(arr).all(axis=(1, 2))
    abs_coord = np.abs(arr)
    out_of_bounds = np.isfinite(abs_coord) & (abs_coord > max_abs_coord)

    if arr.shape[0] > 1:
        step_speed = np.linalg.norm(np.diff(arr, axis=0), axis=2) * fps
        distal_speed = step_speed[:, DISTAL_KEYPOINTS]
        large_distal_frames = np.nanmax(distal_speed, axis=1) > large_speed_threshold
    else:
        step_speed = np.empty((0, 17), dtype=float)
        distal_speed = np.empty((0, len(DISTAL_KEYPOINTS)), dtype=float)
        large_distal_frames = np.empty(0, dtype=bool)

    lengths = limb_lengths(arr)
    if lengths.shape[0] > 1:
        length_change = np.abs(np.diff(lengths, axis=0)) * fps
    else:
        length_change = np.empty((0, lengths.shape[1]), dtype=float)

    zero_frames = finite_frame & (np.nanmax(np.abs(arr), axis=(1, 2)) <= 1e-8)

    return {
        "file": path.name,
        "video_stem": path.stem,
        "n_frames": int(arr.shape[0]),
        "finite_fraction": fraction(finite_keypoint),
        "max_abs_coord": finite_or_nan(abs_coord, np.max),
        "p99_abs_coord": percentile(abs_coord, 99),
        "out_of_bounds_fraction": fraction(out_of_bounds),
        "max_distal_speed": finite_or_nan(distal_speed, np.max),
        "p99_distal_speed": percentile(distal_speed, 99),
        "max_all_keypoint_speed": finite_or_nan(step_speed, np.max),
        "p99_all_keypoint_speed": percentile(step_speed, 99),
        "large_distal_speed_frame_fraction": fraction(large_distal_frames),
        "max_limb_length": finite_or_nan(lengths, np.max),
        "p99_limb_length": percentile(lengths, 99),
        "max_limb_length_change": finite_or_nan(length_change, np.max),
        "p99_limb_length_change": percentile(length_change, 99),
        "zero_frame_fraction": fraction(zero_frames),
        "duplicate_frame_fraction": frame_duplicate_fraction(arr),
    }


def robust_z_scores(rows: list[dict[str, float | str]], metric: str) -> dict[str, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    valid = np.isfinite(values)
    z_scores = np.full(values.shape, np.nan, dtype=float)
    if np.count_nonzero(valid) < 4:
        return {str(row["file"]): float(z) for row, z in zip(rows, z_scores)}

    median = float(np.median(values[valid]))
    mad = float(np.median(np.abs(values[valid] - median)))
    if mad <= 1e-12:
        spread = float(np.std(values[valid]))
        if spread <= 1e-12:
            z_scores[valid] = 0.0
        else:
            z_scores[valid] = (values[valid] - median) / spread
    else:
        z_scores[valid] = 0.6745 * (values[valid] - median) / mad
    return {str(row["file"]): float(z) for row, z in zip(rows, z_scores)}


def add_flags(
    rows: list[dict[str, float | str]],
    *,
    max_abs_coord: float,
    max_distal_speed: float,
    max_limb_length: float,
    robust_z_threshold: float,
) -> None:
    metric_z_by_file = {
        metric: robust_z_scores(rows, metric)
        for metric in STAT_OUTLIER_METRICS
    }

    for row in rows:
        reasons = []
        if float(row["finite_fraction"]) < 1.0:
            reasons.append("nonfinite_coordinates")
        if float(row["max_abs_coord"]) > max_abs_coord:
            reasons.append("coordinate_out_of_bounds")
        if float(row["max_distal_speed"]) > max_distal_speed:
            reasons.append("distal_speed_above_absolute_threshold")
        if float(row["max_limb_length"]) > max_limb_length:
            reasons.append("limb_length_above_absolute_threshold")
        if float(row["n_frames"]) < 2:
            reasons.append("too_few_frames")

        z_fields = []
        for metric, z_by_file in metric_z_by_file.items():
            z = z_by_file[str(row["file"])]
            row[f"{metric}_robust_z"] = z
            if np.isfinite(z) and z > robust_z_threshold:
                reasons.append(f"{metric}_statistical_outlier")
                z_fields.append(f"{metric}={z:.2f}")

        row["statistical_outlier_metrics"] = ";".join(z_fields)
        row["flagged"] = bool(reasons)
        row["flag_reasons"] = ";".join(dict.fromkeys(reasons))


def format_value(value: float | int | str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def write_csv(rows: list[dict[str, float | str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    z_fields = [f"{metric}_robust_z" for metric in STAT_OUTLIER_METRICS]
    fieldnames = [
        "file",
        "video_stem",
        "final_code_for_ai_str",
        *METRIC_FIELDS,
        *z_fields,
        "flagged",
        "flag_reasons",
        "statistical_outlier_metrics",
        "error",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: format_value(row.get(name, "")) for name in fieldnames})


def iter_npy_files(input_dir: Path, max_files: int) -> Iterable[Path]:
    paths = sorted(input_dir.glob("*.npy"))
    if max_files > 0:
        paths = paths[:max_files]
    return paths


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {args.input_dir}")

    metadata_csv = None if str(args.metadata_csv) == "" else args.metadata_csv
    metadata = load_metadata(metadata_csv)
    rows: list[dict[str, float | str]] = []

    for path in iter_npy_files(args.input_dir, args.max_files):
        try:
            row = compute_metrics(
                path,
                fps=args.fps,
                max_abs_coord=args.max_abs_coord,
                large_speed_threshold=args.large_speed_threshold,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch QC moving.
            row = {
                "file": path.name,
                "video_stem": path.stem,
                "error": str(exc),
                "flagged": True,
                "flag_reasons": "load_or_shape_error",
            }
            for metric in METRIC_FIELDS:
                row.setdefault(metric, math.nan)

        meta = metadata.get(path.stem, MetadataRow())
        row["final_code_for_ai_str"] = meta.final_code_for_ai_str
        row.setdefault("error", "")
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"no .npy files found in {args.input_dir}")

    valid_rows = [row for row in rows if not row.get("error")]
    if valid_rows:
        add_flags(
            valid_rows,
            max_abs_coord=args.max_abs_coord,
            max_distal_speed=args.max_distal_speed,
            max_limb_length=args.max_limb_length,
            robust_z_threshold=args.robust_z_threshold,
        )

    write_csv(rows, args.output_csv)

    flagged_count = sum(row.get("flagged") is True for row in rows)
    print(f"Processed {len(rows)} files from {args.input_dir}")
    print(f"Flagged {flagged_count} files")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
