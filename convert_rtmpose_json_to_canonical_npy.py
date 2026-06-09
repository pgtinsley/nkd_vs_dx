#!/usr/bin/env python
"""Convert RTMPose JSON pose estimates to canonical COCO-17 NumPy arrays.

The output shape for each file is (n_frames, 17, 2). By default this processes
only the first five JSON files so the conversion can be checked before running
the full dataset.
"""

from __future__ import annotations

import csv
import json
import math
import argparse

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import savgol_filter


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


LEFT_SHOULDER  = COCO17_NAMES.index("left_shoulder")
RIGHT_SHOULDER = COCO17_NAMES.index("right_shoulder")
LEFT_HIP       = COCO17_NAMES.index("left_hip")
RIGHT_HIP      = COCO17_NAMES.index("right_hip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RTMPose JSON files to canonical smoothed COCO-17 .npy files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("pose_estimate_data_json"),
        help="Directory containing RTMPose JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pose_estimate_data_npy"),
        help="Directory where converted .npy files will be written.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Number of sorted JSON files to process. Use 0 or a negative value for all files.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=11,
        help="Savitzky-Golay smoothing window. 11 frames is a modest human-movement smoother at 30 fps.",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order.",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=None,
        help="Optional CSV or text file listing files to process. CSVs may contain .npy or .json names.",
    )
    parser.add_argument(
        "--file-list-column",
        default="file",
        help="Column to read when --file-list is a CSV.",
    )
    parser.add_argument(
        "--only-flagged",
        action="store_true",
        help="When --file-list is a CSV with a flagged column, process only rows where flagged is true.",
    )
    parser.add_argument(
        "--min-keypoint-score",
        type=float,
        default=0.20,
        help="Keypoints below this RTMPose confidence are interpolated before smoothing.",
    )
    parser.add_argument(
        "--min-torso-score",
        type=float,
        default=0.30,
        help="Torso anchor keypoints below this confidence make the frame anchor invalid.",
    )
    parser.add_argument(
        "--min-torso-fraction",
        type=float,
        default=0.25,
        help="Reject frame torso lengths below this fraction of the recording's median torso length.",
    )
    parser.add_argument(
        "--max-torso-fraction",
        type=float,
        default=4.00,
        help="Reject frame torso lengths above this multiple of the recording's median torso length.",
    )
    parser.add_argument(
        "--max-canonical-abs",
        type=float,
        default=8.0,
        help="Repair frames whose canonical coordinates exceed this absolute torso-normalized value.",
    )
    parser.add_argument(
        "--max-canonical-speed",
        type=float,
        default=100.0,
        help="Repair frames whose max keypoint speed exceeds this torso-normalized speed per second.",
    )
    parser.add_argument(
        "--skip-torso-rotation",
        action="store_true",
        help=(
            "Center on the pelvis and scale by torso length, but do not rotate the "
            "hip-to-shoulder axis to vertical. The y-axis is still flipped so upper "
            "body keypoints are positive in canonical coordinates."
        ),
    )
    return parser.parse_args()


def iter_instances(frames: Iterable[dict]) -> Iterable[dict]:
    for frame in frames:
        for instance in frame.get("instances", []):
            yield instance


def infer_frame_size(frames: list[dict]) -> tuple[float, float]:
    """Infer frame dimensions from bbox/keypoint coordinates when metadata is absent."""
    max_x = 0.0
    max_y = 0.0

    for instance in iter_instances(frames):
        bbox = instance.get("bbox")
        if bbox:
            box = np.asarray(bbox[0] if isinstance(bbox[0], list) else bbox, dtype=float)
            if box.size >= 4:
                max_x = max(max_x, float(box[2]))
                max_y = max(max_y, float(box[3]))

        keypoints = np.asarray(instance.get("keypoints", []), dtype=float)
        if keypoints.ndim == 2 and keypoints.shape[1] >= 2:
            max_x = max(max_x, float(np.nanmax(keypoints[:, 0])))
            max_y = max(max_y, float(np.nanmax(keypoints[:, 1])))

    return max(max_x, 1.0), max(max_y, 1.0)


def get_bbox(instance: dict, keypoints: np.ndarray) -> np.ndarray:
    bbox = instance.get("bbox")
    if bbox:
        box = np.asarray(bbox[0] if isinstance(bbox[0], list) else bbox, dtype=float)
        if box.size >= 4:
            return box[:4]

    valid = np.isfinite(keypoints[:, :2]).all(axis=1)
    if not np.any(valid):
        return np.array([0.0, 0.0, 0.0, 0.0])

    xy = keypoints[valid, :2]
    return np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])


def get_candidate(instance: dict, frame_size: tuple[float, float]) -> dict | None:
    keypoints = np.asarray(instance.get("keypoints", []), dtype=float)
    if keypoints.ndim != 2 or keypoints.shape[0] < 17 or keypoints.shape[1] < 2:
        return None

    keypoints = keypoints[:17, :2]
    keypoint_scores = np.asarray(instance.get("keypoint_scores", []), dtype=float)
    if keypoint_scores.size >= 17:
        keypoint_scores = keypoint_scores[:17]
    else:
        keypoint_scores = np.ones(17, dtype=float)

    box = get_bbox(instance, keypoints)
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    area = width * height
    bbox_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    frame_w, frame_h = frame_size
    frame_area = frame_w * frame_h
    frame_center = np.array([frame_w / 2.0, frame_h / 2.0])

    center_distance = np.linalg.norm((bbox_center - frame_center) / np.array([frame_w, frame_h]))
    center_score = 1.0 - np.clip(center_distance / math.sqrt(0.5), 0.0, 1.0)

    area_fraction = area / frame_area if frame_area > 0 else 0.0
    size_score = 1.0 - np.clip(area_fraction / 0.75, 0.0, 1.0)
    pose_score = float(np.nanmean(keypoint_scores))
    bbox_score = float(instance.get("bbox_score", 0.0))

    return {
        "keypoints": keypoints,
        "keypoint_scores": keypoint_scores,
        "bbox_center": bbox_center,
        "bbox_score": bbox_score,
        "pose_score": pose_score,
        "center_score": center_score,
        "size_score": size_score,
    }


def select_pose_candidates(frames: list[dict], frame_size: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Select a stable single subject, preferring the centered infant with temporal continuity."""
    frame_w, frame_h = frame_size
    norm = np.array([frame_w, frame_h])
    selected_keypoints = []
    selected_scores = []
    previous_center = None

    for frame in frames:
        candidates = [
            candidate
            for instance in frame.get("instances", [])
            if (candidate := get_candidate(instance, frame_size)) is not None
        ]

        if not candidates:
            selected_keypoints.append(np.full((17, 2), np.nan, dtype=float))
            selected_scores.append(np.zeros(17, dtype=float))
            continue

        best_candidate = None
        best_score = -math.inf
        for candidate in candidates:
            if previous_center is None:
                continuity_score = candidate["center_score"]
            else:
                continuity_distance = np.linalg.norm((candidate["bbox_center"] - previous_center) / norm)
                continuity_score = 1.0 - np.clip(continuity_distance / 0.35, 0.0, 1.0)

            score = (
                (0.25 * candidate["bbox_score"])
                + (0.25 * candidate["pose_score"])
                + (0.25 * candidate["center_score"])
                + (0.15 * continuity_score)
                + (0.10 * candidate["size_score"])
            )
            if score > best_score:
                best_score = score
                best_candidate = candidate

        selected_keypoints.append(best_candidate["keypoints"])
        selected_scores.append(best_candidate["keypoint_scores"])
        previous_center = best_candidate["bbox_center"]

    return np.asarray(selected_keypoints, dtype=np.float32), np.asarray(selected_scores, dtype=np.float32)


def fill_missing_1d(values: np.ndarray) -> np.ndarray:
    values = values.astype(float, copy=True)
    finite = np.isfinite(values)

    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)

    x = np.arange(values.size)
    values[~finite] = np.interp(x[~finite], x[finite], values[finite])
    return values


def adjusted_savgol_window(n_frames: int, requested_window: int, polyorder: int) -> int | None:
    if n_frames <= polyorder + 1:
        return None

    window = min(requested_window, n_frames)
    if window % 2 == 0:
        window -= 1
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        return None
    return window


def smooth_keypoints(keypoints: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    smoothed = keypoints.astype(float, copy=True)
    window = adjusted_savgol_window(smoothed.shape[0], window_length, polyorder)

    for point_idx in range(smoothed.shape[1]):
        for coord_idx in range(smoothed.shape[2]):
            series = fill_missing_1d(smoothed[:, point_idx, coord_idx])
            if window is not None:
                series = savgol_filter(series, window_length=window, polyorder=polyorder, mode="interp")
            smoothed[:, point_idx, coord_idx] = series

    return smoothed


def get_midpoints(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mid_hip = (keypoints[:, LEFT_HIP, :] + keypoints[:, RIGHT_HIP, :]) / 2.0
    mid_shoulder = (keypoints[:, LEFT_SHOULDER, :] + keypoints[:, RIGHT_SHOULDER, :]) / 2.0
    torso = mid_shoulder - mid_hip
    torso_length = np.linalg.norm(torso, axis=1)
    return mid_hip, mid_shoulder, torso, torso_length


def get_valid_torso_mask(
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_torso_score: float,
    min_torso_fraction: float,
    max_torso_fraction: float,
) -> np.ndarray:
    _, _, _, torso_length = get_midpoints(keypoints)
    torso_idxs = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    finite_torso = np.isfinite(keypoints[:, torso_idxs, :]).all(axis=(1, 2))
    confident_torso = (scores[:, torso_idxs] >= min_torso_score).all(axis=1)
    positive_torso = np.isfinite(torso_length) & (torso_length > 1e-6)
    initially_valid = finite_torso & confident_torso & positive_torso

    if np.any(initially_valid):
        median_torso = float(np.nanmedian(torso_length[initially_valid]))
    else:
        median_torso = float(np.nanmedian(torso_length[positive_torso])) if np.any(positive_torso) else 1.0

    lower = max(median_torso * min_torso_fraction, 1e-6)
    upper = max(median_torso * max_torso_fraction, lower)
    plausible_torso = (torso_length >= lower) & (torso_length <= upper)
    return finite_torso & confident_torso & positive_torso & plausible_torso


def prepare_keypoints_for_canonicalization(
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_keypoint_score: float,
    min_torso_score: float,
    min_torso_fraction: float,
    max_torso_fraction: float,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    prepared = keypoints.astype(float, copy=True)
    prepared[scores < min_keypoint_score] = np.nan

    valid_torso = get_valid_torso_mask(
        keypoints,
        scores,
        min_torso_score=min_torso_score,
        min_torso_fraction=min_torso_fraction,
        max_torso_fraction=max_torso_fraction,
    )
    prepared[~valid_torso] = np.nan
    return smooth_keypoints(prepared, window_length=window_length, polyorder=polyorder)


def get_stable_anchor(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid_hip, _, torso, torso_length = get_midpoints(keypoints)
    positive = np.isfinite(torso_length) & (torso_length > 1e-6)
    median_torso = float(np.nanmedian(torso_length[positive])) if np.any(positive) else 1.0
    valid_anchor = positive & (torso_length >= median_torso * 0.25) & (torso_length <= median_torso * 4.0)

    stable_mid_hip = mid_hip.astype(float, copy=True)
    stable_torso = torso.astype(float, copy=True)
    stable_mid_hip[~valid_anchor] = np.nan
    stable_torso[~valid_anchor] = np.nan

    for coord_idx in range(2):
        stable_mid_hip[:, coord_idx] = fill_missing_1d(stable_mid_hip[:, coord_idx])
        stable_torso[:, coord_idx] = fill_missing_1d(stable_torso[:, coord_idx])

    stable_torso_length = np.linalg.norm(stable_torso, axis=1)
    bad_length = ~np.isfinite(stable_torso_length) | (stable_torso_length < median_torso * 0.25)
    stable_torso[bad_length] = np.array([0.0, median_torso])
    stable_torso_length = np.linalg.norm(stable_torso, axis=1)
    stable_torso_length = np.nan_to_num(stable_torso_length, nan=median_torso, posinf=median_torso, neginf=median_torso)
    stable_torso_length[stable_torso_length < 1e-6] = median_torso
    return stable_mid_hip, stable_torso, stable_torso_length


def canonicalize_keypoints(keypoints: np.ndarray, rotate_torso_to_upright: bool = True) -> np.ndarray:
    canonical = keypoints.astype(float, copy=True)

    mid_hip, torso, torso_length = get_stable_anchor(canonical)

    centered = canonical - mid_hip[:, None, :]
    x = centered[:, :, 0]
    y = centered[:, :, 1]

    if not rotate_torso_to_upright:
        canonical[:, :, 0] = x / torso_length[:, None]
        canonical[:, :, 1] = -y / torso_length[:, None]
        return canonical

    # Rotate so the hip-to-shoulder axis points upward in canonical coordinates.
    angles = (math.pi / 2.0) - np.arctan2(torso[:, 1], torso[:, 0])
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    rotated_x = (x * cos_angles[:, None]) - (y * sin_angles[:, None])
    rotated_y = (x * sin_angles[:, None]) + (y * cos_angles[:, None])

    canonical[:, :, 0] = rotated_x / torso_length[:, None]
    canonical[:, :, 1] = rotated_y / torso_length[:, None]
    return canonical


def get_bad_canonical_frames(canonical: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    bad_frames = ~np.isfinite(canonical).all(axis=(1, 2)) | (
        np.nanmax(np.abs(canonical), axis=(1, 2)) > args.max_canonical_abs
    )
    if canonical.shape[0] > 1:
        per_step_speed = np.nanmax(np.linalg.norm(np.diff(canonical, axis=0), axis=2), axis=1) * 30.0
        bad_steps = per_step_speed > args.max_canonical_speed
        bad_frames[1:][bad_steps] = True
        bad_frames[:-1][bad_steps] = True
    return bad_frames


def repair_canonical_outliers(
    keypoints: np.ndarray,
    scores: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    prepared = prepare_keypoints_for_canonicalization(
        keypoints,
        scores,
        min_keypoint_score=args.min_keypoint_score,
        min_torso_score=args.min_torso_score,
        min_torso_fraction=args.min_torso_fraction,
        max_torso_fraction=args.max_torso_fraction,
        window_length=args.window_length,
        polyorder=args.polyorder,
    )
    rotate_torso_to_upright = not args.skip_torso_rotation
    canonical = canonicalize_keypoints(prepared, rotate_torso_to_upright=rotate_torso_to_upright)

    repaired = prepared.copy()
    for _ in range(5):
        bad_frames = get_bad_canonical_frames(canonical, args)
        if not np.any(bad_frames):
            return smooth_keypoints(canonical, window_length=args.window_length, polyorder=args.polyorder)
        repaired[bad_frames] = np.nan
        repaired = smooth_keypoints(repaired, window_length=args.window_length, polyorder=args.polyorder)
        canonical = canonicalize_keypoints(repaired, rotate_torso_to_upright=rotate_torso_to_upright)

    remaining_bad = ~np.isfinite(canonical).all(axis=(1, 2))
    if np.any(remaining_bad):
        canonical[remaining_bad] = 0.0
    return smooth_keypoints(canonical, window_length=args.window_length, polyorder=args.polyorder)


def load_selected_keypoints(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with json_path.open("r", encoding="utf-8") as file:
        frames = json.load(file)

    frame_size = infer_frame_size(frames)
    return select_pose_candidates(frames, frame_size)


def convert_file(
    json_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    keypoints, scores = load_selected_keypoints(json_path)
    keypoints = repair_canonical_outliers(keypoints, scores, args)

    output_path = output_dir / f"{json_path.stem}.npy"
    np.save(output_path, keypoints.astype(np.float32))
    return output_path


def get_json_paths(args: argparse.Namespace) -> list[Path]:
    json_paths_by_stem = {path.stem: path for path in args.input_dir.glob("*.json")}
    if args.file_list is None:
        json_paths = sorted(json_paths_by_stem.values())
        if args.max_files > 0:
            json_paths = json_paths[: args.max_files]
        return json_paths

    requested_names = []
    if args.file_list.suffix.lower() == ".csv":
        with args.file_list.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if args.only_flagged and row.get("flagged", "").lower() not in {"true", "1", "yes"}:
                    continue
                requested_names.append(row[args.file_list_column])
    else:
        with args.file_list.open("r", encoding="utf-8") as file:
            requested_names = [line.strip() for line in file if line.strip()]

    json_paths = []
    missing = []
    for name in requested_names:
        stem = Path(name).stem
        path = json_paths_by_stem.get(stem)
        if path is None:
            missing.append(name)
        else:
            json_paths.append(path)

    if missing:
        print(f"Warning: {len(missing)} listed files were not found in {args.input_dir}")
    if args.max_files > 0:
        json_paths = json_paths[: args.max_files]
    return json_paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = get_json_paths(args)
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {args.input_dir}")

    for json_path in json_paths:
        output_path = convert_file(
            json_path,
            args.output_dir,
            args=args,
        )
        arr = np.load(output_path, mmap_mode="r")
        print(f"Wrote {output_path} {arr.shape}")


if __name__ == "__main__":
    main()
