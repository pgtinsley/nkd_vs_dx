#!/usr/bin/env python
"""Build framewise MiniRocket channels from non-rotated canonical COCO-17 poses.

Input pose arrays are expected to have shape (frames, 17, 2), centered at the
pelvis and torso-scaled by convert_rtmpose_json_to_canonical_npy.py with
--skip-torso-rotation. Output feature arrays are saved as (channels, frames),
which is the tsai-friendly layout used by the existing feature builders.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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

IDX = {name: i for i, name in enumerate(COCO17_NAMES)}
DISTAL = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
PAIRWISE_DISTAL = (
    ("wrist_wrist", "left_wrist", "right_wrist"),
    ("ankle_ankle", "left_ankle", "right_ankle"),
    ("left_wrist_left_ankle", "left_wrist", "left_ankle"),
    ("right_wrist_right_ankle", "right_wrist", "right_ankle"),
    ("left_wrist_right_ankle", "left_wrist", "right_ankle"),
    ("right_wrist_left_ankle", "right_wrist", "left_ankle"),
)

DROP_REPETITIVE_CHANNELS = {
    "left_wrist_midline_crossing_indicator",
    "left_wrist_contralateral_indicator",
    "right_wrist_midline_crossing_indicator",
    "right_wrist_contralateral_indicator",
    "left_ankle_midline_crossing_indicator",
    "left_ankle_contralateral_indicator",
    "right_ankle_midline_crossing_indicator",
    "right_ankle_contralateral_indicator",
    "shoulder_midpoint_x",
    "shoulder_midpoint_vx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("pose_estimate_data_npy_codex_notRotated"))
    parser.add_argument("--meta-csv", type=Path, default=Path("df_meta_constructed_notRotated_withQCFlags.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("features_notRotated_withQCFlags"))
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def is_flagged(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def first_derivative(values: np.ndarray, fps: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] > 1:
        out[1:] = np.diff(values, axis=0) * fps
    return out


def norm(values: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(values, axis=axis).astype(np.float32)


def unwrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.unwrap(angle.astype(np.float64)).astype(np.float32)


def angle_velocity(angle: np.ndarray, fps: float) -> np.ndarray:
    return first_derivative(unwrap_angle(angle), fps=fps)


def wrapped_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b)).astype(np.float32)


def safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (num / (den + eps)).astype(np.float32)


def distal_spread_area(points: np.ndarray) -> np.ndarray:
    """Framewise polygon area after ordering the four distal points by angle."""
    centroid = points.mean(axis=1, keepdims=True)
    theta = np.arctan2(points[:, :, 1] - centroid[:, :, 1], points[:, :, 0] - centroid[:, :, 0])
    order = np.argsort(theta, axis=1)
    ordered = np.take_along_axis(points, order[:, :, None], axis=1)
    x = ordered[:, :, 0]
    y = ordered[:, :, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))
    return area.astype(np.float32)


def movement_turning_angle(velocity: np.ndarray, speed: np.ndarray) -> np.ndarray:
    unit = velocity / np.maximum(speed[:, :, None], 1e-6)
    cos_turn = np.sum(unit[1:] * unit[:-1], axis=2)
    enough_motion = (speed[1:] > 0.05) & (speed[:-1] > 0.05)
    turn = np.zeros_like(speed, dtype=np.float32)
    turn[1:] = np.where(enough_motion, np.arccos(np.clip(cos_turn, -1.0, 1.0)), 0.0)
    return turn


def build_feature_names() -> list[str]:
    names: list[str] = []
    for keypoint in DISTAL:
        names.extend(
            [
                f"{keypoint}_x",
                f"{keypoint}_y",
                f"{keypoint}_vx",
                f"{keypoint}_vy",
                f"{keypoint}_ax",
                f"{keypoint}_ay",
                f"{keypoint}_speed",
                f"{keypoint}_acceleration_magnitude",
                f"{keypoint}_jerk_magnitude",
                f"{keypoint}_radial_distance",
                f"{keypoint}_radial_velocity",
                f"{keypoint}_movement_direction_angle",
                f"{keypoint}_turning_angle",
                f"{keypoint}_midline_crossing_indicator",
                f"{keypoint}_contralateral_indicator",
            ]
        )

    names.extend([f"{name}_distance" for name, _, _ in PAIRWISE_DISTAL])
    names.extend(
        [
            "mean_distal_radius",
            "distal_spread_area",
            "left_right_wrist_speed_difference",
            "left_right_wrist_speed_abs_difference",
            "left_right_wrist_speed_product",
            "left_right_ankle_speed_difference",
            "left_right_ankle_speed_abs_difference",
            "left_right_ankle_speed_product",
            "left_right_wrist_radial_distance_difference",
            "left_right_ankle_radial_distance_difference",
            "mean_distal_speed",
            "max_distal_speed",
            "distal_speed_standard_deviation",
            "upper_lower_speed_ratio",
            "left_ipsilateral_wrist_ankle_speed_difference",
            "left_ipsilateral_wrist_ankle_speed_product",
            "right_ipsilateral_wrist_ankle_speed_difference",
            "right_ipsilateral_wrist_ankle_speed_product",
            "left_wrist_right_ankle_speed_difference",
            "left_wrist_right_ankle_speed_product",
            "right_wrist_left_ankle_speed_difference",
            "right_wrist_left_ankle_speed_product",
            "shoulder_midpoint_x",
            "shoulder_midpoint_y",
            "shoulder_midpoint_vx",
            "shoulder_midpoint_vy",
            "shoulder_midpoint_speed",
            "torso_angle",
            "torso_angular_velocity",
            "torso_angular_acceleration",
            "shoulder_line_angle",
            "shoulder_line_angular_velocity",
            "hip_line_angle",
            "hip_line_angular_velocity",
            "shoulder_width",
            "hip_width",
            "shoulder_hip_width_ratio",
            "left_right_shoulder_height_difference",
            "left_right_hip_height_difference",
            "shoulder_hip_angle_difference",
            "torso_angular_velocity_x_mean_distal_speed",
            "torso_angular_velocity_x_left_wrist_speed",
            "torso_angular_velocity_x_right_wrist_speed",
            "torso_angular_velocity_x_left_ankle_speed",
            "torso_angular_velocity_x_right_ankle_speed",
            "shoulder_midpoint_speed_over_mean_distal_speed",
            "shoulder_midpoint_speed_minus_mean_distal_speed",
            "torso_angular_acceleration_x_mean_distal_speed",
            "distal_spread_area_x_torso_angular_velocity",
        ]
    )
    return names


ALL_FEATURE_NAMES = build_feature_names()
FEATURE_NAMES = [name for name in ALL_FEATURE_NAMES if name not in DROP_REPETITIVE_CHANNELS]


def engineer_features(pose: np.ndarray, fps: float) -> np.ndarray:
    if pose.ndim != 3 or pose.shape[1:] != (17, 2):
        raise ValueError(f"Expected pose shape (frames, 17, 2), got {pose.shape}")

    pose = pose.astype(np.float32, copy=False)
    distal_xy = np.stack([pose[:, IDX[name], :] for name in DISTAL], axis=1)
    velocity = first_derivative(distal_xy, fps=fps)
    acceleration = first_derivative(velocity, fps=fps)
    jerk = first_derivative(acceleration, fps=fps)

    speed = norm(velocity, axis=2)
    acceleration_mag = norm(acceleration, axis=2)
    jerk_mag = norm(jerk, axis=2)
    radius = norm(distal_xy, axis=2)
    radial_velocity = first_derivative(radius, fps=fps)
    direction = np.arctan2(velocity[:, :, 1], velocity[:, :, 0]).astype(np.float32)
    turning = movement_turning_angle(velocity, speed)

    midline_crossing = np.zeros_like(radius, dtype=np.float32)
    midline_crossing[1:] = (distal_xy[1:, :, 0] * distal_xy[:-1, :, 0] < 0).astype(np.float32)
    contralateral = np.stack(
        [
            distal_xy[:, 0, 0] < 0.0,
            distal_xy[:, 1, 0] > 0.0,
            distal_xy[:, 2, 0] < 0.0,
            distal_xy[:, 3, 0] > 0.0,
        ],
        axis=1,
    ).astype(np.float32)

    channels: list[np.ndarray] = []
    for i in range(len(DISTAL)):
        channels.extend(
            [
                distal_xy[:, i, 0],
                distal_xy[:, i, 1],
                velocity[:, i, 0],
                velocity[:, i, 1],
                acceleration[:, i, 0],
                acceleration[:, i, 1],
                speed[:, i],
                acceleration_mag[:, i],
                jerk_mag[:, i],
                radius[:, i],
                radial_velocity[:, i],
                direction[:, i],
                turning[:, i],
                midline_crossing[:, i],
                contralateral[:, i],
            ]
        )

    for _, a, b in PAIRWISE_DISTAL:
        channels.append(norm(pose[:, IDX[a], :] - pose[:, IDX[b], :]))

    mean_distal_radius = radius.mean(axis=1).astype(np.float32)
    spread_area = distal_spread_area(distal_xy)
    mean_distal_speed = speed.mean(axis=1).astype(np.float32)
    max_distal_speed = speed.max(axis=1).astype(np.float32)
    distal_speed_std = speed.std(axis=1).astype(np.float32)
    wrist_mean_speed = (speed[:, 0] + speed[:, 1]) / 2.0
    ankle_mean_speed = (speed[:, 2] + speed[:, 3]) / 2.0

    channels.extend(
        [
            mean_distal_radius,
            spread_area,
            speed[:, 0] - speed[:, 1],
            np.abs(speed[:, 0] - speed[:, 1]).astype(np.float32),
            speed[:, 0] * speed[:, 1],
            speed[:, 2] - speed[:, 3],
            np.abs(speed[:, 2] - speed[:, 3]).astype(np.float32),
            speed[:, 2] * speed[:, 3],
            radius[:, 0] - radius[:, 1],
            radius[:, 2] - radius[:, 3],
            mean_distal_speed,
            max_distal_speed,
            distal_speed_std,
            safe_ratio(wrist_mean_speed, ankle_mean_speed),
            speed[:, 0] - speed[:, 2],
            speed[:, 0] * speed[:, 2],
            speed[:, 1] - speed[:, 3],
            speed[:, 1] * speed[:, 3],
            speed[:, 0] - speed[:, 3],
            speed[:, 0] * speed[:, 3],
            speed[:, 1] - speed[:, 2],
            speed[:, 1] * speed[:, 2],
        ]
    )

    left_shoulder = pose[:, IDX["left_shoulder"], :]
    right_shoulder = pose[:, IDX["right_shoulder"], :]
    left_hip = pose[:, IDX["left_hip"], :]
    right_hip = pose[:, IDX["right_hip"], :]
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    mid_hip = (left_hip + right_hip) / 2.0

    shoulder_velocity = first_derivative(mid_shoulder, fps=fps)
    shoulder_speed = norm(shoulder_velocity)
    torso_vec = mid_shoulder - mid_hip
    torso_angle = np.arctan2(torso_vec[:, 1], torso_vec[:, 0]).astype(np.float32)
    torso_angular_velocity = angle_velocity(torso_angle, fps=fps)
    torso_angular_acceleration = first_derivative(torso_angular_velocity, fps=fps)
    shoulder_line = left_shoulder - right_shoulder
    hip_line = left_hip - right_hip
    shoulder_line_angle = np.arctan2(shoulder_line[:, 1], shoulder_line[:, 0]).astype(np.float32)
    hip_line_angle = np.arctan2(hip_line[:, 1], hip_line[:, 0]).astype(np.float32)
    shoulder_line_angular_velocity = angle_velocity(shoulder_line_angle, fps=fps)
    hip_line_angular_velocity = angle_velocity(hip_line_angle, fps=fps)
    shoulder_width = norm(shoulder_line)
    hip_width = norm(hip_line)
    shoulder_hip_angle_difference = wrapped_angle_diff(shoulder_line_angle, hip_line_angle)

    channels.extend(
        [
            mid_shoulder[:, 0],
            mid_shoulder[:, 1],
            shoulder_velocity[:, 0],
            shoulder_velocity[:, 1],
            shoulder_speed,
            unwrap_angle(torso_angle),
            torso_angular_velocity,
            torso_angular_acceleration,
            unwrap_angle(shoulder_line_angle),
            shoulder_line_angular_velocity,
            unwrap_angle(hip_line_angle),
            hip_line_angular_velocity,
            shoulder_width,
            hip_width,
            safe_ratio(shoulder_width, hip_width),
            left_shoulder[:, 1] - right_shoulder[:, 1],
            left_hip[:, 1] - right_hip[:, 1],
            shoulder_hip_angle_difference,
            torso_angular_velocity * mean_distal_speed,
            torso_angular_velocity * speed[:, 0],
            torso_angular_velocity * speed[:, 1],
            torso_angular_velocity * speed[:, 2],
            torso_angular_velocity * speed[:, 3],
            safe_ratio(shoulder_speed, mean_distal_speed),
            shoulder_speed - mean_distal_speed,
            torso_angular_acceleration * mean_distal_speed,
            spread_area * torso_angular_velocity,
        ]
    )

    full_features = np.vstack(channels).astype(np.float32)
    keep_indices = [i for i, name in enumerate(ALL_FEATURE_NAMES) if name not in DROP_REPETITIVE_CHANNELS]
    features = full_features[keep_indices]
    if features.shape != (len(FEATURE_NAMES), pose.shape[0]):
        raise ValueError(f"Feature shape {features.shape} does not match expected {(len(FEATURE_NAMES), pose.shape[0])}")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def metadata_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = metadata_rows(args.meta_csv)
    selected = [row for row in rows if row.get("file") and not is_flagged(row.get("flagged"))]

    manifest = []
    missing = []
    for row in selected:
        pose_path = args.pose_dir / row["file"]
        if not pose_path.exists():
            missing.append(row["file"])
            continue

        pose = np.load(pose_path)
        features = engineer_features(pose, fps=args.fps)
        out_path = args.output_dir / pose_path.name
        np.save(out_path, features)
        manifest.append(
            {
                "file": pose_path.name,
                "pose_path": str(pose_path),
                "feature_path": str(out_path),
                "frames": str(pose.shape[0]),
                "n_channels": str(features.shape[0]),
                "feature_layout": "channels_first",
                "flagged": str(row.get("flagged", "")),
                "flag_reasons": str(row.get("flag_reasons", "")),
                "record_id": str(row.get("record_id", "")),
                "subject_unique_id": str(row.get("subject_unique_id", "")),
                "final_code_for_ai": str(row.get("final_code_for_ai", "")),
                "final_code_for_ai_str": str(row.get("final_code_for_ai_str", "")),
                "diagnosis": str(row.get("diagnosis", "")),
                "diagnosis_singular": str(row.get("diagnosis_singular", "")),
                "adjusted_age_weeks": str(row.get("adjusted_age_weeks", "")),
            }
        )

    manifest_path = args.output_dir / "feature_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(manifest[0].keys()) if manifest else ["file"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    with (args.output_dir / "feature_names.json").open("w", encoding="utf-8") as file:
        json.dump(FEATURE_NAMES, file, indent=2)

    if missing:
        with (args.output_dir / "missing_pose_files.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["file"])
            writer.writerows([[name] for name in missing])

    print(f"metadata rows: {len(rows)}")
    print(f"unflagged metadata rows with file: {len(selected)}")
    print(f"feature arrays written: {len(manifest)}")
    print(f"missing pose files: {len(missing)}")
    print(f"channels per array: {len(FEATURE_NAMES)}")
    print(f"output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
