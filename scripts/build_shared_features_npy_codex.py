#!/usr/bin/env python
"""Build an expanded shared feature set for all active pose arrays.

This preserves the original 40 features from build_features_npy_codex.py and
adds clinically interpretable shared features for younger and older infants:
variability, smoothness, synchrony, complexity, midline behavior, and posture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_features_npy_codex import (
    COCO17_NAMES,
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    IDX,
    angle_series,
    engineer_features as engineer_base_features,
    safe_float,
)


DISTAL = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
EXTRA_FEATURE_NAMES = (
    "distal_speed_variability_1s",
    "distal_acceleration_variability_1s",
    "distal_normalized_jerk_1s",
    "speed_burstiness_3s",
    "upper_limb_speed_correlation_3s",
    "lower_limb_speed_correlation_3s",
    "cross_limb_speed_synchrony_3s",
    "simultaneous_limb_activation_fraction_1s",
    "limb_speed_dispersion_1s",
    "fidgety_micro_movement_density_3s",
    "multi_directional_movement_rate_3s",
    "direction_entropy_3s",
    "limb_activation_pattern_entropy_3s",
    "active_limb_repertoire_count_3s",
    "wrist_midline_crossing_rate_3s",
    "ankle_midline_crossing_rate_3s",
    "wrist_antigravity_fraction_3s",
    "elbow_angle_variability_3s",
    "knee_angle_variability_3s",
    "trunk_lean_variability_3s",
)
FEATURE_NAMES = tuple(BASE_FEATURE_NAMES) + EXTRA_FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("pose_estimate_data_npy_codex"))
    parser.add_argument("--meta-csv", type=Path, default=Path("df_meta_constructed_with_pose_qc_codex.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("features_npy_codex_shared"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--three-months-weeks", type=float, default=52.0)
    return parser.parse_args()


def first_derivative(x: np.ndarray, fps: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    out[1:] = np.diff(x, axis=0) * fps
    return out


def rolling_mean_1d(x: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    x = np.asarray(x, dtype=np.float32)
    if window == 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def rolling_std_1d(x: np.ndarray, window: int) -> np.ndarray:
    mean = rolling_mean_1d(x, window)
    mean_sq = rolling_mean_1d(np.square(x), window)
    return np.sqrt(np.maximum(mean_sq - np.square(mean), 0.0)).astype(np.float32)


def rolling_corr_1d(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    mx = rolling_mean_1d(x, window)
    my = rolling_mean_1d(y, window)
    cov = rolling_mean_1d(x * y, window) - mx * my
    vx = rolling_mean_1d(x * x, window) - mx * mx
    vy = rolling_mean_1d(y * y, window) - my * my
    return (cov / np.sqrt(np.maximum(vx * vy, 1e-8))).astype(np.float32)


def rolling_binary_event_rate(event: np.ndarray, window: int) -> np.ndarray:
    return rolling_mean_1d(event.astype(np.float32), window)


def entropy_from_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-8, 1.0)
    entropy = -np.sum(probs * np.log2(probs), axis=0)
    normalizer = np.log2(probs.shape[0]) if probs.shape[0] > 1 else 1.0
    return (entropy / normalizer).astype(np.float32)


def rolling_category_entropy(codes: np.ndarray, n_categories: int, window: int) -> np.ndarray:
    indicators = np.zeros((n_categories, codes.size), dtype=np.float32)
    valid = (codes >= 0) & (codes < n_categories)
    indicators[codes[valid], np.flatnonzero(valid)] = 1.0
    probs = np.vstack([rolling_mean_1d(indicators[i], window) for i in range(n_categories)])
    return entropy_from_probabilities(probs)


def rolling_category_count(codes: np.ndarray, n_categories: int, window: int) -> np.ndarray:
    indicators = np.zeros((n_categories, codes.size), dtype=np.float32)
    valid = (codes >= 0) & (codes < n_categories)
    indicators[codes[valid], np.flatnonzero(valid)] = 1.0
    seen = np.vstack([rolling_mean_1d(indicators[i], window) > 0 for i in range(n_categories)])
    return seen.sum(axis=0).astype(np.float32)


def get_extra_features(pose: np.ndarray, fps: float) -> np.ndarray:
    pose = pose.astype(np.float32, copy=False)
    n = pose.shape[0]
    w1 = round(fps)
    w3 = round(fps * 3)

    distal_xy = np.stack([pose[:, IDX[name], :] for name in DISTAL], axis=1)
    distal_velocity = first_derivative(distal_xy, fps)
    distal_accel_vec = first_derivative(distal_velocity, fps)
    distal_jerk_vec = first_derivative(distal_accel_vec, fps)
    distal_speed = np.linalg.norm(distal_velocity, axis=2).astype(np.float32)
    distal_accel = np.linalg.norm(distal_accel_vec, axis=2).astype(np.float32)
    distal_jerk = np.linalg.norm(distal_jerk_vec, axis=2).astype(np.float32)
    distal_mean_speed = distal_speed.mean(axis=1)
    distal_mean_accel = distal_accel.mean(axis=1)
    distal_mean_jerk = distal_jerk.mean(axis=1)

    speed_var_1s = rolling_std_1d(distal_mean_speed, w1)
    accel_var_1s = rolling_std_1d(distal_mean_accel, w1)
    normalized_jerk_1s = rolling_mean_1d(distal_mean_jerk / np.maximum(distal_mean_speed, 0.05), w1)
    speed_burstiness_3s = rolling_std_1d(distal_mean_speed, w3) / (rolling_mean_1d(distal_mean_speed, w3) + 1e-4)

    upper_corr_3s = rolling_corr_1d(distal_speed[:, 0], distal_speed[:, 1], w3)
    lower_corr_3s = rolling_corr_1d(distal_speed[:, 2], distal_speed[:, 3], w3)
    cross_pairs = [
        rolling_corr_1d(distal_speed[:, 0], distal_speed[:, 2], w3),
        rolling_corr_1d(distal_speed[:, 0], distal_speed[:, 3], w3),
        rolling_corr_1d(distal_speed[:, 1], distal_speed[:, 2], w3),
        rolling_corr_1d(distal_speed[:, 1], distal_speed[:, 3], w3),
    ]
    cross_sync_3s = np.mean(np.vstack(cross_pairs), axis=0).astype(np.float32)

    active = distal_speed > 0.10
    simultaneous_activation_1s = rolling_mean_1d((active.sum(axis=1) >= 3).astype(np.float32), w1)
    limb_speed_dispersion_1s = rolling_mean_1d(np.std(distal_speed, axis=1), w1)
    micro = (distal_speed >= 0.05) & (distal_speed <= 0.60)
    fidgety_micro_density_3s = rolling_mean_1d((micro.sum(axis=1) >= 2).astype(np.float32), w3)

    velocity_norm = np.maximum(distal_speed, 1e-6)
    unit_velocity = distal_velocity / velocity_norm[:, :, None]
    cos_turn = np.sum(unit_velocity[1:] * unit_velocity[:-1], axis=2)
    enough_motion = (distal_speed[1:] > 0.05) & (distal_speed[:-1] > 0.05)
    directional_turns = ((cos_turn < 0.5) & enough_motion).sum(axis=1) >= 2
    multi_directional_rate_3s = rolling_mean_1d(np.r_[0.0, directional_turns.astype(np.float32)], w3)

    angles = np.arctan2(distal_velocity[:, :, 1], distal_velocity[:, :, 0])
    bins = np.floor(((angles + np.pi) / (2 * np.pi)) * 8).astype(np.int16)
    bins = np.clip(bins, 0, 7)
    moving = distal_speed > 0.05
    direction_indicators = np.zeros((8, n), dtype=np.float32)
    for b in range(8):
        direction_indicators[b] = ((bins == b) & moving).sum(axis=1)
    direction_total = np.maximum(direction_indicators.sum(axis=0), 1.0)
    direction_probs = np.vstack([rolling_mean_1d(direction_indicators[b] / direction_total, w3) for b in range(8)])
    direction_entropy_3s = entropy_from_probabilities(direction_probs)

    activation_codes = (
        active[:, 0].astype(np.int16)
        + (active[:, 1].astype(np.int16) << 1)
        + (active[:, 2].astype(np.int16) << 2)
        + (active[:, 3].astype(np.int16) << 3)
    )
    activation_entropy_3s = rolling_category_entropy(activation_codes, 16, w3)
    activation_count_3s = rolling_category_count(activation_codes, 16, w3)

    left_wrist_x = distal_xy[:, 0, 0]
    right_wrist_x = distal_xy[:, 1, 0]
    left_ankle_x = distal_xy[:, 2, 0]
    right_ankle_x = distal_xy[:, 3, 0]
    wrist_crossings = (
        (left_wrist_x[1:] * left_wrist_x[:-1] < 0)
        | (right_wrist_x[1:] * right_wrist_x[:-1] < 0)
    )
    ankle_crossings = (
        (left_ankle_x[1:] * left_ankle_x[:-1] < 0)
        | (right_ankle_x[1:] * right_ankle_x[:-1] < 0)
    )
    wrist_midline_crossing_rate_3s = rolling_binary_event_rate(np.r_[False, wrist_crossings], w3)
    ankle_midline_crossing_rate_3s = rolling_binary_event_rate(np.r_[False, ankle_crossings], w3)

    left_shoulder_y = pose[:, IDX["left_shoulder"], 1]
    right_shoulder_y = pose[:, IDX["right_shoulder"], 1]
    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    wrist_antigravity_3s = rolling_mean_1d(
        ((pose[:, IDX["left_wrist"], 1] > shoulder_y) | (pose[:, IDX["right_wrist"], 1] > shoulder_y)).astype(np.float32),
        w3,
    )

    left_elbow = angle_series(pose[:, IDX["left_shoulder"], :], pose[:, IDX["left_elbow"], :], pose[:, IDX["left_wrist"], :])
    right_elbow = angle_series(pose[:, IDX["right_shoulder"], :], pose[:, IDX["right_elbow"], :], pose[:, IDX["right_wrist"], :])
    left_knee = angle_series(pose[:, IDX["left_hip"], :], pose[:, IDX["left_knee"], :], pose[:, IDX["left_ankle"], :])
    right_knee = angle_series(pose[:, IDX["right_hip"], :], pose[:, IDX["right_knee"], :], pose[:, IDX["right_ankle"], :])
    elbow_angle_var_3s = rolling_std_1d((left_elbow + right_elbow) / 2, w3)
    knee_angle_var_3s = rolling_std_1d((left_knee + right_knee) / 2, w3)
    mid_shoulder = (pose[:, IDX["left_shoulder"], :] + pose[:, IDX["right_shoulder"], :]) / 2.0
    mid_hip = (pose[:, IDX["left_hip"], :] + pose[:, IDX["right_hip"], :]) / 2.0
    trunk = mid_shoulder - mid_hip
    trunk_lean = np.arctan2(trunk[:, 0], np.maximum(trunk[:, 1], 1e-6)).astype(np.float32)
    trunk_lean_var_3s = rolling_std_1d(trunk_lean, w3)

    channels = [
        speed_var_1s,
        accel_var_1s,
        normalized_jerk_1s,
        speed_burstiness_3s,
        upper_corr_3s,
        lower_corr_3s,
        cross_sync_3s,
        simultaneous_activation_1s,
        limb_speed_dispersion_1s,
        fidgety_micro_density_3s,
        multi_directional_rate_3s,
        direction_entropy_3s,
        activation_entropy_3s,
        activation_count_3s,
        wrist_midline_crossing_rate_3s,
        ankle_midline_crossing_rate_3s,
        wrist_antigravity_3s,
        elbow_angle_var_3s,
        knee_angle_var_3s,
        trunk_lean_var_3s,
    ]
    extra = np.vstack(channels).astype(np.float32)
    if extra.shape != (len(EXTRA_FEATURE_NAMES), n):
        raise ValueError(f"Unexpected extra feature shape {extra.shape}")
    return np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)


def age_regime(adjusted_age_weeks: float, threshold: float) -> str:
    if not np.isfinite(adjusted_age_weeks):
        return "unknown"
    if adjusted_age_weeks < threshold:
        return "early_general_movements_under_3mo_adjusted"
    return "later_fidgety_or_voluntary_over_3mo_adjusted"


def summarize_features(features: np.ndarray) -> dict[str, float]:
    summary = {}
    for i, name in enumerate(FEATURE_NAMES):
        values = features[i]
        summary[f"{name}__mean"] = float(np.mean(values))
        summary[f"{name}__std"] = float(np.std(values))
        summary[f"{name}__p95"] = float(np.percentile(values, 95))
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta_csv) if args.meta_csv.exists() else pd.DataFrame()
    by_file = {}
    if "file" in meta.columns:
        meta_with_file = meta[meta["file"].notna()].drop_duplicates(subset=["file"], keep="first")
        by_file = meta_with_file.set_index("file", drop=False).to_dict("index")

    rows = []
    summaries = []
    pose_paths = sorted(path for path in args.pose_dir.glob("*.npy") if not path.stem.endswith("_preInterpolation"))
    for pose_path in pose_paths:
        pose = np.load(pose_path)
        if pose.ndim != 3 or pose.shape[1:] != (17, 2):
            raise ValueError(f"{pose_path} has shape {pose.shape}; expected (frames, 17, 2)")
        base = engineer_base_features(pose, args.fps)
        extra = get_extra_features(pose, args.fps)
        features = np.vstack([base, extra]).astype(np.float32)
        out_path = args.output_dir / pose_path.name
        np.save(out_path, features)

        meta_row = by_file.get(pose_path.name, {})
        adjusted_age = safe_float(meta_row.get("adjusted_age_weeks"))
        label = meta_row.get("final_code_for_ai_str") or meta_row.get("diagnosis") or ""
        row = {
            "file": pose_path.name,
            "pose_path": str(pose_path),
            "feature_path": str(out_path),
            "frames": int(pose.shape[0]),
            "duration_sec": float(pose.shape[0] / args.fps),
            "n_features": len(FEATURE_NAMES),
            "n_base_features": len(BASE_FEATURE_NAMES),
            "n_extra_features": len(EXTRA_FEATURE_NAMES),
            "feature_layout": "channels_first",
            "class_label": label,
            "adjusted_age_weeks": adjusted_age,
            "age_regime": age_regime(adjusted_age, args.three_months_weeks),
            "has_metadata": bool(meta_row),
        }
        rows.append(row)
        summaries.append({"file": pose_path.name, **row, **summarize_features(features)})

    pd.DataFrame(rows).to_csv(args.output_dir / "feature_manifest.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "feature_summary.csv", index=False)
    with (args.output_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(list(FEATURE_NAMES), f, indent=2)
    with (args.output_dir / "extra_feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(list(EXTRA_FEATURE_NAMES), f, indent=2)

    print(f"Wrote {len(rows)} shared feature arrays to {args.output_dir}")
    print(f"Each array has {len(FEATURE_NAMES)} channels: {len(BASE_FEATURE_NAMES)} base + {len(EXTRA_FEATURE_NAMES)} extra")
    print(f"Skipped {len(list(args.pose_dir.glob('*_preInterpolation.npy')))} backup arrays")


if __name__ == "__main__":
    main()
