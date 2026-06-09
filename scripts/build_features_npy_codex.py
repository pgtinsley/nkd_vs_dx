#!/usr/bin/env python
"""Build interpretable movement feature time series from canonical COCO-17 poses.

Input pose arrays are expected to have shape (frames, 17, 2), as created by
convert_rtmpose_json_to_canonical_npy.py. Output feature arrays are saved as
(channels, frames), which is the common tsai layout.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
JOINTS = (
    ("left_elbow_angle", "left_shoulder", "left_elbow", "left_wrist"),
    ("right_elbow_angle", "right_shoulder", "right_elbow", "right_wrist"),
    ("left_knee_angle", "left_hip", "left_knee", "left_ankle"),
    ("right_knee_angle", "right_hip", "right_knee", "right_ankle"),
)

FEATURE_NAMES = (
    "left_wrist_x_midline",
    "right_wrist_x_midline",
    "left_ankle_x_midline",
    "right_ankle_x_midline",
    "left_wrist_y",
    "right_wrist_y",
    "left_ankle_y",
    "right_ankle_y",
    "left_wrist_radius",
    "right_wrist_radius",
    "left_ankle_radius",
    "right_ankle_radius",
    "left_wrist_speed",
    "right_wrist_speed",
    "left_ankle_speed",
    "right_ankle_speed",
    "left_wrist_acceleration",
    "right_wrist_acceleration",
    "left_ankle_acceleration",
    "right_ankle_acceleration",
    "left_wrist_jerk",
    "right_wrist_jerk",
    "left_ankle_jerk",
    "right_ankle_jerk",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "upper_limb_speed_asymmetry",
    "lower_limb_speed_asymmetry",
    "wrist_to_wrist_distance",
    "ankle_to_ankle_distance",
    "shoulder_width",
    "hip_width",
    "trunk_lean_angle",
    "trunk_extension",
    "whole_body_mean_speed",
    "distal_to_whole_body_activity_ratio",
    "distal_stagnation_fraction_1s",
    "distal_direction_change_rate_1s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("pose_estimate_data_npy_codex"))
    parser.add_argument("--meta-csv", type=Path, default=Path("df_meta_constructed_with_pose_qc_codex.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("features_npy_codex"))
    parser.add_argument("--visual-dir", type=Path, default=Path("features_npy_codex_visuals"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--three-months-weeks",
        type=float,
        default=52.0,
        help="Adjusted-age threshold for early writhing/fidgety versus later voluntary movement context.",
    )
    return parser.parse_args()


def safe_float(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "file" not in df.columns and "path" in df.columns:
        df["file"] = df["path"].map(lambda x: Path(str(x)).name)
    return df


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(int(window), 1)
    if window == 1:
        return values.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values.astype(np.float32), (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def first_derivative(xy: np.ndarray, fps: float) -> np.ndarray:
    delta = np.empty_like(xy, dtype=np.float32)
    delta[0] = 0.0
    delta[1:] = np.diff(xy, axis=0) * fps
    return delta


def vector_norm(values: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(values, axis=axis).astype(np.float32)


def angle_series(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = vector_norm(ba) * vector_norm(bc)
    denom = np.maximum(denom, 1e-6)
    cosang = np.sum(ba * bc, axis=1) / denom
    return np.arccos(np.clip(cosang, -1.0, 1.0)).astype(np.float32)


def age_regime(adjusted_age_weeks: float, threshold: float) -> str:
    if not np.isfinite(adjusted_age_weeks):
        return "unknown"
    if adjusted_age_weeks < threshold:
        return "early_general_movements_under_3mo_adjusted"
    return "later_fidgety_or_voluntary_over_3mo_adjusted"


def engineer_features(pose: np.ndarray, fps: float) -> np.ndarray:
    pose = pose.astype(np.float32, copy=False)
    n_frames = pose.shape[0]
    distal_xy = np.stack([pose[:, IDX[name], :] for name in DISTAL], axis=1)
    distal_velocity = first_derivative(distal_xy, fps=fps)
    distal_accel_vec = first_derivative(distal_velocity, fps=fps)
    distal_jerk_vec = first_derivative(distal_accel_vec, fps=fps)

    distal_speed = vector_norm(distal_velocity, axis=2)
    distal_accel = vector_norm(distal_accel_vec, axis=2)
    distal_jerk = vector_norm(distal_jerk_vec, axis=2)
    distal_radius = vector_norm(distal_xy, axis=2)

    all_speed = vector_norm(first_derivative(pose, fps=fps), axis=2)
    whole_body_mean_speed = np.nanmean(all_speed, axis=1).astype(np.float32)
    distal_mean_speed = np.nanmean(distal_speed, axis=1).astype(np.float32)
    activity_ratio = distal_mean_speed / (whole_body_mean_speed + 1e-4)

    stagnating = (distal_speed < 0.05).mean(axis=1).astype(np.float32)
    stagnation_fraction = rolling_mean(stagnating, round(fps))

    velocity_norm = np.maximum(distal_speed, 1e-6)
    unit_velocity = distal_velocity / velocity_norm[:, :, None]
    cos_turn = np.sum(unit_velocity[1:] * unit_velocity[:-1], axis=2)
    enough_motion = (distal_speed[1:] > 0.05) & (distal_speed[:-1] > 0.05)
    turns = ((cos_turn < 0.5) & enough_motion).mean(axis=1).astype(np.float32)
    turns = np.concatenate([[0.0], turns])
    direction_change_rate = rolling_mean(turns, round(fps))

    joint_angles = [
        angle_series(pose[:, IDX[a], :], pose[:, IDX[b], :], pose[:, IDX[c], :])
        for _, a, b, c in JOINTS
    ]

    left_wrist_speed, right_wrist_speed, left_ankle_speed, right_ankle_speed = distal_speed.T
    upper_speed_asym = np.abs(left_wrist_speed - right_wrist_speed) / (
        left_wrist_speed + right_wrist_speed + 1e-4
    )
    lower_speed_asym = np.abs(left_ankle_speed - right_ankle_speed) / (
        left_ankle_speed + right_ankle_speed + 1e-4
    )

    wrist_distance = vector_norm(pose[:, IDX["left_wrist"], :] - pose[:, IDX["right_wrist"], :])
    ankle_distance = vector_norm(pose[:, IDX["left_ankle"], :] - pose[:, IDX["right_ankle"], :])
    shoulder_width = vector_norm(pose[:, IDX["left_shoulder"], :] - pose[:, IDX["right_shoulder"], :])
    hip_width = vector_norm(pose[:, IDX["left_hip"], :] - pose[:, IDX["right_hip"], :])
    mid_shoulder = (pose[:, IDX["left_shoulder"], :] + pose[:, IDX["right_shoulder"], :]) / 2.0
    mid_hip = (pose[:, IDX["left_hip"], :] + pose[:, IDX["right_hip"], :]) / 2.0
    trunk = mid_shoulder - mid_hip
    trunk_lean = np.arctan2(trunk[:, 0], np.maximum(trunk[:, 1], 1e-6)).astype(np.float32)
    trunk_extension = vector_norm(trunk)

    channels = [
        distal_xy[:, 0, 0],
        distal_xy[:, 1, 0],
        distal_xy[:, 2, 0],
        distal_xy[:, 3, 0],
        distal_xy[:, 0, 1],
        distal_xy[:, 1, 1],
        distal_xy[:, 2, 1],
        distal_xy[:, 3, 1],
        *[distal_radius[:, i] for i in range(4)],
        *[distal_speed[:, i] for i in range(4)],
        *[distal_accel[:, i] for i in range(4)],
        *[distal_jerk[:, i] for i in range(4)],
        *joint_angles,
        upper_speed_asym.astype(np.float32),
        lower_speed_asym.astype(np.float32),
        wrist_distance,
        ankle_distance,
        shoulder_width,
        hip_width,
        trunk_lean,
        trunk_extension,
        whole_body_mean_speed,
        activity_ratio.astype(np.float32),
        stagnation_fraction,
        direction_change_rate,
    ]
    features = np.vstack(channels).astype(np.float32)
    if features.shape != (len(FEATURE_NAMES), n_frames):
        raise ValueError(f"Unexpected feature shape {features.shape}; expected {(len(FEATURE_NAMES), n_frames)}")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def summarize_features(features: np.ndarray) -> dict[str, float]:
    summary = {}
    for i, name in enumerate(FEATURE_NAMES):
        values = features[i]
        summary[f"{name}__mean"] = float(np.mean(values))
        summary[f"{name}__std"] = float(np.std(values))
        summary[f"{name}__p95"] = float(np.percentile(values, 95))
    return summary


def build_manifest(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.visual_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(args.meta_csv)
    if "file" in meta.columns:
        for stale_name in ("metadata_rows_missing_pose_file.csv", "duplicate_metadata_file_rows.csv"):
            stale_path = args.output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        missing_file_rows = meta[meta["file"].isna()]
        if not missing_file_rows.empty:
            missing_file_rows.to_csv(args.output_dir / "metadata_rows_missing_pose_file.csv", index=False)
        meta_with_file = meta[meta["file"].notna()].copy()
        duplicate_meta_files = meta_with_file[meta_with_file["file"].duplicated(keep=False)].sort_values("file")
        if not duplicate_meta_files.empty:
            duplicate_meta_files.to_csv(args.output_dir / "duplicate_metadata_file_rows.csv", index=False)
        meta_lookup = meta_with_file.drop_duplicates(subset=["file"], keep="first")
        by_file = meta_lookup.set_index("file", drop=False).to_dict("index")
    else:
        by_file = {}
    rows = []
    summaries = []

    for pose_path in sorted(args.pose_dir.glob("*.npy")):
        pose = np.load(pose_path)
        if pose.ndim != 3 or pose.shape[1:] != (17, 2):
            raise ValueError(f"{pose_path} has shape {pose.shape}; expected (frames, 17, 2)")

        features = engineer_features(pose, fps=args.fps)
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
            "feature_layout": "channels_first",
            "class_label": label,
            "adjusted_age_weeks": adjusted_age,
            "age_regime": age_regime(adjusted_age, args.three_months_weeks),
            "has_metadata": bool(meta_row),
        }
        rows.append(row)
        summaries.append({"file": pose_path.name, **row, **summarize_features(features)})

    manifest = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summaries)
    manifest.to_csv(args.output_dir / "feature_manifest.csv", index=False)
    summary_df.to_csv(args.output_dir / "feature_summary.csv", index=False)
    with (args.output_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(list(FEATURE_NAMES), f, indent=2)
    return manifest, summary_df


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(b) - np.mean(a)) / (pooled + 1e-8))


def create_visuals(summary_df: pd.DataFrame, args: argparse.Namespace) -> list[Path]:
    paths = []
    labeled = summary_df[summary_df["class_label"].isin(["NKD", "DX"])].copy()

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    counts = labeled.groupby(["age_regime", "class_label"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", ax=ax, color=["#2f6f73", "#b84a62"])
    ax.set_title("Available labeled recordings by class and adjusted-age regime")
    ax.set_xlabel("")
    ax.set_ylabel("recordings")
    ax.tick_params(axis="x", rotation=15)
    out = args.visual_dir / "class_counts_by_age_regime.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    paths.append(out)

    metric_cols = [c for c in labeled.columns if c.endswith("__mean") or c.endswith("__std") or c.endswith("__p95")]
    effect_rows = []
    for regime_name, regime_df in [("all_labeled", labeled), *list(labeled.groupby("age_regime"))]:
        if set(regime_df["class_label"]) >= {"NKD", "DX"}:
            nkd = regime_df[regime_df["class_label"] == "NKD"]
            dx = regime_df[regime_df["class_label"] == "DX"]
            for col in metric_cols:
                effect_rows.append({"age_regime": regime_name, "metric": col, "cohens_d_dx_minus_nkd": cohens_d(nkd[col], dx[col])})
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(args.output_dir / "feature_separability_effect_sizes.csv", index=False)

    top = effects[effects["age_regime"] == "all_labeled"].copy()
    top["abs_d"] = top["cohens_d_dx_minus_nkd"].abs()
    top = top.sort_values("abs_d", ascending=False).head(20).sort_values("cohens_d_dx_minus_nkd")
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    colors = np.where(top["cohens_d_dx_minus_nkd"] >= 0, "#b84a62", "#2f6f73")
    ax.barh(top["metric"].str.replace("__", "\n", regex=False), top["cohens_d_dx_minus_nkd"], color=colors)
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_title("Largest univariate NKD vs DX separability signals")
    ax.set_xlabel("Cohen's d, DX minus NKD")
    out = args.visual_dir / "top_feature_effect_sizes.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    paths.append(out)

    plot_metrics = [
        "whole_body_mean_speed__mean",
        "upper_limb_speed_asymmetry__mean",
        "lower_limb_speed_asymmetry__mean",
        "distal_stagnation_fraction_1s__mean",
        "distal_direction_change_rate_1s__mean",
        "left_wrist_radius__std",
    ]
    available = [m for m in plot_metrics if m in labeled.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), available):
        data = [labeled.loc[labeled["class_label"] == cls, metric].dropna() for cls in ["NKD", "DX"]]
        ax.boxplot(data, tick_labels=["NKD", "DX"], showfliers=False, patch_artist=True)
        ax.set_title(metric.replace("__", "\n"))
    for ax in axes.ravel()[len(available) :]:
        ax.axis("off")
    fig.suptitle("Clinical feature distributions in labeled metadata")
    out = args.visual_dir / "clinical_feature_distributions.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    paths.append(out)

    example_paths = []
    for regime in labeled["age_regime"].dropna().unique()[:2]:
        for cls in ["NKD", "DX"]:
            sample = labeled[(labeled["age_regime"] == regime) & (labeled["class_label"] == cls)].head(1)
            if not sample.empty:
                example_paths.append(Path(sample.iloc[0]["feature_path"]))
    if example_paths:
        fig, axes = plt.subplots(len(example_paths), 1, figsize=(13, 2.4 * len(example_paths)), sharex=False, constrained_layout=True)
        if len(example_paths) == 1:
            axes = [axes]
        feature_lookup = {name: i for i, name in enumerate(FEATURE_NAMES)}
        for ax, path in zip(axes, example_paths):
            arr = np.load(path)
            n = min(arr.shape[1], int(args.fps * 45))
            t = np.arange(n) / args.fps
            for name in ["whole_body_mean_speed", "upper_limb_speed_asymmetry", "distal_stagnation_fraction_1s", "distal_direction_change_rate_1s"]:
                ax.plot(t, arr[feature_lookup[name], :n], linewidth=0.9, label=name)
            row = labeled[labeled["file"] == path.name].iloc[0]
            ax.set_title(f"{path.name} | {row['class_label']} | {row['age_regime']}")
            ax.set_ylabel("feature value")
            ax.legend(ncol=2, fontsize=8)
        axes[-1].set_xlabel("seconds")
        out = args.visual_dir / "example_feature_timeseries.png"
        fig.savefig(out, dpi=170)
        plt.close(fig)
        paths.append(out)

    return paths


def main() -> None:
    args = parse_args()
    manifest, summary_df = build_manifest(args)
    visual_paths = create_visuals(summary_df, args)
    print(f"Wrote {len(manifest)} feature arrays to {args.output_dir}")
    print(f"Each array has {len(FEATURE_NAMES)} channels in channels-first layout.")
    print(f"Metadata matches: {int(manifest['has_metadata'].sum())}; unmatched pose files: {int((~manifest['has_metadata']).sum())}")
    for path in visual_paths:
        print(f"visual: {path}")


if __name__ == "__main__":
    main()
