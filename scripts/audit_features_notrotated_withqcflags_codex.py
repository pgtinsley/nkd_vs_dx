#!/usr/bin/env python
"""Hard-anomaly audit for features_notRotated_withQCFlags arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("features_notRotated_withQCFlags"))
    parser.add_argument("--output-dir", type=Path, default=Path("features_notRotated_withQCFlags_hard_anomaly_check"))
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def load_feature_names(feature_dir: Path) -> list[str]:
    path = feature_dir / "feature_names.json"
    if path.exists():
        return json.loads(path.read_text())
    first = next(feature_dir.glob("*.npy"))
    return [f"feature_{i}" for i in range(np.load(first, mmap_mode="r").shape[0])]


def hard_cap(name: str) -> tuple[float | None, str]:
    if name.endswith(("_x", "_y")):
        return 4.0, "coord_abs_gt_4_torso_lengths"
    if any(token in name for token in ["radial_distance", "_distance", "shoulder_width", "hip_width"]):
        return 4.0, "distance_abs_gt_4_torso_lengths"
    if name == "distal_spread_area":
        return 16.0, "distal_spread_area_gt_16"
    if name.endswith(("_vx", "_vy")) or "velocity" in name or name.endswith("_speed") or name in {
        "mean_distal_speed",
        "max_distal_speed",
        "distal_speed_standard_deviation",
    }:
        if "angular_velocity_x" in name:
            return 50.0, "angular_velocity_speed_product_abs_gt_50"
        if "angular_velocity" in name:
            return 30.0, "angular_velocity_abs_gt_30_rad_per_sec"
        return 15.0, "speed_or_velocity_abs_gt_15_torso_lengths_per_sec"
    if "acceleration" in name or name.endswith(("_ax", "_ay")):
        if "angular_acceleration_x" in name:
            return 5000.0, "angular_acceleration_speed_product_abs_gt_5000"
        if "angular_acceleration" in name:
            return 900.0, "angular_acceleration_abs_gt_900_rad_per_sec2"
        return 350.0, "acceleration_abs_gt_350_torso_lengths_per_sec2"
    if "jerk" in name:
        return 10000.0, "jerk_abs_gt_10000_torso_lengths_per_sec3"
    if name.endswith("_speed_product"):
        return 225.0, "speed_product_gt_225"
    if name == "upper_lower_speed_ratio":
        return 20.0, "upper_lower_speed_ratio_gt_20"
    if name == "shoulder_midpoint_speed_over_mean_distal_speed":
        return 20.0, "shoulder_speed_ratio_gt_20"
    if name == "shoulder_hip_width_ratio":
        return 10.0, "shoulder_hip_width_ratio_gt_10"
    return None, ""


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    starts = np.r_[idx[0], idx[breaks]]
    ends = np.r_[idx[breaks - 1], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.feature_dir.glob("*.npy"))
    if not paths:
        raise SystemExit(f"No .npy files found in {args.feature_dir}")

    names = load_feature_names(args.feature_dir)
    expected_channels = len(names)
    caps = [hard_cap(name) for name in names]
    file_rows: list[dict] = []
    run_rows: list[dict] = []
    shape_rows: list[dict] = []

    for path in paths:
        arr = np.load(path, mmap_mode="r")
        shape_rows.append(
            {
                "file": path.name,
                "channels": int(arr.shape[0]) if arr.ndim >= 1 else np.nan,
                "frames": int(arr.shape[1]) if arr.ndim == 2 else np.nan,
                "dtype": str(arr.dtype),
                "shape_ok": bool(arr.ndim == 2 and arr.shape[0] == expected_channels),
            }
        )
        if arr.ndim != 2 or arr.shape[0] != expected_channels:
            continue

        file_flagged_frames: set[int] = set()
        file_flagged_values = 0
        nonfinite_values = int((~np.isfinite(arr)).sum())
        max_abs_value = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
        max_abs_by_feature = np.nanmax(np.abs(arr), axis=1)

        for ch, name in enumerate(names):
            values = np.asarray(arr[ch], dtype=np.float64)
            finite = np.isfinite(values)
            cap, reason = caps[ch]
            mask = ~finite
            if cap is not None:
                if name in {"upper_lower_speed_ratio", "shoulder_midpoint_speed_over_mean_distal_speed"}:
                    mask = mask | (finite & ((values < -1e-4) | (values > cap)))
                elif name == "shoulder_hip_width_ratio":
                    mask = mask | (finite & ((values < -1e-4) | (values > cap)))
                else:
                    mask = mask | (finite & (np.abs(values) > cap))

            if not mask.any():
                continue

            flagged_idx = np.flatnonzero(mask)
            file_flagged_values += int(flagged_idx.size)
            file_flagged_frames.update(flagged_idx.tolist())
            for start, end in contiguous_runs(mask):
                segment = values[start : end + 1]
                abs_segment = np.abs(segment)
                peak_offset = int(np.nanargmax(abs_segment)) if np.isfinite(abs_segment).any() else 0
                peak_frame = start + peak_offset
                run_rows.append(
                    {
                        "file": path.name,
                        "channel": ch,
                        "feature": name,
                        "reason": "nonfinite" if (~finite[start : end + 1]).any() else reason,
                        "cap": cap,
                        "start_frame": int(start),
                        "end_frame": int(end),
                        "n_frames": int(end - start + 1),
                        "peak_frame": int(peak_frame),
                        "peak_time_sec": float(peak_frame / args.fps),
                        "peak_value": float(values[peak_frame]),
                        "min_value": float(np.nanmin(segment)),
                        "max_value": float(np.nanmax(segment)),
                    }
                )

        file_rows.append(
            {
                "file": path.name,
                "frames": int(arr.shape[1]),
                "hard_flagged_values": int(file_flagged_values),
                "hard_flagged_frames": int(len(file_flagged_frames)),
                "nonfinite_values": nonfinite_values,
                "max_abs_value": max_abs_value,
                "max_abs_coord": float(max(max_abs_by_feature[i] for i, n in enumerate(names) if n.endswith(("_x", "_y")))),
                "max_abs_velocity_or_speed": float(
                    max(
                        max_abs_by_feature[i]
                        for i, n in enumerate(names)
                        if n.endswith(("_vx", "_vy", "_speed")) or n in {"mean_distal_speed", "max_distal_speed"}
                    )
                ),
                "max_abs_acceleration": float(
                    max(
                        max_abs_by_feature[i]
                        for i, n in enumerate(names)
                        if "acceleration" in n or n.endswith(("_ax", "_ay"))
                    )
                ),
                "max_abs_jerk": float(max(max_abs_by_feature[i] for i, n in enumerate(names) if "jerk" in n)),
                "max_upper_lower_speed_ratio": float(max_abs_by_feature[names.index("upper_lower_speed_ratio")]),
                "max_shoulder_speed_ratio": float(max_abs_by_feature[names.index("shoulder_midpoint_speed_over_mean_distal_speed")]),
            }
        )

    shape_df = pd.DataFrame(shape_rows)
    file_df = pd.DataFrame(file_rows).sort_values(
        ["hard_flagged_frames", "hard_flagged_values", "max_abs_value"], ascending=False
    )
    run_df = pd.DataFrame(run_rows)
    if not run_df.empty:
        channel_df = (
            run_df.groupby(["channel", "feature", "reason"], dropna=False)
            .agg(
                n_files=("file", "nunique"),
                n_runs=("file", "size"),
                n_frames=("n_frames", "sum"),
                max_abs_value=("peak_value", lambda s: float(np.nanmax(np.abs(s)))),
                min_value=("min_value", "min"),
                max_value=("max_value", "max"),
            )
            .reset_index()
            .sort_values(["n_files", "n_frames", "max_abs_value"], ascending=False)
        )
    else:
        channel_df = pd.DataFrame()

    shape_df.to_csv(args.output_dir / "feature_hard_anomaly_shapes.csv", index=False)
    file_df.to_csv(args.output_dir / "feature_hard_anomaly_file_summary.csv", index=False)
    run_df.to_csv(args.output_dir / "feature_hard_anomaly_runs.csv", index=False)
    channel_df.to_csv(args.output_dir / "feature_hard_anomaly_channel_summary.csv", index=False)

    print(f"Scanned {len(paths)} arrays in {args.feature_dir}")
    print(f"Shape problems: {int((~shape_df['shape_ok']).sum())}")
    print(f"Files with hard anomalies: {int((file_df['hard_flagged_values'] > 0).sum())}")
    print(f"Hard anomaly runs: {len(run_df)}")
    if not channel_df.empty:
        print("Top hard-anomaly channels:")
        print(channel_df.head(15).to_string(index=False))
    print(f"Wrote reports to {args.output_dir}")


if __name__ == "__main__":
    main()
