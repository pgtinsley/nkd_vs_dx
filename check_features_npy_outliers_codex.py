#!/usr/bin/env python
"""Audit engineered feature arrays for implausible or extreme values.

The report is intentionally channel-aware: each feature channel has different
units, so this combines robust-z screening with simple physiological caps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ABS_CAPS = {
    # Canonical coordinates and distances are in torso-length units.
    "x_midline": 4.0,
    "_y": 4.0,
    "radius": 4.0,
    "distance": 4.0,
    "shoulder_width": 3.0,
    "hip_width": 3.0,
    "trunk_extension": 3.0,
    # Distal limb derivatives are torso-length units per second and higher derivatives.
    "speed": 15.0,
    "whole_body_mean_speed": 10.0,
    "acceleration": 350.0,
    "jerk": 10000.0,
    # Angles and ratios/fractions should be bounded by construction.
    "angle": np.pi,
    "asymmetry": 1.05,
    "activity_ratio": 20.0,
    "stagnation_fraction": 1.05,
    "direction_change_rate": 1.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("features_npy_codex"))
    parser.add_argument("--output-dir", type=Path, default=Path("features_npy_codex_outlier_check"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sample-stride", type=int, default=30)
    parser.add_argument("--robust-z-threshold", type=float, default=25.0)
    parser.add_argument("--top-points-per-channel", type=int, default=25)
    return parser.parse_args()


def get_abs_cap(name: str) -> float | None:
    if name == "trunk_lean_angle":
        return float(np.pi / 2.0)
    if any(key in name for key in ["asymmetry", "stagnation_fraction", "direction_change_rate"]):
        return 1.05
    if "activity_ratio" in name:
        return 20.0
    if name in {"left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"}:
        return float(np.pi)
    for key, cap in DEFAULT_ABS_CAPS.items():
        if key in name:
            return float(cap)
    return None


def load_feature_names(feature_dir: Path) -> list[str]:
    path = feature_dir / "feature_names.json"
    if path.exists():
        return json.loads(path.read_text())
    first = next(feature_dir.glob("*.npy"))
    return [f"feature_{i}" for i in range(np.load(first, mmap_mode="r").shape[0])]


def iter_feature_paths(feature_dir: Path) -> list[Path]:
    return sorted(p for p in feature_dir.glob("*.npy") if p.is_file())


def estimate_channel_stats(paths: list[Path], n_channels: int, sample_stride: int) -> pd.DataFrame:
    samples = [[] for _ in range(n_channels)]
    for path in paths:
        arr = np.load(path, mmap_mode="r")
        sampled = arr[:, ::sample_stride]
        for ch in range(n_channels):
            values = np.asarray(sampled[ch], dtype=np.float32)
            values = values[np.isfinite(values)]
            if values.size:
                samples[ch].append(values)

    rows = []
    for ch, parts in enumerate(samples):
        values = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        if values.size:
            median = float(np.median(values))
            q01, q05, q25, q75, q95, q99 = np.percentile(values, [1, 5, 25, 75, 95, 99])
            mad = float(np.median(np.abs(values - median)))
            robust_scale = max(1.4826 * mad, 1e-6)
        else:
            median = q01 = q05 = q25 = q75 = q95 = q99 = mad = robust_scale = np.nan
        rows.append(
            {
                "channel": ch,
                "median": median,
                "mad": mad,
                "robust_scale": robust_scale,
                "p01": q01,
                "p05": q05,
                "p25": q25,
                "p75": q75,
                "p95": q95,
                "p99": q99,
                "sampled_values": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    starts = np.r_[idx[0], idx[breaks]]
    ends = np.r_[idx[breaks - 1], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def scan_paths(
    paths: list[Path],
    feature_names: list[str],
    stats: pd.DataFrame,
    fps: float,
    robust_z_threshold: float,
    top_points_per_channel: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = stats["median"].to_numpy(dtype=np.float64)
    scale = stats["robust_scale"].to_numpy(dtype=np.float64)
    caps = np.array([get_abs_cap(name) if get_abs_cap(name) is not None else np.nan for name in feature_names])
    top_points = {ch: [] for ch in range(len(feature_names))}
    run_rows = []
    file_rows = []

    for path in paths:
        arr = np.load(path, mmap_mode="r")
        file_flagged_values = 0
        file_flagged_runs = 0
        file_max_abs_robust_z = 0.0
        file_max_abs_value = 0.0

        for ch, name in enumerate(feature_names):
            values = np.asarray(arr[ch], dtype=np.float64)
            finite = np.isfinite(values)
            rz = np.zeros_like(values, dtype=np.float64)
            if np.isfinite(scale[ch]) and scale[ch] > 0:
                rz[finite] = (values[finite] - med[ch]) / scale[ch]
            abs_rz = np.abs(rz)
            robust_mask = finite & (abs_rz >= robust_z_threshold)
            cap = caps[ch]
            if name in {"left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"}:
                cap_mask = finite & ((values < -1e-3) | (values > np.pi + 1e-3))
            elif name == "trunk_lean_angle":
                cap_mask = finite & (np.abs(values) > cap)
            elif any(k in name for k in ["asymmetry", "stagnation_fraction", "direction_change_rate"]):
                cap_mask = finite & ((values < -1e-4) | (values > 1.05))
            else:
                cap_mask = finite & (np.abs(values) > cap) if np.isfinite(cap) else np.zeros_like(finite)
            mask = robust_mask | cap_mask | ~finite

            if finite.any():
                file_max_abs_robust_z = max(file_max_abs_robust_z, float(np.nanmax(abs_rz[finite])))
                file_max_abs_value = max(file_max_abs_value, float(np.nanmax(np.abs(values[finite]))))

            flagged_idx = np.flatnonzero(mask)
            file_flagged_values += int(flagged_idx.size)
            if flagged_idx.size:
                candidate = flagged_idx[np.argsort(abs_rz[flagged_idx])[-top_points_per_channel:]]
                for frame_idx in candidate:
                    top_points[ch].append(
                        {
                            "file": path.name,
                            "channel": ch,
                            "feature": name,
                            "frame_idx": int(frame_idx),
                            "time_sec": float(frame_idx / fps),
                            "value": float(values[frame_idx]),
                            "robust_z": float(rz[frame_idx]),
                            "abs_robust_z": float(abs_rz[frame_idx]),
                            "abs_cap": cap if np.isfinite(cap) else np.nan,
                            "reason": ",".join(
                                reason
                                for reason, hit in [
                                    ("nonfinite", not finite[frame_idx]),
                                    ("robust_z", robust_mask[frame_idx]),
                                    ("absolute_cap", cap_mask[frame_idx]),
                                ]
                                if hit
                            ),
                        }
                    )

                for start, end in contiguous_runs(mask):
                    segment = slice(start, end + 1)
                    seg_abs_rz = abs_rz[segment]
                    seg_values = values[segment]
                    local = int(np.nanargmax(seg_abs_rz))
                    peak_frame = start + local
                    run_rows.append(
                        {
                            "file": path.name,
                            "channel": ch,
                            "feature": name,
                            "start_frame": int(start),
                            "end_frame": int(end),
                            "start_time_sec": float(start / fps),
                            "end_time_sec": float(end / fps),
                            "n_frames": int(end - start + 1),
                            "peak_frame": int(peak_frame),
                            "peak_time_sec": float(peak_frame / fps),
                            "peak_value": float(values[peak_frame]),
                            "peak_robust_z": float(rz[peak_frame]),
                            "peak_abs_robust_z": float(abs_rz[peak_frame]),
                            "min_value": float(np.nanmin(seg_values)),
                            "max_value": float(np.nanmax(seg_values)),
                            "abs_cap": cap if np.isfinite(cap) else np.nan,
                            "reason": ",".join(
                                reason
                                for reason, hit in [
                                    ("nonfinite", np.any(~finite[segment])),
                                    ("robust_z", np.any(robust_mask[segment])),
                                    ("absolute_cap", np.any(cap_mask[segment])),
                                ]
                                if hit
                            ),
                        }
                    )
                    file_flagged_runs += 1

        file_rows.append(
            {
                "file": path.name,
                "frames": int(arr.shape[1]),
                "flagged_values": file_flagged_values,
                "flagged_runs": file_flagged_runs,
                "max_abs_robust_z": file_max_abs_robust_z,
                "max_abs_value": file_max_abs_value,
            }
        )

    top_rows = []
    for ch, rows in top_points.items():
        rows = sorted(rows, key=lambda r: r["abs_robust_z"], reverse=True)[:top_points_per_channel]
        top_rows.extend(rows)
    return pd.DataFrame(run_rows), pd.DataFrame(file_rows), pd.DataFrame(top_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = iter_feature_paths(args.feature_dir)
    feature_names = load_feature_names(args.feature_dir)
    if not paths:
        raise SystemExit(f"No .npy files found in {args.feature_dir}")

    stats = estimate_channel_stats(paths, len(feature_names), args.sample_stride)
    stats.insert(1, "feature", feature_names)
    stats["abs_cap"] = [get_abs_cap(name) for name in feature_names]
    stats.to_csv(args.output_dir / "feature_channel_robust_stats.csv", index=False)

    run_df, file_df, top_df = scan_paths(
        paths=paths,
        feature_names=feature_names,
        stats=stats,
        fps=args.fps,
        robust_z_threshold=args.robust_z_threshold,
        top_points_per_channel=args.top_points_per_channel,
    )
    run_df.to_csv(args.output_dir / "feature_outlier_runs.csv", index=False)
    file_df.to_csv(args.output_dir / "feature_outlier_file_summary.csv", index=False)
    top_df.to_csv(args.output_dir / "feature_outlier_top_points.csv", index=False)

    channel_summary = (
        run_df.groupby(["channel", "feature", "reason"], dropna=False)
        .agg(
            n_files=("file", "nunique"),
            n_runs=("file", "size"),
            n_flagged_frames=("n_frames", "sum"),
            max_abs_robust_z=("peak_abs_robust_z", "max"),
            max_abs_value=("max_value", "max"),
            min_value=("min_value", "min"),
        )
        .reset_index()
        .sort_values(["n_files", "n_flagged_frames"], ascending=False)
        if not run_df.empty
        else pd.DataFrame()
    )
    channel_summary.to_csv(args.output_dir / "feature_outlier_channel_summary.csv", index=False)

    print(f"Scanned {len(paths)} arrays in {args.feature_dir}")
    print(f"Outlier runs: {len(run_df)}")
    print(f"Files with flagged values: {int((file_df['flagged_values'] > 0).sum())}")
    if not channel_summary.empty:
        print("Top flagged channels:")
        print(channel_summary.head(12).to_string(index=False))
    print(f"Wrote reports to {args.output_dir}")


if __name__ == "__main__":
    main()
