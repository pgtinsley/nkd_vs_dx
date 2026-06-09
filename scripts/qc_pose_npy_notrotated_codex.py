#!/usr/bin/env python
"""Quality-check pelvis-centered, torso-scaled COCO-17 pose .npy arrays."""

from __future__ import annotations

import argparse
import csv
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

IDX = {name: idx for idx, name in enumerate(COCO17_NAMES)}

BONES = {
    "shoulder_width": ("left_shoulder", "right_shoulder"),
    "hip_width": ("left_hip", "right_hip"),
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "left_lower_arm": ("left_elbow", "left_wrist"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "right_lower_arm": ("right_elbow", "right_wrist"),
    "left_upper_leg": ("left_hip", "left_knee"),
    "left_lower_leg": ("left_knee", "left_ankle"),
    "right_upper_leg": ("right_hip", "right_knee"),
    "right_lower_leg": ("right_knee", "right_ankle"),
    "left_trunk": ("left_shoulder", "left_hip"),
    "right_trunk": ("right_shoulder", "right_hip"),
}

DISTANCE_CAPS = {
    "shoulder_width": 2.5,
    "hip_width": 2.0,
    "left_upper_arm": 2.5,
    "left_lower_arm": 2.5,
    "right_upper_arm": 2.5,
    "right_lower_arm": 2.5,
    "left_upper_leg": 3.0,
    "left_lower_leg": 3.0,
    "right_upper_leg": 3.0,
    "right_lower_leg": 3.0,
    "left_trunk": 2.5,
    "right_trunk": 2.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("pose_estimate_data_npy_codex_notRotated"))
    parser.add_argument("--output-dir", type=Path, default=Path("pose_estimate_data_npy_codex_notRotated_qc"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--coord-cap", type=float, default=4.0)
    parser.add_argument("--speed-cap", type=float, default=15.0)
    parser.add_argument("--torso-y-min", type=float, default=0.20)
    parser.add_argument("--robust-z", type=float, default=10.0)
    return parser.parse_args()


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad < 1e-12:
        std = np.nanstd(values)
        return (values - np.nanmean(values)) / std if std > 0 else np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    starts = np.r_[idx[0], idx[breaks]]
    ends = np.r_[idx[breaks - 1], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def load_arrays(paths: list[Path]) -> dict[str, np.ndarray]:
    arrays = {}
    for path in paths:
        arrays[path.name] = np.load(path)
    return arrays


def get_midpoints(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid_hip = (arr[:, IDX["left_hip"], :] + arr[:, IDX["right_hip"], :]) / 2.0
    mid_shoulder = (arr[:, IDX["left_shoulder"], :] + arr[:, IDX["right_shoulder"], :]) / 2.0
    return mid_hip, mid_shoulder


def bone_lengths(arr: np.ndarray) -> dict[str, np.ndarray]:
    lengths = {}
    for name, (start, end) in BONES.items():
        lengths[name] = np.linalg.norm(arr[:, IDX[start], :] - arr[:, IDX[end], :], axis=1)
    return lengths


def build_global_bone_stats(arrays: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    stats = {}
    for bone in BONES:
        values = []
        for arr in arrays.values():
            vals = bone_lengths(arr)[bone]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                values.append(vals[:: max(1, vals.size // 2000)])
        all_values = np.concatenate(values) if values else np.array([], dtype=float)
        median = float(np.nanmedian(all_values)) if all_values.size else np.nan
        mad = float(np.nanmedian(np.abs(all_values - median))) if all_values.size else np.nan
        scale = max(1.4826 * mad, 1e-6) if np.isfinite(mad) else np.nan
        stats[bone] = (median, scale)
    return stats


def add_run_rows(
    rows: list[dict],
    file: str,
    reason: str,
    mask: np.ndarray,
    values: np.ndarray,
    threshold: float,
    detail: str,
    fps: float,
) -> None:
    for start, end in contiguous_runs(mask):
        segment = values[start : end + 1]
        if np.all(~np.isfinite(segment)):
            peak_frame = start
            peak_value = np.nan
        else:
            local = int(np.nanargmax(np.abs(segment)))
            peak_frame = start + local
            peak_value = float(values[peak_frame])
        rows.append(
            {
                "file": file,
                "reason": reason,
                "detail": detail,
                "start_frame": start,
                "end_frame": end,
                "n_frames": end - start + 1,
                "peak_frame": peak_frame,
                "peak_time_sec": peak_frame / fps,
                "peak_value": peak_value,
                "threshold": threshold,
            }
        )


def summarize_file(
    file: str,
    arr: np.ndarray,
    bone_stats: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    frame_rows = []
    finite_frame = np.isfinite(arr).all(axis=(1, 2))
    nonfinite_mask = ~finite_frame
    abs_coord = np.nanmax(np.abs(arr), axis=(1, 2))
    coord_mask = abs_coord > args.coord_cap

    mid_hip, mid_shoulder = get_midpoints(arr)
    torso = mid_shoulder - mid_hip
    torso_length = np.linalg.norm(torso, axis=1)
    torso_y = torso[:, 1]
    pelvis_offset = np.linalg.norm(mid_hip, axis=1)

    torso_len_mask = np.abs(torso_length - 1.0) > 0.10
    torso_y_mask = torso_y < args.torso_y_min
    pelvis_mask = pelvis_offset > 1e-4

    speed = np.linalg.norm(np.diff(arr, axis=0), axis=2) * args.fps if arr.shape[0] > 1 else np.zeros((0, 17))
    max_speed_by_frame = np.r_[0.0, np.nanmax(speed, axis=1)] if speed.size else np.zeros(arr.shape[0])
    speed_mask = max_speed_by_frame > args.speed_cap

    add_run_rows(frame_rows, file, "missing_or_nonfinite", nonfinite_mask, nonfinite_mask.astype(float), 1.0, "any", args.fps)
    add_run_rows(frame_rows, file, "coordinate_abs_cap", coord_mask, abs_coord, args.coord_cap, "any_keypoint", args.fps)
    add_run_rows(frame_rows, file, "torso_length_not_one", torso_len_mask, torso_length, 0.10, "mid_shoulder-mid_hip", args.fps)
    add_run_rows(frame_rows, file, "upper_body_not_positive_y", torso_y_mask, torso_y, args.torso_y_min, "mid_shoulder_y", args.fps)
    add_run_rows(frame_rows, file, "pelvis_not_centered", pelvis_mask, pelvis_offset, 1e-4, "mid_hip_norm", args.fps)
    add_run_rows(frame_rows, file, "extreme_framewise_difference", speed_mask, max_speed_by_frame, args.speed_cap, "max_keypoint_speed", args.fps)

    lengths = bone_lengths(arr)
    max_bone_z = 0.0
    max_bone_z_name = ""
    for bone, values in lengths.items():
        median, scale = bone_stats[bone]
        z = (values - median) / scale if np.isfinite(scale) and scale > 0 else np.zeros_like(values)
        abs_z = np.abs(z)
        max_idx = int(np.nanargmax(abs_z)) if abs_z.size else 0
        if abs_z.size and float(abs_z[max_idx]) > max_bone_z:
            max_bone_z = float(abs_z[max_idx])
            max_bone_z_name = bone

        cap = DISTANCE_CAPS[bone]
        low_cap = max(0.02, median - args.robust_z * scale) if np.isfinite(median) and np.isfinite(scale) else 0.02
        high_cap = min(cap, median + args.robust_z * scale) if np.isfinite(median) and np.isfinite(scale) else cap
        bone_mask = (values < low_cap) | (values > high_cap)
        add_run_rows(
            frame_rows,
            file,
            "anatomically_implausible_bone_length",
            bone_mask,
            values,
            high_cap,
            bone,
            args.fps,
        )

    anomaly_frames = set()
    reasons = set()
    for row in frame_rows:
        anomaly_frames.update(range(row["start_frame"], row["end_frame"] + 1))
        reasons.add(row["reason"])

    summary = {
        "file": file,
        "frames": int(arr.shape[0]),
        "finite": bool(np.isfinite(arr).all()),
        "missing_values": int(np.size(arr) - np.isfinite(arr).sum()),
        "max_abs_coord": float(np.nanmax(abs_coord)),
        "max_keypoint_speed": float(np.nanmax(max_speed_by_frame)),
        "median_torso_length": float(np.nanmedian(torso_length)),
        "min_torso_y": float(np.nanmin(torso_y)),
        "median_torso_y": float(np.nanmedian(torso_y)),
        "max_pelvis_offset": float(np.nanmax(pelvis_offset)),
        "max_bone_robust_z": max_bone_z,
        "max_bone_robust_z_name": max_bone_z_name,
        "flagged": bool(frame_rows),
        "flag_reasons": ";".join(sorted(reasons)),
        "n_flagged_runs": len(frame_rows),
        "n_flagged_frames": len(anomaly_frames),
        "flagged_frame_fraction": len(anomaly_frames) / arr.shape[0] if arr.shape[0] else 0.0,
    }
    return summary, frame_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")

    arrays = load_arrays(paths)
    bone_stats = build_global_bone_stats(arrays)
    summaries = []
    frame_rows = []
    for file, arr in arrays.items():
        summary, rows = summarize_file(file, arr, bone_stats, args)
        summaries.append(summary)
        frame_rows.extend(rows)

    summary_fields = [
        "file",
        "frames",
        "finite",
        "missing_values",
        "max_abs_coord",
        "max_keypoint_speed",
        "median_torso_length",
        "min_torso_y",
        "median_torso_y",
        "max_pelvis_offset",
        "max_bone_robust_z",
        "max_bone_robust_z_name",
        "flagged",
        "flag_reasons",
        "n_flagged_runs",
        "n_flagged_frames",
        "flagged_frame_fraction",
    ]
    frame_fields = [
        "file",
        "reason",
        "detail",
        "start_frame",
        "end_frame",
        "n_frames",
        "peak_frame",
        "peak_time_sec",
        "peak_value",
        "threshold",
    ]

    summary_path = args.output_dir / "pose_notRotated_qc_file_summary.csv"
    frame_path = args.output_dir / "pose_notRotated_qc_flagged_runs.csv"
    inspect_path = args.output_dir / "pose_notRotated_qc_manual_inspection_list.csv"

    write_csv(summary_path, summaries, summary_fields)
    write_csv(frame_path, frame_rows, frame_fields)

    inspect = sorted(
        [row for row in summaries if row["flagged"]],
        key=lambda row: (
            row["missing_values"] > 0,
            row["max_keypoint_speed"],
            row["max_abs_coord"],
            row["max_bone_robust_z"],
            row["n_flagged_frames"],
        ),
        reverse=True,
    )
    write_csv(inspect_path, inspect, summary_fields)

    print(f"Scanned {len(summaries)} files")
    print(f"Files with missing/nonfinite values: {sum(row['missing_values'] > 0 for row in summaries)}")
    print(f"Files flagged for manual inspection: {len(inspect)}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {frame_path}")
    print(f"Wrote {inspect_path}")
    for row in inspect[:25]:
        print(
            f"{row['file']}: {row['flag_reasons']} "
            f"(max_speed={row['max_keypoint_speed']:.2f}, max_abs={row['max_abs_coord']:.2f}, "
            f"flagged_frames={row['n_flagged_frames']})"
        )


if __name__ == "__main__":
    main()
