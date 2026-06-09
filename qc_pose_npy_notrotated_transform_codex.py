#!/usr/bin/env python
"""QC not-rotated canonical pose arrays without 2D bone-length screening."""

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

LEFT_SHOULDER = COCO17_NAMES.index("left_shoulder")
RIGHT_SHOULDER = COCO17_NAMES.index("right_shoulder")
LEFT_HIP = COCO17_NAMES.index("left_hip")
RIGHT_HIP = COCO17_NAMES.index("right_hip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("pose_estimate_data_npy_codex_notRotated"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pose_estimate_data_npy_codex_notRotated_qc_transform_only"),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--coord-cap", type=float, default=4.0)
    parser.add_argument("--speed-cap", type=float, default=15.0)
    parser.add_argument("--torso-length-tolerance", type=float, default=0.10)
    parser.add_argument("--pelvis-offset-cap", type=float, default=1e-4)
    parser.add_argument("--torso-y-min", type=float, default=0.20)
    return parser.parse_args()


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    starts = np.r_[idx[0], idx[breaks]]
    ends = np.r_[idx[breaks - 1], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def add_runs(
    rows: list[dict],
    file: str,
    reason: str,
    detail: str,
    mask: np.ndarray,
    values: np.ndarray,
    threshold: float,
    fps: float,
) -> None:
    for start, end in contiguous_runs(mask):
        segment = values[start : end + 1]
        if segment.size and np.isfinite(segment).any():
            local = int(np.nanargmax(np.abs(segment)))
            peak_frame = start + local
            peak_value = float(values[peak_frame])
        else:
            peak_frame = start
            peak_value = np.nan
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


def summarize_file(path: Path, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    arr = np.load(path)
    rows = []

    mid_hip = (arr[:, LEFT_HIP, :] + arr[:, RIGHT_HIP, :]) / 2.0
    mid_shoulder = (arr[:, LEFT_SHOULDER, :] + arr[:, RIGHT_SHOULDER, :]) / 2.0
    torso = mid_shoulder - mid_hip
    torso_length = np.linalg.norm(torso, axis=1)
    torso_y = torso[:, 1]
    pelvis_offset = np.linalg.norm(mid_hip, axis=1)

    finite_frame = np.isfinite(arr).all(axis=(1, 2))
    missing_mask = ~finite_frame
    abs_coord = np.nanmax(np.abs(arr), axis=(1, 2))
    coord_mask = abs_coord > args.coord_cap
    torso_length_error = np.abs(torso_length - 1.0)
    torso_length_mask = torso_length_error > args.torso_length_tolerance
    pelvis_mask = pelvis_offset > args.pelvis_offset_cap
    torso_y_mask = torso_y < args.torso_y_min
    negative_torso_y_mask = torso_y < 0.0

    if arr.shape[0] > 1:
        speed = np.linalg.norm(np.diff(arr, axis=0), axis=2) * args.fps
        max_speed = np.r_[0.0, np.nanmax(speed, axis=1)]
    else:
        max_speed = np.zeros(arr.shape[0])
    speed_mask = max_speed > args.speed_cap

    add_runs(rows, path.name, "missing_or_nonfinite", "any", missing_mask, missing_mask.astype(float), 1.0, args.fps)
    add_runs(rows, path.name, "coordinate_abs_cap", "any_keypoint", coord_mask, abs_coord, args.coord_cap, args.fps)
    add_runs(
        rows,
        path.name,
        "torso_length_not_unit",
        "abs(mid_shoulder-mid_hip length - 1)",
        torso_length_mask,
        torso_length_error,
        args.torso_length_tolerance,
        args.fps,
    )
    add_runs(rows, path.name, "pelvis_not_centered", "mid_hip_norm", pelvis_mask, pelvis_offset, args.pelvis_offset_cap, args.fps)
    add_runs(rows, path.name, "upper_body_not_positive_y", "mid_shoulder_y", torso_y_mask, torso_y, args.torso_y_min, args.fps)
    add_runs(rows, path.name, "extreme_framewise_difference", "max_keypoint_speed", speed_mask, max_speed, args.speed_cap, args.fps)

    anomaly_frames = set()
    reasons = set()
    for row in rows:
        anomaly_frames.update(range(row["start_frame"], row["end_frame"] + 1))
        reasons.add(row["reason"])

    summary = {
        "file": path.name,
        "frames": int(arr.shape[0]),
        "finite": bool(np.isfinite(arr).all()),
        "missing_values": int(np.size(arr) - np.isfinite(arr).sum()),
        "max_abs_coord": float(np.nanmax(abs_coord)),
        "max_keypoint_speed": float(np.nanmax(max_speed)),
        "median_torso_length": float(np.nanmedian(torso_length)),
        "max_torso_length_error": float(np.nanmax(torso_length_error)),
        "min_torso_y": float(np.nanmin(torso_y)),
        "median_torso_y": float(np.nanmedian(torso_y)),
        "negative_torso_y_frames": int(negative_torso_y_mask.sum()),
        "upper_body_not_positive_y_frames": int(torso_y_mask.sum()),
        "max_pelvis_offset": float(np.nanmax(pelvis_offset)),
        "flagged": bool(rows),
        "flag_reasons": ";".join(sorted(reasons)),
        "n_flagged_runs": len(rows),
        "n_flagged_frames": len(anomaly_frames),
        "flagged_frame_fraction": len(anomaly_frames) / arr.shape[0] if arr.shape[0] else 0.0,
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")

    summaries = []
    run_rows = []
    for path in paths:
        summary, rows = summarize_file(path, args)
        summaries.append(summary)
        run_rows.extend(rows)

    summary_fields = [
        "file",
        "frames",
        "finite",
        "missing_values",
        "max_abs_coord",
        "max_keypoint_speed",
        "median_torso_length",
        "max_torso_length_error",
        "min_torso_y",
        "median_torso_y",
        "negative_torso_y_frames",
        "upper_body_not_positive_y_frames",
        "max_pelvis_offset",
        "flagged",
        "flag_reasons",
        "n_flagged_runs",
        "n_flagged_frames",
        "flagged_frame_fraction",
    ]
    run_fields = [
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

    flagged = [row for row in summaries if row["flagged"]]
    flagged = sorted(
        flagged,
        key=lambda row: (
            row["missing_values"] > 0,
            row["max_abs_coord"],
            row["max_keypoint_speed"],
            row["upper_body_not_positive_y_frames"],
            row["max_torso_length_error"],
            row["n_flagged_frames"],
        ),
        reverse=True,
    )

    summary_path = args.output_dir / "pose_notRotated_transform_qc_file_summary.csv"
    runs_path = args.output_dir / "pose_notRotated_transform_qc_flagged_runs.csv"
    inspect_path = args.output_dir / "pose_notRotated_transform_qc_manual_inspection_list.csv"
    files_path = args.output_dir / "pose_notRotated_transform_qc_manual_inspection_files.txt"

    write_csv(summary_path, summaries, summary_fields)
    write_csv(runs_path, run_rows, run_fields)
    write_csv(inspect_path, flagged, summary_fields)
    files_path.write_text("\n".join(row["file"] for row in flagged) + ("\n" if flagged else ""), encoding="utf-8")

    print(f"Scanned {len(summaries)} files")
    print(f"Flagged {len(flagged)} files")
    print(f"Files with missing/nonfinite values: {sum(row['missing_values'] > 0 for row in summaries)}")
    print(f"Files with coordinate cap violations: {sum(row['max_abs_coord'] > args.coord_cap for row in summaries)}")
    print(f"Files with speed cap violations: {sum(row['max_keypoint_speed'] > args.speed_cap for row in summaries)}")
    print(
        "Files with non-unit torso lengths: "
        f"{sum(row['max_torso_length_error'] > args.torso_length_tolerance for row in summaries)}"
    )
    print(f"Files with upper body y below {args.torso_y_min}: {sum(row['upper_body_not_positive_y_frames'] > 0 for row in summaries)}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {runs_path}")
    print(f"Wrote {inspect_path}")
    print(f"Wrote {files_path}")


if __name__ == "__main__":
    main()
