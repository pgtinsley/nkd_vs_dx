#!/usr/bin/env python
"""Find frame-level anomalies in canonical pose arrays and plot QC visuals."""

from __future__ import annotations

import argparse
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

EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)

DISTANCE_PAIRS = {
    "wrist_to_wrist_distance": (9, 10, 4.0),
    "ankle_to_ankle_distance": (15, 16, 4.0),
    "shoulder_width": (5, 6, 3.0),
    "hip_width": (11, 12, 3.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("pose_estimate_data_npy_codex"))
    parser.add_argument("--output-dir", type=Path, default=Path("pose_estimate_data_npy_codex_anomaly_qc"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--speed-cap", type=float, default=15.0)
    parser.add_argument("--coord-cap", type=float, default=4.0)
    parser.add_argument("--context-sec", type=float, default=2.0)
    parser.add_argument("--max-plots", type=int, default=0, help="0 means plot every flagged file.")
    return parser.parse_args()


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    starts = np.r_[idx[0], idx[breaks]]
    ends = np.r_[idx[breaks - 1], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def summarize_pose(path: Path, fps: float, speed_cap: float, coord_cap: float) -> tuple[list[dict], dict]:
    arr = np.load(path)
    rows = []
    finite = np.isfinite(arr).all(axis=(1, 2))
    nonfinite_mask = ~finite

    if arr.shape[0] > 1:
        speed = np.linalg.norm(np.diff(arr, axis=0), axis=2) * fps
        speed_mask = speed > speed_cap
        speed_frame_mask = np.r_[False, speed_mask.any(axis=1)]
        for start, end in contiguous_runs(speed_frame_mask):
            seg = speed[max(start - 1, 0) : end]
            local = np.unravel_index(np.nanargmax(seg), seg.shape)
            peak_frame = max(start - 1, 0) + local[0] + 1
            keypoint_idx = int(local[1])
            rows.append(
                {
                    "file": path.name,
                    "reason": "high_keypoint_speed",
                    "start_frame": start,
                    "end_frame": end,
                    "n_frames": end - start + 1,
                    "peak_frame": peak_frame,
                    "peak_time_sec": peak_frame / fps,
                    "keypoint": COCO17_NAMES[keypoint_idx],
                    "peak_value": float(speed[peak_frame - 1, keypoint_idx]),
                    "threshold": speed_cap,
                }
            )
    else:
        speed = np.zeros((0, 17), dtype=float)
        speed_frame_mask = np.zeros(arr.shape[0], dtype=bool)

    coord_mask = np.nanmax(np.abs(arr), axis=(1, 2)) > coord_cap
    for start, end in contiguous_runs(coord_mask):
        seg = np.abs(arr[start : end + 1])
        local = np.unravel_index(np.nanargmax(seg), seg.shape)
        peak_frame = start + int(local[0])
        keypoint_idx = int(local[1])
        coord_idx = int(local[2])
        rows.append(
            {
                "file": path.name,
                "reason": "coord_abs_cap",
                "start_frame": start,
                "end_frame": end,
                "n_frames": end - start + 1,
                "peak_frame": peak_frame,
                "peak_time_sec": peak_frame / fps,
                "keypoint": f"{COCO17_NAMES[keypoint_idx]}_{'xy'[coord_idx]}",
                "peak_value": float(arr[peak_frame, keypoint_idx, coord_idx]),
                "threshold": coord_cap,
            }
        )

    for name, (i, j, cap) in DISTANCE_PAIRS.items():
        distance = np.linalg.norm(arr[:, i, :] - arr[:, j, :], axis=1)
        dist_mask = distance > cap
        for start, end in contiguous_runs(dist_mask):
            local = int(np.nanargmax(distance[start : end + 1]))
            peak_frame = start + local
            rows.append(
                {
                    "file": path.name,
                    "reason": name,
                    "start_frame": start,
                    "end_frame": end,
                    "n_frames": end - start + 1,
                    "peak_frame": peak_frame,
                    "peak_time_sec": peak_frame / fps,
                    "keypoint": f"{COCO17_NAMES[i]}-{COCO17_NAMES[j]}",
                    "peak_value": float(distance[peak_frame]),
                    "threshold": cap,
                }
            )

    for start, end in contiguous_runs(nonfinite_mask):
        rows.append(
            {
                "file": path.name,
                "reason": "nonfinite",
                "start_frame": start,
                "end_frame": end,
                "n_frames": end - start + 1,
                "peak_frame": start,
                "peak_time_sec": start / fps,
                "keypoint": "any",
                "peak_value": np.nan,
                "threshold": np.nan,
            }
        )

    summary = {
        "file": path.name,
        "frames": int(arr.shape[0]),
        "duration_sec": float(arr.shape[0] / fps),
        "max_abs_coord": float(np.nanmax(np.abs(arr))),
        "max_keypoint_speed": float(np.nanmax(speed)) if speed.size else 0.0,
        "n_anomaly_runs": len(rows),
        "n_anomaly_frames": int(len(set().union(*[set(range(r["start_frame"], r["end_frame"] + 1)) for r in rows])) if rows else 0),
        "path": str(path),
    }
    return rows, summary


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    valid = np.isfinite(points).all(axis=-1)
    xy = points[valid]
    if xy.size == 0:
        return
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    center = (lo + hi) / 2
    radius = max(float((hi - lo).max()) / 2, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def plot_skeleton(ax: plt.Axes, pose: np.ndarray, title: str, color: str) -> None:
    ax.scatter(pose[:, 0], pose[:, 1], s=22, color=color)
    for i, j in EDGES:
        ax.plot([pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]], color=color, alpha=0.75, linewidth=1.4)
    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    ax.set_title(title)


def plot_anomaly_file(path: Path, runs: pd.DataFrame, output_dir: Path, fps: float, context_sec: float) -> Path:
    arr = np.load(path)
    speed = np.linalg.norm(np.diff(arr, axis=0), axis=2) * fps if arr.shape[0] > 1 else np.zeros((0, 17))
    mean_speed = np.nanmean(speed, axis=1) if speed.size else np.zeros(1)
    max_speed = np.nanmax(speed, axis=1) if speed.size else np.zeros(1)
    t = np.arange(mean_speed.size) / fps
    top = runs.sort_values(["peak_value", "n_frames"], ascending=False).head(3)
    peak_frame = int(top.iloc[0]["peak_frame"])
    context = int(round(context_sec * fps))
    lo = max(0, peak_frame - context)
    hi = min(arr.shape[0] - 1, peak_frame + context)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.plot(t, mean_speed, linewidth=0.8, label="mean keypoint speed")
    ax_ts.plot(t, max_speed, linewidth=0.8, alpha=0.75, label="max keypoint speed")
    for _, row in runs.iterrows():
        ax_ts.axvspan(row["start_frame"] / fps, row["end_frame"] / fps, color="#b84a62", alpha=0.18)
    ax_ts.set_title(f"{path.name}: pose anomaly timeline")
    ax_ts.set_xlabel("time, seconds")
    ax_ts.set_ylabel("torso lengths / sec")
    ax_ts.legend(fontsize=8)

    frames = np.unique(np.clip([lo, max(0, peak_frame - 1), peak_frame, min(arr.shape[0] - 1, peak_frame + 1), hi], 0, arr.shape[0] - 1))
    colors = ["#2f6f73", "#7b6ba8", "#b84a62", "#d18f3f", "#555555"]
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)] + [fig.add_subplot(gs[2, i]) for i in range(3)]
    for ax in axes:
        ax.axis("off")
    for ax, frame, color in zip(axes, frames, colors):
        ax.axis("on")
        plot_skeleton(ax, arr[frame], f"frame {frame}", color)
        set_equal_axes(ax, arr[frames])

    text_ax = axes[-1]
    text_ax.axis("off")
    lines = [
        f"{r.reason}: frames {int(r.start_frame)}-{int(r.end_frame)}, peak {int(r.peak_frame)}, {r.keypoint}, value {float(r.peak_value):.2f}"
        for r in top.itertuples()
    ]
    text_ax.text(0, 0.9, "\n".join(lines), va="top", fontsize=9)

    out = output_dir / f"{path.stem}_pose_anomaly_qc.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    pose_paths = sorted(path for path in args.pose_dir.glob("*.npy") if not path.stem.endswith("_preInterpolation"))
    for path in pose_paths:
        rows, summary = summarize_pose(path, args.fps, args.speed_cap, args.coord_cap)
        all_rows.extend(rows)
        summaries.append(summary)

    runs = pd.DataFrame(all_rows)
    summary = pd.DataFrame(summaries)
    runs.to_csv(args.output_dir / "pose_anomaly_runs.csv", index=False)
    summary.to_csv(args.output_dir / "pose_anomaly_file_summary.csv", index=False)

    flagged = summary[summary["n_anomaly_runs"] > 0].sort_values(
        ["n_anomaly_frames", "max_keypoint_speed", "max_abs_coord"], ascending=False
    )
    if args.max_plots > 0:
        flagged = flagged.head(args.max_plots)

    plot_rows = []
    for row in flagged.itertuples():
        file_runs = runs[runs["file"] == row.file]
        out = plot_anomaly_file(Path(row.path), file_runs, plot_dir, args.fps, args.context_sec)
        plot_rows.append({"file": row.file, "plot_path": str(out)})

    pd.DataFrame(plot_rows).to_csv(args.output_dir / "pose_anomaly_plot_manifest.csv", index=False)
    print(f"Scanned {len(summary)} pose arrays")
    print(f"Flagged {len(flagged)} files using speed>{args.speed_cap}, coord>{args.coord_cap}, and body-distance caps")
    print(f"Wrote runs: {args.output_dir / 'pose_anomaly_runs.csv'}")
    print(f"Wrote file summary: {args.output_dir / 'pose_anomaly_file_summary.csv'}")
    print(f"Wrote {len(plot_rows)} plots to {plot_dir}")
    if not flagged.empty:
        print(flagged.head(20)[["file", "n_anomaly_runs", "n_anomaly_frames", "max_keypoint_speed", "max_abs_coord"]].to_string(index=False))


if __name__ == "__main__":
    main()
