#!/usr/bin/env python
"""Plot canonical RTMPose NumPy arrays for visual sanity checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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

LIMB_POINTS = {
    "left wrist": 9,
    "right wrist": 10,
    "left ankle": 15,
    "right ankle": 16,
}

PROXIMAL_POINTS = {
    "left wrist": 5,
    "right wrist": 6,
    "left ankle": 11,
    "right ankle": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pose .npy files created by the Codex converter.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("pose_estimate_data_npy_codex"),
        help="Directory containing converted .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pose_estimate_data_npy_codex_plots"),
        help="Directory where plot PNG files will be written.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Number of sorted .npy files to plot. Use 0 or a negative value for all files.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second used for the time axis.",
    )
    return parser.parse_args()


def set_equal_axes(ax: plt.Axes, arr: np.ndarray) -> None:
    valid = np.isfinite(arr).all(axis=-1)
    xy = arr[valid]
    if xy.size == 0:
        return

    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    radius = max(float((max_xy - min_xy).max()) / 2.0, 0.75)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def plot_skeleton(ax: plt.Axes, pose: np.ndarray, color: str, label: str) -> None:
    ax.scatter(pose[:, 0], pose[:, 1], s=18, color=color, alpha=0.9, label=label)
    for start, end in EDGES:
        ax.plot(
            [pose[start, 0], pose[end, 0]],
            [pose[start, 1], pose[end, 1]],
            color=color,
            alpha=0.65,
            linewidth=1.5,
        )


def plot_file(npy_path: Path, output_dir: Path, fps: float) -> dict[str, float | str | int]:
    arr = np.load(npy_path)
    n_frames = arr.shape[0]
    time = np.arange(n_frames) / fps
    sample_frames = np.unique(np.linspace(0, n_frames - 1, min(6, n_frames), dtype=int))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(sample_frames)))
    for frame_idx, color in zip(sample_frames, colors):
        plot_skeleton(axes[0], arr[frame_idx], color, f"frame {frame_idx}")
    set_equal_axes(axes[0], arr[sample_frames])
    axes[0].axhline(0, color="0.85", linewidth=0.8)
    axes[0].axvline(0, color="0.85", linewidth=0.8)
    axes[0].set_title("Canonical skeleton snapshots")
    axes[0].set_xlabel("x, torso lengths from mid-hip")
    axes[0].set_ylabel("y, torso lengths from mid-hip")
    axes[0].legend(fontsize=8, loc="best")

    for name, idx in LIMB_POINTS.items():
        axes[1].plot(arr[:, idx, 0], arr[:, idx, 1], linewidth=0.9, alpha=0.75, label=name)
    set_equal_axes(axes[1], arr[:, list(LIMB_POINTS.values()), :])
    axes[1].axhline(0, color="0.85", linewidth=0.8)
    axes[1].axvline(0, color="0.85", linewidth=0.8)
    axes[1].set_title("Hand and foot trajectories")
    axes[1].set_xlabel("x, torso lengths from mid-hip")
    axes[1].set_ylabel("y, torso lengths from mid-hip")
    axes[1].legend(fontsize=8, loc="best")

    scatter_path = output_dir / f"{npy_path.stem}_scatter_trajectories.png"
    fig.suptitle(npy_path.name)
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    for name, idx in LIMB_POINTS.items():
        limb_centroid = np.nanmean(arr[:, idx, :], axis=0)
        limb_centroid_distance = np.linalg.norm(arr[:, idx, :] - limb_centroid, axis=1)
        axes[0].plot(time, limb_centroid_distance, linewidth=0.9, label=name)
    axes[0].set_title("Limb distance from its time-series centroid")
    axes[0].set_ylabel("torso lengths")
    axes[0].legend(ncol=4, fontsize=8)

    for name, idx in LIMB_POINTS.items():
        proximal_idx = PROXIMAL_POINTS[name]
        proximal_distance = np.linalg.norm(arr[:, idx, :] - arr[:, proximal_idx, :], axis=1)
        axes[1].plot(time, proximal_distance, linewidth=0.9, label=name)
    axes[1].set_title("Distal limb distance to proximal joint")
    axes[1].set_ylabel("torso lengths")
    axes[1].legend(ncol=4, fontsize=8)

    per_frame_speed = np.linalg.norm(np.diff(arr, axis=0), axis=2)
    mean_speed = np.nanmean(per_frame_speed, axis=1) * fps
    max_speed = np.nanmax(per_frame_speed, axis=1) * fps
    axes[2].plot(time[1:], mean_speed, color="tab:blue", linewidth=0.8, label="mean keypoint speed")
    axes[2].plot(time[1:], max_speed, color="tab:red", linewidth=0.7, alpha=0.7, label="max keypoint speed")
    axes[2].set_title("Frame-to-frame movement")
    axes[2].set_ylabel("torso lengths / second")
    axes[2].legend(fontsize=8)

    left_wrist = arr[:, LIMB_POINTS["left wrist"], 1]
    right_wrist = arr[:, LIMB_POINTS["right wrist"], 1]
    left_ankle = arr[:, LIMB_POINTS["left ankle"], 1]
    right_ankle = arr[:, LIMB_POINTS["right ankle"], 1]
    axes[3].plot(time, left_wrist, label="left wrist y", linewidth=0.8)
    axes[3].plot(time, right_wrist, label="right wrist y", linewidth=0.8)
    axes[3].plot(time, left_ankle, label="left ankle y", linewidth=0.8)
    axes[3].plot(time, right_ankle, label="right ankle y", linewidth=0.8)
    axes[3].set_title("Vertical limb position")
    axes[3].set_xlabel("time, seconds")
    axes[3].set_ylabel("torso lengths")
    axes[3].legend(ncol=4, fontsize=8)

    line_path = output_dir / f"{npy_path.stem}_lineplots.png"
    fig.suptitle(npy_path.name)
    fig.savefig(line_path, dpi=160)
    plt.close(fig)

    return {
        "file": npy_path.name,
        "frames": n_frames,
        "duration_sec": n_frames / fps,
        "mean_speed": float(np.nanmean(mean_speed)),
        "p95_speed": float(np.nanpercentile(mean_speed, 95)),
        "max_keypoint_speed": float(np.nanmax(max_speed)),
        "scatter_path": str(scatter_path),
        "line_path": str(line_path),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npy_paths = sorted(args.input_dir.glob("*.npy"))
    if args.max_files > 0:
        npy_paths = npy_paths[: args.max_files]
    if not npy_paths:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")

    print("file,frames,duration_sec,mean_speed,p95_speed,max_keypoint_speed")
    for npy_path in npy_paths:
        summary = plot_file(npy_path, args.output_dir, args.fps)
        print(
            "{file},{frames},{duration_sec:.1f},{mean_speed:.4f},{p95_speed:.4f},{max_keypoint_speed:.4f}".format(
                **summary
            )
        )


if __name__ == "__main__":
    main()
