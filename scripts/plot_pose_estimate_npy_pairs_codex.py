#!/usr/bin/env python
"""Render paired trajectory sanity plots for two pose .npy directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot matched pose trajectories from two .npy directories.")
    parser.add_argument("--rotated-dir", type=Path, default=Path("pose_estimate_data_npy_codex"))
    parser.add_argument("--not-rotated-dir", type=Path, default=Path("pose_estimate_data_npy_codex_notRotated"))
    parser.add_argument("--output-dir", type=Path, default=Path("pose_estimate_data_npy_codex_notRotated_pair_plots"))
    parser.add_argument("--max-pairs", type=int, default=4)
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


def plot_skeletons(ax: plt.Axes, arr: np.ndarray, title: str) -> None:
    sample_frames = np.unique(np.linspace(0, arr.shape[0] - 1, min(6, arr.shape[0]), dtype=int))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(sample_frames)))

    for frame_idx, color in zip(sample_frames, colors):
        pose = arr[frame_idx]
        ax.scatter(pose[:, 0], pose[:, 1], s=10, color=color, alpha=0.9)
        for start, end in EDGES:
            ax.plot(
                [pose[start, 0], pose[end, 0]],
                [pose[start, 1], pose[end, 1]],
                color=color,
                alpha=0.55,
                linewidth=1.0,
            )

    set_equal_axes(ax, arr[sample_frames])
    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("x, torso lengths from mid-hip")
    ax.set_ylabel("y, torso lengths from mid-hip")


def plot_trajectories(ax: plt.Axes, arr: np.ndarray, title: str) -> None:
    for name, idx in LIMB_POINTS.items():
        ax.plot(arr[:, idx, 0], arr[:, idx, 1], linewidth=0.8, alpha=0.75, label=name)

    set_equal_axes(ax, arr[:, list(LIMB_POINTS.values()), :])
    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("x, torso lengths from mid-hip")
    ax.set_ylabel("y, torso lengths from mid-hip")
    ax.legend(fontsize=8, loc="best")


def select_common_files(rotated_dir: Path, not_rotated_dir: Path, max_pairs: int) -> list[str]:
    rotated = {path.name for path in rotated_dir.glob("*.npy")}
    not_rotated = {path.name for path in not_rotated_dir.glob("*.npy")}
    common = sorted(rotated & not_rotated)
    if max_pairs > 0 and len(common) > max_pairs:
        indices = np.unique(np.linspace(0, len(common) - 1, max_pairs, dtype=int))
        common = [common[idx] for idx in indices]
    return common


def plot_pair(name: str, rotated_dir: Path, not_rotated_dir: Path, output_dir: Path) -> Path:
    rotated = np.load(rotated_dir / name)
    not_rotated = np.load(not_rotated_dir / name)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    plot_skeletons(axes[0, 0], rotated, "Rotated canonical skeleton snapshots")
    plot_skeletons(axes[0, 1], not_rotated, "Not-rotated skeleton snapshots")
    plot_trajectories(axes[1, 0], rotated, "Rotated hand and foot trajectories")
    plot_trajectories(axes[1, 1], not_rotated, "Not-rotated hand and foot trajectories")

    fig.suptitle(name)
    output_path = output_dir / f"{Path(name).stem}_rotated_vs_notRotated.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = select_common_files(args.rotated_dir, args.not_rotated_dir, args.max_pairs)
    if not common:
        raise FileNotFoundError("No matching .npy files found between the two directories.")

    print("file,plot_path")
    for name in common:
        output_path = plot_pair(name, args.rotated_dir, args.not_rotated_dir, args.output_dir)
        print(f"{name},{output_path}")


if __name__ == "__main__":
    main()
