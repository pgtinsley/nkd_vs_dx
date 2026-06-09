#!/usr/bin/env python
"""Plot high-confidence LOOCV mistakes with pose trajectory context."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loocv-csv", type=Path, default=Path("loocv_inceptiontime_young_shared_features.csv"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("df_meta_constructed_with_pose_qc_codex.csv"))
    parser.add_argument("--pose-dir", type=Path, default=Path("pose_estimate_data_npy_codex"))
    parser.add_argument("--output-dir", type=Path, default=Path("loocv_problematic_trajectory_plots"))
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    valid = np.isfinite(points).all(axis=-1)
    xy = points[valid]
    if xy.size == 0:
        return
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max(float((hi - lo).max()) / 2.0, 0.75)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def draw_skeleton(ax: plt.Axes, pose: np.ndarray, color: str = "#d95f02") -> None:
    ax.scatter(pose[:, 0], pose[:, 1], s=10, color="black", zorder=3)
    for i, j in EDGES:
        if np.isfinite(pose[[i, j]]).all():
            ax.plot([pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]], color=color, linewidth=1.1, zorder=2)


def finite_frame(arr: np.ndarray) -> int | None:
    finite = np.isfinite(arr).all(axis=(1, 2))
    if not finite.any():
        return None
    center = arr.shape[0] // 2
    candidates = np.flatnonzero(finite)
    return int(candidates[np.argmin(np.abs(candidates - center))])


def summarize_pose(arr: np.ndarray, fps: float) -> dict[str, float]:
    missing = float(1.0 - np.isfinite(arr).all(axis=2).mean())
    if arr.shape[0] > 1:
        step = np.linalg.norm(np.diff(arr, axis=0), axis=2)
        p99_step = float(np.nanpercentile(step, 99))
        speed = step * fps
        mean_speed = float(np.nanmean(np.nanmean(speed, axis=1)))
        max_speed = float(np.nanmax(speed))
    else:
        p99_step = np.nan
        mean_speed = np.nan
        max_speed = np.nan
    max_abs_coord = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else np.nan
    return {
        "missing_keypoint_fraction": missing,
        "p99_step": p99_step,
        "mean_keypoint_speed": mean_speed,
        "max_keypoint_speed": max_speed,
        "max_abs_coord": max_abs_coord,
    }


def panel_label(row: pd.Series, pose_stats: dict[str, float]) -> str:
    parts = [
        f"{row.file}",
        f"age {row.adjusted_age_weeks:.1f}w",
        f"true {row.y_true_label} -> pred {row.y_pred_label}",
        f"DX p={row.dx_proba:.3f}, conf={row.pred_confidence:.3f}",
        f"dx: {row.get('diagnosis_singular', 'NA')}",
        f"prem {row.get('prematurity', 'NA')}, sex {row.get('sex', 'NA')}",
        f"GMOS gj {row.get('gmos_globaljudgement_final', 'NA')}",
        f"flagged {row.get('flagged', 'NA')}",
        f"missing {pose_stats['missing_keypoint_fraction']:.1%}, p99 step {pose_stats['p99_step']:.2f}",
    ]
    return "\n".join(parts)


def plot_cloud_panel(ax: plt.Axes, arr: np.ndarray, title: str, info: str) -> None:
    limb_arr = arr[:, list(LIMB_POINTS.values()), :]
    xy = limb_arr.reshape(-1, 2)
    valid = np.isfinite(xy).all(axis=1)
    if valid.any():
        ax.scatter(xy[valid, 0], xy[valid, 1], s=3, color="0.55", alpha=0.18, linewidths=0)
    frame = finite_frame(arr)
    if frame is None:
        ax.text(0.5, 0.5, "no fully finite frames", transform=ax.transAxes, ha="center", va="center", color="#b84a62")
    else:
        draw_skeleton(ax, arr[frame])
        ax.text(0.02, 0.02, f"frame {frame}", transform=ax.transAxes, fontsize=7, va="bottom")
    set_equal_axes(ax, arr)
    ax.axhline(0, color="0.90", linewidth=0.7)
    ax.axvline(0, color="0.90", linewidth=0.7)
    ax.grid(color="0.92", linewidth=0.5)
    ax.set_title(title, fontsize=8)
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.88},
    )


def plot_individual(path: Path, arr: np.ndarray, row: pd.Series, pose_stats: dict[str, float], out_dir: Path, fps: float) -> Path:
    time = np.arange(arr.shape[0]) / fps
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    plot_cloud_panel(axes[0], arr, "Coordinate cloud + representative skeleton", panel_label(row, pose_stats))
    for name, idx in LIMB_POINTS.items():
        axes[1].plot(arr[:, idx, 0], arr[:, idx, 1], linewidth=0.8, alpha=0.75, label=name)
    set_equal_axes(axes[1], arr[:, list(LIMB_POINTS.values()), :])
    axes[1].axhline(0, color="0.90", linewidth=0.7)
    axes[1].axvline(0, color="0.90", linewidth=0.7)
    axes[1].set_title("Hand and foot trajectories")
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle(f"{row.error_rank}. {path.name}")
    out = out_dir / f"{row.error_rank:02d}_{path.stem}_problematic_trajectory.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for name, idx in LIMB_POINTS.items():
        axes[0].plot(time, arr[:, idx, 0], linewidth=0.7, label=f"{name} x")
        axes[1].plot(time, arr[:, idx, 1], linewidth=0.7, label=f"{name} y")
    speed = np.linalg.norm(np.diff(arr, axis=0), axis=2) * fps if arr.shape[0] > 1 else np.zeros((0, 17))
    if speed.size:
        axes[2].plot(time[1:], np.nanmean(speed, axis=1), linewidth=0.8, label="mean keypoint speed")
        axes[2].plot(time[1:], np.nanmax(speed, axis=1), linewidth=0.7, alpha=0.75, label="max keypoint speed")
    axes[0].set_ylabel("x")
    axes[1].set_ylabel("y")
    axes[2].set_ylabel("torso lengths/sec")
    axes[2].set_xlabel("time, seconds")
    for ax in axes:
        ax.legend(ncol=4, fontsize=7)
        ax.grid(color="0.92", linewidth=0.5)
    fig.suptitle(panel_label(row, pose_stats))
    line_out = out_dir / f"{row.error_rank:02d}_{path.stem}_problematic_timeseries.png"
    fig.savefig(line_out, dpi=170)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = args.output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    loocv = pd.read_csv(args.loocv_csv)
    loocv["nkd_proba"] = 1.0 - loocv["dx_proba"]
    loocv["pred_confidence"] = np.where(loocv["y_pred"].eq(1), loocv["dx_proba"], loocv["nkd_proba"])
    loocv["true_label_proba"] = np.where(loocv["y_true"].eq(1), loocv["dx_proba"], loocv["nkd_proba"])
    mistakes = loocv[loocv["y_true"].ne(loocv["y_pred"])].copy()
    mistakes = mistakes.sort_values(["pred_confidence", "true_label_proba"], ascending=[False, True]).head(args.top_n)

    metadata_cols = [
        "file",
        "record_id",
        "subject_unique_id",
        "diagnosis",
        "diagnosis_singular",
        "prematurity",
        "sex",
        "race",
        "gmos_globaljudgement_final",
        "gmos_totalscore_final",
        "mos_totalscore_final",
        "consensus_normalyn_final",
        "duration_sec",
        "frames",
        "flagged",
        "reasons",
        "path",
    ]
    meta = pd.read_csv(args.metadata_csv, usecols=lambda c: c in metadata_cols).drop_duplicates("file")
    selected = mistakes.merge(meta, on="file", how="left", suffixes=("", "_meta"))
    selected.insert(0, "error_rank", np.arange(1, len(selected) + 1))

    ncols = 4
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.7 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    manifest_rows = []
    for ax, row in zip(axes, selected.itertuples(index=False)):
        pose_path = Path(row.path) if isinstance(row.path, str) and row.path else args.pose_dir / row.file
        if not pose_path.exists():
            ax.text(0.5, 0.5, f"missing pose file\n{row.file}", transform=ax.transAxes, ha="center", va="center")
            ax.axis("off")
            manifest_rows.append(row._asdict() | {"pose_path": str(pose_path), "plot_path": "", "pose_file_found": False})
            continue
        arr = np.load(pose_path)
        pose_stats = summarize_pose(arr, args.fps)
        info = panel_label(pd.Series(row._asdict()), pose_stats)
        plot_cloud_panel(ax, arr, f"{row.error_rank}. {row.record_id}", info)
        individual_path = plot_individual(pose_path, arr, pd.Series(row._asdict()), pose_stats, individual_dir, args.fps)
        manifest_rows.append(
            row._asdict()
            | pose_stats
            | {"pose_path": str(pose_path), "plot_path": str(individual_path), "pose_file_found": True}
        )

    for ax in axes[len(selected) :]:
        ax.axis("off")

    fig.suptitle("High-confidence LOOCV mistakes: coordinate cloud and representative skeleton", fontsize=16, weight="bold")
    grid_path = args.output_dir / "top_problematic_loocv_trajectory_skeletons.png"
    fig.savefig(grid_path, dpi=170)
    plt.close(fig)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "top_problematic_loocv_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"Wrote grid plot: {grid_path}")
    print(f"Wrote individual plots: {individual_dir}")
    print(f"Wrote manifest: {manifest_path}")
    print(
        manifest[
            [
                "error_rank",
                "file",
                "adjusted_age_weeks",
                "diagnosis_singular",
                "y_true_label",
                "y_pred_label",
                "dx_proba",
                "pred_confidence",
                "true_label_proba",
                "flagged",
                "pose_file_found",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
