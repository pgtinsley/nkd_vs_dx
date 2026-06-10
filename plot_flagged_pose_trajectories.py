#!/usr/bin/env python
"""Plot trajectory summaries for pose arrays listed in a QC CSV."""

from __future__ import annotations

import argparse
import csv
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from qc_pose_npy_statistical_outliers import COCO17_NAMES, DISTAL_KEYPOINTS, LIMB_SEGMENTS


DISTAL_COLORS = {
    "left_wrist": "#0072B2",
    "right_wrist": "#D55E00",
    "left_ankle": "#009E73",
    "right_ankle": "#CC79A7",
}

METADATA_LABEL_FIELDS = (
    ("prematurity", "Prematurity"),
    ("adjusted_age_weeks", "Adjusted age"),
    ("final_code_for_ai_str", "Final code"),
    ("diagnosis", "Diagnosis"),
)


@dataclass(frozen=True)
class MetadataRow:
    start_frame: int
    stop_frame: int
    labels: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectory plots for arrays listed in a QC CSV."
    )
    parser.add_argument(
        "--qc-csv",
        type=Path,
        default=Path("qc_pose_npy_statistical_outliers.csv"),
        help="QC CSV containing flagged rows from qc_pose_npy_statistical_outliers.py.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("pose_estimate_data_npy"),
        help="Directory containing the .npy pose arrays.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp_qc_pose_trajectory_plots"),
        help="Root directory where PNG plots will be written.",
    )
    parser.add_argument(
        "--selection",
        choices=("flagged", "unflagged"),
        default="flagged",
        help="Which QC rows to plot. Plots are written to a matching subdirectory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate used for velocity traces.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("df_meta.csv"),
        help=(
            "Metadata CSV used for plot labels, and required for snippet frame "
            "ranges when --gma-snippet is enabled."
        ),
    )
    parser.add_argument(
        "--gma-snippet",
        action="store_true",
        help=(
            "Plot only frames between gma_video_start_1_fnum and "
            "gma_video_stop_1_fnum. Rows without matching usable metadata are skipped."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit the number of selected files plotted. Use 0 for all.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random"),
        default="first",
        help="How to choose rows when --max-files is smaller than the selected set.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used when --sample-mode random is selected.",
    )
    return parser.parse_args()


def read_selected_rows(
    qc_csv: Path,
    *,
    selection: str,
    max_files: int,
    sample_mode: str,
    random_seed: int,
) -> list[dict[str, str]]:
    rows = []
    with qc_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            is_flagged = row.get("flagged", "").lower() == "true"
            if (selection == "flagged" and is_flagged) or (selection == "unflagged" and not is_flagged):
                rows.append(row)
    if sample_mode == "random":
        rng = np.random.default_rng(random_seed)
        order = rng.permutation(len(rows))
        rows = [rows[int(idx)] for idx in order]
    if max_files > 0 and not rows:
        return rows
    return rows


def parse_int_field(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(round(float(value)))
    except ValueError:
        return None


def load_metadata(metadata_csv: Path) -> dict[str, MetadataRow]:
    metadata = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_stem = row.get("video_stem", "").strip()
            if not video_stem:
                continue
            start_frame = parse_int_field(row.get("gma_video_start_1_fnum", ""))
            stop_frame = parse_int_field(row.get("gma_video_stop_1_fnum", ""))
            if start_frame is None or stop_frame is None:
                continue
            labels = {
                label: row.get(field, "").strip()
                for field, label in METADATA_LABEL_FIELDS
            }
            metadata[video_stem] = MetadataRow(
                start_frame=start_frame,
                stop_frame=stop_frame,
                labels=labels,
            )
    return metadata


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def load_pose(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[1:] != (17, 2):
        raise ValueError(f"expected shape (n_frames, 17, 2), got {arr.shape}")
    return arr.astype(float, copy=False)


def crop_to_gma_snippet(arr: np.ndarray, metadata: MetadataRow) -> tuple[np.ndarray, tuple[int, int]] | None:
    start = max(metadata.start_frame, 0)
    stop = min(metadata.stop_frame, arr.shape[0] - 1)
    if stop < start:
        return None
    cropped = arr[start : stop + 1]
    if cropped.shape[0] < 2:
        return None
    return cropped, (start, stop)


def keypoint_speed(arr: np.ndarray, fps: float) -> np.ndarray:
    if arr.shape[0] < 2:
        return np.empty((0, arr.shape[1]), dtype=float)
    return np.linalg.norm(np.diff(arr, axis=0), axis=2) * fps


def limb_lengths(arr: np.ndarray) -> np.ndarray:
    lengths = []
    for _, start_name, end_name in LIMB_SEGMENTS:
        start = COCO17_NAMES.index(start_name)
        end = COCO17_NAMES.index(end_name)
        lengths.append(np.linalg.norm(arr[:, start, :] - arr[:, end, :], axis=1))
    return np.stack(lengths, axis=1)


def add_colored_trajectory(ax: plt.Axes, xy: np.ndarray, color: str, label: str) -> None:
    finite = np.isfinite(xy).all(axis=1)
    if np.count_nonzero(finite) < 2:
        return

    xy = xy[finite]
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fade = np.linspace(0.18, 0.95, len(segments))
    collection = LineCollection(segments, colors=color, linewidths=1.2, alpha=0.85)
    collection.set_array(fade)
    ax.add_collection(collection)
    ax.scatter(xy[0, 0], xy[0, 1], color=color, s=18, marker="o", label=f"{label} start")
    ax.scatter(xy[-1, 0], xy[-1, 1], color=color, s=32, marker="x", label=f"{label} end")


def add_metadata_legend(ax: plt.Axes, metadata: MetadataRow | None) -> None:
    if metadata is None:
        return
    lines = []
    for label, value in metadata.labels.items():
        lines.append(f"{label}: {value or 'unknown'}")
    ax.text(
        0.01,
        0.99,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "alpha": 0.88,
        },
    )


def plot_one(
    row: dict[str, str],
    input_dir: Path,
    output_dir: Path,
    fps: float,
    *,
    metadata: MetadataRow | None = None,
    gma_snippet: bool = False,
) -> Path | None:
    pose_path = input_dir / row["file"]
    arr = load_pose(pose_path)
    snippet_range = None
    if gma_snippet:
        if metadata is None:
            return None
        cropped = crop_to_gma_snippet(arr, metadata)
        if cropped is None:
            return None
        arr, snippet_range = cropped

    speeds = keypoint_speed(arr, fps)
    distal_speeds = speeds[:, DISTAL_KEYPOINTS] if speeds.size else np.empty((0, len(DISTAL_KEYPOINTS)))
    lengths = limb_lengths(arr)
    max_limb_length = np.nanmax(lengths, axis=1) if lengths.size else np.empty(0)

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    ax_xy = fig.add_subplot(grid[:, 0])
    ax_speed = fig.add_subplot(grid[0, 1])
    ax_limb = fig.add_subplot(grid[1, 1])

    for keypoint_idx, keypoint_name in enumerate(COCO17_NAMES):
        xy = arr[:, keypoint_idx, :]
        finite = np.isfinite(xy).all(axis=1)
        if np.count_nonzero(finite) >= 2:
            ax_xy.plot(xy[finite, 0], xy[finite, 1], color="#B8B8B8", linewidth=0.55, alpha=0.35)

    for keypoint_idx in DISTAL_KEYPOINTS:
        keypoint_name = COCO17_NAMES[keypoint_idx]
        add_colored_trajectory(
            ax_xy,
            arr[:, keypoint_idx, :],
            DISTAL_COLORS[keypoint_name],
            keypoint_name.replace("_", " "),
        )

    finite_xy = arr[np.isfinite(arr).all(axis=2)]
    if finite_xy.size:
        xy_min = finite_xy.min(axis=0)
        xy_max = finite_xy.max(axis=0)
        span = np.maximum(xy_max - xy_min, 1e-6)
        pad = span * 0.08
        ax_xy.set_xlim(xy_min[0] - pad[0], xy_max[0] + pad[0])
        ax_xy.set_ylim(xy_min[1] - pad[1], xy_max[1] + pad[1])

    ax_xy.set_title("2D keypoint trajectories")
    ax_xy.set_xlabel("canonical x")
    ax_xy.set_ylabel("canonical y")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.22)
    ax_xy.legend(loc="best", fontsize=8, frameon=False, ncols=2)
    add_metadata_legend(ax_xy, metadata)

    if distal_speeds.size:
        t_speed = np.arange(1, arr.shape[0]) / fps
        for i, keypoint_idx in enumerate(DISTAL_KEYPOINTS):
            keypoint_name = COCO17_NAMES[keypoint_idx]
            ax_speed.plot(
                t_speed,
                distal_speeds[:, i],
                color=DISTAL_COLORS[keypoint_name],
                linewidth=0.85,
                label=keypoint_name.replace("_", " "),
            )
        ax_speed.plot(t_speed, np.nanmax(distal_speeds, axis=1), color="#222222", linewidth=1.0, label="max distal")
    ax_speed.set_title("Distal speed")
    ax_speed.set_xlabel("time (s)")
    ax_speed.set_ylabel("coord units / s")
    ax_speed.grid(True, alpha=0.22)
    ax_speed.legend(loc="best", fontsize=8, frameon=False)

    t_frame = np.arange(arr.shape[0]) / fps
    for segment_idx, (segment_name, _, _) in enumerate(LIMB_SEGMENTS):
        ax_limb.plot(t_frame, lengths[:, segment_idx], linewidth=0.65, alpha=0.55, label=segment_name)
    if max_limb_length.size:
        ax_limb.plot(t_frame, max_limb_length, color="#222222", linewidth=1.0, label="max limb")
    ax_limb.set_title("Limb segment lengths")
    ax_limb.set_xlabel("time (s)")
    ax_limb.set_ylabel("coord units")
    ax_limb.grid(True, alpha=0.22)
    ax_limb.legend(loc="best", fontsize=7, frameon=False, ncols=2)

    reasons = row.get("flag_reasons", "")
    title = f"{row['file']} | {arr.shape[0]} frames"
    if snippet_range is not None:
        title += f" | GMA frames {snippet_range[0]}-{snippet_range[1]}"
    if reasons:
        title += "\n" + textwrap.fill(reasons.replace(";", "; "), width=120)
    else:
        title += "\nnot flagged by QC"
    fig.suptitle(title, fontsize=12)

    output_path = output_dir / f"{safe_stem(row['file'])}_trajectory.png"
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not args.qc_csv.exists():
        raise FileNotFoundError(f"QC CSV does not exist: {args.qc_csv}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {args.input_dir}")
    if args.gma_snippet and not args.metadata_csv.exists():
        raise FileNotFoundError(f"metadata CSV does not exist: {args.metadata_csv}")
    metadata_by_stem = load_metadata(args.metadata_csv) if args.metadata_csv.exists() else {}

    rows = read_selected_rows(
        args.qc_csv,
        selection=args.selection,
        max_files=args.max_files,
        sample_mode=args.sample_mode,
        random_seed=args.random_seed,
    )
    if not rows:
        raise ValueError(f"no {args.selection} rows found in {args.qc_csv}")

    output_dir = args.output_dir / args.selection
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    skipped = 0
    for row in rows:
        metadata = metadata_by_stem.get(row["video_stem"])
        output_path = plot_one(
            row,
            args.input_dir,
            output_dir,
            args.fps,
            metadata=metadata,
            gma_snippet=args.gma_snippet,
        )
        if output_path is None:
            skipped += 1
            continue
        written.append(output_path)
        if args.max_files > 0 and len(written) >= args.max_files:
            break

    if not written:
        raise ValueError(
            f"no plots were written for {args.selection}; "
            "check metadata availability and GMA frame ranges"
        )

    print(f"Wrote {len(written)} {args.selection} trajectory plots to {output_dir}")
    if skipped:
        print(f"Skipped {skipped} rows without usable metadata/snippet frames")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
