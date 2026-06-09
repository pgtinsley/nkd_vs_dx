#!/usr/bin/env python
"""Flag converted pose arrays that deserve visual QC."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from plot_pose_estimate_npy_codex import plot_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find and plot potential outliers in converted pose arrays.")
    parser.add_argument("--input-dir", type=Path, default=Path("pose_estimate_data_npy_codex"))
    parser.add_argument("--plot-dir", type=Path, default=Path("pose_estimate_data_npy_codex_outlier_plots"))
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("pose_estimate_data_npy_codex_outlier_summary.csv"),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--short-frames", type=int, default=2700)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad < 1e-12:
        std = np.nanstd(values)
        return (values - np.nanmean(values)) / std if std > 0 else np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def summarize_file(path: Path, fps: float) -> dict[str, float | int | str | bool]:
    arr = np.load(path)
    finite = bool(np.isfinite(arr).all())

    per_frame_speed = np.linalg.norm(np.diff(arr, axis=0), axis=2) if arr.shape[0] > 1 else np.zeros((0, 17))
    mean_speed = np.nanmean(per_frame_speed, axis=1) * fps if per_frame_speed.size else np.array([0.0])
    max_speed = np.nanmax(per_frame_speed, axis=1) * fps if per_frame_speed.size else np.array([0.0])

    limb_idxs = [9, 10, 15, 16]
    limb_centroid_distances = []
    for idx in limb_idxs:
        centroid = np.nanmean(arr[:, idx, :], axis=0)
        limb_centroid_distances.append(np.linalg.norm(arr[:, idx, :] - centroid, axis=1))
    limb_centroid_distances = np.concatenate(limb_centroid_distances)

    return {
        "file": path.name,
        "frames": int(arr.shape[0]),
        "duration_sec": float(arr.shape[0] / fps),
        "finite": finite,
        "max_abs_coord": float(np.nanmax(np.abs(arr))),
        "coord_range": float(np.nanmax(arr) - np.nanmin(arr)),
        "mean_speed": float(np.nanmean(mean_speed)),
        "p95_speed": float(np.nanpercentile(mean_speed, 95)),
        "max_keypoint_speed": float(np.nanmax(max_speed)),
        "p95_limb_centroid_distance": float(np.nanpercentile(limb_centroid_distances, 95)),
        "path": str(path),
    }


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("*.npy"))
    summaries = [summarize_file(path, args.fps) for path in paths]
    if not summaries:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")

    metric_names = [
        "max_abs_coord",
        "coord_range",
        "mean_speed",
        "p95_speed",
        "max_keypoint_speed",
        "p95_limb_centroid_distance",
    ]
    metric_arrays = {name: np.asarray([row[name] for row in summaries], dtype=float) for name in metric_names}
    metric_z = {name: robust_z(values) for name, values in metric_arrays.items()}

    for row_idx, row in enumerate(summaries):
        reasons = []
        if not row["finite"]:
            reasons.append("nonfinite")
        if row["frames"] < args.short_frames:
            reasons.append(f"short_frames<{args.short_frames}")
        for name in metric_names:
            z = metric_z[name][row_idx]
            row[f"{name}_robust_z"] = float(z)
            if z > 6.0:
                reasons.append(f"high_{name}")
        row["reasons"] = ";".join(reasons)
        row["flagged"] = bool(reasons)

    flagged = [row for row in summaries if row["flagged"]]
    flagged_by_severity = sorted(
        flagged,
        key=lambda row: (
            not row["finite"],
            max(abs(row[f"{name}_robust_z"]) for name in metric_names),
            args.short_frames - row["frames"] if row["frames"] < args.short_frames else 0,
        ),
        reverse=True,
    )
    to_plot = flagged_by_severity[: args.top_n]

    fieldnames = [
        "file",
        "frames",
        "duration_sec",
        "finite",
        "max_abs_coord",
        "coord_range",
        "mean_speed",
        "p95_speed",
        "max_keypoint_speed",
        "p95_limb_centroid_distance",
        *[f"{name}_robust_z" for name in metric_names],
        "flagged",
        "reasons",
        "path",
    ]
    with args.summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Scanned {len(summaries)} files")
    print(f"Flagged {len(flagged)} files")
    print(f"Wrote {args.summary_csv}")
    print(f"Plotting {len(to_plot)} highest-severity flagged files to {args.plot_dir}")
    for row in to_plot:
        plot_file(Path(row["path"]), args.plot_dir, args.fps)
        print(f"plotted {row['file']} :: {row['reasons']}")


if __name__ == "__main__":
    main()
