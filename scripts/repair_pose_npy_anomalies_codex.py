#!/usr/bin/env python
"""Back up and repair short pose anomalies identified by the pose QC audit."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

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
KEYPOINT_INDEX = {name: idx for idx, name in enumerate(COCO17_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=Path("pose_estimate_data_npy_codex"))
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("pose_estimate_data_npy_codex_anomaly_qc/pose_anomaly_runs.csv"),
    )
    parser.add_argument("--metadata-csv", type=Path, default=Path("df_meta_constructed_with_pose_qc_codex.csv"))
    parser.add_argument("--bad-file", default="25171-7ba47f82-211206102642.npy")
    parser.add_argument("--bad-reason", default="Bad pose estimate data")
    return parser.parse_args()


def backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_preInterpolation{path.suffix}")


def make_backup(path: Path) -> Path:
    out = backup_path(path)
    if not out.exists():
        shutil.copy2(path, out)
    return out


def merge_runs(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not runs:
        return []
    runs = sorted(runs)
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def interpolate_keypoint_runs(arr: np.ndarray, keypoint_idx: int, runs: list[tuple[int, int]]) -> int:
    n_frames = arr.shape[0]
    changed = 0
    for start, end in merge_runs(runs):
        start = max(int(start), 0)
        end = min(int(end), n_frames - 1)
        if start > end:
            continue

        left = start - 1
        right = end + 1
        if left < 0 and right >= n_frames:
            continue
        if left < 0:
            arr[start : end + 1, keypoint_idx, :] = arr[right, keypoint_idx, :]
        elif right >= n_frames:
            arr[start : end + 1, keypoint_idx, :] = arr[left, keypoint_idx, :]
        else:
            x = np.arange(start, end + 1, dtype=np.float32)
            xp = np.array([left, right], dtype=np.float32)
            for coord in range(2):
                fp = arr[[left, right], keypoint_idx, coord].astype(np.float32)
                arr[start : end + 1, keypoint_idx, coord] = np.interp(x, xp, fp)
        changed += end - start + 1
    return changed


def repair_pose_files(args: argparse.Namespace) -> pd.DataFrame:
    runs = pd.read_csv(args.runs_csv)
    repair_runs = runs[(runs["file"] != args.bad_file) & (runs["reason"] == "high_keypoint_speed")].copy()
    records = []

    for file_name, file_df in repair_runs.groupby("file"):
        path = args.pose_dir / file_name
        if not path.exists():
            records.append({"file": file_name, "status": "missing", "changed_frames": 0, "backup_path": ""})
            continue

        backup = make_backup(path)
        arr = np.load(path).astype(np.float32, copy=True)
        changed_total = 0
        touched = []
        for keypoint, kp_df in file_df.groupby("keypoint"):
            if keypoint not in KEYPOINT_INDEX:
                records.append(
                    {
                        "file": file_name,
                        "status": f"unknown_keypoint:{keypoint}",
                        "changed_frames": 0,
                        "backup_path": str(backup),
                    }
                )
                continue
            kp_idx = KEYPOINT_INDEX[keypoint]
            kp_runs = list(zip(kp_df["start_frame"], kp_df["end_frame"]))
            changed = interpolate_keypoint_runs(arr, kp_idx, kp_runs)
            changed_total += changed
            touched.append(f"{keypoint}:{changed}")

        np.save(path, arr.astype(np.float32))
        records.append(
            {
                "file": file_name,
                "status": "repaired",
                "changed_frames": changed_total,
                "keypoints": ";".join(touched),
                "backup_path": str(backup),
            }
        )

    return pd.DataFrame(records)


def flag_bad_file_in_metadata(args: argparse.Namespace) -> str:
    with args.metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "file" not in fieldnames or "flagged" not in fieldnames or "reasons" not in fieldnames:
        raise ValueError("Metadata CSV must include file, flagged, and reasons columns.")

    matched = False
    for row in rows:
        if row.get("file") == args.bad_file:
            row["flagged"] = "True"
            row["reasons"] = args.bad_reason
            matched = True

    if not matched:
        row = {name: "" for name in fieldnames}
        stem = Path(args.bad_file).stem
        row["record_id"] = stem.split("-", 1)[0]
        row["video_stem"] = stem
        row["file"] = args.bad_file
        row["path"] = str(args.pose_dir / args.bad_file)
        row["flagged"] = "True"
        row["reasons"] = args.bad_reason
        rows.append(row)

    with args.metadata_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return "updated_existing_row" if matched else "appended_missing_pose_row"


def main() -> None:
    args = parse_args()
    bad_path = args.pose_dir / args.bad_file
    if bad_path.exists():
        make_backup(bad_path)

    repair_log = repair_pose_files(args)
    log_path = args.pose_dir / "pose_interpolation_repair_log.csv"
    repair_log.to_csv(log_path, index=False)
    metadata_status = flag_bad_file_in_metadata(args)

    print(f"Repair log: {log_path}")
    print(f"Metadata status for {args.bad_file}: {metadata_status}")
    print(f"Repaired files: {int((repair_log['status'] == 'repaired').sum())}")
    print(f"Total interpolated keypoint frames: {int(repair_log['changed_frames'].sum())}")
    print(repair_log.sort_values('changed_frames', ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
