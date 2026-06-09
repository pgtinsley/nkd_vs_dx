#!/usr/bin/env python
"""Apply manual whole-array rotations to selected not-rotated pose arrays."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


ROTATE_90_CCW = (
    "585-ca141535-F-72-2010211206.npy",
    "24659-4dfd44d3-210625104846.npy",
    "24998-9adcee24-211013160559.npy",
    "25581-46f04e84-220421102139.npy",
)

ROTATE_180_CCW = (
    "24308-baac499a-2103081311_C.npy",
    "24429-3c6a50f1-2104130923_A.npy",
    "24438-1a47ff39-2104141041_A.npy",
    "24600-554662f4-2106090006_C.npy",
    "25272-6c4725c9-220126162217.npy",
    "25285-b87cfd90-2112140719_A.npy",
    "25314-b1984729-2202221338_A.npy",
    "25375-2e235f6c-220322152447.npy",
    "25376-2c0a2244-220322154745.npy",
    "25377-0a729bb6-220322162814.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("pose_estimate_data_npy_codex_notRotated"))
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("pose_estimate_data_npy_codex_notRotated_manual_rotation_backup"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("pose_estimate_data_npy_codex_notRotated_manual_rotation_manifest.csv"),
    )
    return parser.parse_args()


def rotate(arr: np.ndarray, degrees_ccw: int) -> np.ndarray:
    rotated = arr.copy()
    x = arr[:, :, 0]
    y = arr[:, :, 1]
    if degrees_ccw == 90:
        rotated[:, :, 0] = -y
        rotated[:, :, 1] = x
    elif degrees_ccw == 180:
        rotated[:, :, 0] = -x
        rotated[:, :, 1] = -y
    else:
        raise ValueError(f"Unsupported rotation: {degrees_ccw}")
    return rotated


def main() -> None:
    args = parse_args()
    args.backup_dir.mkdir(parents=True, exist_ok=True)

    requested = {name: 90 for name in ROTATE_90_CCW}
    requested.update({name: 180 for name in ROTATE_180_CCW})

    rows = []
    for name, degrees in sorted(requested.items()):
        path = args.input_dir / name
        if not path.exists():
            raise FileNotFoundError(path)

        backup_path = args.backup_dir / name
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

        arr = np.load(path)
        rotated = rotate(arr, degrees)
        np.save(path, rotated.astype(arr.dtype, copy=False))
        rows.append(
            {
                "file": name,
                "degrees_ccw": degrees,
                "input_path": str(path),
                "backup_path": str(backup_path),
                "shape": "x".join(str(dim) for dim in arr.shape),
                "dtype": str(arr.dtype),
            }
        )

    with args.manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file", "degrees_ccw", "input_path", "backup_path", "shape", "dtype"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rotated {len(rows)} files")
    print(f"Wrote backups to {args.backup_dir}")
    print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
