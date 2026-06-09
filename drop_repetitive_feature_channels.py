#!/usr/bin/env python
"""Drop repetitive channels from features_notRotated_withQCFlags in place."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


FEATURE_DIR = Path("features_notRotated_withQCFlags")
DROP_CHANNELS = {
    "left_wrist_midline_crossing_indicator",
    "left_wrist_contralateral_indicator",
    "right_wrist_midline_crossing_indicator",
    "right_wrist_contralateral_indicator",
    "left_ankle_midline_crossing_indicator",
    "left_ankle_contralateral_indicator",
    "right_ankle_midline_crossing_indicator",
    "right_ankle_contralateral_indicator",
    "shoulder_midpoint_x",
    "shoulder_midpoint_vx",
}


def main() -> None:
    feature_names_path = FEATURE_DIR / "feature_names.json"
    names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    missing = sorted(DROP_CHANNELS - set(names))
    if missing:
        raise ValueError(f"Drop channels missing from feature_names.json: {missing}")

    keep_indices = [i for i, name in enumerate(names) if name not in DROP_CHANNELS]
    drop_indices = [i for i, name in enumerate(names) if name in DROP_CHANNELS]
    new_names = [names[i] for i in keep_indices]

    npy_paths = sorted(FEATURE_DIR.glob("*.npy"))
    for path in npy_paths:
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] != len(names):
            raise ValueError(f"{path} has shape {arr.shape}; expected ({len(names)}, frames)")
        np.save(path, arr[keep_indices].astype(np.float32, copy=False))

    backup_names_path = FEATURE_DIR / "feature_names_before_drop_repetitive.json"
    if not backup_names_path.exists():
        backup_names_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    feature_names_path.write_text(json.dumps(new_names, indent=2), encoding="utf-8")

    manifest_path = FEATURE_DIR / "feature_manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    for row in rows:
        row["n_channels"] = str(len(new_names))
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    dropped_path = FEATURE_DIR / "dropped_repetitive_channels.json"
    dropped_path.write_text(
        json.dumps(
            {
                "dropped_channels": sorted(DROP_CHANNELS),
                "dropped_indices_original": drop_indices,
                "old_n_channels": len(names),
                "new_n_channels": len(new_names),
                "n_arrays_rewritten": len(npy_paths),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"rewritten arrays: {len(npy_paths)}")
    print(f"dropped channels: {len(drop_indices)}")
    print(f"old channels: {len(names)}")
    print(f"new channels: {len(new_names)}")


if __name__ == "__main__":
    main()
