#!/usr/bin/env python
"""10 FPS temporal channel relevance screens for NKD/DX labels.

The input for each recording is the exact 2,700-frame GMA window specified by
metadata, downsampled by 3 to 900 frames. Channels are tested within adjusted-age
strata: <52 weeks and >=52 weeks.

This avoids testing individual 30 FPS frame numbers against labels. Instead it
screens each channel using:
1. raw_10fps_waveform_logistic: the whole 900-step waveform.
2. recurrence_pattern_logistic: autocorrelation, spectral power, and derivative
   recurrence descriptors computed from the 900-step waveform.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_DIR = Path("features_notRotated_withQCFlags")
META_CSV = Path("df_meta_constructed_notRotated_withQCFlags.csv")
OUT_DIR = FEATURE_DIR / "temporal_label_relevance_10fps_by_age"
ORIGINAL_SEGMENT_LEN = 2700
SEQ_LEN = 900
DOWNSAMPLE_STEP = 3
FPS = 10.0
AGE_CUT_WEEKS = 52.0
RANDOM_STATE = 20260605
N_SPLITS = 5
AUTOCORR_LAGS = (1, 2, 5, 10, 20, 50, 100)
SPECTRAL_BANDS = ((0.05, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0))


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    names = json.loads((FEATURE_DIR / "feature_names.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((FEATURE_DIR / "feature_manifest.csv").open(newline="", encoding="utf-8")))
    meta_rows = list(csv.DictReader(META_CSV.open(newline="", encoding="utf-8")))
    meta_by_file = {row["file"]: row for row in meta_rows if row.get("file")}

    selected = []
    for row in rows:
        label = row.get("final_code_for_ai_str", "")
        age = safe_float(row.get("adjusted_age_weeks", ""))
        meta_row = meta_by_file.get(row["file"])
        if not meta_row:
            continue
        start = safe_float(meta_row.get("gma_video_start_1_fnum", ""))
        stop = safe_float(meta_row.get("gma_video_stop_1_fnum", ""))
        if label in {"NKD", "DX"} and np.isfinite(age) and np.isfinite(start) and np.isfinite(stop):
            selected.append({**row, "gma_start": int(round(start)), "gma_stop": int(round(stop))})

    X = np.empty((len(selected), len(names), SEQ_LEN), dtype=np.float32)
    y = np.empty(len(selected), dtype=np.int8)
    ages = np.empty(len(selected), dtype=np.float32)
    files = []
    for i, row in enumerate(selected):
        arr = np.load(row["feature_path"])
        start = int(row["gma_start"])
        stop = int(row["gma_stop"])
        if stop - start != ORIGINAL_SEGMENT_LEN:
            raise ValueError(f"{row['file']} segment is {stop - start}, expected {ORIGINAL_SEGMENT_LEN}")
        if start < 0 or stop > arr.shape[1]:
            raise ValueError(f"{row['file']} segment [{start}:{stop}] is out of bounds for {arr.shape[1]}")
        X[i] = arr[:, start:stop:DOWNSAMPLE_STEP]
        y[i] = 1 if row["final_code_for_ai_str"] == "DX" else 0
        ages[i] = safe_float(row["adjusted_age_weeks"])
        files.append(row["file"])
    return X, y, ages, names, files


def fold_count(y: np.ndarray) -> int:
    return int(min(N_SPLITS, np.bincount(y, minlength=2).min()))


def score_predictions(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float | str | int]:
    auc = roc_auc_score(y, score)
    if auc < 0.5:
        auc_directional = 1.0 - auc
        direction = "lower_in_dx"
    else:
        auc_directional = auc
        direction = "higher_in_dx"
    n_correct = int((pred == y).sum())
    majority = max(np.mean(y), 1.0 - np.mean(y))
    return {
        "auc": float(auc),
        "auc_directional": float(auc_directional),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "n_correct": n_correct,
        "accuracy_binom_p_vs_majority": float(binomtest(n_correct, len(y), p=majority, alternative="greater").pvalue),
        "direction": direction,
    }


def crossval_logistic(X: np.ndarray, y: np.ndarray, c: float = 0.1) -> dict[str, float | str | int]:
    cv = StratifiedKFold(n_splits=fold_count(y), shuffle=True, random_state=RANDOM_STATE)
    pred = np.zeros_like(y)
    score = np.zeros(y.shape[0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for train_idx, test_idx in cv.split(X, y):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="l2",
                    C=c,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                ),
            )
            model.fit(X[train_idx], y[train_idx])
            score[test_idx] = model.decision_function(X[test_idx])
            pred[test_idx] = model.predict(X[test_idx])
    return score_predictions(y, pred, score)


def recurrence_pattern_features(Xc: np.ndarray) -> tuple[np.ndarray, list[str]]:
    features = []
    names = []
    X = Xc.astype(np.float64, copy=False)
    centered = X - X.mean(axis=1, keepdims=True)
    scale = X.std(axis=1, keepdims=True) + 1e-12
    Z = centered / scale

    for lag in AUTOCORR_LAGS:
        ac = np.mean(Z[:, :-lag] * Z[:, lag:], axis=1)
        features.append(ac)
        names.append(f"autocorr_lag_{lag}")

    fft = np.fft.rfft(centered, axis=1)
    power = np.square(np.abs(fft))
    freqs = np.fft.rfftfreq(X.shape[1], d=1.0 / FPS)
    total_power = power[:, freqs > 0].sum(axis=1) + 1e-12
    for low, high in SPECTRAL_BANDS:
        mask = (freqs >= low) & (freqs < high)
        band_power = power[:, mask].sum(axis=1) / total_power
        features.append(band_power)
        names.append(f"spectral_power_frac_{low:g}_{high:g}hz")

    diff = np.diff(Z, axis=1)
    abs_diff = np.abs(diff)
    features.extend(
        [
            np.mean(abs_diff, axis=1),
            np.std(diff, axis=1),
            np.mean(np.diff(np.signbit(diff), axis=1) != 0, axis=1),
            np.percentile(abs_diff, 95, axis=1),
        ]
    )
    names.extend(["mean_abs_derivative", "std_derivative", "derivative_turn_rate", "p95_abs_derivative"])

    return np.column_stack(features).astype(np.float32), names


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, ages, channel_names, files = load_dataset()
    rows: list[dict[str, object]] = []
    counts = []
    pattern_feature_names = None

    for age_group, mask in [("lt52w", ages < AGE_CUT_WEEKS), ("ge52w", ages >= AGE_CUT_WEEKS)]:
        Xg = X[mask]
        yg = y[mask]
        counts.append(
            {
                "age_group": age_group,
                "n": int(yg.size),
                "n_dx": int(yg.sum()),
                "n_nkd": int((yg == 0).sum()),
                "majority_baseline": float(max(np.mean(yg), 1.0 - np.mean(yg))),
            }
        )
        if yg.size == 0 or np.bincount(yg, minlength=2).min() < 2:
            continue
        for channel_idx, channel in enumerate(channel_names):
            Xc = Xg[:, channel_idx, :]
            raw_scores = crossval_logistic(Xc, yg, c=0.1)
            rows.append(
                {
                    "age_group": age_group,
                    "method": "raw_10fps_waveform_logistic",
                    "channel_idx": channel_idx,
                    "channel": channel,
                    "n": int(yg.size),
                    "n_dx": int(yg.sum()),
                    "n_nkd": int((yg == 0).sum()),
                    **raw_scores,
                }
            )

            Xp, pattern_feature_names = recurrence_pattern_features(Xc)
            pattern_scores = crossval_logistic(Xp, yg, c=1.0)
            rows.append(
                {
                    "age_group": age_group,
                    "method": "recurrence_pattern_logistic",
                    "channel_idx": channel_idx,
                    "channel": channel,
                    "n": int(yg.size),
                    "n_dx": int(yg.sum()),
                    "n_nkd": int((yg == 0).sum()),
                    **pattern_scores,
                }
            )
            print(age_group, channel_idx + 1, len(channel_names), channel, flush=True)

    consensus = []
    for age_group in ("lt52w", "ge52w"):
        for channel in channel_names:
            subset = [r for r in rows if r["age_group"] == age_group and r["channel"] == channel]
            if not subset:
                continue
            consensus.append(
                {
                    "age_group": age_group,
                    "channel": channel,
                    "best_auc_directional": max(float(r["auc_directional"]) for r in subset),
                    "mean_auc_directional": float(np.mean([float(r["auc_directional"]) for r in subset])),
                    "best_balanced_accuracy": max(float(r["balanced_accuracy"]) for r in subset),
                    "mean_balanced_accuracy": float(np.mean([float(r["balanced_accuracy"]) for r in subset])),
                    "best_method": max(subset, key=lambda r: float(r["auc_directional"]))["method"],
                }
            )
    consensus.sort(key=lambda r: (str(r["age_group"]), -float(r["best_auc_directional"])))

    write_csv(OUT_DIR / "temporal_channel_relevance_10fps.csv", rows)
    write_csv(OUT_DIR / "temporal_channel_relevance_10fps_consensus.csv", consensus)
    write_csv(OUT_DIR / "age_group_counts.csv", counts)
    if pattern_feature_names is not None:
        (OUT_DIR / "recurrence_pattern_feature_names.json").write_text(
            json.dumps(pattern_feature_names, indent=2),
            encoding="utf-8",
        )

    print(f"records: {len(files)}")
    print(f"channels: {len(channel_names)}")
    print(f"sequence_length_10fps: {SEQ_LEN}")
    print(f"output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
