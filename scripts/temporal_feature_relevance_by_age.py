#!/usr/bin/env python
"""Temporal channel relevance screens for NKD/DX labels by adjusted-age stratum.

This intentionally avoids collapsed time-series summary statistics. It tests
each channel as the 2,700-frame GMA segment defined by metadata start/stop
frame numbers.

Methods:
1. raw_temporal_logistic: L2 logistic regression on the full GMA waveform.
2. max_timepoint_label_correlation: strongest label correlation at any GMA
   timepoint, with permutation control.
3. minirocket: optional PyTorch MiniRocket features for that single channel
   plus ridge. Enable with RUN_MINIROCKET=1.
"""

from __future__ import annotations

import csv
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tsai.models.MINIROCKETPlus_Pytorch import MiniRocketFeaturesPlus


FEATURE_DIR = Path("features_notRotated_withQCFlags")
META_CSV = Path("df_meta_constructed_notRotated_withQCFlags.csv")
OUT_DIR = FEATURE_DIR / "temporal_label_relevance_by_age"
SEQ_LEN = 2700
AGE_CUT_WEEKS = 52.0
RANDOM_STATE = 20260605
MINIROCKET_FEATURES = 1008
N_SPLITS = 5
N_PERMUTATIONS = 200
RUN_MINIROCKET = os.environ.get("RUN_MINIROCKET", "0") == "1"


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
            row = {**row, "gma_start": int(round(start)), "gma_stop": int(round(stop))}
            selected.append(row)

    X = np.empty((len(selected), len(names), SEQ_LEN), dtype=np.float32)
    y = np.empty(len(selected), dtype=np.int8)
    ages = np.empty(len(selected), dtype=np.float32)
    files = []
    for i, row in enumerate(selected):
        arr = np.load(row["feature_path"])
        start = int(row["gma_start"])
        stop = int(row["gma_stop"])
        if stop - start != SEQ_LEN:
            raise ValueError(f"{row['file']} segment is {stop - start} frames, expected {SEQ_LEN}")
        if start < 0 or stop > arr.shape[1]:
            raise ValueError(f"{row['file']} segment [{start}:{stop}] is out of bounds for {arr.shape[1]} frames")
        X[i] = arr[:, start:stop].astype(np.float32, copy=False)
        y[i] = 1 if row["final_code_for_ai_str"] == "DX" else 0
        ages[i] = safe_float(row["adjusted_age_weeks"])
        files.append(row["file"])
    return X, y, ages, names, files


def fold_count(y: np.ndarray) -> int:
    class_counts = np.bincount(y, minlength=2)
    return int(min(N_SPLITS, class_counts.min()))


def score_predictions(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y, score)
    if auc < 0.5:
        auc_directional = 1.0 - auc
        direction = "lower_in_dx"
    else:
        auc_directional = auc
        direction = "higher_in_dx"
    accuracy = accuracy_score(y, pred)
    n_correct = int((pred == y).sum())
    p_acc = binomtest(n_correct, len(y), p=max(np.mean(y), 1.0 - np.mean(y)), alternative="greater").pvalue
    return {
        "auc": float(auc),
        "auc_directional": float(auc_directional),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy),
        "n_correct": n_correct,
        "accuracy_binom_p_vs_majority": float(p_acc),
        "direction": direction,
    }


def raw_temporal_logistic(Xc: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n_splits = fold_count(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pred = np.zeros_like(y)
    score = np.zeros(y.shape[0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for train_idx, test_idx in cv.split(Xc, y):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="l2",
                    C=0.1,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                ),
            )
            model.fit(Xc[train_idx], y[train_idx])
            score[test_idx] = model.decision_function(Xc[test_idx])
            pred[test_idx] = model.predict(Xc[test_idx])
    return score_predictions(y, pred, score)


def max_timepoint_label_correlation(Xc: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Find the strongest label association at any frame, with max-stat permutation."""
    rng = np.random.default_rng(RANDOM_STATE)
    Xz = Xc.astype(np.float64, copy=False)
    Xz = Xz - Xz.mean(axis=0, keepdims=True)
    Xz = Xz / (Xz.std(axis=0, keepdims=True) + 1e-12)
    yz = y.astype(np.float64)
    yz = yz - yz.mean()
    yz = yz / (yz.std() + 1e-12)

    corr = yz @ Xz / Xz.shape[0]
    abs_corr = np.abs(corr)
    observed = float(abs_corr.max())
    best_timepoint = int(abs_corr.argmax())
    signed_corr = float(corr[best_timepoint])

    null = np.empty(N_PERMUTATIONS, dtype=np.float64)
    for i in range(N_PERMUTATIONS):
        yp = rng.permutation(yz)
        null[i] = float(np.abs(yp @ Xz / Xz.shape[0]).max())
    p_value = float((np.sum(null >= observed) + 1) / (N_PERMUTATIONS + 1))

    return {
        "max_abs_timepoint_corr": observed,
        "best_timepoint": best_timepoint,
        "best_time_sec": float(best_timepoint / 30.0),
        "signed_corr_at_best_timepoint": signed_corr,
        "permutation_p": p_value,
    }


def minirocket_ridge(Xc: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n_splits = fold_count(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pred = np.zeros_like(y)
    score = np.zeros(y.shape[0], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for train_idx, test_idx in cv.split(Xc, y):
        X_train = Xc[train_idx, None, :].astype(np.float32, copy=False)
        X_test = Xc[test_idx, None, :].astype(np.float32, copy=False)
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        mrf = MiniRocketFeaturesPlus(1, SEQ_LEN, num_features=MINIROCKET_FEATURES).to(device)
        mrf.fit(X_train)
        with torch.no_grad():
            Z_train = mrf(torch.from_numpy(X_train).to(device)).detach().cpu().numpy()
            Z_test = mrf(torch.from_numpy(X_test).to(device)).detach().cpu().numpy()
        model = make_pipeline(
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), class_weight="balanced"),
        )
        model.fit(Z_train, y[train_idx])
        score[test_idx] = model.decision_function(Z_test)
        pred[test_idx] = model.predict(Z_test)
    return score_predictions(y, pred, score)


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
    X, y, ages, names, files = load_dataset()
    rows: list[dict[str, object]] = []
    counts = []

    for age_group, mask in [
        ("lt52w", ages < AGE_CUT_WEEKS),
        ("ge52w", ages >= AGE_CUT_WEEKS),
    ]:
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
        for channel_idx, channel in enumerate(names):
            Xc = Xg[:, channel_idx, :]
            raw_scores = raw_temporal_logistic(Xc, yg)
            corr_scores = max_timepoint_label_correlation(Xc, yg)
            rows.append(
                {
                    "age_group": age_group,
                    "method": "raw_temporal_logistic",
                    "channel_idx": channel_idx,
                    "channel": channel,
                    "n": int(yg.size),
                    "n_dx": int(yg.sum()),
                    "n_nkd": int((yg == 0).sum()),
                    **raw_scores,
                }
            )
            rows.append(
                {
                    "age_group": age_group,
                    "method": "max_timepoint_label_correlation",
                    "channel_idx": channel_idx,
                    "channel": channel,
                    "n": int(yg.size),
                    "n_dx": int(yg.sum()),
                    "n_nkd": int((yg == 0).sum()),
                    **corr_scores,
                }
            )
            if RUN_MINIROCKET:
                rocket_scores = minirocket_ridge(Xc, yg)
                rows.append(
                    {
                        "age_group": age_group,
                        "method": "single_channel_minirocket_ridge",
                        "channel_idx": channel_idx,
                        "channel": channel,
                        "n": int(yg.size),
                        "n_dx": int(yg.sum()),
                        "n_nkd": int((yg == 0).sum()),
                        **rocket_scores,
                    }
                )
            print(age_group, channel_idx + 1, len(names), channel, flush=True)

    write_csv(OUT_DIR / "temporal_channel_relevance.csv", rows)
    write_csv(OUT_DIR / "age_group_counts.csv", counts)

    consensus = []
    for age_group in ("lt52w", "ge52w"):
        for channel in names:
            subset = [r for r in rows if r["age_group"] == age_group and r["channel"] == channel]
            if not subset:
                continue
            auc_values = [float(r["auc_directional"]) for r in subset if "auc_directional" in r]
            bal_values = [float(r["balanced_accuracy"]) for r in subset if "balanced_accuracy" in r]
            corr_values = [float(r["max_abs_timepoint_corr"]) for r in subset if "max_abs_timepoint_corr" in r]
            p_values = [float(r["permutation_p"]) for r in subset if "permutation_p" in r]
            consensus.append(
                {
                    "age_group": age_group,
                    "channel": channel,
                    "best_auc_directional": max(auc_values) if auc_values else float("nan"),
                    "mean_auc_directional": float(np.mean(auc_values)) if auc_values else float("nan"),
                    "best_balanced_accuracy": max(bal_values) if bal_values else float("nan"),
                    "mean_balanced_accuracy": float(np.mean(bal_values)) if bal_values else float("nan"),
                    "best_max_abs_timepoint_corr": max(corr_values) if corr_values else float("nan"),
                    "best_corr_permutation_p": min(p_values) if p_values else float("nan"),
                    "best_method": max(
                        subset,
                        key=lambda r: float(r.get("auc_directional", r.get("max_abs_timepoint_corr", float("nan")))),
                    )["method"],
                }
            )
    consensus.sort(key=lambda r: (str(r["age_group"]), -float(r["best_auc_directional"])))
    write_csv(OUT_DIR / "temporal_channel_relevance_consensus.csv", consensus)

    print(f"records: {len(files)}")
    print(f"channels: {len(names)}")
    print(f"output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
