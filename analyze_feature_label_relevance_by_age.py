#!/usr/bin/env python
"""Test feature-channel relevance to NKD/DX labels within adjusted-age strata."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


FEATURE_DIR = Path("features_notRotated_withQCFlags")
OUT_DIR = FEATURE_DIR / "label_relevance_by_age"
AGE_CUT_WEEKS = 52.0
SUMMARY_FUNCS = {
    "mean": lambda x: np.mean(x, axis=1),
    "std": lambda x: np.std(x, axis=1),
    "p05": lambda x: np.percentile(x, 5, axis=1),
    "p50": lambda x: np.percentile(x, 50, axis=1),
    "p95": lambda x: np.percentile(x, 95, axis=1),
    "iqr": lambda x: np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1),
    "rms": lambda x: np.sqrt(np.mean(np.square(x), axis=1)),
}


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def cohens_d(dx: np.ndarray, nkd: np.ndarray) -> float:
    if dx.size < 2 or nkd.size < 2:
        return float("nan")
    pooled = np.sqrt(
        ((dx.size - 1) * np.var(dx, ddof=1) + (nkd.size - 1) * np.var(nkd, ddof=1))
        / max(dx.size + nkd.size - 2, 1)
    )
    return float((np.mean(dx) - np.mean(nkd)) / (pooled + 1e-12))


def rank_biserial_from_u(u: float, n_dx: int, n_nkd: int) -> float:
    return float((2.0 * u / max(n_dx * n_nkd, 1)) - 1.0)


def fdr_bh(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q.tolist()
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = ranked.size
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(pv)
    out[order] = adjusted
    q[valid] = out
    return q.tolist()


def summarize_recordings(rows: list[dict[str, str]], names: list[str]) -> list[dict[str, str | float]]:
    summary_rows: list[dict[str, str | float]] = []
    for row in rows:
        label = row.get("final_code_for_ai_str", "")
        if label not in {"NKD", "DX"}:
            continue
        age = safe_float(row.get("adjusted_age_weeks", ""))
        if not np.isfinite(age):
            continue
        arr = np.load(row["feature_path"])
        summaries = {summary_name: func(arr) for summary_name, func in SUMMARY_FUNCS.items()}
        out: dict[str, str | float] = {
            "file": row["file"],
            "label": label,
            "y_dx": 1 if label == "DX" else 0,
            "adjusted_age_weeks": age,
            "age_group": "lt52w" if age < AGE_CUT_WEEKS else "ge52w",
        }
        for channel_idx, channel_name in enumerate(names):
            for summary_name, values in summaries.items():
                out[f"{channel_name}__{summary_name}"] = float(values[channel_idx])
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_summaries(summary_rows: list[dict[str, str | float]], names: list[str]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for age_group in ("lt52w", "ge52w"):
        group_rows = [r for r in summary_rows if r["age_group"] == age_group]
        n_dx = sum(r["label"] == "DX" for r in group_rows)
        n_nkd = sum(r["label"] == "NKD" for r in group_rows)
        if n_dx < 2 or n_nkd < 2:
            continue
        y = np.asarray([r["y_dx"] for r in group_rows], dtype=int)
        for channel_name in names:
            for summary_name in SUMMARY_FUNCS:
                metric = f"{channel_name}__{summary_name}"
                values = np.asarray([r[metric] for r in group_rows], dtype=float)
                dx = values[y == 1]
                nkd = values[y == 0]
                try:
                    u_result = mannwhitneyu(dx, nkd, alternative="two-sided")
                    p_value = float(u_result.pvalue)
                    rank_biserial = rank_biserial_from_u(float(u_result.statistic), dx.size, nkd.size)
                except ValueError:
                    p_value = float("nan")
                    rank_biserial = float("nan")
                try:
                    auc = float(roc_auc_score(y, values))
                except ValueError:
                    auc = float("nan")
                if np.isfinite(auc) and auc < 0.5:
                    auc_directional = 1.0 - auc
                    direction = "lower_in_dx"
                else:
                    auc_directional = auc
                    direction = "higher_in_dx"
                results.append(
                    {
                        "age_group": age_group,
                        "n_dx": dx.size,
                        "n_nkd": nkd.size,
                        "channel": channel_name,
                        "summary": summary_name,
                        "metric": metric,
                        "mean_dx": float(np.mean(dx)),
                        "mean_nkd": float(np.mean(nkd)),
                        "median_dx": float(np.median(dx)),
                        "median_nkd": float(np.median(nkd)),
                        "cohens_d_dx_minus_nkd": cohens_d(dx, nkd),
                        "rank_biserial_dx_minus_nkd": rank_biserial,
                        "auc": auc,
                        "auc_directional": auc_directional,
                        "direction": direction,
                        "mannwhitney_p": p_value,
                    }
                )
    for age_group in ("lt52w", "ge52w"):
        idx = [i for i, r in enumerate(results) if r["age_group"] == age_group]
        q_values = fdr_bh([float(results[i]["mannwhitney_p"]) for i in idx])
        for i, q in zip(idx, q_values):
            results[i]["mannwhitney_q_fdr_within_age"] = q
    return results


def best_channel_rows(test_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best = {}
    for row in test_rows:
        key = (row["age_group"], row["channel"])
        current = best.get(key)
        if current is None or float(row["auc_directional"]) > float(current["auc_directional"]):
            best[key] = row
    best_rows = list(best.values())
    best_rows.sort(key=lambda r: (str(r["age_group"]), -float(r["auc_directional"])))
    return best_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = json.loads((FEATURE_DIR / "feature_names.json").read_text(encoding="utf-8"))
    manifest = list(csv.DictReader((FEATURE_DIR / "feature_manifest.csv").open(newline="", encoding="utf-8")))
    summary_rows = summarize_recordings(manifest, names)
    test_rows = test_summaries(summary_rows, names)
    best_rows = best_channel_rows(test_rows)

    write_csv(OUT_DIR / "recording_channel_summaries.csv", summary_rows)
    write_csv(OUT_DIR / "channel_summary_label_tests.csv", test_rows)
    write_csv(OUT_DIR / "best_summary_per_channel_label_tests.csv", best_rows)

    counts = {}
    for r in summary_rows:
        key = (r["age_group"], r["label"])
        counts[key] = counts.get(key, 0) + 1
    print(f"recordings analyzed: {len(summary_rows)}")
    for key, value in sorted(counts.items()):
        print(f"{key[0]} {key[1]}: {value}")
    print(f"channels: {len(names)}")
    print(f"tests: {len(test_rows)}")
    print(f"output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
