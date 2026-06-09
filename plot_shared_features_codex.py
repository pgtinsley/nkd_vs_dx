#!/usr/bin/env python
"""Plot the additional shared movement features for clinical/ML review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("features_npy_codex_shared"))
    parser.add_argument("--output-dir", type=Path, default=Path("features_npy_codex_shared_visuals"))
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def cohens_d(nkd: pd.Series, dx: pd.Series) -> float:
    a = pd.to_numeric(nkd, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(dx, errors="coerce").dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(b) - np.mean(a)) / (pooled + 1e-8))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.feature_dir / "feature_summary.csv")
    manifest = pd.read_csv(args.feature_dir / "feature_manifest.csv")
    extra_names = json.loads((args.feature_dir / "extra_feature_names.json").read_text())
    feature_names = json.loads((args.feature_dir / "feature_names.json").read_text())
    feature_idx = {name: i for i, name in enumerate(feature_names)}

    labeled = summary[summary["class_label"].isin(["NKD", "DX"])].copy()

    effect_rows = []
    for regime_name, regime_df in [("all_labeled", labeled), *list(labeled.groupby("age_regime"))]:
        if set(regime_df["class_label"]) >= {"NKD", "DX"}:
            nkd = regime_df[regime_df["class_label"] == "NKD"]
            dx = regime_df[regime_df["class_label"] == "DX"]
            for name in extra_names:
                for stat in ["mean", "std", "p95"]:
                    metric = f"{name}__{stat}"
                    effect_rows.append(
                        {
                            "age_regime": regime_name,
                            "metric": metric,
                            "feature": name,
                            "stat": stat,
                            "cohens_d_dx_minus_nkd": cohens_d(nkd[metric], dx[metric]),
                        }
                    )
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(args.output_dir / "shared_extra_feature_effect_sizes.csv", index=False)

    top = effects[effects["age_regime"] == "all_labeled"].copy()
    top["abs_d"] = top["cohens_d_dx_minus_nkd"].abs()
    top = top.sort_values("abs_d", ascending=False).head(20).sort_values("cohens_d_dx_minus_nkd")

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    colors = np.where(top["cohens_d_dx_minus_nkd"] >= 0, "#b84a62", "#2f6f73")
    labels = top["metric"].str.replace("__", "\n", regex=False)
    ax.barh(labels, top["cohens_d_dx_minus_nkd"], color=colors)
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_title("New Shared Features: Largest NKD vs DX Scalar Differences")
    ax.set_xlabel("Cohen's d, DX minus NKD")
    out = args.output_dir / "new_feature_effect_sizes.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)

    selected = [
        "distal_speed_variability_1s__mean",
        "distal_normalized_jerk_1s__mean",
        "upper_limb_speed_correlation_3s__mean",
        "simultaneous_limb_activation_fraction_1s__mean",
        "direction_entropy_3s__mean",
        "active_limb_repertoire_count_3s__mean",
        "wrist_midline_crossing_rate_3s__mean",
        "wrist_antigravity_fraction_3s__mean",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), selected):
        data = [labeled.loc[labeled["class_label"] == cls, metric].dropna() for cls in ["NKD", "DX"]]
        bp = ax.boxplot(data, tick_labels=["NKD", "DX"], showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#2f6f73", "#b84a62"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(metric.replace("__", "\n"), fontsize=10)
    fig.suptitle("New Shared Feature Distributions")
    out = args.output_dir / "new_feature_distributions.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)

    regimes = [r for r in manifest["age_regime"].dropna().unique() if r != "unknown"]
    example_rows = []
    for regime in regimes[:2]:
        for cls in ["NKD", "DX"]:
            rows = manifest[(manifest["age_regime"] == regime) & (manifest["class_label"] == cls)]
            if not rows.empty:
                example_rows.append(rows.iloc[0])

    trace_features = [
        "distal_speed_variability_1s",
        "distal_normalized_jerk_1s",
        "upper_limb_speed_correlation_3s",
        "direction_entropy_3s",
        "active_limb_repertoire_count_3s",
        "wrist_antigravity_fraction_3s",
    ]

    fig, axes = plt.subplots(len(example_rows), 1, figsize=(14, max(4, 2.8 * len(example_rows))), constrained_layout=True)
    if len(example_rows) == 1:
        axes = [axes]
    for ax, row in zip(axes, example_rows):
        arr = np.load(row["feature_path"])
        n = min(arr.shape[1], int(args.fps * 60))
        t = np.arange(n) / args.fps
        for name in trace_features:
            values = arr[feature_idx[name], :n]
            p95 = np.percentile(values, 95)
            scale = p95 if p95 > 1e-6 else 1.0
            ax.plot(t, values / scale, linewidth=0.9, label=name)
        ax.set_title(f"{row['file']} | {row['class_label']} | {row['age_regime']}")
        ax.set_ylabel("scaled value")
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("seconds")
    out = args.output_dir / "new_feature_example_timeseries.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)

    # Age/regime view of the features most tied to fidgety/general-movement descriptions.
    age_metrics = [
        "fidgety_micro_movement_density_3s__mean",
        "direction_entropy_3s__mean",
        "limb_activation_pattern_entropy_3s__mean",
        "active_limb_repertoire_count_3s__mean",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, age_metrics):
        groups = []
        labels = []
        for regime in regimes[:2]:
            for cls in ["NKD", "DX"]:
                vals = labeled.loc[(labeled["age_regime"] == regime) & (labeled["class_label"] == cls), metric].dropna()
                groups.append(vals)
                labels.append(f"{cls}\n{'early' if 'under' in regime else 'older'}")
        bp = ax.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor("#2f6f73" if label.startswith("NKD") else "#b84a62")
            patch.set_alpha(0.75)
        ax.set_title(metric.replace("__", "\n"), fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Complexity/Repertoire Features by Class and Age Regime")
    out = args.output_dir / "new_feature_age_regime_distributions.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)

    print(f"Wrote plots to {args.output_dir}")
    for path in sorted(args.output_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
