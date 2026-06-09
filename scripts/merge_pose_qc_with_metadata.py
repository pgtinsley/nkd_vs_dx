#!/usr/bin/env python
"""Merge constructed metadata with pose QC summary and plot modeling overview."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


META_CSV = Path("df_meta_constructed.csv")
QC_CSV = Path("pose_estimate_data_npy_codex_outlier_summary.csv")
MERGED_CSV = Path("df_meta_constructed_with_pose_qc_codex.csv")
META_WITH_STEM_CSV = Path("df_meta_constructed_with_video_stem.csv")
QC_WITH_STEM_CSV = Path("pose_estimate_data_npy_codex_outlier_summary_with_video_stem.csv")
PLOT_DIR = Path("pose_modeling_overview_plots")


def add_video_stem_columns(meta: pd.DataFrame, qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = meta.copy()
    qc = qc.copy()

    meta["video_stem"] = meta["fname_mkv"].map(lambda value: Path(str(value)).stem if pd.notna(value) else np.nan)
    qc["video_stem"] = qc["file"].map(lambda value: Path(str(value)).stem if pd.notna(value) else np.nan)
    return meta, qc


def save_count_table(series: pd.Series, output_path: Path) -> pd.DataFrame:
    table = (
        series.fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis(series.name)
        .reset_index(name="count")
    )
    table["percent"] = table["count"] / table["count"].sum() * 100.0
    table.to_csv(output_path, index=False)
    return table


def bar_count(ax: plt.Axes, table: pd.DataFrame, label_col: str, title: str) -> None:
    plot_table = table.copy()
    plot_table[label_col] = plot_table[label_col].astype(str)
    ax.barh(plot_table[label_col][::-1], plot_table["count"][::-1], color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel("records")
    ax.tick_params(axis="y", labelsize=8)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_CSV)
    qc = pd.read_csv(QC_CSV)
    meta, qc = add_video_stem_columns(meta, qc)
    meta.to_csv(META_WITH_STEM_CSV, index=False)
    qc.to_csv(QC_WITH_STEM_CSV, index=False)

    merged = meta.merge(
        qc,
        on="video_stem",
        how="left",
        suffixes=("", "_pose_qc"),
        validate="many_to_one",
    )
    merged.to_csv(MERGED_CSV, index=False)

    final_code_table = save_count_table(
        merged["final_code_for_ai_str"],
        PLOT_DIR / "final_code_for_ai_str_breakdown.csv",
    )
    diagnosis_table = save_count_table(
        merged["diagnosis_singular"],
        PLOT_DIR / "diagnosis_singular_breakdown.csv",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    bar_count(axes[0], final_code_table, "final_code_for_ai_str", "final_code_for_ai_str")
    bar_count(axes[1], diagnosis_table, "diagnosis_singular", "diagnosis_singular")
    fig.savefig(PLOT_DIR / "class_breakdowns.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    merged["duration_sec"].dropna().hist(ax=axes[0, 0], bins=35, color="#59a14f")
    axes[0, 0].set_title("Pose recording duration")
    axes[0, 0].set_xlabel("seconds")
    axes[0, 0].set_ylabel("records")

    merged["frames"].dropna().hist(ax=axes[0, 1], bins=35, color="#f28e2b")
    axes[0, 1].set_title("Pose frame count")
    axes[0, 1].set_xlabel("frames")
    axes[0, 1].set_ylabel("records")

    flagged_counts = (
        merged["flagged"]
        .fillna("Missing pose QC")
        .replace({True: "Flagged", False: "Not flagged"})
        .value_counts()
    )
    axes[1, 0].bar(flagged_counts.index.astype(str), flagged_counts.values, color="#e15759")
    axes[1, 0].set_title("Pose QC status")
    axes[1, 0].set_ylabel("records")
    axes[1, 0].tick_params(axis="x", rotation=20)

    axes[1, 1].scatter(
        merged["mean_speed"],
        merged["p95_limb_centroid_distance"],
        c=merged["flagged"].fillna(False).map({True: "#e15759", False: "#4c78a8"}),
        alpha=0.65,
        s=18,
    )
    axes[1, 1].set_title("Movement spread vs mean speed")
    axes[1, 1].set_xlabel("mean speed")
    axes[1, 1].set_ylabel("p95 limb-centroid distance")
    fig.savefig(PLOT_DIR / "pose_qc_overview.png", dpi=170)
    plt.close(fig)

    plot_df = merged.dropna(subset=["final_code_for_ai_str", "mean_speed", "p95_limb_centroid_distance"]).copy()
    plot_df["final_code_for_ai_str"] = plot_df["final_code_for_ai_str"].astype(str)
    order = plot_df["final_code_for_ai_str"].value_counts().index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    plot_df.boxplot(column="mean_speed", by="final_code_for_ai_str", ax=axes[0], grid=False, rot=35)
    axes[0].set_title("Mean Speed By AI Code")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("mean speed")

    plot_df.boxplot(column="p95_limb_centroid_distance", by="final_code_for_ai_str", ax=axes[1], grid=False, rot=35)
    axes[1].set_title("Limb Excursion By AI Code")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("p95 limb-centroid distance")
    fig.suptitle("")
    fig.savefig(PLOT_DIR / "movement_metrics_by_final_code.png", dpi=170)
    plt.close(fig)

    crosstab = pd.crosstab(
        merged["final_code_for_ai_str"].fillna("Missing"),
        merged["diagnosis_singular"].fillna("Missing"),
        margins=True,
    )
    crosstab.to_csv(PLOT_DIR / "final_code_by_diagnosis_crosstab.csv")

    print(f"meta_rows={len(meta)}")
    print(f"qc_rows={len(qc)}")
    print(f"merged_rows={len(merged)}")
    print(f"matched_pose_qc={int(merged['file'].notna().sum())}")
    print(f"missing_pose_qc={int(merged['file'].isna().sum())}")
    print(f"meta_with_video_stem_csv={META_WITH_STEM_CSV}")
    print(f"qc_with_video_stem_csv={QC_WITH_STEM_CSV}")
    print(f"merged_csv={MERGED_CSV}")
    print(f"plot_dir={PLOT_DIR}")
    print("\nfinal_code_for_ai_str")
    print(final_code_table.to_string(index=False))
    print("\ndiagnosis_singular")
    print(diagnosis_table.to_string(index=False))


if __name__ == "__main__":
    main()
