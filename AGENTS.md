# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python-based pose analysis and movement feature workflow specifically geared towards modeling Normal versus Abnormal infant movement patterns. The most raw version of the pose estimate data lives in `pose_estimate_data_json`; `pose_estimate_data_*` houses different versions and formats of the data as specified by the suffix. `df_meta.csv` provides sample-wise metadata and summary inputs, keyed with the `video_stem` column; the target feature column is `final_code_for_ai_str`, where `NKD` refers to Normal movement, and `DX` refers to Abnormal movement. Top-level `*.py` scripts perform data conversion, quality control, feature generation, exploratory analysis, and plotting. Jupyter notebooks hold exploratory or model-development work (especially those in the `tsai` directory). Generated pose arrays, feature arrays, experiment outputs, and temporary artifacts are generally ignored by `.gitignore` via patterns such as `pose_estimate_data_*`, `features_*`, `tsai_*`, `*.npy`, and `*.pkl`.

## Build, Test, and Development Commands

Unless otherwise specified, use the tsai-codex-gpu environment:

```bash
mamba env create -f environment_tsai_codex_gpu.yml
mamba run -n tsai-codex-gpu python convert_rtmpose_json_to_canonical_npy.py --max-files 5
```

Run scripts directly from the repository root. Examples:

```bash
python convert_rtmpose_json_to_canonical_npy.py --help
python qc_pose_npy_notrotated_codex.py --help
```

There is no package build step. Validate changes by running the affected script with `--help` and, when data is available, a small representative input set. Check defaults before launching full-dataset runs. 

## Coding Style & Naming Conventions

Use Python 3.11. Follow PEP 8 conventions with 4-space indentation, descriptive `snake_case` names for functions and variables, and uppercase constants for stable feature lists or keypoint mappings. Keep command-line interfaces in `argparse` and prefer `pathlib.Path` for filesystem paths, matching existing scripts. Write outputs to explicit `--output-dir` or similarly named arguments instead of hard-coded scratch paths. Preserve existing filename patterns: processing scripts often use action prefixes such as `build_`, `audit_`, `qc_`, `plot_`, and `repair_`; generated outputs should include the dataset or transform name, for example `notRotated` or `withQCFlags`.

## Testing Guidelines

No formal test suite is currently present. For script changes, perform targeted smoke tests with `--help`, run the modified code path on a small sample if possible, and inspect generated CSV/plot/model artifacts for shape, column, and naming consistency. 

## Security & Configuration Tips

Do not commit raw pose arrays, model checkpoints, logs, or local environment files unless explicitly required. Check `.gitignore` before adding large artifacts, especially `*.npy`, `*.pkl`, `*.pth`.
