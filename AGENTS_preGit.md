# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python analysis workspace for RTMPose-to-feature processing and NKD/DX modeling. 
Root-level `*.py` files are executable pipeline scripts. 
Important inputs include `pose_estimate_data_json/`, metadata CSVs such as `df_meta_constructed_notRotated_withQCFlags.csv`, and canonical movement tables. 
Derived arrays and QC outputs are stored in directories such as `pose_estimate_data_npy_codex*`, `features_npy_codex*`, `features_notRotated_withQCFlags/`, and `*_plots/`. 
Modeling outputs live under `tsai_notRotated_framewise_loocv_results*` and `mil_notRotated_withQCFlags_results/`. 
The `tsai/` directory is a local vendored dependency; avoid editing it unless the task is explicitly about that package.

## Build, Test, and Development Commands

Use the tsai-codex-gpu mamba environment unless instructed otherwise:

```bash
mamba env create -f environment_tsai_codex_gpu.yml
mamba run -n tsai-codex-gpu python convert_rtmpose_json_to_canonical_npy.py --max-files 5
mamba run -n tsai-codex-gpu python build_features_notRotated_withQCFlags.py
mamba run -n tsai-codex-gpu python audit_features_notrotated_withqcflags_codex.py
```

Most scripts expose `--help`; check defaults before launching full-dataset runs. 
Use small smoke runs first when a script supports limits such as `--max-files`.

## Coding Style & Naming Conventions

Use Python 3.11, 4-space indentation, `pathlib.Path` for filesystem paths, and `argparse` for script CLIs. 
Keep constants near the top of scripts and use descriptive snake_case names. 
Preserve existing filename patterns: processing scripts often use action prefixes such as `build_`, `audit_`, `qc_`, `plot_`, and `repair_`; generated outputs should include the dataset or transform name, for example `notRotated` or `withQCFlags`.

## Testing Guidelines

There is no formal pytest suite in the root workspace. 
For code changes, at minimum run syntax checks and a targeted smoke command:

```bash
mamba run -n tsai-codex-gpu python -m py_compile *.py
mamba run -n tsai-codex-gpu python <changed_script>.py --help
```

When changing data transformations, validate output shapes, manifests, and a small sample of generated plots or QC summaries before running the full dataset.

## Commit & Pull Request Guidelines

Root Git history is not available in this workspace, so use concise imperative commit subjects such as `Add feature anomaly audit` or `Fix notRotated metadata join`. 
PRs should describe the affected pipeline stage, list commands run, note any regenerated files or large artifacts, and include screenshots or plot paths when visual QC changes.

## Agent-Specific Instructions

Default to `mamba run -n tsai-codex-gpu ...` for execution. 
Do not remove or overwrite generated data directories without explicit confirmation.