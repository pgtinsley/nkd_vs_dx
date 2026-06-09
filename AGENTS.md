# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python-based pose-analysis and movement-feature workflow. Top-level `*.py` scripts perform data conversion, quality control, feature generation, analysis, and plotting. Notebooks such as `composite_movement_score.ipynb` and `mil_normal_vs_abnormal_notRotated_withQCFlags.ipynb` hold exploratory or model-development work. Tracked CSV files such as `canonical_movement_features.csv`, `canonical_movement_scores.csv`, and `df_meta_constructed*.csv` provide metadata and summary inputs. Generated pose arrays, feature arrays, experiment outputs, and temporary artifacts are generally ignored by `.gitignore` via patterns such as `pose_estimate_data_*`, `features_*`, `tsai_*`, `*.npy`, `*.pkl`, and `exp/*`.

## Build, Test, and Development Commands

Unless otherwise specified, use the GPU environment:

```bash
mamba env create -f environment_tsai_codex_gpu.yml
mamba activate tsai-codex-gpu
```

Use the CPU-oriented environment only when GPU dependencies are not needed:

```bash
mamba env create -f environment_tsai_codex.yml
```

Run scripts directly from the repository root. Examples:

```bash
python convert_rtmpose_json_to_canonical_npy.py --help
python build_features_npy_codex.py --pose-dir pose_estimate_data_npy_codex --output-dir features_npy_codex
python qc_pose_npy_notrotated_codex.py --help
```

There is no package build step. Validate changes by running the affected script with `--help` and, when data is available, a small representative input set.

## Coding Style & Naming Conventions

Use Python 3.11. Follow PEP 8 conventions with 4-space indentation, descriptive `snake_case` names for functions and variables, and uppercase constants for stable feature lists or keypoint mappings. Keep command-line interfaces in `argparse` and prefer `pathlib.Path` for filesystem paths, matching existing scripts. Write outputs to explicit `--output-dir` or similarly named arguments instead of hard-coded scratch paths.

## Testing Guidelines

No formal test suite is currently present. For script changes, perform targeted smoke tests with `--help`, run the modified code path on a small sample if possible, and inspect generated CSV/plot/model artifacts for shape, column, and naming consistency. When adding tests, place them under `tests/` and name files `test_<module>.py`.

## Commit & Pull Request Guidelines

Git history currently contains only an initial commit, so no project-specific commit convention is established. Use short, imperative commit messages such as `Add pose QC audit summary` or `Fix feature channel ordering`. Pull requests should describe the changed workflow, list input and output paths affected, mention any generated artifacts intentionally excluded by `.gitignore`, and include plots or tables when visual/model results change.

## Security & Configuration Tips

Do not commit raw pose arrays, model checkpoints, logs, or local environment files unless explicitly required. Check `.gitignore` before adding large artifacts, especially `*.npy`, `*.pkl`, `*.pt`, and experiment directories.
