# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python-based pose analysis pipeline specifically designed to classify Normal (NKD) versus Aberrant (DX) movement patterns in infants between zero and six months adjusted age. The raw pose estimate data from RTMPose lives in `pose_estimate_data_json`; `pose_estimate_data_npy_rotated` contains pre-processed pose data; the time series in these directories have already been smoothed (Savitzy-Golay), normalized (centered at the hip center and scaled by torso length), and rotated to be vertically aligned. `df_meta_merged.csv` provides sample-wise metadata and summary inputs, keyed with the `video_stem` column; the target feature column is `final_code_for_ai_str`, where `NKD` refers to a normal diagnosis, and `DX` refers to an aberrant diagnosis. Top-level `*.py` scripts can help perform data conversion, quality control, feature generation, exploratory analysis, and plotting. Jupyter notebooks hold exploratory and model-development work. Unless otherwise specified, only consider samples in the metadata that have NKD or DX labels. Unless specified, do not use entire array sequences; instead use the snippet between `gma_video_start_1_fnum` and `gma_video_stop_1_fnum` in the metadata. Do not use anything in the `_archive` directory.

## Build, Test, and Development Commands

By default, use the tsai-codex-gpu environment, and run scripts directly from the repository root:

```bash
mamba env create -f environment_tsai_codex_gpu.yml
mamba run -n tsai-codex-gpu python convert_rtmpose_json_to_canonical_npy.py --max-files 5
```

There are no formal package build steps or test suites. Validate changes to scripts to the affected script with `--help`. When data is available, run edited scripts on a small representative input set (including both NKD and DX samples). Check defaults before launching full-dataset runs. 

## Coding Style & Naming Conventions

Always keep code simple and readable. Follow PEP 8 conventions with 4-space indentation, descriptive `snake_case` names for functions and variables, and uppercase constants for stable feature lists or keypoint mappings. Keep command-line interfaces in `argparse` and prefer `pathlib.Path` for filesystem paths, matching existing scripts. Write outputs to explicit `--output-dir` or similarly named arguments instead of hard-coded scratch paths. Use the plotly library whenever producing visualizations.

## Security & Configuration Tips

Do not commit raw pose arrays, model checkpoints, logs, or local environment files unless explicitly required. Check `.gitignore` before adding large artifacts, especially `*.npy`, `*.pkl`, `*.pth`.