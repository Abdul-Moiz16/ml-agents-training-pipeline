# Project 2.1 - AI and Machine Learning
Semester project for Computer Science programme, AI and Machine Learning
module at Maastricht University.
Based on [The Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) (ML-Agents). The central topic is collection of data from real-time 3D video game enviroments with AI-controlled agents and predicting certain features of the training runs by using Machine Learning techniques with the collected data.

## Requirements

- **Python 3.10.12**
- **Conda** (Miniconda/Anaconda)

Recommended:

- **Unity Editor 2023.2.12f1** (only needed if you want to modify/rebuild the Unity environment)
- **VS Code** (recommended for development)

> Note: Prebuilt Unity environments for 3DBall are included under `training_manager/builds/` for Linux / macOS / Windows.



## Setup (Conda)

### 1) Create and activate environment
After downloading miniconda or conda create and activate an environment using the following commands:
```bash
conda create -n group8 python=3.10.12 -y
conda activate group8
```
> Note: `python=3.10.12` is important.

> Note: Then name `group8` can be changed.

### 2) Install project
While in the root directory and if the conda environment is active, run:
```bash
python -m pip install --upgrade pip
python -m pip install -e .
```
This installs:

- the training CLI: mlrun
- the prediction CLI: predrun
- required python dependecies

For CLI help run:
```bash
predrun --help
```
for the **prediction** pipeline.

Run:
```bash
mlagents --help
```
for the **training** pipeline.

## Running training (training_manager)

For batch training run:
```bash
mlrun --batch
```
To rebuild the main csv:
```bash
mlrun --rebuild-main
```

To rebuild the main csv and add newly collected data to it, type:
```bash
mlrun --batch --rebuild-main
```

Also a lot of changes that are tied to the way the program classifies a run as converged can be changed like:
```bash
mlrun --batch --max-steps
mlrun --batch --target-mean
mlrun --batch --window-rows
mlrun --batch --cv-max
mlrun --batch --mean-jitter
mlrun --batch --min-steps-before-check
```

Results are saved in `training_manager/experiments/results`

> Note: Stop at anytime using `Ctrl+c`


## Running prediction models (prediction_models)

### 0) Rebuild Main (Optinal if already present)

```bash
mlrun --rebuild-main
```

### 1) Typical full pipeline
Runs holdout -> encoding -> (optional feature selection) -> model analysis -> build predictor -> backtest:
```bash
predrun all --per-machine 5
```
Gridsearch can be skipped with:
```bash
predrun all --per-machine 5 --skip-gridsearch
```
Same thing with feature selection:
```bash
predrun all --per-machine 5 --skip-feature-select
```
## Prediction pipeline steps (manual)

### 0) Rebuild Main (Optinal if already present)

```bash
mlrun --rebuild-main
```

### 1) Holdout

This selects runs form the main CSV which will **NOT** be used in training for model and will be used to compare Actual vs Predicted values.

Defaults to 1 per machine and only converged + complete:

```bash
predrun holdout --per-machine 5
```
This writes:
- `prediction_models/predictors/actual_and_predicted/holdout_runs.csv`
- `prediction_models/predictors/actual_and_predicted/holdout_run_ids.txt`
- `prediction_models/predictors/configs_to_predict/holdout/*.yaml`

### 2) Dataset encoding

Create encoded datasets. Make SURE you exclude holdouts.

```bash
predrun encode time_to_convergence --exclude prediction_models/predictors/actual_and_predicted/holdout_run_ids.txt

predrun encode avg_ram_usage --exclude prediction_models/predictors/actual_and_predicted/holdout_run_ids.txt
```

### 3) Feature selection (optional, The best has already been selected.)

This will generate best features that works best together for a given target feature to be used for prediction.

```bash
predrun feature-select --target time_to_convergence

predrun feature-select --target avg_ram_usage

```
Outputs are written to `prediction_models/feature-selection/results/`.

### 4) Model analysis

Compute mean_ae, r2, and median_ae per model. You can run tuned (default) or baseline defaults:

```bash
predrun model-analysis time_to_convergence

predrun model-analysis avg_ram_usage
```
To compare against sklearn defaults:

```bash
predrun model-analysis time_to_convergence --use-defaults

predrun model-analysis avg_ram_usage --use-defaults
```
Outputs:
- `prediction_models/models_analysis/models_comparisons/model_comparison_<target>_tuned.csv`
- `prediction_models/models_analysis/models_comparisons/model_comparison_<target>_default.csv`

### 5) Grid search (optional, this might take long time. The best has already been selected)

Generates best hyperparameters per target. Results are picked up automatically by model analysis and predictor build.

```bash
predrun gridsearch time_to_convergence

predrun gridsearch avg_ram_usage
```
Results are saved to `prediction_models/gridsearch/gridsearch_results_<target>.csv`.

### 6) Build predictor

Creates and saves the best model (selected by median_ae) as a .pkl:

```bash
predrun build time_to_convergence

predrun build avg_ram_usage
```

### 7) Backtest

Actual vs Predicted values.

```bash
predrun backtest time_to_convergence

predrun backtest avg_ram_usage
```
Results are saved to `prediction_models/predictors/actual_and_predicted/`.

### 8) Predict a new config

Put configs in `prediction_models/predictors/configs_to_predict/` and run:

```bash
predrun predict time_to_convergence prediction_models/predictors/configs_to_predict/<file>.yaml

predrun predict avg_ram_usage prediction_models/predictors/configs_to_predict/<file>.yaml
```

### 9) Predict with custom hardware

Put configs (with a `hardware:` block) in prediction_models/predictors/custom_configs_with_hw/ and run:

```bash
predrun predict-hw time_to_convergence prediction_models/predictors/custom_configs_with_hw

predrun predict-hw avg_ram_usage prediction_models/predictors/custom_configs_with_hw
```

Combined outputs are saved to `prediction_models/predictors/predicted_custom_hw/.`