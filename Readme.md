# Project 2.1 - AI and Machine Learning
Semester project for Computer Science programme, AI and Machine Learning
module at Maastricht University.
Based on [The Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) (ML-Agents). The central topic is collection of data from real-time 3D video game enviroments with AI-controlled agents and predicting certain features of the training runs by using Machine Learning techniques with the collected data.

<!---
maybe add research question here
-->

## Requirements

- **Python 3.10.12** - optional if using Docker
- **Unity Editor 2023.2.12f1**
- **ML-Agents 21** - optional if using Docker

Recommended:

- **Visual Studio Code**
- **Miniconda** - optional if using Docker


## ML-Agents Setup with Conda and Unity

### 1. Download and install the [Unity game engine](https://unity.com/)

Using 2023.2.12f1 Editor version

### 2. Download and install [Miniconda](https://www.anaconda.com/download/success) enviroment menager

Using latest (25.7.0) version

### 3. Create Conda Environment

Create a new Conda environment with Python 3.10.12:

```bash
conda create -n mlagents python=3.10.12
conda activate mlagents
```

### 4. Install Ml-Agents

Install ML-Agents and ML-Agents Environments from the local source. Navigate to directory where you have this repository downloaded, then run:

```bash
cd path\to\ml-agents
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents
```
### 5. Run the training
Then to run batch training, also in the terminal, type:

```bash
mlrun --batch
```

To rebuild the main csv, type:
```bash
mlrun --rebuild-main
```
To rebuild the main csv and add newly collected data to it, type:
```bash
mlrun --batch --rebuild-main
```
Stop at anytime using `Ctrl+c`

Results are saved in training_manager/experiments/results

## ML-Agents Setup with Dokcer

### 1. Download and install [Docker Desktop](https://docs.docker.com/desktop/)

If on Windows: ensure **WSL 2** is enabled - Docker Desktop will guide you.

If on Linux: Install Docker using your distro’s package manager and start the daemon. Optionally you can add your user to the `docker` group to avoid `sudo`. Note that you have to logout and log back in or reboot.

On all OSs: Ensure the Docker service is running

You can verify using this bash commands:
```bash
docker --version
docker run --rm hello-world
```

### 2. Setup run scripts

All commands below must be run from the repository root

#### Linux/macOs
Make sure run.sh is executable:
```bash
chmod +x scripts/run.sh
```

#### Windows
Run commands from PowerShell (no chmod needed).

### 3. Run the training
All commands below must be run from the repository root
#### 3.1. Start continuous training (batch mode) ***and*** rebuild aggregated results (main.csv) after every run

 Linux/macOS
```bash
./scripts/run.sh --batch
```

 Windows
```powershell
.\scripts\run.ps1 --batch
```

Stop at anytime using `Ctrl+c`


#### 3.2. Start continuous training (batch mode) ***without*** rebuilding aggregated results (main.csv)

Linux/macOS
```bash
./scripts/run.sh --batch
```

Windows
```powershell
.\scripts\run.ps1 --batch
```

Stop at anytime using `Ctrl+c`


#### 3.3. Rebuild aggregated results (main.csv)

Linux/macOS
```bash
./scripts/run.sh --rebuild-main
```

Windows
```powershell
.\scripts\run.ps1 --rebuild-main
```
Results are saved in training_manager/experiments/results

## Running prediction models

### 0) Rebuild Main (Optinal if already present)

```bash
mlrun --rebuild-main
```
### 1) Holdout

This select runs form the main CSV which will NOT be used in training for model and will be used to compare Actual vs Predicted values.

Defaults to 1 per machine and only converged + complete:

```bash
python prediction_models/predictors/create_holdout.py --per-machine 5
```
This writes:
- prediction_models/predictors/actual_and_predicted/holdout_runs.csv
- prediction_models/predictors/actual_and_predicted/holdout_run_ids.txt
- prediction_models/predictors/configs_to_predict/holdout/*.yaml

### 2) Dataset encoding

Create encoded datasets. Make SURE you exclude holdouts.

```bash
python prediction_models/dataset_encoder.py time_to_convergence --exclude prediction_models/predictors/actual_and_predicted/holdout_run_ids.txt

python prediction_models/dataset_encoder.py avg_ram_usage --exclude prediction_models/predictors/actual_and_predicted/holdout_run_ids.txt
```

### 3) Feature selection (optional, The best has already been selected.)

This will generate best features that works best together for a given target feature to be used for prediction.

```bash
python prediction_models/feature-selection/feature_selection_runner.py --target time_to_convergence

python prediction_models/feature-selection/feature_selection_runner.py --target avg_ram_usage
```
Outputs are written to prediction_models/feature-selection/results/.

### 4) Model analysis

Compute mean_ae, r2, and median_ae per model. You can run tuned (default) or baseline defaults:

```bash
python prediction_models/models_analysis/run_all_models.py time_to_convergence

python prediction_models/models_analysis/run_all_models.py avg_ram_usage
```
To compare against sklearn defaults:

```bash
python prediction_models/models_analysis/run_all_models.py time_to_convergence --use-defaults

python prediction_models/models_analysis/run_all_models.py avg_ram_usage --use-defaults
```
Outputs:
- prediction_models/models_analysis/models_comparisons/model_comparison_<target>_tuned.csv
- prediction_models/models_analysis/models_comparisons/model_comparison_<target>_default.csv

### 5) Grid search (optional, this might take long time. The best has already been selected)

Generates best hyperparameters per target. Results are picked up automatically by model analysis and predictor build.

```bash
python prediction_models/gridsearch/run_all_gridsearch.py time_to_convergence

python prediction_models/gridsearch/run_all_gridsearch.py avg_ram_usage
```
Results are saved to prediction_models/gridsearch/gridsearch_results_<target>.csv.

### 6) Build predictor

Creates and saves the best model (selected by median_ae) as a .pkl:

```bash
python prediction_models/predictors/build_generator.py time_to_convergence

python prediction_models/predictors/build_generator.py avg_ram_usage
```

### 7) Backtest

Actual vs Predicted values.

```bash
python prediction_models/predictors/run_holdout_backtest.py time_to_convergence

python prediction_models/predictors/run_holdout_backtest.py avg_ram_usage
```
Results are saved to prediction_models/predictors/actual_and_predicted/.

### 8) Predict a new config

Put configs in prediction_models/predictors/configs_to_predict/ and run:

```bash
python prediction_models/predictors/build_accessor.py time_to_convergence prediction_models/predictors/configs_to_predict/<yaml_file>

python prediction_models/predictors/build_accessor.py avg_ram_usage prediction_models/predictors/configs_to_predict/<yaml_file>
```

### 9) Predict with custom hardware

Put configs (with a `hardware:` block) in prediction_models/predictors/custom_configs_with_hw/ and run:

```bash
python prediction_models/predictors/build_accessor.py time_to_convergence prediction_models/predictors/custom_configs_with_hw

python prediction_models/predictors/build_accessor.py avg_ram_usage prediction_models/predictors/custom_configs_with_hw
```

Combined outputs are saved to prediction_models/predictors/predicted_custom_hw/.

## Data collection
The data are stored as the CSV files. Data include hardware used, environment specs, DRL algorithm, [hyperparameters configuration](https://unity-technologies.github.io/ml-agents/Training-Configuration-File/) together with runtime and performance outcomes. Proper documentation of parameters that we collect can be found in the repository mentioned below under the **Group 8** directory.

Firstly, group members add the generated data to the [forked repository](https://github.com/qba24qba/BCS2720-project-data-collection-2025-2026.git). When they are ready to be shared for use by the rest of the groups we submit a **pull request** to the [collaborative collection repository](https://github.com/DennisSoemers/BCS2720-project-data-collection-2025-2026).


## Project Status

Polishing the prediction models...

