# Project 2.1 - AI and Machine Learning
Semester project for Computer Science programme, AI and Machine Learning
module at Maastricht University.
Based on [The Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) (ML-Agents). The central topic is collection of data from real- time 3D video game enviroments with AI-controlled agents and applying Machine Learning techniques to that data.

## Requirements
- **Unity Editor 2023.2.12f1**

Recommended:
- **Visual Studio Code**


## 1) Install Docker

### Windows (Docker Desktop)
1. Install **Docker Desktop** (Windows).
2. Ensure **WSL 2** is enabled (Docker Desktop will guide you).
3. Start Docker Desktop and wait until it says “Docker is running”.

### macOS (Docker Desktop)
1. Install **Docker Desktop** (Mac).
2. Start Docker Desktop and wait until it says “Docker is running”.

### Linux
Install Docker using your distro’s package manager and start the daemon:
- Ensure the Docker service is running
- Optional: add your user to the `docker` group to avoid `sudo`. Note you have to logout and log back in or reboot.

### Verify
You can verify using this bash commands:
```bash
docker --version
docker run --rm hello-world
```

## 2) Setup run scripts

All commands below must be run from the repository root

### Linux/macOs
Make sure run.sh is executable:
```bash
chmod +x scripts/run.sh
```

### Windows
Run commands from PowerShell (no chmod needed).




## 3) How to run
All commands below must be run from the repository root

### 3.1) Show help / all CLI options

#### Linux/macOS
```bash
./scripts/run.sh --help
```

#### Windows
```powershell
.\scripts\run.ps1 --help
```

### 3.2) Start continuous training (batch mode) ***and*** rebuild aggregated results (main.csv) after every run

#### Linux/macOS
```bash
./scripts/run.sh --batch
```

#### Windows
```powershell
.\scripts\run.ps1 --batch
```

Stop at anytime using `Ctrl+c`


### 3.3) Start continuous training (batch mode) ***without*** rebuilding aggregated results (main.csv)

#### Linux/macOS
```bash
./scripts/run.sh --batch
```

#### Windows
```powershell
.\scripts\run.ps1 --batch
```

Stop at anytime using `Ctrl+c`


### 3.4) Rebuild aggregated results (main.csv)

#### Linux/macOS
```bash
./scripts/run.sh --rebuild-main
```

#### Windows
```powershell
.\scripts\run.ps1 --rebuild-main
```

### 3.5) Dataset encoding
```bash
python prediction_models/dataset_encoder.py time_to_convergence
python prediction_models/dataset_encoder.py avg_ram_usage
```

### 3.6) Feature selection
TODO simpify so no encoded_dataset path needed --> deduce from target
```bash
python prediction_models/feature-selection/feature_selection_runner.py --encoded prediction_models/encoded_datasets/encoded_time_to_convergence.csv --target time_to_convergence
python prediction_models/feature-selection/feature_selection_runner.py --encoded prediction_models/encoded_datasets/encoded_avg_ram_usage.csv --target avg_ram_usage 
```
### 3.7) Model analysis
```bash
python prediction_models/models_analysis/run_all_models.py time_to_convergence
python prediction_models/models_analysis/run_all_models.py avg_ram_usage
```

### 3.8) Build generator
Creates the prediction model and saves it as pkl
```bash
python prediction_models/predictors/build_generator.py time_to_convergence
python prediction_models/predictors/build_generator.py avg_ram_usage
```

### 3.9) Build accessor
Allows access to saved prediction model. Predicts target value based on inputted yaml file which is to be saved in:
```bash
prediction_models/runs_to_predict
```
```bash
python prediction_models/predictors/build_accessor.py time_to_convergence <yaml_file>
python prediction_models/predictors/build_accessor.py avg_ram_usage <yaml_file>
```

## Data collection
The data are stored as the CSV files. Data include hardware used, environment specs, DRL algorithm, [hyperparameters configuration](https://unity-technologies.github.io/ml-agents/Training-Configuration-File/) together with runtime and performance outcomes. Proper documentation of parameters that we collect can be found in the repository mentioned below under the **Group 8** directory.

Firstly, group members add the generated data to the [forked repository](https://github.com/qba24qba/BCS2720-project-data-collection-2025-2026.git). When they are ready to be shared for use by the rest of the groups we submit a **pull request** to the [collaborative collection repository](https://github.com/DennisSoemers/BCS2720-project-data-collection-2025-2026).


## Project Status

In development. Gathering data from trainings, working on ML models...
