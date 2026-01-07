# Project 2.1 - AI and Machine Learning

Semester project for Computer Science programme, AI and Machine Learning
module at Maastricht University.
Based on [The Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) (ML-Agents). The central topic is collection of data from real- time 3D video game enviroments with AI-controlled agents and applying Machine Learning techniques to that data.

## Requirements

- **Python 3.10.12**
- **Unity Editor 2023.2.12f1**
- **ML-Agents 21**

Recommended:

- **Visual Studio Code**
- **Miniconda**

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

## How to run training

Currently works on Windows, Linux and MacOS

Whilst in the root folder be sure to be in your python virtual env and have mlagents installed.

If it is the first time running, in the terminal type:

```bash
pip install -e .
```

For help type:
```bash
mlrun -h
```

Then to run batch training, also in the terminal, type:
```bash
mlrun --batch
```

To rebuild the main csv, type:
```bash
mlrun --rebuild-main
```

You can also run batch training and rebuild after every trained model with:
```bash
mlrun --batch --rebuild-main
```

Changes to the code are automatically applied to the package unless they are major and change absolutely everything.

Single run training currently not working

Results are saved in training_manager/experiments/results

## Data collection

Right now we are developing the tool to automate the process of running the tests and storing data afterwards.

The data are stored as the CSV files. Data include hardware used, environment specs, DRL algorithm, [hyperparameters configuration](https://unity-technologies.github.io/ml-agents/Training-Configuration-File/) together with runtime and performance outcomes. Proper documentation of parameters that we collect can be found in the repositories mentioned below under the **Group 8** directory.

Firstly, group members add the generated data to the [forked repository](https://github.com/qba24qba/BCS2720-project-data-collection-2025-2026.git). When they are ready to be shared for use by the rest of the groups we submit a **pull request** to the [collaborative collection repository](https://github.com/DennisSoemers/BCS2720-project-data-collection-2025-2026).


## Project Status

In development. Gathering data from trainings..
