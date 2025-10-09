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
Choose the agent and find the right file in */config*, then run:
```bash
 mlagents-learn config/subdir/Some_agent.yaml --run-id=TrainRun1 --train
```

To see results navigate to */results*


## Project Status
In development. This is only basic setup, more coming soon...