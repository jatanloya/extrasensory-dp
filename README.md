# ExtraSensory DP

Applying differential privacy to the extrasensory dataset. 

## Introduction 

This repository contains code for the experiments of CMU's Engineering Privacy in Software course project: "Using Differential Privacy: Mobile Activity". 

## Setting Up

Python: 3.9

### Dataset

Download and unzip: 
1. http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip
2. http://extrasensory.ucsd.edu/data/cv5Folds.zip

### Dependencies

Create a virtual environment using venv or conda with Python 3.9.

Navigate to the project directory using your terminal.

Then, run `pip install -r requirements.txt` in the virtual environment. 

## Running the Code

Navigate to the project directory using your terminal.

For a simple neural network: run `python run_experiment.py exp_config_files/simple_nn.yaml`.

For a convolutional neural network: run `python run_experiment.py exp_config_files/cnn.yaml`

## Reproducibility

## Credits

## License