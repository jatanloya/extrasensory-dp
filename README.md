# ExtraSensory DP

## Introduction 

This repository contains code for the experiments of CMU's Engineering Privacy in Software course project: "Using Differential Privacy: Mobile Activity". 

We apply differential privacy to human-activity recognition (HAR) machine learning models trained on the ExtraSensory dataset [1]. We use Opacus [2], a differential privacy library for training PyTorch models, to train our private models. We empirically demonstrate how this reduces privacy risk by comparing the attack performance of a membership inference adversary on non-private and private versions of our HAR models.

## Setup

The code was developed and tested in the following environment:

1. Operating System: Ubuntu 20.04 LTS
1. Architecture: x86_64
1. Python: 3.9

Please note that training models with differential privacy on ARM architectures (e.g. Apple Silicon) can result in run-time errors. 

We have thus provided the experiment configuration files, trained models, and data used for generating visualizations shown in the project report. We have also provided a Jupyter notebook for experimenting with the code (recommended to be run in Google Colab).

### Dependencies

Follow these instructions to download the required dependencies:

1. Create a virtual environment using your preferred environment manager (e.g. `venv`, `conda`). 
1. Activate the virtual environment. 
1. Navigate to the project directory.
1. Run `pip install -r requirements.txt` to download the dependencies. 

### Dataset

We have included the original ExtraSensory dataset and cross-validation splits in this repository. 

If you want to download the files from the source: 

1. http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip
1. http://extrasensory.ucsd.edu/data/cv5Folds.zip

Remember to change the paths to the parent directories of the unzipped files in your experiment configuration file. More details on this file are given below.

## Usage

Navigate to the project directory.

### Examples

For a simple neural network: run `python run_experiment.py exp_config_files/simple_nn.yaml`.

For a convolutional neural network: run `python run_experiment.py exp_config_files/cnn.yaml`

## Reproducibility

## Credits

[1] Vaizman, Yonatan, Katherine Ellis, and Gert Lanckriet. "Recognizing detailed human context in the wild from smartphones and smartwatches." IEEE pervasive computing 16.4 (2017): 62-74.

[2] Yousefpour, Ashkan, et al. "Opacus: User-friendly differential privacy library in PyTorch." arXiv preprint arXiv:2109.12298 (2021).

## License

This project is licensed under the terms of the MIT license, as found in the [LICENSE](https://github.com/jatanloya/extrasensory-dp/blob/main/LICENSE) file.