# Differentially Private Human Activity Recognition

This repository contains code for the experiments of CMU's Engineering Privacy in Software course project: "Differentially Private Human Activity Recognition".

## Introduction

We apply differential privacy to human-activity recognition (HAR) machine learning models trained on the ExtraSensory dataset [1]. We use Opacus [2], a differential privacy library for training PyTorch models, to train our private models. We empirically demonstrate how this reduces privacy risk by comparing the attack performance of a membership inference adversary on non-private and private versions of our HAR models.

## Setup

The code was developed and tested in the following environment:

1. Operating System: Ubuntu 20.04 LTS
1. Architecture: x86_64
1. Python: 3.9

Please note that training models with differential privacy on ARM architectures (e.g. Apple Silicon) can result in run-time errors (due to Opacus).

We have thus provided the experiment configuration files, trained models, and data used for generating visualizations shown in the project report: [Link to zipped data](https://drive.google.com/file/d/11F7eC4ae_Sv3k-eh6o9JSAuhkdO33Kwp/view?usp=share_link). 

We have also provided a [Jupyter notebook](https://github.com/jatanloya/extrasensory-dp/blob/main/extrasensorypytorch.ipynb) for experimenting with the code:
<a target="_blank" href="https://colab.research.google.com/github/jatanloya/extrasensory-dp/blob/main/extrasensorypytorch.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Dependencies

Follow these instructions to download the required dependencies:

1. Create a virtual environment using your preferred environment manager (e.g. `venv`, `conda`).
1. Activate the virtual environment.
1. Navigate to the project directory.
1. Run `pip install -r requirements.txt` to download the dependencies.

### Dataset

We have already included the original ExtraSensory dataset and cross-validation splits in this repository.

If you want to download the files from the source:

1. http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip
1. http://extrasensory.ucsd.edu/data/cv5Folds.zip

Remember to change the paths to the parent directories of the unzipped files in your experiment configuration file. More details on this file are given below.

## Usage

Run an experiment using `python run_experiment.py [PATH-TO-CONFIG-FILE]`.

### Quickstart

We have provided the experiment configuration files used in the project [here](https://github.com/jatanloya/extrasensory-dp/tree/main/exp_config_files):

1. For our experiments on a simple neural network, run `python run_experiment.py exp_config_files/simple_nn.yaml`.
1. For our experiments on a convolutional neural network, run `python run_experiment.py exp_config_files/cnn.yaml`

### Modifying Configuration Files

An experiment configuration file lets you specify filepaths and hyperparameters you would need for an experiment. We use the `YAML` file format.

A configuration file must follow a strict format:

```yaml
exp_id: # unique experiment ID for setting filepaths
model_type: # model architecture to be used for non-private and private models, see supported values below
user_data_files_directory: "./ExtraSensory.per_uuid_features_labels" # directory where user data files are stored, change to desired path
exp_data_directory: "./exp_data" # directory where generated files should be stored, change to desired path
train_split_uuid_filepaths: # list of users (identified by their UUIDs) to be included in the train data
  # here we use fold 0
  - "./cv_5_folds/fold_0_train_android_uuids.txt"
  - "./cv_5_folds/fold_0_train_iphone_uuids.txt"
test_split_uuid_filepaths: # list of users (identified by their UUIDs) to be included in the test data
  # here we use fold 0
  - "./cv_5_folds/fold_0_test_android_uuids.txt"
  - "./cv_5_folds/fold_0_test_iphone_uuids.txt"
sensors_to_use: # list of sensor measurements to be used by the HAR model
  -  # see supported values below
target_labels: # list of activity labels to be learnt by the HAR model
  -  # see supported values below
batch_size: int # batch size for training non-private and private models
shuffle_train: boolean # true: shuffle order of train data, false: fix order of train data
epochs: int # number of epochs for training non-private and private models
lr: float # learning rate for training non-private and private models
private_args:
  epsilon: float # epsilon for training the private model with (epsilon-delta)-differential privacy
  delta: float # delta for training the private model with (epsilon-delta)-differential privacy
  clipping_norm: float # clipping norm used for private training
```

<details>
<summary>Supported Model Architectures</summary>

```
simple_nn
cnn
```

</details>

<details>
<summary>Supported Sensor Measurements</summary>

```
Acc
Gyro
Magnet
WAcc
Compass
Loc
Aud
AP
PS
LF
```

</details>

<details>
<summary>Supported Activity Labels</summary>

```
PHONE_ON_TABLE
SITTING
OR_indoors
LOC_home
LYING_DOWN
TALKING
SLEEPING
LOC_main_workplace
PHONE_IN_POCKET
EATING
WATCHING_TV
SURFING_THE_INTERNET
OR_standing
FIX_walking
OR_outside
WITH_FRIENDS
PHONE_IN_HAND
COMPUTER_WORK
WITH_CO-WORKERS
DRESSING
COOKING
WASHING_DISHES
ON_A_BUS
GROOMING
DRIVE_-_I_M_THE_DRIVER
TOILET
AT_SCHOOL
IN_A_CAR
DRINKING__ALCOHOL_
IN_A_MEETING
DRIVE_-_I_M_A_PASSENGER
BATHING_-_SHOWER
STROLLING
SINGING
SHOPPING
FIX_restaurant
DOING_LAUNDRY
FIX_running
OR_exercise
STAIRS_-_GOING_UP
STAIRS_-_GOING_DOWN
BICYCLING
LAB_WORK
IN_CLASS
CLEANING
AT_A_PARTY
AT_A_BAR
LOC_beach
AT_THE_GYM
ELEVATOR
PHONE_IN_BAG
```

</details>

### Adding New Model Architectures

You can add a new model architecture by following these steps:

1. Add the model's class definition to `models.py`.
1. Give your model architecture an identifier. For this example, let's assume the identifier is `custom_nn`.
1. In the `models.get_input_size` function, add an if branch for returning the input size needed by the `custom_nn` model object.
1. In the `models.get_model_obj` function, add an if branch for returning the `custom_nn` model object.
1. Optional, update if `custom_nn` needs a particular input data format: In the `models.get_channels_format` function, add an if branch for returning the channels format needed by the `custom_nn` model object. If you need a new channels format, give it an identifier.

If your model architecture uses a new channels format, you will need to make additional changes:

1. In `train_test.py`, add an if branch in the `train` and `test` functions to modify the data shape according to your new channels format.
1. In `model_performance.py`, add an if branch in the `compute_accuracy` function to modify the data shape according to your new channels format.
1. In `membership_inference.py`, add an if branch in the `get_loss_values` function to modify the data shape according to your new channels format.

## Reproducibility

To reproduce the visualizations shown in the project, run `python repro_visual.py`.

## Credits

[1] Vaizman, Yonatan, Katherine Ellis, and Gert Lanckriet. "Recognizing detailed human context in the wild from smartphones and smartwatches." IEEE pervasive computing 16.4 (2017): 62-74.

[2] Yousefpour, Ashkan, et al. "Opacus: User-friendly differential privacy library in PyTorch." arXiv preprint arXiv:2109.12298 (2021).

## License

This project is licensed under the terms of the MIT license, as found in the [LICENSE](https://github.com/jatanloya/extrasensory-dp/blob/main/LICENSE) file.
