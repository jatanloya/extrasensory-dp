import argparse
import yaml
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine

from dataset import get_uuids_from_filepaths, get_train_and_test_data
from models import get_model_obj, get_channels_format, get_input_size
from train_test import train_test_model
from model_performance import plot_train_and_test_loss_per_epoch, plot_train_and_test_acc_per_epoch
from membership_inference import get_loss_values, plot_train_and_test_losses, get_mia_model_roc_curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments on the ExtraSensory dataset.')
    parser.add_argument('config_file', type=str, help='Filepath for experiment configuration file.')

    args = parser.parse_args()
    config_filepath = args.config_file

    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)

    exp_id = config['exp_id']
    model_type = config['model_type']
    exp_data_dir = config['exp_data_directory']

    # Make directories for storing exp_data if they don't exist
    if not os.path.isdir(exp_data_dir):
        os.mkdir(exp_data_dir)
        os.mkdir(f"{exp_data_dir}/models")
        os.mkdir(f"{exp_data_dir}/processed_datasets")
        os.mkdir(f"{exp_data_dir}/plots_data")

    processed_dataset_filepath = f"{exp_data_dir}/processed_datasets/{exp_id}_data.npz"

    if os.path.exists(processed_dataset_filepath):
        print("Loading processed dataset for experiment.")
        data = np.load(processed_dataset_filepath)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
    else:
        print("Generating processed dataset for experiment.")
        directory = config['user_data_files_directory']

        train_uuids = get_uuids_from_filepaths(config['train_split_uuid_filepaths'])
        test_uuids = get_uuids_from_filepaths(config['test_split_uuid_filepaths'])

        sensors_to_use = config['sensors_to_use']
        target_labels = config['target_labels']
        X_train, y_train, X_test, y_test = get_train_and_test_data(directory, train_uuids, test_uuids,
                                                                   sensors_to_use, target_labels)
        np.savez(processed_dataset_filepath,
                 X_train=X_train, y_train=y_train,
                 X_test=X_test, y_test=y_test)

    # Transform data to PyTorch tensors
    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)

    # Hyperparameters
    batch_size = config['batch_size']
    shuffle_train = config['shuffle_train']
    input_size = X_test.shape[1]
    num_classes = y_test.shape[1]
    lr = config['lr']
    epochs = config['epochs']

    # Create PyTorch dataloaders
    train_dataloader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=batch_size,
                                  shuffle=shuffle_train)
    test_dataloader = DataLoader(TensorDataset(X_test, y_test),
                                 batch_size=batch_size,
                                 shuffle=False)

    # Set device for training/testing models
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Define args for training model without DP
    non_private_args = {
        'epochs': epochs,  # number of epochs for training model
        'lr': lr,  # learning rate for training model
        'disable_dp': True,  # non-private training
    }
    input_size = get_input_size(model_type, input_size)
    non_private_model = get_model_obj(model_type, input_size, num_classes).to(device)
    channels_format = get_channels_format(model_type)

    non_private_model_id = f'{exp_id}_non_private'
    non_private_optimizer = optim.Adam(non_private_model.parameters(),
                                       non_private_args['lr'])
    privacy_engine = None

    non_private_model_filepath = f"{exp_data_dir}/models/{non_private_model_id}"
    if os.path.exists(non_private_model_filepath):
        print("Loading model trained without DP for experiment.")
        checkpoint = torch.load(non_private_model_filepath)
        non_private_model.load_state_dict(checkpoint['model_state_dict'])
        non_private_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Loading model performance stats
        data = np.load(f"{non_private_model_filepath}_perf.npz")
        non_private_train_losses = data['train_losses']
        non_private_test_losses = data['test_losses']
        non_private_train_accs = data['train_accs']
        non_private_test_accs = data['test_accs']
    else:
        print("Training model without DP for experiment.")
        # Train and test model without DP
        non_private_model, non_private_train_losses, non_private_test_losses, non_private_train_accs, non_private_test_accs = train_test_model(
            model=non_private_model,
            args=non_private_args,
            device=device,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=non_private_optimizer,
            privacy_engine=privacy_engine,
            channels_format=channels_format
        )

        # Saving model
        torch.save({
            'model_state_dict': non_private_model.state_dict(),
            'optimizer_state_dict': non_private_optimizer.state_dict(),
        }, non_private_model_filepath)

        # Saving model performance stats
        np.savez(f"{non_private_model_filepath}_perf.npz",
                 train_losses=non_private_train_losses,
                 test_losses=non_private_test_losses,
                 train_accs=non_private_train_accs,
                 test_accs=non_private_test_accs)

    # Plot model performance graphs
    non_private_model_id_str = non_private_model_id.replace(".", "_")
    plot_train_and_test_loss_per_epoch(
        "Non-private Model", 
        non_private_train_losses, 
        non_private_test_losses,
        f"{exp_data_dir}/plots_data/{non_private_model_id_str}_loss_per_epoch"
    )
    plot_train_and_test_acc_per_epoch(
        "Non-private model", 
        non_private_train_accs, 
        non_private_test_accs,
        f"{exp_data_dir}/plots_data/{non_private_model_id_str}_acc_per_epoch"
    )

    # Run membership inference attack on model without DP
    per_sample_loss_values_filepath = f"{exp_data_dir}/plots_data/{exp_id}_per_sample_loss_values_non_private.npz"
    non_private_train_loss_values, non_private_test_loss_values = [], []
    if os.path.exists(per_sample_loss_values_filepath):
        print("Loading per-sample train and test loss values.")
        data = np.load(per_sample_loss_values_filepath)
        non_private_train_loss_values = data['per_sample_train_losses']
        non_private_test_loss_values = data['per_sample_test_losses']
    else:
        print("Generating per-sample train and test loss values.")
        non_private_train_loss_values = get_loss_values(non_private_model,
                                                        device,
                                                        train_dataloader,
                                                        channels_format)
        non_private_test_loss_values = get_loss_values(non_private_model,
                                                       device, 
                                                       test_dataloader,
                                                       channels_format)
        np.savez(per_sample_loss_values_filepath,
                 per_sample_train_losses=non_private_train_loss_values,
                 per_sample_test_losses=non_private_test_loss_values)
    plot_train_and_test_losses(
        non_private_train_loss_values, 
        non_private_test_loss_values,
        f"{exp_data_dir}/plots_data/{non_private_model_id_str}_per_sample_losses"
    )
    fpr, tpr = get_mia_model_roc_curve(
        non_private_train_loss_values,
        non_private_test_loss_values,
        f"{exp_data_dir}/plots_data/{non_private_model_id_str}_mia_roc"
    )
    np.savez(f"{exp_data_dir}/plots_data/{non_private_model_id}_mia_roc_data.npz", fpr=fpr, tpr=tpr)

    # Define args for training model with DP
    private_args = {
        'epochs': epochs,  # number of epochs for training model
        'lr': lr,  # learning rate for training model
        'disable_dp': False,  # non-private training,
        'secure_rng': False,  # flag to enable secure RNG to have trustworthy privacy guarantees
        'epsilon': config['private_args']['epsilon'],  # target epsilon for (eps, delta)-DP
        'delta': config['private_args']['delta'],  # target delta for (eps, delta)-DP
        'clipping_norm': config['private_args']['clipping_norm']  # per-sample clipping norm for DP-SGD
    }

    input_size = get_input_size(model_type, input_size)
    private_model = get_model_obj(model_type, input_size, num_classes).to(device)
    channels_format = get_channels_format(model_type)

    private_model_id = f'{exp_id}_private'
    private_optimizer = optim.Adam(
        private_model.parameters(),
        lr=lr,
    )
    privacy_engine = PrivacyEngine(secure_mode=private_args['secure_rng'])

    private_model, private_optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
        module=private_model,
        optimizer=private_optimizer,
        data_loader=train_dataloader,
        epochs=epochs,
        target_epsilon=private_args['epsilon'],
        target_delta=private_args['delta'],
        max_grad_norm=private_args['clipping_norm']
    )

    private_model_filepath = f"{exp_data_dir}/models/{private_model_id}"
    if os.path.exists(private_model_filepath):
        print("Loading model trained with DP for experiment.")
        checkpoint = torch.load(private_model_filepath)
        private_model.load_state_dict(checkpoint['model_state_dict'])
        private_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Loading model performance stats
        data = np.load(private_model_filepath + "_perf.npz")
        private_train_losses = data['train_losses']
        private_test_losses = data['test_losses']
        private_train_accs = data['train_accs']
        private_test_accs = data['test_accs']
    else:
        print("Training model with DP for experiment.")
        # Train and test model without DP
        private_model, private_train_losses, private_test_losses, private_train_accs, private_test_accs = train_test_model(
            model=private_model,
            args=private_args,
            device=device,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=private_optimizer,
            privacy_engine=privacy_engine,
            channels_format=channels_format
        )

        # Saving model
        torch.save({
            'model_state_dict': private_model.state_dict(),
            'optimizer_state_dict': private_optimizer.state_dict(),
        }, private_model_filepath)

        # Saving model performance stats
        np.savez(f"{private_model_filepath}_perf.npz",
                 train_losses=private_train_losses,
                 test_losses=private_test_losses,
                 train_accs=private_train_accs,
                 test_accs=private_test_accs)

    # Plot model performance graphs
    private_model_id_str = private_model_id.replace(".", "_")
    plot_train_and_test_loss_per_epoch(
        "Private Model", 
        private_train_losses, 
        private_test_losses,
        f"{exp_data_dir}/plots_data/{private_model_id_str}_loss_per_epoch"
    )
    plot_train_and_test_acc_per_epoch(
        "Private model", 
        private_train_accs, 
        private_test_accs,
        f"{exp_data_dir}/plots_data/{private_model_id_str}_acc_per_epoch"
    )

    # Run membership inference attack on model without DP
    per_sample_loss_values_filepath = f"{exp_data_dir}/plots_data/{exp_id}_per_sample_loss_values_private.npz"
    private_train_loss_values, private_test_loss_values = [], []
    if os.path.exists(per_sample_loss_values_filepath):
        print("Loading per-sample train and test loss values.")
        data = np.load(per_sample_loss_values_filepath)
        private_train_loss_values = data['per_sample_train_losses']
        private_test_loss_values = data['per_sample_test_losses']
    else:
        print("Generating per-sample train and test loss values.")
        private_train_loss_values = get_loss_values(private_model,
                                                    device, 
                                                    train_dataloader,
                                                    channels_format)
        private_test_loss_values = get_loss_values(private_model,
                                                   device, 
                                                   test_dataloader,
                                                   channels_format)
        np.savez(per_sample_loss_values_filepath,
                 per_sample_train_losses=private_train_loss_values,
                 per_sample_test_losses=private_test_loss_values)
    plot_train_and_test_losses(
        private_train_loss_values, 
        private_test_loss_values,
        f"{exp_data_dir}/plots_data/{private_model_id_str}_per_sample_losses"
    )
    fpr, tpr = get_mia_model_roc_curve(
        private_train_loss_values,
        private_test_loss_values,
        f"{exp_data_dir}/plots_data/{private_model_id_str}_mia_roc"
    )
    np.savez(f"{exp_data_dir}/plots_data/{private_model_id}_mia_roc_data.npz", fpr=fpr, tpr=tpr)

