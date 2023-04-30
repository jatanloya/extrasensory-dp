import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from opacus.utils.batch_memory_manager import BatchMemoryManager

from model_performance import compute_accuracy


def train(args, model, device, train_loader,
          optimizer, privacy_engine, epoch,
          channels_format=None):
    """
    Train a model with or without differential privacy.

    :param args: Argument object with hyperparameters for non-private and private training.
    :param model: Model object that needs to be trained.
    :param device: Device (e.g. CPU, CUDA) to be used for training.
    :param train_loader: Training dataset.
    :param optimizer: Optimizer (non-private or private) to be used for training.
    :param privacy_engine: PrivacyEngine object, used to compute the resultant privacy budget.
    :param epoch: Current epoch of training.
    :param channels_format: Format of input samples that the model object requires.
    :return: Loss of the model on the training dataset, averaged over all samples.
    """
    model.train(True)
    criterion = nn.BCELoss()  # Can't use cross entropy for multi-label
    
    losses = []
    if not args["disable_dp"]:
        # Avoid OOM errors for private training: encapsulate data loader using Opacus BatchMemoryManager
        with BatchMemoryManager(
            data_loader=train_loader, max_physical_batch_size=2, optimizer=optimizer
        ) as new_train_loader:

            for _batch_idx, (data, target) in enumerate(tqdm(new_train_loader)):
                data, target = data.to(device), target.to(device)

                if channels_format == 'channels_first':
                    data = torch.unsqueeze(data, 2)
                    data = data.permute(0, 2, 1)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()
                losses.append(loss.item())
    else:
        for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)

            if channels_format == 'channels_first':
                data = torch.unsqueeze(data, 2)
                data = data.permute(0, 2, 1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    if not args["disable_dp"]:
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            # f"(Epsilon = {privacy_engine.accountant.get_epsilon(delta=args['delta'])}, Delta = {args['delta']})"
            f"(Epsilon = {args['epsilon']}, Delta = {args['delta']})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

    return np.mean(losses)


def test(model, device, test_loader, channels_format=None):
    """
    Compute accuracy of a model on the test dataset.

    :param model: Model object that needs to be evaluated.
    :param device: Device (e.g. CPU, CUDA) that was used for training the model.
    :param test_loader: Test dataset.
    :param channels_format: Format of input samples that the model object requires.
    :return: Loss of the model on the test dataset, averaged over all samples.
    """
    model.eval()
    criterion = nn.BCELoss()
    losses = []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)

            if channels_format == 'channels_first':
                data = torch.unsqueeze(data, 2)
                data = data.permute(0, 2, 1)

            output = model(data)
            loss = criterion(output, target).item()
            losses.append(loss)

    test_loss = np.mean(losses)
    print(f"Test Loss: {test_loss:.6f}")
    return test_loss


def train_test_model(model, args, device, train_dataloader, test_dataloader,
                     optimizer, privacy_engine, channels_format):
    """
    Perform a train step and a test step for the model in every epoch.

    :param model: Model object that needs to be trained.
    :param args: Argument object with hyperparameters for non-private and private training.
    :param device: Device (e.g. CPU, CUDA) to be used for training.
    :param train_dataloader: Training dataset.
    :param test_dataloader: Test dataset.
    :param optimizer: Optimizer (non-private or private) to be used for training.
    :param privacy_engine: PrivacyEngine object, used to compute the resultant privacy budget.
    :param channels_format: Format of input samples that the model object requires.
    :return: Trained model object, train and test accuracies per epoch, average train and test loss values per epoch.
    """
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(1, args['epochs'] + 1):
        # Train step
        train_loss = train(
            args,
            model,
            device,
            train_dataloader,
            optimizer,
            privacy_engine,
            epoch,
            channels_format
        )
        train_losses.append(train_loss)

        train_acc = compute_accuracy(model, device, train_dataloader, channels_format)
        train_accs.append(train_acc)

        # Test step
        test_loss = test(model, device, test_dataloader, channels_format)
        test_losses.append(test_loss)

        test_acc = compute_accuracy(model, device, test_dataloader, channels_format)
        test_accs.append(test_acc)

    return model, train_losses, test_losses, train_accs, test_accs

