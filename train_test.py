import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from opacus.utils.batch_memory_manager import BatchMemoryManager

from model_performance import compute_accuracy


def train(args, model, device, train_loader,
          optimizer, privacy_engine, epoch,
          channels_format=None):
    model.train(True)
    criterion = nn.BCELoss()  # can't use cross entropy for multi-label
    
    losses = []
    if not args["disable_dp"]:
        # to avoid OOM errors for private training
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
            f"(Epsilon = {args['epsilon']}, Delta = {args['delta']})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

    return np.mean(losses)


def test(model, device, test_loader, channels_format=None):
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


def train_test_model(model, args, device, train_dataloader, test_dataloader, optimizer, privacy_engine, channels_format):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(1, args['epochs'] + 1):
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

        test_loss = test(model, device, test_dataloader, channels_format)
        test_losses.append(test_loss)

        test_acc = compute_accuracy(model, device, test_dataloader, channels_format)
        test_accs.append(test_acc)

    return model, train_losses, test_losses, train_accs, test_accs

