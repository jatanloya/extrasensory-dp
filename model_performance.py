import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def compute_accuracy(model, device, dataloader, channels_format=None):
    """
    Compute accuracy of the model on the specified dataset.

    :param model: Model object.
    :param device: Device (e.g. CPU, CUDA) that was used for training the model.
    :param dataloader: Dataset for which accuracy needs to be computed.
    :param channels_format: Format of input samples that the model object requires.
    :return:
    """
    model.eval()

    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)

            if channels_format == 'channels_first':
                data = torch.unsqueeze(data, 2)
                data = data.permute(0, 2, 1)

            scores = model(data)

            preds = scores > 0.5
            num_correct += accuracy_score(target.to('cpu'),
                                          preds.to('cpu'),
                                          normalize=False)
            num_samples += len(data)

    model.train(True)

    total_acc = num_correct / num_samples
    print("Computed accuracy:", total_acc)
    return total_acc


def plot_train_and_test_loss_per_epoch(model_id, train_losses, test_losses, filename):
    """
    Plot train and test loss values per epoch for the specified model.

    :param model_id: Identifier for model.
    :param train_losses: Average train loss per epoch.
    :param test_losses: Average test loss per epoch.
    :param filename: Filename where plot will be saved.
    """
    fig, ax = plt.subplots()

    ax.plot(train_losses, label='Train Loss')
    ax.plot(test_losses, label='Test Loss')

    ax.set_title(f"Model: {model_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.legend()
    plt.savefig(filename, dpi=500)


def plot_train_and_test_acc_per_epoch(model_id, train_accs, test_accs, filename):
    """
    Plot train and test accuracies per epoch for the specified model.

    :param model_id: Identifier for model.
    :param train_accs: Train accuracy per epoch.
    :param test_accs: Test accuracy per epoch.
    :param filename: Filename where plot will be saved.
    """
    fig, ax = plt.subplots()

    ax.plot(train_accs, label='Train Acc')
    ax.plot(test_accs, label='Test Acc')

    ax.set_title(f"Model: {model_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Acc")

    ax.legend()
    plt.savefig(filename, dpi=500)
