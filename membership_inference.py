import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns


def get_loss_values(model, device, dataloader, channels_format=None):
    model.eval()
    criterion = nn.BCELoss()

    loss_values = []
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data, target = data.to(device), target.to(device)

            if channels_format == 'channels_last':
                data = torch.unsqueeze(data, 2)
                data = data.permute(0, 2, 1)

            output = model(data)

            for one_output, one_target in zip(target, output):
                loss = criterion(one_output, one_target).item()
                loss_values.append(loss)

    return np.array(loss_values)


def plot_train_and_test_losses(train_loss_values, test_loss_values, filename):
    loss_value_types = ['Train'] * len(train_loss_values)
    loss_value_types.extend(['Test'] * len(test_loss_values))
    data_dict = {
        'loss_values': np.concatenate([train_loss_values, test_loss_values]),
        'Data': loss_value_types
    }

    fig, ax = plt.subplots()
    sns.histplot(data=data_dict,
                 x='loss_values',
                 hue='Data',
                 stat="probability",
                 # binwidth=0.01,
                 log_scale=True,
                 kde=True,
                 hue_order=['Train', 'Test'],
                 palette=['g', 'b'],
                 common_norm=False)
    ax.set_xlabel('Loss Value')
    ax.set_ylabel('Fraction')
    plt.tight_layout()

    plt.savefig(filename, dpi=500)


def get_mia_model_roc_curve(train_loss_values, test_loss_values, filename):
    loss_values = np.concatenate(
        [train_loss_values, test_loss_values]
    ).reshape(-1, 1)
    true_membership_labels = [1] * len(train_loss_values)
    true_membership_labels.extend([0] * len(test_loss_values))

    mia_attack_model = LogisticRegression(
        class_weight='balanced'
    )
    mia_attack_model.fit(loss_values, true_membership_labels)

    RocCurveDisplay.from_estimator(
        mia_attack_model, loss_values, true_membership_labels
    )
    plt.savefig(filename, dpi=500)

    y_preds = mia_attack_model.predict_proba(loss_values)[:, 1]
    fpr, tpr, _ = roc_curve(true_membership_labels, y_preds)

    return fpr, tpr
