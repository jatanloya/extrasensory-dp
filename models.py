import torch.nn as nn
import torch.nn.functional as F


def get_input_size(model_type, input_size):
    supported_model_types = ["simple_nn", "cnn"]
    if model_type == "simple_nn":
        return input_size
    elif model_type == "cnn":
        return 1
    else:
        raise ValueError(f"{model_type} unsupported. Please use one of {supported_model_types} instead!")


def get_model_obj(model_type, input_size, num_classes):
    supported_model_types = ["simple_nn", "cnn"]
    if model_type == "simple_nn":
        return SimpleNN(input_size, num_classes)
    elif model_type == "cnn":
        return CNN(input_size, num_classes)
    else:
        raise ValueError(f"{model_type} unsupported. Please use one of {supported_model_types} instead!")


def get_channels_format(model_type):
    supported_model_types = ["simple_nn", "cnn"]
    if model_type == "simple_nn":
        return None
    elif model_type == "cnn":
        return "channels_last"
    else:
        raise ValueError(f"{model_type} unsupported. Please use one of {supported_model_types} instead!")


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.sigm = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigm(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, 5)
        self.conv2 = nn.Conv1d(64, 64, 5)
        self.conv3 = nn.Conv1d(64, 64, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6592, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigm = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.sigm(x)
        return x
