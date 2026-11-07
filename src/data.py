import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_csv(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    x = data[:, :-1]
    y = data[:, -1].astype(np.int64)
    # conv1d wants (batch, channels, length)
    x = torch.from_numpy(x).unsqueeze(1)
    return x, torch.from_numpy(y)


def get_loaders(data_dir="data", batch_size=128):
    x_train, y_train = load_csv(f"{data_dir}/mitbih_train.csv")
    x_test, y_test = load_csv(f"{data_dir}/mitbih_test.csv")
    train = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)
    return train, test, y_train