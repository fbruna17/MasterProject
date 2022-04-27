import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.Datahandler import SequenceDataset

X = np.random.rand(*(18, 4))
X = pd.DataFrame(X, columns=["1", "2", "3", "4"])

class SlidingDataset(Dataset):
    def __init__(self, dataframe, target, features, memory, horizon):
        self.features = features
        self.target = target
        self.memory = memory
        self.horizon = horizon
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return X.shape[0] - self.memory - self.horizon

    def __getitem__(self, index):
        start_y = index + self.horizon
        _x = self.X[index: index + self.memory, :]
        _y = self.y[start_y: start_y + self.memory]
        return _x, _y

dataset = SlidingDataset(dataframe=X, target="1", features=["1", "2", "3", "4"], memory=3, horizon=2)
dl = DataLoader(dataset=dataset, batch_size=1, drop_last=True)
