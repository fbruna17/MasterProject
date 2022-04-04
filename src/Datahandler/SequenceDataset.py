import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, memory, horizon):
        self.features = features
        self.target = target
        self.memory = memory
        self.horizon = horizon
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0] - self.memory - self.horizon + 1

    def __getitem__(self, i):
        start_y = i + self.memory
        _x = self.X[i: start_y, :]  # ':' means all columns
        _y = self.y[start_y: start_y + self.horizon]
        return _x, _y


class SubSequenceDataset(SequenceDataset):
    def __init__(self, dataframe, target, features, memory, horizon):
        super(SubSequenceDataset, self).__init__(dataframe, target, features, memory, horizon)

    def __getitem__(self, i):
        start_y = i + self.memory
        _x = self.X[i: start_y]
        _x2 = self.X[start_y - self.horizon: start_y]  # ':' means all columns
        _y = self.y[start_y: start_y + self.horizon]
        torch.equal(_x[-4:], _x2)
        return _x, _x2, _y


def make_torch_dataset(train: torch.Tensor, val: torch.Tensor, test: torch.Tensor, memory: int, horizon: int,
                       batch: int, target: str):
    train_sequence = SequenceDataset(train,
                                     target=target,
                                     features=list(train.columns),
                                     memory=memory,
                                     horizon=horizon)
    val_sequence = SequenceDataset(val,
                                   target=target,
                                   features=list(val.columns),
                                   memory=memory,
                                   horizon=horizon)
    test_sequence = SequenceDataset(test,
                                    target=target,
                                    features=list(test.columns),
                                    memory=memory,
                                    horizon=horizon)

    train_dataset = DataLoader(train_sequence, batch_size=batch)
    val_dataset = DataLoader(val_sequence, batch_size=batch)
    test_dataset = DataLoader(test_sequence, batch_size=1)

    return train_dataset, val_dataset, test_dataset
