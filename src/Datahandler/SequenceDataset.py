import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


class WeatherSequence(Dataset):
    def __init__(self, dataframe, features, memory):
        self.dataframe = dataframe
        self.features = features
        self.memory = memory
        self.X = torch.tensor(dataframe.values).float()

    def __len__(self):
        return self.X.shape[0] - self.memory

    def __getitem__(self, i):
        start_y = i + self.memory
        _x = self.X[i: start_y, :]  # ':' means all columns
        return _x


class SlidingDataset(Dataset):
    def __init__(self, dataframe, target, features, memory, horizon):
        self.features = features
        self.target = target
        self.memory = memory
        self.horizon = horizon
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0] - (self.memory - self.horizon)

    def __getitem__(self, index):
        start_y = index + self.horizon
        _x = self.X[index: index + self.memory, :]
        _y = self.y[start_y: start_y + self.memory]
        return _x, _y

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


def make_torch_dataset(train, val, test, memory: int, horizon: int,
                       batch: int, target: str, sliding, drop_last=True):

    if sliding == True:
        train_sequence = SlidingDataset(train,
                                        target=target,
                                        features=list(train.columns),
                                        memory=memory,
                                        horizon=horizon)

        val_sequence = SlidingDataset(val,
                                        target=target,
                                        features=list(val.columns),
                                        memory=memory,
                                        horizon=horizon)

        test_sequence = SlidingDataset(test,
                                        target=target,
                                        features=list(test.columns),
                                        memory=memory,
                                        horizon=horizon)
        train_dataset = DataLoader(train_sequence, batch_size=batch, drop_last=True)
        val_dataset = DataLoader(val_sequence, batch_size=batch, drop_last=True)
        test_dataset = DataLoader(test_sequence, batch_size=batch, drop_last=True)
        return train_dataset, val_dataset, test_dataset

    else:
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

        train_dataset = DataLoader(train_sequence, batch_size=batch, drop_last=drop_last)
        val_dataset = DataLoader(val_sequence, batch_size=batch, drop_last=drop_last)
        test_dataset = DataLoader(test_sequence, batch_size=1, drop_last=drop_last)

        return train_dataset, val_dataset, test_dataset



def make_weather_dataset(data: pd.DataFrame, memory, batch, drop_last=True):
    return DataLoader(WeatherSequence(dataframe=data, features=len(data.columns), memory=memory), batch_size=batch, drop_last=drop_last)