import warnings

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.Datahandler import scaling

warnings.simplefilter(action="ignore")
# %% LOAD DATA
from src.Datahandler.scaling import TimeSeriesTransformer

path = 'data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)
warnings.simplefilter(action="ignore")


# %% Outlier Removal and Feature Extraction


def generate_noise(series: pd.Series):
    min_value = series.min()
    max_value = series.max()
    noise = series.apply(lambda x: x - 0.015 * ((min_value - max_value) * np.random.random() + min_value))
    noise = np.maximum(noise, min_value)
    return noise


def generate_leads(df: pd.DataFrame, series: pd.Series, col_name, number_of_leads: int = 5):
    for i in range(1, number_of_leads):
        df[f"{col_name} t+{i}"] = series.shift(-i)


# %% TRAIN, VALIDATION, TEST SPLIT
split_train = int(len(df) * 0.8)
split_val = split_train + int(len(df) * 0.1)

train = df[:split_train]
val = df[split_train:split_val]
test = df[split_val:]

# %% SCALING

transformer = TimeSeriesTransformer()
transformer.standardizer_fit(train)
train = transformer.standardizer_transform(train)
val = transformer.standardizer_transform(val)
test = transformer.standardizer_transform(test)

inv_transform = transformer.inverse_transform_target(y=torch.tensor(train["GHI"]).float(),
                                                     y_hat=torch.tensor(train["GHI"]).float())


# %%


# %% Torch Dataset


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


memory = 15
horizon = 4

train_sequence = SequenceDataset(train, target='GHI', features=list(df.columns), memory=memory, horizon=horizon)
val_sequence = SequenceDataset(val, target='GHI', features=list(df.columns), memory=memory, horizon=horizon)
test_sequence = SequenceDataset(test, target='GHI', features=list(df.columns), memory=memory, horizon=horizon)
## Torch Dataloader

train_data = DataLoader(train_sequence, batch_size=128)
val_data = DataLoader(val_sequence, batch_size=128)
test_data = DataLoader(test_sequence)


## MODEL

class SimpleLSTM(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_size,
                 num_layers,
                 dropout,
                 output_length,
                 ):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_features = n_features
        self.dropout = dropout
        self.output_length = output_length
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, self.output_length)

    def forward(self, x):
        lstm_out, other = self.lstm(x)  # x shape: (batch, memory, n_features = 3),
        #  lstm_out: (batch, memory, hidden_size)
        y_pred = self.linear(lstm_out[:, -1])
        # y_pred: (batch, horizon)
        return y_pred


## Training
epochs = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleLSTM(n_features=len(train_sequence.features), hidden_size=48,
                   num_layers=1,
                   dropout=0.1,
                   output_length=horizon)

optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.L1Loss()

training_loss = []
validation_loss = []

for i in range(epochs):
    model.train()

    temp_losses = []
    for x, y in train_data:
        x = x.to(device).float()
        y = y.to(device).float()

        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_function(y, y_hat)
        temp_losses.append(loss.cpu().detach().numpy())

        loss.backward()  # Calculates gradients w.r.t. loss
        optimizer.step()  # Changes the weights

    training_loss.append(np.mean(np.stack(temp_losses)))

    temp_losses = []
    model.eval()
    for x, y in val_data:
        x = x.to(device).float()
        y = y.to(device).float()

        # Predict and calculate loss
        y_hat = model(x)
        loss = loss_function(y, y_hat)
        temp_losses.append(loss.cpu().detach().numpy())

    validation_loss.append(np.mean(np.stack(temp_losses)))

    if i % 10 == 0:
        print(f"Loss at epoch {i} is {training_loss[-1]}")

## Testing

with torch.no_grad():
    temp_losses = []
    for x, y in test_data:
        x = x.to(device).float()
        y = y.to(device).float()

        # Make prediction(s)
        y_hat = model(x).unsqueeze(-1)

        # Calculate error
        y, y_hat = transformer.inverse_transform_target(y, y_hat)
        error = loss_function(y, y_hat)
        temp_losses.append(error)

test_error = np.mean(np.stack(temp_losses))
