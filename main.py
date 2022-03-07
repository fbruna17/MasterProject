import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.logger.logger import Logger

## LOAD DATA

path = 'data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)[:500]

## Outlier Removal and Feature Extraction
weather_copy = df[['Cloudopacity', 'DewPoint', 'Pressure', 'WindDir', 'WindVel', 'Pw', 'Tamb']]
df = df.loc[(df.Hour < 22) & (df.Hour > 5)]

## TRAIN, VALIDATION, TEST SPLIT
split_train = int(len(df) * 0.8)
split_val = split_train + int(len(df) * 0.1)

train = df[:split_train]
val = df[split_train:split_val]
test = df[split_val:]

## SCALING
...


## Torch Dataset


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
horizon = 5
batch = 1

train_sequence = SequenceDataset(train, target='GHI', features=['GHI', 'Tamb'], memory=memory, horizon=horizon)
val_sequence = SequenceDataset(val, target='GHI', features=['GHI', 'Tamb'], memory=memory, horizon=horizon)
test_sequence = SequenceDataset(test, target='GHI', features=['GHI', 'Tamb'], memory=memory, horizon=horizon)
## Torch Dataloader

train_data = DataLoader(train_sequence, batch_size=batch)
val_data = DataLoader(val_sequence, batch_size=batch)
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
model = SimpleLSTM(n_features=len(train_sequence.features), hidden_size=16,
                   num_layers=1,
                   dropout=0.1,
                   output_length=horizon)

optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.L1Loss()

log = Logger()

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

    train_loss = np.mean(np.stack(temp_losses))

    temp_losses = []
    model.eval()
    for x, y in val_data:
        x = x.to(device).float()
        y = y.to(device).float()

        # Predict and calculate loss
        y_hat = model(x)
        loss = loss_function(y, y_hat)
        temp_losses.append(loss.cpu().detach().numpy())

    val_loss = np.mean(np.stack(temp_losses))

    log.append_loss(int(i+1), train_loss, val_loss, str(loss_function))

    if i % 10 == 0:
        print(f"Loss at epoch {i} is {round(train_loss, 5)}")

log.plot_losses()

## Testing

with torch.no_grad():
    temp_losses = []
    for x, y in test_data:
        x = x.to(device).float()
        y = y.to(device).float()

        # Make prediction(s)
        y_hat = model(x)

        # inverse..

        # Calculate error
        error = loss_function(y.squeeze(), y_hat.squeeze())

        temp_losses.append(error)
        log.add_prediction(X=x, y=y, y_hat=y_hat, error=error, loss_function=str(loss_function))

log.plot_prediction(15)
