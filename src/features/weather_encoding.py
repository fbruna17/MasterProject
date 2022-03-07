import os

import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.Datahandler.scaling import TimeSeriesTransformer
import numpy as np
# %%
path = "/Users/jonathanjonler/PycharmProjects/MasterProject/data/raw/irradiance_data_NL_2007_2022.pkl"
df = pd.read_pickle(path)

def separate_weather_features(df: pd.DataFrame, weather_cols=None) -> pd.DataFrame:
    if weather_cols is None:
        weather_cols = ["Tamb", "Cloudopacity", "DewPoint", "Pw", "Pressure", "WindDir", "WindVel"]
    return df[weather_cols]


def generate_noise(series: pd.Series):
    min_value = series.min()
    max_value = series.max()
    noise = series.apply(lambda x: x - 0.015 * ((min_value - max_value) * np.random.random() + min_value))
    noise = np.maximum(noise, min_value)
    return noise

noisy_weather = weather_features.apply(generate_noise)

def generate_leads(df: pd.DataFrame, series: pd.Series, col_name, number_of_leads: int = 5):
    for i in range(1, number_of_leads):
        df[f"{col_name} t+{i}"] = series.shift(-i)


#%%
split_train = int(len(df) * 0.8)
split_val = split_train + int(len(df) * 0.1)

train = df[:split_train]
val = df[split_train:split_val]
test = df[split_val:]

# %%
transfomer = TimeSeriesTransformer(train)
transfomer.standardizer_fit()
train = transfomer.standardizer_transform(train)
val = transfomer.standardizer_transform(val)
test = transfomer.standardizer_transform(test)

# %%

seq = torch.tensor(weather_features.values).float()
train_data = DataLoader(seq, batch_size=64)


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 7))


    def forward(self, x):
        x = self.encoder(x)
        return x

# %%

model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in train_data:
        img = data
        img = Variable(img)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        out = output.cpu()
        print(out)

