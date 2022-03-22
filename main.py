import warnings
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.Datahandler.scaling import scale_data
from src.Datahandler.prepare_data import split_data
from src.Datahandler.SequenceDataset import make_torch_dataset, SubSequenceDataset
from src.models.architecture import BayesianLSTM, LSTMEncoderDecoder
from src.models.train_model import train_model
from src.models.test_model import test_model

warnings.simplefilter(action="ignore")
# %% LOAD DATA

path = 'data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)[:2000]

# %% MODEL AND TRAINING PARAMETERS
columns = ['GHI']
df = df[columns]
memory = 15
horizon = 5
batch = 64
target = "GHI"
n_features = len(df.columns)
latent_dim = 5
learning_rate = 0.05
EPOCHS = 10
train_new_model = False

data = DataLoader(SubSequenceDataset(dataframe=df, memory=memory, horizon=horizon, features=columns, target=target),
                  batch_size=batch)

model = LSTMEncoderDecoder(n_features, 16, 16, 1, horizon)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for e in range(50):
    l = 0
    for x1,x2,y in data:
        y_hat = model(x1, x2)
        loss = torch.sqrt(F.mse_loss(y_hat, y))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        l += loss.detach()

    if e % 5 == 0:
        print(l / 2000)
# # %% TRAIN, VALIDATION, TEST SPLIT
# train, val, test = split_data(df)
#
# # %% SCALING
# train, val, test, transformer = scale_data(train, val, test)
#
# # %% TORCH DATASET
# train_dataset, val_dataset, test_dataset = make_torch_dataset(train, val, test, memory, horizon, batch, target)
#
# # %% INITIALIZE  MODEL
# model = BayesianLSTM(n_features, latent_dim, batch)
#
# # %% TRAIN MODEL
# trained_model = train_model(model, learning_rate, EPOCHS, train_dataset)
#
# # %% TEST MODEL
# error, ys, y_hats = test_model(trained_model, test_dataset, transformer)
