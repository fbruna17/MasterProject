import warnings
import pandas as pd
import torch.utils.data
import pytorch_lightning as pl
from pytorch_forecasting import QuantileLoss
import torch.nn as nn
from pipeline import TrainingParameters, DataParameters, Pipeline
from src.Datahandler.SequenceDataset import make_torch_dataset
from src.features import build_features as bf
from src.helpers import plot_losses
from src.models.archs import WhateverNet2
from src.likelihood_models import QuantileRegression, WeibullLikelihood, GammaLikelihood, BetaLikelihood, \
    GaussianLikelihood

warnings.simplefilter(action="ignore")

# %% Preparing data and building features.
path = 'data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)
df = df.drop(columns="Minute")
df = bf.build_features(df)
memory = 24
horizon = 4
batch = 128

weather_df = df[['Month_sin', 'Month_cos',
                 'Hour_sin', 'Hour_cos',
                 'Year_sin', 'Year_cos',
                 'Day_sin', 'Day_cos',
                 'Tamb', 'Cloudopacity',
                 'DewPoint', 'Pw',
                 'Pressure', 'WindVel',
                 'AlbedoDaily', 'Zenith_sin',
                 'Zenith_cos', 'Azimuth_sin', 'Azimuth_cos']]
weather_columns = weather_df.columns.to_list()
df_columns = df.columns.to_list()
df_order = pd.unique(weather_columns + df_columns)

df = df[df_order]
df.insert(0, 'GHI', df.pop('GHI'))
df.to_csv("irradiance_dataset.csv")
# %% Specify features and other constants.

n_features = len(df.columns)
target = 'GHI'
learning_rate = 0.0005
dropout = 0.05
weather_features = len(weather_df.columns)

data_params = DataParameters(memory=memory, horizon=horizon, batch_size=batch, target=target, sliding=True, input_size=n_features)

tcn_params = {"num_filters": 5,
              "dilation_base": 2,
              "kernel_size": 3}
model_params = {"hidden_size": 80,
                "bigru_layers": 1,
                "dropout": dropout,
                "attention_head_size": 4,
                "tcn_params": tcn_params}

training_params = TrainingParameters(epochs=25, learning_rate=learning_rate, loss_function=nn.MSELoss(), scheduler=True)

pipe = Pipeline(data=df,
                data_params=data_params,
                training_params=training_params,
                target="GHI",
                likelihood=QuantileRegression(quantiles=[0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]),
                model_params=model_params)

pipe.train(plot=True)
