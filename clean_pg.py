import warnings
import pandas as pd
import torch.utils.data
import pytorch_lightning as pl
from pytorch_forecasting import QuantileLoss

from pipeline import TrainingParameters, DataParameters, Pipeline
from src.Datahandler.SequenceDataset import make_torch_dataset
from src.features import build_features as bf
from src.helpers import plot_losses
from src.models.architecture import *
from src.models.archs import WhateverNet2, WhateverNet3

warnings.simplefilter(action="ignore")

# %% Preparing data and building features.
path = 'data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)[:4000]
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
df.to_csv("irradiance_dataset")
# %% Specify features and other constants.

n_features = len(df.columns)
target = 'GHI'
learning_rate = 0.0005
dropout = 0.4
weather_features = len(weather_df.columns)


data_params = DataParameters(memory=memory, horizon=horizon, batch_size=batch, target=target)

tcn_params = {"num_filters": 5,
              "dilation_base": 2,
              "kernel_size": 3}

model2 = WhateverNet2(input_size=n_features,
                      memory=memory,
                      target_size=1,
                      horizon=horizon,
                      hidden_size=56,
                      bigru_layers=1,
                      attention_head_size=4,
                      tcn_params=tcn_params,
                      nr_parameters=1
                      )

model_optimizer = torch.optim.Adam(params=model2.parameters(), lr=learning_rate, weight_decay=0.001)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=model_optimizer, gamma=0.93)

training_params = TrainingParameters(epochs=25, loss_function=nn.MSELoss(), optimiser=model_optimizer)

pipe = Pipeline(data=df, model=model2, data_params=data_params, training_params=training_params, target="GHI")

pipe.train(plot=True)
