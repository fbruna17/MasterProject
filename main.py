import warnings
import pandas as pd
import torch.optim
from matplotlib import pyplot as plt

from pipeline import TrainingParameters, DataParameters, Pipeline, train_model, test_model, prepare_data, monte_carlo
from src.features import build_features as bf
from src.helpers import plot_losses
from src.models.architecture import *

warnings.simplefilter(action="ignore")
# %% LOAD DATA

path = 'data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)[:30000]

# %% BUILD FEATURES
df = bf.build_features(df)

# %% TRAIN, VALIDATION, TEST SPLIT
column_order = ['GHI', 'Month_sin',
                'Month_cos', 'Hour_sin',
                'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos', 'Tamb', 'Cloudopacity', 'DewPoint', 'DHI',
                'DNI',
                'EBH', 'Pw', 'Pressure', 'WindVel', 'AlbedoDaily', 'WindDir_sin',
                'WindDir_cos', 'Zenith_sin', 'Zenith_cos', 'Azimuth_sin', 'Azimuth_cos',
                'Azimuth t+1', 'Azimuth t+2', 'Azimuth t+3', 'Zenith t+1', 'Zenith t+2',
                'Zenith t+3']

df = df[column_order]
enc_df = df[['GHI', 'Month_sin',
             'Month_cos', 'Hour_sin',
             'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos']]

# %% MODEL AND TRAINING PARAMETERS

memory = 48
horizon = 4
batch = 64
n_features = len(df.columns)
enc_features = len(enc_df.columns)
target = 'GHI'
learning_rate = 0.001
dropout = 0.4

encoder_params = {
    'input_size': enc_features,
    'hs_1': 12,
    'hs_2': 8,
    'output_size': 1,
    'dropout': dropout,
}
decoder_params = {
    'input_size': encoder_params['output_size'],
    'hs_1': 12,
    'hs_2': 24,
    'output_size': 12,
    'dropout': dropout
}

# %% MODEL INSTANTIATION

data_parameters = DataParameters(memory=memory, horizon=horizon, batch_size=batch, target=target)

train, test, transformer = prepare_data(df, data_parameters)

x,y = test.dataset[10]

blstm = BayesianLSTM(n_features=len(df.columns), output_length=horizon, batch_size=batch)

lstm = SimpleLSTM(input_size=len(df.columns), output_size=horizon, hs_1=64, hs_2=32)

optimiser = torch.optim.Adam(params=lstm.parameters(), lr=learning_rate)

training_parameters = TrainingParameters(epochs=10, optimiser=optimiser, loss_function=F.mse_loss)

trained, losses = train_model(model=lstm, data=train, training_params=training_parameters, plot=True)

test_results = test_model(trained, test, transformer)


mu, eta, y_inversed = monte_carlo(trained, x.unsqueeze(0), y, 100, transformer)

plt.plot(y_inversed, label='y')
plt.plot(mu.flatten(), label='mu')
plt.plot((mu + 2 * eta).flatten(), label='mu+2')
plt.plot((mu - 2 * eta).flatten(), label='mu-2')
plt.legend()
plt.show()

