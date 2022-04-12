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

path = 'data/raw/15min.pkl'
df = pd.read_pickle(path)[:5000]

# %% BUILD FEATURES
# df = bf.build_features(df)

target = 'AirTemp'

# %% TRAIN, VALIDATION, TEST SPLIT
# column_order = [target, 'Month_sin',
#                 'Month_cos', 'Hour_sin',
#                 'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos', 'Cloudopacity', 'DewPoint', 'DHI',
#                 'DNI', # todo include tamb
#                 'EBH', 'Pw', 'Pressure', 'WindVel', 'AlbedoDaily', 'WindDir_sin',
#                 'WindDir_cos', 'Zenith_sin', 'Zenith_cos', 'Azimuth_sin', 'Azimuth_cos',
#                 'Azimuth t+1', 'Azimuth t+2', 'Azimuth t+3', 'Zenith t+1', 'Zenith t+2',
#                 'Zenith t+3']
#
# df = df[column_order]
# enc_df = df[[target, 'Month_sin',
#              'Month_cos', 'Hour_sin',
#              'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos']]

cols = ['Ghi', 'AirTemp', 'Azimuth',
        'CloudOpacity', 'DewpointTemp', 'Dhi', 'Dni', 'Ebh',
        'GtiFixedTilt', 'GtiTracking', 'PrecipitableWater', 'RelativeHumidity',
        'SnowWater', 'SurfacePressure', 'WindDirection10m', 'WindSpeed10m',
        'Zenith', 'AlbedoDaily', 'Month', 'Day', 'Hour', 'Minute', 'Year']

df = df[cols]
# %% MODEL AND TRAINING PARAMETERS

memory = 24
horizon = 4 * 4
batch = 128
n_features = len(df.columns)
# enc_features = len(enc_df.columns)

learning_rate = 0.001
dropout = 0.4
epochs = 100

# encoder_params = {
#     'input_size': enc_features,
#     'hs_1': 12,
#     'hs_2': 8,
#     'output_size': 1,
#     'dropout': dropout,
# }
# decoder_params = {
#     'input_size': encoder_params['output_size'],
#     'hs_1': 12,
#     'hs_2': 24,
#     'output_size': 12,
#     'dropout': dropout
# }

# %% MODEL INSTANTIATION

data_parameters = DataParameters(memory=memory, horizon=horizon, batch_size=batch, target=target)

train, test, transformer = prepare_data(df, data_parameters)

x, y = test.dataset[20]

blstm = BayesianLSTM(n_features=len(df.columns), hs_1=64, hs_2=32, output_length=horizon, batch_size=batch,
                     dropout=dropout)

lstm = SimpleLSTM(input_size=len(df.columns), output_size=horizon, hs_1=128, hs_2=64, dropout=dropout)

gru = SimpleGru(input_size=len(df.columns), output_size=horizon, hs=64)

hidden = int(len(df.columns) * memory) / 3

var_lstm = SimpleVarLSTM(input_size=len(df.columns), hs=128, output_size=horizon, dropout=dropout)
model = var_lstm

optimiser = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


def rmse(y, y_hat):
    return torch.sqrt(F.mse_loss(y, y_hat))


training_parameters = TrainingParameters(epochs=epochs, optimiser=optimiser, loss_function=rmse)

# mu, eta, y_inv = monte_carlo(trained, x.unsqueeze(0), y, 100, transformer)


trained, losses = train_model(model=model, data=train, training_params=training_parameters, plot=True, plot_freq=20)

plt.plot(losses['train'], label='train')
plt.plot(losses['validation'], label='val')
plt.show()

test_results = test_model(trained, test, transformer)


def relu(x):
    return max(0, x)


def plot_interval1(y, mu, eta):
    xx = list(range(len(y)))
    plt.plot(y, label='y')
    plt.plot(mu.flatten(), label='median')
    lower1 = (mu - eta).flatten()
    upper1 = (mu + eta).flatten()
    plt.fill_between(xx, lower1, upper1, alpha=0.3)
    lower2 = (mu - 2 * eta).flatten()
    upper2 = (mu + 2 * eta).flatten()
    plt.fill_between(xx, lower2, upper2, alpha=0.15)
    plt.legend()
    plt.show()


def plot_interval(lower, median, upper, true):
    xx = list(range(len(median)))

    plt.plot(xx, true, label='True', color='b')
    plt.plot(xx, median, label='Median', linestyle='--')
    plt.fill_between(xx, lower, upper, alpha=0.2, label='90% interval')
    plt.legend()
    plt.show()


for i in range(horizon, 100, horizon):
    x, y = test.dataset[i]
    _, prev_y = test.dataset[i - horizon + 1]
    lower, median, upper, y_inv = monte_carlo(trained, x.unsqueeze(0), y, 1000, transformer, lower_quantile=0.01, upper_quantile=0.99)
    plt.plot(list(range(-horizon + 1, 1)), prev_y, label='Previous Y', color='b')
    plt.ylim(0, 1)
    plot_interval(lower, median, upper, y_inv)
##
