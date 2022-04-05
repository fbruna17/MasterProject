import warnings
import pandas as pd
from pipeline import TrainingParameters, DataParameters, Pipeline
from src.Datahandler.prepare_data import split_data
from src.features import build_features as bf
from src.models.weather_forecasting import WeatherForecasting, LSTNet
from src.helpers import plot_losses
from src.models.architecture import *

warnings.simplefilter(action="ignore")
# %% LOAD DATA

path = 'data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)
df = df.drop(columns="Minute")
df = pd.read_pickle(path)[:30000]

# %% BUILD FEATURES
df = bf.build_features(df)

# %% TRAIN, VALIDATION, TEST SPLIT
train, val, test = split_data(df)
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

weather_df = df[['Month_sin',
                'Month_cos', 'Hour_sin',
                'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos', 'Tamb', 'Cloudopacity', 'DewPoint', 'DHI',
                'DNI',
                'EBH', 'Pw', 'Pressure', 'WindVel', 'AlbedoDaily', 'WindDir_sin',
                'WindDir_cos', 'Zenith_sin', 'Zenith_cos', 'Azimuth_sin', 'Azimuth_cos']]

# %% MODEL AND TRAINING PARAMETERS

memory = 24
horizon = 5
batch = 128
n_features = len(df.columns)
target = 'Cloudopacity'
learning_rate = 0.005
dropout = 0.
enc_features = len(enc_df.columns)
weather_features = len(weather_df.columns)

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

weather_params = {

}

# %% SCALING and TORCH DATALOADER

# %% MODEL INSTANTIATION

data_params = DataParameters(memory=memory, horizon=horizon, batch_size=batch, target=target)

model = LSMTEncoderDecoder1(encoder_params=encoder_params, decoder_params=decoder_params, memory=memory,
                            fc_hidden_size=624, output_size=horizon)

weather_model = WeatherForecasting(input_size=n_features, hidden_size1=32, hidden_size2=48, output_size=horizon)

weather_pipeline =


optimizer = torch.optim.Adam(params=weather_model.parameters(), lr=learning_rate)

training_params = TrainingParameters(epochs=50,
                                     loss_function=F.mse_loss,
                                     optimiser=optimizer)

encoder_decoder_pipeline = Pipeline(data=enc_df, model=model, data_params=data_params, training_params=training_params)

params = {
    'n_extracted_features': encoder_params['output_size'],
    'n_external_features': len(df.columns),
    'predict_hidden_1': 288,
    'predict_hidden_2': 64,
    'n_output_steps': horizon,
}

# pred_net = PredictionNet(params, dropout)
train_new_model = False

if train_new_model:
    trained_model, losses = encoder_decoder_pipeline.train(plot=True)
    encoder_decoder_pipeline.save('enc_dec_pretrained.pkl')

pred_params = {
    'input_size': (encoder_params['output_size'] + n_features) * memory,
    'hs_1': 128,
    'hs_2': 64,
    'output': horizon
}

pretrained = torch.load('src/models/enc_dec_pretrained.pkl')

pred_net = PredNet(encoder=pretrained.encoder, params=pred_params, dropout=0.1, n_univariate_featues=enc_features)

optimizer = torch.optim.Adam(params=pred_net.parameters(), lr=learning_rate)

training_params = TrainingParameters(epochs=50,
                                     loss_function=F.mse_loss,
                                     optimiser=optimizer)

pred_pipeline = Pipeline(data=df, model=pred_net, data_params=data_params, training_params=training_params)
mdl, losses = pred_pipeline.train(plot=True)

plot_losses(losses)
# # %% TEST
#
# test_results = test_model(trained_model, test, transformer)
#
# plot_predictions(test_results, 100, horizon)
