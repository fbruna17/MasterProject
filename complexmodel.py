import warnings
import pandas as pd
import torch.utils.data

from pipeline import TrainingParameters, DataParameters, Pipeline, WeatherPipeline
from src.Datahandler.SequenceDataset import make_torch_dataset
from src.features import build_features as bf
from src.helpers import plot_losses
from src.models.architecture import *
warnings.simplefilter(action="ignore")
# %% LOAD DATA

path = 'data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)[:5000]
df = df.drop(columns="Minute")
df = bf.build_features(df)
column_order = ['GHI', 'Month_sin', 'Month_cos',
                'Hour_sin', 'Hour_cos', 'Year_sin',
                'Year_cos', 'Day_sin', 'Day_cos',
                'Tamb', 'Cloudopacity', 'DewPoint', 'DHI',
                'DNI', 'EBH', 'Pw',
                'Pressure', 'WindVel', 'AlbedoDaily',
                'WindDir_sin','WindDir_cos', 'Zenith_sin',
                'Zenith_cos', 'Azimuth_sin', 'Azimuth_cos',
                'Azimuth_sin t+1', 'Azimuth_sin t+2', 'Azimuth_sin t+3',
                'Azimuth_cos t+1', 'Azimuth_cos t+2', 'Azimuth_cos t+3',
                'Zenith_sin t+1', 'Zenith_sin t+2', 'Zenith_sin t+3',
                'Zenith_cos t+1', 'Zenith_cos t+2', 'Zenith_cos t+3'
                ]

df = df[column_order]
# %% BUILD FEATURES
memory = 24
horizon = 5
batch = 128

weather_df = df[['Month_sin', 'Month_cos',
                 'Hour_sin', 'Hour_cos',
                 'Year_sin', 'Year_cos',
                 'Day_sin', 'Day_cos',
                 'Tamb', 'Cloudopacity',
                 'DewPoint', 'DHI',
                 'DNI', 'EBH',
                 'Pw', 'Pressure',
                 'WindVel', 'AlbedoDaily',
                 'WindDir_sin', 'WindDir_cos',
                 'Zenith_sin', 'Zenith_cos',
                 'Azimuth_sin', 'Azimuth_cos']]


enc_df = df[['GHI', 'Month_sin',
             'Month_cos', 'Hour_sin',
             'Hour_cos', 'Year_sin',
             'Year_cos', 'Day_sin',
             'Day_cos']]


# %% MODEL AND TRAINING PARAMETERS


n_features = len(df.columns)
target = 'GHI'
weather_targets = ['Tamb', "Cloudopacity", "DewPoint", "Pw", "Pressure", "WindVel"]
learning_rate = 0.001
dropout = 0.4
enc_features = len(enc_df.columns)
weather_features = len(weather_df.columns)

encoder_params = {
    'input_size': enc_features,
    'hs_1': 24,
    'hs_2': 12,
    'output_size': 1,
    'dropout': dropout,
}
decoder_params = {
    'input_size': encoder_params['output_size'],
    'hs_1': 32,
    'hs_2': 48,
    'output_size': 12,
    'dropout': dropout
}

weather_params = {
    'input_size': weather_features,
    'hs_1': 32,
    'hs_2': 48,
    'output_size': horizon
}

# %% MODEL INSTANTIATION

data_params = DataParameters(memory=memory, horizon=horizon, batch_size=batch, target=target)

weather_data_params = [DataParameters(memory=memory,
                                      horizon=horizon,
                                      batch_size=batch,
                                      target=weather_targets[i]) for i in range(len(weather_targets))]

model = LSMTEncoderDecoder1(encoder_params=encoder_params,
                            decoder_params=decoder_params,
                            memory=memory,
                            fc_hidden_size=128,
                            output_size=horizon)

fc_was = 624

weather_model = SimpleLSTM(**weather_params)
weather_optimizer = torch.optim.Adam(params=weather_model.parameters(), lr=learning_rate)
weather_lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(weather_optimizer, T_0=10, T_mult=2)
model_optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
model_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(model_optimizer, T_0=10, T_mult=2)
weather_training_params = TrainingParameters(epochs=2,
                                             loss_function=F.mse_loss,
                                             optimiser=weather_optimizer)


training_params = TrainingParameters(epochs=2,
                                     loss_function=F.mse_loss,
                                     optimiser=model_optimizer)

encoder_decoder_pipeline = Pipeline(data=enc_df,
                                    model=model,
                                    data_params=data_params,
                                    training_params=training_params)

weather_pipelines = [Pipeline(data=weather_df,
                              model=weather_model,
                              data_params=weather_data_params[i],
                              training_params=weather_training_params,
                              target=weather_data_params[i].target) for i in range(len(weather_data_params))]

weather_models = [i.train(plot=True, scheduler2=weather_lr_scheduler2) for i in weather_pipelines]
weather_models = [weather_models[i][0] for i in range(len(weather_models))]

trained_model, losses = encoder_decoder_pipeline.train(plot=False, scheduler2=model_scheduler2)

tcn_model = TCNEncoderDecoderModel(TCNEncoder(encoder=trained_model,
                                              encoder_univariate_features=enc_features,
                                              sequence_length=memory,
                                              encoder_output_size=memory,
                                              past_covariates_size=len(df.columns)-1,
                                              kernel_size=3,
                                              dropout=0.1,
                                              output_size=memory),
                                   TCNDecoder(encoder_output_size=memory,
                                              horizon=horizon,
                                              kernel_size=4,
                                              sequence_length=memory,
                                              dropout=0.1),
                                   weather_predictor=WeatherPredictor(*weather_models, as_feats=False))


tcn_model2 = TCNEncoderDecoderModel(PastCovariatesEncoder(encoder=trained_model,
                                              input_size=memory,
                                              hidden_size=24,
                                              encoder_out_feat=14,
                                              output_size=12,
                                              univar_features=enc_features
                                              ),
                                   FutureCovariatesDecoder(input_size=12,
                                                           hidden_size=24,
                                                           output_size=29,
                                                           weather_features=6),
                                    weather_predictor=WeatherPredictor(*weather_models, as_feats=False))



tcn_optimizer = torch.optim.Adam(params=tcn_model.parameters(), lr=learning_rate)

tcn_training_params = TrainingParameters(epochs=100, loss_function=F.mse_loss, optimiser=tcn_optimizer)

tcn_data_params = DataParameters(24, 5, 128, "GHI")

tcn_pipeline = Pipeline(df, tcn_model2, data_params=tcn_data_params, training_params=tcn_training_params, target="GHI")

model, losses = tcn_pipeline.train(True)