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
df = pd.read_pickle(path)
df = df.drop(columns="Minute")
df = bf.build_features(df)
column_order = ['GHI', 'Month_sin',
                'Month_cos', 'Hour_sin',
                'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos', 'Tamb', 'Cloudopacity', 'DewPoint', 'DHI',
                'DNI',
                'EBH', 'Pw', 'Pressure', 'WindVel', 'AlbedoDaily', 'WindDir_sin',
                'WindDir_cos',
                'Zenith_sin', 'Zenith_cos', 'Azimuth_sin', 'Azimuth_cos',
                'Azimuth_sin t+1', 'Azimuth_sin t+2', 'Azimuth_sin t+3',
                'Azimuth_cos t+1', 'Azimuth_cos t+2', 'Azimuth_cos t+3',
                'Zenith_sin t+1', 'Zenith_sin t+2', 'Zenith_sin t+3',
                'Zenith_cos t+1', 'Zenith_cos t+2', 'Zenith_cos t+3'
                ]

df = df[column_order]
# %% BUILD FEATURES
memory = 24
horizon = 5
batch = 1024
prefix = 'src/models/weather_model_'
weather_nets = [torch.load(prefix + 'Tamb'),
                torch.load(prefix + 'Cloudopacity'),
                 torch.load(prefix + 'DewPoint'),
                 torch.load(prefix + 'Pw'),
                 torch.load(prefix + 'Pressure'),
                 torch.load(prefix + 'WindVel')]

weather_df = df[['Month_sin',
                'Month_cos', 'Hour_sin',
                'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos', 'Tamb', 'Cloudopacity', 'DewPoint', 'DHI',
                'DNI',
                'EBH', 'Pw', 'Pressure', 'WindVel', 'AlbedoDaily', 'WindDir_sin',
                'WindDir_cos', 'Zenith_sin', 'Zenith_cos', 'Azimuth_sin', 'Azimuth_cos']]

weather_pred = WeatherPredictor(*weather_nets)
#
weather_feat_params = DataParameters(memory=memory, horizon=horizon, batch_size=1, target="GHI")
#
weather_pred_pipe = WeatherPipeline(data=weather_df,
                                    model=weather_pred,
                                    data_params=weather_feat_params,
                                    horizon=horizon)
#
dff = weather_pred_pipe.make_preprocessed_dataset()
df = pd.concat([dff, df[memory:]], axis=1)


# %% TRAIN, VALIDATION, TEST SPLIT

enc_df = df[['GHI', 'Month_sin',
             'Month_cos', 'Hour_sin',
             'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos']]



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

# %% SCALING and TORCH DATALOADER

# %% MODEL INSTANTIATION

data_params = DataParameters(memory=memory, horizon=horizon, batch_size=batch, target=target)

weather_data_params = [DataParameters(memory=memory,
                                      horizon=horizon,
                                      batch_size=batch,
                                      target=weather_targets[i]
                                      )
                       for i in range(len(weather_targets))
                       ]

model = LSMTEncoderDecoder1(encoder_params=encoder_params,
                            decoder_params=decoder_params,
                            memory=memory,
                            fc_hidden_size=624,
                            output_size=horizon
                            )


weather_model = SimpleLSTM(**weather_params)

weather_optimizer = torch.optim.Adam(params=weather_model.parameters(), lr=learning_rate)
#weather_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(weather_optimizer, gamma=0.8)
weather_lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(weather_optimizer, T_0=10, T_mult=2)

model_optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
#model_scheduler = torch.optim.lr_scheduler.ExponentialLR(model_optimizer, gamma=0.8)
model_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(model_optimizer, T_0=10, T_mult=2)
weather_training_params = TrainingParameters(epochs=100,
                                            loss_function=F.mse_loss,
                                            optimiser=weather_optimizer
                                            )

training_params = TrainingParameters(epochs=100,
                                     loss_function=F.mse_loss,
                                     optimiser=model_optimizer
                                     )

encoder_decoder_pipeline = Pipeline(data=enc_df,
                                    model=model,
                                    data_params=data_params,
                                    training_params=training_params
                                    )

weather_pipelines = [Pipeline(data=weather_df,
                              model=weather_model,
                              data_params=weather_data_params[i],
                              training_params=weather_training_params,
                              target=weather_data_params[i].target
                              )
                     for i in range(len(weather_data_params))
                     ]

params = {
    'n_extracted_features': encoder_params['output_size'],
    'n_external_features': len(df.columns),
    'predict_hidden_1': 256,
    'predict_hidden_2': 128,
    'n_output_steps': horizon,
}



# pred_net = PredictionNet(params, dropout)
train_new_model = True

if train_new_model:
    trained_model, losses = encoder_decoder_pipeline.train(plot=True, scheduler2=model_scheduler2)
    [i.train(plot=True, scheduler2=weather_lr_scheduler2) for i in weather_pipelines]
    encoder_decoder_pipeline.save('enc_dec_pretrained.pkl')
    [i.save(f"weather_model_{i.target}") for i in weather_pipelines]

pred_params = {
    'input_size': (encoder_params['output_size'] + n_features) * memory,
    'hs_1': 256,
    'hs_2': 64,
    'output': horizon
}



pretrained = torch.load('src/models/enc_dec_pretrained.pkl')


pred_net = PredNet(encoder=pretrained.encoder, params=pred_params, dropout=0.1, n_univariate_featues=enc_features)

optimizer = torch.optim.Adam(params=pred_net.parameters(), lr=learning_rate)
scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.9)
training_params = TrainingParameters(epochs=150,
                                     loss_function=F.mse_loss,
                                     optimiser=optimizer)

pred_pipeline = Pipeline(data=df, model=pred_net, data_params=data_params, training_params=training_params)
mdl, losses = pred_pipeline.train(plot=True, scheduler1=scheduler1, scheduler2=scheduler2)

plot_losses(losses)
# # %% TEST
#
# test_results = test_model(trained_model, test, transformer)
#
# plot_predictions(test_results, 100, horizon)
