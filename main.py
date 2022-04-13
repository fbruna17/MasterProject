import warnings
import pandas as pd
import torch.utils.data

from pipeline import TrainingParameters, DataParameters, Pipeline
from src.Datahandler.SequenceDataset import make_torch_dataset
from src.features import build_features as bf
from src.helpers import plot_losses
from src.models.architecture import *

warnings.simplefilter(action="ignore")

train_weather = False
train_encoder = False
train_pred = True
train_simple_pred = False

path = 'data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)[:30000]
df = df.drop(columns="Minute")
df = bf.build_features(df)
# column_order = ['GHI', 'Month_sin',
#                 'Month_cos', 'Hour_sin',
#                 'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos', 'Tamb', 'Cloudopacity', 'DewPoint', 'DHI',
#                 'DNI',
#                 'EBH', 'Pw', 'Pressure', 'WindVel', 'AlbedoDaily', 'WindDir_sin',
#                 'WindDir_cos',
#                 'Zenith_sin', 'Zenith_cos', 'Azimuth_sin', 'Azimuth_cos',
#                 'Azimuth_sin t+1', 'Azimuth_sin t+2', 'Azimuth_sin t+3',
#                 'Azimuth_cos t+1', 'Azimuth_cos t+2', 'Azimuth_cos t+3',
#                 'Zenith_sin t+1', 'Zenith_sin t+2', 'Zenith_sin t+3',
#                 'Zenith_cos t+1', 'Zenith_cos t+2', 'Zenith_cos t+3'
#                 ]
#
# df = df[column_order]
memory = 24
horizon = 5
batch = 1440 * 2

df.insert(0, 'GHI', df.pop('GHI'))
weather_df = df[['Month_sin',
                 'Month_cos', 'Hour_sin',
                 'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos', 'Tamb', 'Cloudopacity', 'DewPoint', 'DHI',
                 'DNI',
                 'EBH', 'Pw', 'Pressure', 'WindVel', 'AlbedoDaily', 'WindDir_sin',
                 'WindDir_cos', 'Zenith_sin', 'Zenith_cos', 'Azimuth_sin', 'Azimuth_cos']]

# %% TRAIN, VALIDATION, TEST SPLIT
enc_df = df[['GHI', 'Month_sin',
             'Month_cos', 'Hour_sin',
             'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos']]

# %% MODEL AND TRAINING PARAMETERS
n_features = len(df.columns)
target = 'GHI'
weather_targets = ['Tamb', "Cloudopacity", "DewPoint", "Pw", "Pressure", "WindVel"]
weather_target = 'Tamb'
learning_rate = 0.001
dropout = 0.1
enc_features = len(enc_df.columns)
weather_features = len(weather_df.columns)

encoder_params = {
    'input_size': enc_features,
    'hs_1': enc_features * 3,
    'hs_2': enc_features * 4,
    'output_size': horizon,
    'dropout': dropout,
}
decoder_params = {
    'input_size': encoder_params['output_size'],
    'hs_1': 32,
    'hs_2': 48,
    'output_size': horizon,
    'dropout': dropout
}

weather_encoder_params = {
    'input_size': weather_features,
    'hs_1': weather_features*3,
    'hs_2': weather_features*4,
    'output_size': horizon,
    'dropout': dropout,
}
weather_decoder_params = {
    'input_size': 64,
    'hs_1': 42,
    'hs_2': 54,
    'output_size': 42,
    'dropout': dropout
}

data_params = DataParameters(memory=memory, horizon=horizon, batch_size=batch, target=target)

# %% LOAD DATA
if train_weather:
        # weather_data_param = DataParameters(memory=memory,
        #                                     horizon=horizon,
        #                                     batch_size=batch,
        #                                     target=weather_target)
        #
        # weather_model = WeatherEncoderDecoder(encoder_params=weather_encoder_params,
        #                                       decoder_params=weather_decoder_params,
        #                                       memory=memory,
        #                                       output_size=horizon)
        # weather_optimizer = torch.optim.Adam(params=weather_model.parameters(), lr=learning_rate)
        #
        # weather_training_param = TrainingParameters(epochs=50,
        #                                             loss_function=F.mse_loss,
        #                                             optimiser=weather_optimizer)
        # weather_pipeline = Pipeline(data=weather_df,
        #                             model=weather_model,
        #                             data_params=weather_data_param,
        #                             training_params=weather_training_param,
        #                             target=weather_target)
        # trained_model = weather_pipeline.train(plot=True)

        weather_data_params = [DataParameters(memory=memory,
                                              horizon=horizon,
                                              batch_size=batch,
                                              target=weather_targets[i]
                                              )
                               for i in range(len(weather_targets))
                               ]
        weather_model = WeatherEncoderDecoder(encoder_params=weather_encoder_params,
                                              decoder_params=weather_decoder_params,
                                              memory=memory,
                                              output_size=horizon
                                              )

        weather_optimizers = [torch.optim.Adam(params=weather_model.parameters(), lr=learning_rate)
                              for i in range(len(weather_targets))]

        weather_training_params = [TrainingParameters(epochs=75,
                                                      loss_function=F.l1_loss,
                                                      optimiser=weather_optimizers[i]
                                                      ) for i in range(len(weather_optimizers))]
        weather_pipelines = [Pipeline(data=weather_df,
                                      model=weather_model,
                                      data_params=weather_data_params[i],
                                      training_params=weather_training_params[i],
                                      target=weather_data_params[i].target
                                      )
                             for i in range(len(weather_data_params))
                             ]
        [i.train(plot=True) for i in weather_pipelines]
        [i.save(f"weather_model_{i.target}") for i in weather_pipelines]

if train_encoder:
    # %% BUILD FEATURES

    # %% SCALING and TORCH DATALOADER

    # %% MODEL INSTANTIATION

    model = WeatherEncoderDecoder(encoder_params=encoder_params,
                                decoder_params=decoder_params,
                                memory=memory,
                                output_size=horizon
                                )

    model_optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(model_optimizer, T_0=10, T_mult=2)

    training_params = TrainingParameters(epochs=50,
                                         loss_function=F.l1_loss,
                                         optimiser=model_optimizer
                                         )

    encoder_decoder_pipeline = Pipeline(data=enc_df,
                                        model=model,
                                        data_params=data_params,
                                        training_params=training_params,
                                        target="GHI"
                                        )

    trained_model, losses = encoder_decoder_pipeline.train(plot=True, scheduler2=model_scheduler2)
    encoder_decoder_pipeline.save('enc_dec_pretrained.pkl')

if train_pred:
    pred_params = {
        'input_size': 15576,
        'hs_1': 500,
        'hs_2': 250,
        'output': horizon
    }
    prefix = 'weather_model_'
    weather_nets = [torch.load(prefix + 'Tamb'),
                    torch.load(prefix + 'Cloudopacity'),
                    torch.load(prefix + 'DewPoint'),
                    torch.load(prefix + 'Pw'),
                    torch.load(prefix + 'Pressure'),
                    torch.load(prefix + 'WindVel')]

    weather_pred = WeatherPredictor(*weather_nets)

    pretrained = torch.load('enc_dec_pretrained.pkl')

    pred_net = PredNet(encoder=pretrained.encoder, params=pred_params, dropout=0.1, n_univariate_featues=enc_features,
                       weather_pred=weather_pred)

    optimizer_pred = torch.optim.Adam(params=pred_net.parameters(), lr=learning_rate)
    # scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.9)
    training_params = TrainingParameters(epochs=50,
                                         loss_function=F.l1_loss,
                                         optimiser=optimizer_pred)

    pred_pipeline = Pipeline(data=df, model=pred_net, data_params=data_params, training_params=training_params,
                             target="GHI")
    mdl, losses = pred_pipeline.train(plot=True)

if train_simple_pred:
    simple_pred_params = {
        'input_size': 888,
        'hs_1': 444,
        'hs_2': 222,
        'output': horizon
    }
    simple_pred_net = SimplePredNet(dropout=dropout, params=simple_pred_params)
    optimizer_simple_pred_net = torch.optim.Adam(params=simple_pred_net.parameters(), lr=learning_rate)
    training_params = TrainingParameters(epochs=50,
                                         loss_function=F.mse_loss,
                                         optimiser=optimizer_simple_pred_net)
    simple_pred_net_pipeline = Pipeline(data=df, model=simple_pred_net, data_params=data_params,
                                        training_params=training_params, target="GHI")
    mdl, losses = simple_pred_net_pipeline.train(plot=True)

# plot_losses(losses)
# # %% TEST
#
# test_results = test_model(trained_model, test, transformer)
#
# plot_predictions(test_results, 100, horizon)

# prefix = 'src/models/weather_model_'
# weather_nets = [torch.load(prefix + 'Tamb'),
#                 torch.load(prefix + 'Cloudopacity'),
#                  torch.load(prefix + 'DewPoint'),
#                  torch.load(prefix + 'Pw'),
#                  torch.load(prefix + 'Pressure'),
#                  torch.load(prefix + 'WindVel')]
#
# weather_df = df[['Month_sin',
#                 'Month_cos', 'Hour_sin',
#                 'Hour_cos', 'Year_sin', 'Year_cos', 'Day_sin', 'Day_cos', 'Tamb', 'Cloudopacity', 'DewPoint', 'DHI',
#                 'DNI',
#                 'EBH', 'Pw', 'Pressure', 'WindVel', 'AlbedoDaily', 'WindDir_sin',
#                 'WindDir_cos', 'Zenith_sin', 'Zenith_cos', 'Azimuth_sin', 'Azimuth_cos']]
#
# weather_pred = WeatherPredictor(*weather_nets)
# #
# weather_feat_params = DataParameters(memory=memory, horizon=horizon, batch_size=1, target="GHI")
# #
# weather_pred_pipe = WeatherPipeline(data=weather_df,
#                                     model=weather_pred,
#                                     data_params=weather_feat_params,
#                                     horizon=horizon)
# #
# dff = weather_pred_pipe.make_preprocessed_dataset()
# df = pd.concat([dff, df[memory:]], axis=1)
# weather_model = SimpleLSTM(**weather_params)
# weather_optimizer = torch.optim.Adam(params=weather_model.parameters(), lr=learning_rate)
# weather_lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(weather_optimizer, T_0=10, T_mult=2)

# weather_data_params = [DataParameters(memory=memory,
#                                           horizon=horizon,
#                                           batch_size=batch,
#                                           target=weather_targets[i]
#                                           )
#                            for i in range(len(weather_targets))
#                            ]
# weather_training_params = TrainingParameters(epochs=100,
#                                              loss_function=F.mse_loss,
#                                              optimiser=weather_optimizer
#                                              )
# weather_pipelines = [Pipeline(data=weather_df,
#                               model=weather_model,
#                               data_params=weather_data_params[i],
#                               training_params=weather_training_params,
#                               target=weather_data_params[i].target
#                               )
#                      for i in range(len(weather_data_params))
#                      ]
