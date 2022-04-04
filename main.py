import warnings
import pandas as pd
from src.Datahandler.scaling import scale_data
from src.Datahandler.prepare_data import split_data
from src.Datahandler.SequenceDataset import make_torch_dataset
from src.models.architecture import *
from src.models.train_model import train_model
from src.models.test_model import test_model
from src.helpers import plot_losses, plot_predictions
from src.features import build_features as bf


warnings.simplefilter(action="ignore")
# %% LOAD DATA

path = 'data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)[:5000]

# %% BUILD FEATURES
df = bf.build_features(df)


# %% TRAIN, VALIDATION, TEST SPLIT
train, val, test = split_data(df)

# %% MODEL AND TRAINING PARAMETERS

memory = 24
horizon = 5
batch = 128
n_features = len(train.columns)
target = 'GHI'
learning_rate = 0.001
dropout = 0.

encoder_params = {
    'input_size': n_features,
    'hs_1': 32,
    'hs_2': 8,
    'output_size': 6,
    'dropout': dropout,
}
decoder_params = {
    'input_size': encoder_params['output_size'],
    'hs_1': 12,
    'hs_2': 6,
    'output_size': 4,
    'dropout': dropout
}

# %% SCALING and TORCH DATALOADER
train, val, test, transformer = scale_data(train, val, test)

train, val, test = make_torch_dataset(train, val, test, memory=memory, horizon=horizon, batch=batch, target=target)

# %% MODEL INSTANTIATION

model = LSMTEncoderDecoder1(encoder_params=encoder_params,
                            decoder_params=decoder_params,
                            fc_hidden_size=64,
                            output_size=horizon,
                            memory=memory)

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

training_params = {'epochs': 50,
                   'learning_rate': learning_rate,
                   'loss_function': F.mse_loss
                   }

# %% TRAIN

trained_model, losses = train_model(model=model,
                                    training_dataloader=train,
                                    training_params=training_params,
                                    validation_dataloader=val)
plot_losses(losses)

# %% TEST

test_results = test_model(trained_model, test, transformer)

plot_predictions(test_results, 100, horizon)
