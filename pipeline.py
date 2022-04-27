import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from tqdm import tqdm

from src.Datahandler.SequenceDataset import make_torch_dataset, make_weather_dataset
from src.Datahandler.prepare_data import split_data
from src.Datahandler.scaling import scale_data
from src.models.archs import WhateverNet2


class DataParameters:
    def __init__(self, memory, horizon, batch_size, target, sliding, input_size):
        self.memory = memory
        self.horizon = horizon
        self.batch_size = batch_size
        self.target = target
        self.sliding = sliding
        self.input_size = input_size


class TrainingParameters:
    def __init__(self, epochs, learning_rate, weight_decay=0.001, loss_function = None, scheduler: bool = False):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_function
        self.scheduler = scheduler


class Pipeline:
    def __init__(self, data: pd.DataFrame, data_params: DataParameters,
                 training_params: TrainingParameters, target, model_params: dict, likelihood=None):
        self.data = data
        self.model = None
        self.model_params = model_params
        self.data_params = data_params
        self.training_params = training_params
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.transformer = None
        self.likelihood = likelihood
        self.target = target
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_function = None
        self.prepare_data()
        self.create_model()
        self.create_optimizer()

    def prepare_data(self):
        split = split_data(self.data)
        train, val, test, self.transformer = scale_data(*split, self.target)
        train_data, val_data, test_data = make_torch_dataset(train, val, test,
                                                                            self.data_params.memory,
                                                                            self.data_params.horizon,
                                                                            self.data_params.batch_size,
                                                                            self.data_params.target,
                                                                            self.data_params.sliding)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train(self, plot=False):
        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimiser = self.optimizer

        loss_fn = self.loss_function
        losses = {'train': [], 'validation': []}

        epochs = tqdm(range(1, self.training_params.epochs + 1))
        model.train()
        for epoch in epochs:
            epoch_loss = []
            for i, (x, y) in enumerate(self.train_data):
                x, y = x.to(device), y.to(device)
                out = model(x)

                loss = self.likelihood.compute_loss(out.view(self.data_params.batch_size, self.data_params.memory, 1, -1),
                                                    y.view(self.data_params.batch_size, self.data_params.memory, -1)) if self.likelihood else loss_fn(out.squeeze(), y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss.append(loss.item())
            if self.lr_scheduler:
                self.lr_scheduler.step()

            losses['train'].append(np.mean(epoch_loss))


            # VALIDATION
            model.eval()
            val_loss = []
            for x, y in self.val_data:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = self.likelihood.compute_loss(
                    out.view(self.data_params.batch_size, self.data_params.memory, 1, -1),
                    y.view(self.data_params.batch_size, self.data_params.memory, -1)) if self.likelihood else loss_fn(
                    out.squeeze(), y)
                val_loss.append(loss.item())

            losses['validation'].append(np.mean(val_loss))

            epochs.set_description(
                f"Epoch {epoch} of {len(epochs)} \t | \t Training Loss: {round(losses['train'][-1], 4)} \t "
                f"Validation Loss: {round(losses['validation'][-1], 4)} \t"
                f"Progress: {0 if len(losses['train'])<2 else - round(losses['train'][-2] - losses['train'][-1], 4)}")

            if plot:
                if epoch % 2 == 0:
                    plt.plot(y.view(128, 24, 1)[::self.data_params.memory].flatten()[:100], label="Ground Truth")
                    plt.plot(out[:self.data_params.memory:3].flatten()[:100].detach(), label="Predictions")
                    plt.legend()
                    plt.xlabel("Timesteps")
                    plt.ylabel("Global Horizontal Irradiance")
                    plt.title("Solar Energy Forecast")
                    plt.suptitle(f"Model output after {epoch} epochs")
                    plt.show()
        self.model = model
        return model, losses

    def save(self, filepath):
        torch.save(self.model, filepath)

    def create_model(self):
        self.model = WhateverNet2(input_size=self.data_params.input_size,
                                  memory=self.data_params.memory,
                                  target_size=1,
                                  horizon=self.data_params.horizon,
                                  hidden_size=self.model_params["hidden_size"],
                                  nr_parameters= self.likelihood.num_parameters if self.likelihood else 1,
                                  bigru_layers=self.model_params["bigru_layers"],
                                  dropout=self.model_params["dropout"],
                                  attention_head_size=self.model_params["attention_head_size"],
                                  tcn_params=self.model_params["tcn_params"])

    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.training_params.learning_rate)
        self.loss_function = self.training_params.loss_function


    def create_scheduler(self):
        if self.training_params.scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9)

def inverse_transform_column_range(array, col_start, col_end, min_val, max_val):
    return (array[:, col_start: col_end]*(max_val - min_val)) + min_val


