import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from src.Datahandler.SequenceDataset import make_torch_dataset
from src.Datahandler.prepare_data import split_data
from src.Datahandler.scaling import scale_data


class DataParameters:
    def __init__(self, memory, horizon, batch_size, target):
        self.memory = memory
        self.horizon = horizon
        self.batch_size = batch_size
        self.target = target


class TrainingParameters:
    def __init__(self, epochs, loss_function, optimiser):
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimiser = optimiser


class Pipeline:
    def __init__(self, data: pd.DataFrame, model, data_params: DataParameters,
                 training_params: TrainingParameters):
        self.data = data
        if type(model) == str:
            self.model = torch.load(model)
        else:
            self.model = model
        self.memory = data_params.memory
        self.horizon = data_params.horizon
        self.batch_size = data_params.batch_size
        self.target = data_params.target
        self.epochs = training_params.epochs
        self.optimiser = training_params.optimiser
        self.loss_function = training_params.loss_function

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.transformer = None

        self.prepare_data()

    def prepare_data(self):
        split = split_data(self.data)
        train, val, test, self.transformer = scale_data(*split)
        self.train_data, self.val_data, self.test_data = make_torch_dataset(train, val, test,
                                                                            self.memory,
                                                                            self.horizon,
                                                                            self.batch_size,
                                                                            self.target)

    def train(self, plot=False):
        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimiser = self.optimiser

        loss_fn = self.loss_function
        losses = {'train': [], 'validation': []}

        epochs = tqdm(range(1, self.epochs + 1))

        for epoch in epochs:
            model.train()
            epoch_loss = []
            for i, (x, y) in enumerate(self.train_data):
                x, y = x.to(device), y.to(device)
                out = model(x)

                optimiser.zero_grad()
                loss = torch.sqrt(loss_fn(out, y))
                loss.backward()
                optimiser.step()

                epoch_loss.append(loss.item())

            losses['train'].append(np.mean(epoch_loss))

            # VALIDATION
            model.eval()
            val_loss = []
            for x, y in self.val_data:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss.append(loss.item())

            losses['validation'].append(np.mean(val_loss))

            epochs.set_description(
                f"Epoch {epoch} of {len(epochs)} \t | \t Training Loss: {round(losses['train'][-1], 4)} \t "
                f"Validation Loss: {round(losses['validation'][-1], 4)} \t"
                f"Progress: {0 if len(losses['train']) < 2 else - round(losses['train'][-2] - losses['train'][-1], 4)}")

            if plot:
                if epoch % 5 == 0:
                    plt.plot(y[::self.horizon].flatten()[:100])
                    plt.plot(out[::self.horizon].flatten()[:100].detach())
                    plt.show()
        self.model = model
        return model, losses

    def test(self):
        model = self.model
        model.eval()
        data_table = {'True': [], 'Predictions': [], 'Loss': []}
        loss_function = self.loss_function
        with torch.no_grad():
            for x, y in self.test_data:
                y_hat = model(x)

                y, y_hat = self.transformer.inverse_transform_target(y, y_hat)
                error = torch.sqrt(loss_function(y_hat, y))

                data_table['True'].append(y.squeeze().tolist())
                data_table['Predictions'].append(y_hat.detach().squeeze().tolist())
                data_table['Loss'].append(error.item())

        return data_table

    def save(self, filepath):
        torch.save(self.model, filepath)

    def mc_dropout(self, n_samples, dropout=True):
        model = self.model
        model.train() if dropout else model.eval()

        device = "cpu"

        y_mus = []
        y_etas = []
        ys = []
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(self.test_data)):
                x, y = x.to(device), y.to(device)
                y_hat_n = []
                for n in range(n_samples):
                    y_hat = model(x)
                    y_hat_n.append(y_hat.numpy())

                n_mu = np.mean(y_hat_n, axis=0)
                n_eta = np.var(y_hat_n, axis=0)

                y_mus.append(n_mu)
                y_etas.append(n_eta)
                ys.append(y)



        return y_mus, y_etas, ys





