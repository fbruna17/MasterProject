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
    def __init__(self, data: pd.DataFrame, model: nn.Module, data_params: DataParameters,
                 training_params: TrainingParameters, target: str = "GHI"):
        self.data = data
        self.model = model

        self.data_params = data_params
        self.training_params = training_params

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.transformer = None

        self.target = target
        self.prepare_data()

    def prepare_data(self):
        split = split_data(self.data)
        train, val, test, self.transformer = scale_data(*split, self.target)
        self.train_data, self.val_data, self.test_data = make_torch_dataset(train, val, test,
                                                                            self.data_params.memory,
                                                                            self.data_params.horizon,
                                                                            self.data_params.batch_size,
                                                                            self.data_params.target)

    def train(self, plot=False, scheduler1=None, scheduler2=None):
        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimiser = self.training_params.optimiser

        loss_fn = self.training_params.loss_function
        losses = {'train': [], 'validation': []}

        epochs = tqdm(range(1, self.training_params.epochs + 1))

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
            if scheduler1:
                scheduler1.step()
            if scheduler2:
                scheduler2.step()

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
                f"Progress: {0 if len(losses['train'])<2 else - round(losses['train'][-2] - losses['train'][-1], 4)}")

            if plot:
                if epoch % 5 == 0:
                    plt.plot(y[::self.data_params.horizon].flatten()[:100])
                    plt.plot(out[::self.data_params.horizon].flatten()[:100].detach())
                    plt.show()
        self.model = model
        return model, losses

    def save(self, filepath):
        torch.save(self.model, filepath)


class WeatherPipeline:
    def __init__(self, data, model, data_params, horizon):
        self.data = data
        self.index = data.index[data_params.memory:]
        self.features = data.columns
        self.model = model
        self.data_params = data_params
        self.horizon = horizon
        self.transformer = None

        self.prepare_data()

    def prepare_data(self):
        self.transformer = MinMaxScaler()
        self.data = self.transformer.fit_transform(self.data)
        self.weather_scales = {k:v for k, v in zip(
            ("Tamb", "Cloudopacity", "DewPoint", "Pw", "Pressure", "WindVel"),
            ((self.transformer.data_min_[8], self.transformer.data_max_[8]),
            (self.transformer.data_min_[9], self.transformer.data_max_[9]),
            (self.transformer.data_min_[10], self.transformer.data_max_[10]),
            (self.transformer.data_min_[14], self.transformer.data_max_[14]),
            (self.transformer.data_min_[15], self.transformer.data_max_[15]),
            (self.transformer.data_min_[16], self.transformer.data_max_[16])
             ))}


        self.data = pd.DataFrame(data=self.data, columns=self.features)

        self.data = make_weather_dataset(data=self.data,
                                         memory=self.data_params.memory,
                                         batch=3000,
                                         drop_last=False)

    def make_preprocessed_dataset(self):
        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        features = ["Tamb", "Cloudopacity", "DewPoint", "Pw", "Pressure", "WindVel"]
        columns = [f"{f}_{i}" for f in features for i in range(1, self.horizon + 1)]
        _temp = np.zeros((0, len(columns)))
        for x in self.data:
            out = model(x)
            out = out.detach().numpy()
            out = np.hstack([
                inverse_transform_column_range(
                    array=out,
                    col_start=i,
                    col_end= i + self.data_params.horizon,
                    min_val=self.weather_scales[feature][0],
                    max_val=self.weather_scales[feature][1]
                ) for feature, i in zip(features, [0, 5, 9, 14, 19, 24])])

            _temp = np.vstack((_temp, out))
        df = pd.DataFrame(data=_temp, columns=columns, index=self.index)
        return df


def inverse_transform_column_range(array, col_start, col_end, min_val, max_val):
    return (array[:, col_start: col_end]*(max_val - min_val)) + min_val