import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from src.helpers import rmse


def train_model1(model, learning_rate, EPOCHS, train_dataset):
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train_data()

    for e in range(EPOCHS):
        error = 0
        for _x, _y in train_dataset:
            output = model(_x)
            loss = rmse(output, _y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            error += loss.detach()

        if e % 10 == 0:
            print('epoch', e, 'loss: ', error)
    return model


def train_model(model: torch.nn.Module, training_dataloader: DataLoader, training_params: dict,
          validation_dataloader: DataLoader = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimiser = torch.optim.Adam(lr=training_params['learning_rate'], params=model.parameters())
    loss_fn = training_params.get('loss_function', F.mse_loss)
    losses = {'train': [], 'validation': []}

    epochs = tqdm(range(1, training_params['epochs'] + 1))

    for epoch in epochs:
        model.train()
        epoch_loss = []
        for i, (x, y) in enumerate(training_dataloader):
            x, y = x.to(device), y.to(device)
            out = model(x)

            optimiser.zero_grad()
            loss = loss_fn(out, y)
            loss.backward()
            optimiser.step()

            epoch_loss.append(loss.item())

        losses['train'].append(np.mean(epoch_loss))

        # VALIDATION
        model.eval()
        val_loss = []
        for x, y in validation_dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            val_loss.append(loss.item())

        losses['validation'].append(np.mean(val_loss))

        epochs.set_description(
            f"Epoch {epoch} of {len(epochs)} \t | \t Training Loss: {losses['train'][-1].round(5)} \t "
            f"Validation Loss: {losses['validation'][-1].round(5)}")

        if epoch % 5 == 0:
            horizon = y.shape[1]
            plt.plot(y[::horizon].flatten())
            plt.plot(out[::horizon].flatten().detach())
            plt.show()

    return model, losses
