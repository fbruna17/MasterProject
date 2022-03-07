import pandas as pd
import warnings
import matplotlib.pyplot as plt
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)


class Logger:
    def __init__(self):
        self.loss_evolution = pd.DataFrame(columns=['Epoch', 'Training', 'Validation', 'Loss Function'])
        self.test_df = pd.DataFrame(columns=['X', 'y', 'y_hat', 'Error', 'Loss Function'])
        self.test_error = -1

    def append_loss(self, epoch, training_loss: float, validation_loss: float, loss_function: str):
        entry = pd.Series([epoch, training_loss, validation_loss, loss_function], index=self.loss_evolution.columns)
        self.loss_evolution = self.loss_evolution.append(entry, ignore_index=True)

    def add_prediction(self, X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, error: float, loss_function: str):
        entry = pd.Series([X.squeeze().cpu().detach().numpy().tolist(),
                           y.squeeze().cpu().detach().numpy().tolist(),
                           y_hat.squeeze().cpu().detach().numpy().tolist(),
                           error.cpu().detach().numpy().item(),
                           loss_function], index=self.test_df.columns)
        self.test_df = self.test_df.append(entry, ignore_index=True)

    def plot_losses(self):
        self.loss_evolution[['Training', 'Validation']].plot()
        plt.xlabel('Epochs')
        plt.ylabel(self.loss_evolution['Loss Function'].iloc[0])
        plt.show()

    def plot_prediction(self, idx=0):
        if idx > len(self.test_df) - 1:
            print('Index out of bounds')
        else:
            row = self.test_df.iloc[idx]
            xx = range(len(row['y']))
            plt.plot(xx, row['y'], label='y')
            plt.plot(xx, row['y_hat'], label='y_hat')
            plt.xlabel('timestep')
            plt.ylabel('GHI')
            plt.legend()
            plt.grid()
            plt.show()
