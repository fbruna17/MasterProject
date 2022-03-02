import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)


class Logger:
    def __init__(self):
        self.loss_evolution = pd.DataFrame(columns=['Epoch', 'Training', 'Validation', 'Loss Function'])
        self.test_error = -1

    def append_loss(self, epoch, training_loss: float, validation_loss: float, loss_function: str):
        entry = pd.Series([epoch, training_loss, validation_loss, loss_function], index=self.loss_evolution.columns)
        self.loss_evolution = self.loss_evolution.append(entry, ignore_index=True)

    def plot_losses(self):
        self.loss_evolution[['Training', 'Validation']].plot()
        plt.xlabel('Epochs')
        plt.ylabel(self.loss_evolution['Loss Function'].iloc[0])
        plt.show()
