from matplotlib import pyplot as plt
from torch import sqrt, mean, square, abs
from itertools import product
from numpy import sqrt as numpy_sqrt


class RMSE:
    def __call__(self, y_hat, y):
        return sqrt(mean(square((y - y_hat))))


def rmse(y, y_hat):
    # Loss function definition
    loss = sqrt(mean(square((y - y_hat))))
    return loss


def cartesian_product(configurations):
    return list(dict(zip(configurations, x)) for x in product(*configurations.values()))


def mae(y, y_hat):
    loss = mean(abs((y - y_hat)))
    return loss


def mape(y, y_hat):
    loss = mean(abs((y - y_hat) / y)).item() * 100
    return loss


def train():
    ...


def flatten(t):
    return [item for sublist in t for item in sublist]


def plot_losses(losses):
    plt.plot(losses['train'], label='Training')
    plt.plot(losses['validation'], label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Loss evolution')
    plt.legend()
    plt.show()


def plot_predictions(test_results, number_of_timesteps_to_plot, horizon):
    rmse = numpy_sqrt(sum(test_results['Loss']) / len(test_results['Loss']))
    preds = flatten(test_results['Predictions'][::horizon])[:number_of_timesteps_to_plot]
    true = flatten(test_results['True'][::horizon])[:number_of_timesteps_to_plot]
    plt.plot(true, label='True labels')
    plt.plot(preds, label='Predictions')
    plt.title(f'RMSE {round(rmse, 3)}')
    plt.xlabel('Timesteps')
    plt.ylabel('GHI')
    plt.legend()
    plt.show()


