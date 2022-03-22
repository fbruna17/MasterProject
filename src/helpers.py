from torch import sqrt, mean, square, abs
from itertools import product


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
