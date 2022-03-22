from torch import device as torch_device
from torch import no_grad
from src.helpers import rmse
from numpy import array
from src.logger.logger import Logger


def test_model(model, test_dataset, transformer):
    model.eval()
    device = torch_device('cpu')
    ys, y_hats = [], []
    temp_losses = []
    loss_function = rmse
    log = Logger()

    with no_grad():
        for x, y in test_dataset:
            x = x.to(device).float()
            y = y.to(device).float()

            # MAKE PREDICTION
            y_hat = model(x)

            # INVERSE TRANSFORM
            y, y_hat = transformer.inverse_transform_target(y, y_hat)

            # Calculate error
            error = loss_function(y, y_hat)
            temp_losses.append(error)
            log.add_prediction(X=x, y=y, y_hat=y_hat, error=error, loss_function=str(loss_function))

    print(f"Test error: {log.test_df['Error'].mean()}")
    log.plot_prediction()

    return y, y_hat, error
