from torch import no_grad
from torch.nn import functional as F


def test_model(model, dataloader, transformer, loss_function=F.mse_loss):
    model.eval()
    data_table = {'True': [], 'Predictions': [], 'Loss': []}
    with no_grad():
        for x, y in dataloader:
            y_hat = model(x)

            y, y_hat = transformer.inverse_transform_target(y, y_hat)
            error = loss_function(y_hat, y)

            data_table['True'].append(y.squeeze().tolist())
            data_table['Predictions'].append(y_hat.detach().squeeze().tolist())
            data_table['Loss'].append(error.item())

    return data_table



