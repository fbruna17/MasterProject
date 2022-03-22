from torch.optim import Adam
from src.helpers import rmse


def train_model(model, learning_rate, EPOCHS, train_dataset):
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train()

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
