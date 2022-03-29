import pandas as pd
from numpy import stack, mean
from src.Datahandler.scaling import scale_train
from src.models.architecture import LSTMEncoderDecoder
from src.Datahandler.SequenceDataset import make_torch_dataset, SubSequenceDataset
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt

path = '../../data/raw/irradiance_data_NL_2007_2022.pkl'
df = pd.read_pickle(path)[:20000]

columns = ['GHI']
df = df[columns]
memory = 32
horizon = 4
encoder_output = 4
hidden_size = 12
encoder_input = memory
num_layer = 1
batch = 64
target = "GHI"
learning_rate = 0.005
EPOCHS = 30
train_new_model = False
decoder_input = horizon + encoder_output

df = df[columns]
df = scale_train(df)
data = DataLoader(SubSequenceDataset(dataframe=df, memory=memory, horizon=horizon, features=columns, target=target),
                  batch_size=batch, drop_last=True)

model = LSTMEncoderDecoder(
    encoder_input=encoder_input,
    encoder_output=encoder_output,
    hidden_size=hidden_size,
    decoder_input=decoder_input,
    num_layers=num_layer,
    horizon=horizon)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

model.train()
for epoch in range(EPOCHS):
    loss_1 = []
    for x1, x2, y in data:
        y_hat = model(x1.squeeze(), x2.squeeze())
        optimizer.zero_grad()
        loss = torch.sqrt(mse_loss(y_hat, y))
        loss.backward()
        optimizer.step()
        loss_1.append(loss.cpu().detach().numpy())
    print(mean(stack(loss_1)))
    if epoch % 5 == 0:
        plt.plot(y[::4].flatten())
        plt.plot(y_hat[::4].flatten().detach())
        plt.show()
