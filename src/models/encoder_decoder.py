import torch.nn as nn
from torch import device, cuda, optim, tensor
from numpy import stack, mean
import matplotlib.pyplot as plt
from pandas import read_pickle, Series
from torch.utils.data import Dataset, DataLoader
from src.Datahandler.scaling import scale_train

# %% Torch Dataset
class GHIDataset(Dataset):
    def __init__(self, data):
        self.X = tensor(data.values).float()

    def __len__(self):
        return self.X.shape[0] - 24

    def __getitem__(self, i):
        return self.X[i:i + 24]



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.lstm1 = nn.LSTM(self.input_size, self.latent_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        lstm_out, (hidden_state, cell_state) = self.lstm1(x)
        #last_lstm_layer_hidden_state = hidden_state[-1, :]
        return hidden_state


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, input_size):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(self.latent_dim, self.input_size, num_layers=1, batch_first=True)

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        # x = x.reshape((-1, self.seq_len, self.hidden_size))

        return lstm_out


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_size, hidden_size, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_size, input_size)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


input_size = 24
hidden_size = 20
latent_dim = 16

autoencoder = AutoEncoder(input_size, hidden_size, latent_dim)

device = device("cuda:0" if cuda.is_available() else "cpu")
loss_function = nn.L1Loss()


def train(autoencoder, data, epochs=20):
    opt = optim.Adam(autoencoder.parameters())
    loss_1 = []
    for epoch in range(epochs):
        for x in data:
            x = x.to(device)  # GPU
            x_hat = autoencoder(x)
            loss = loss_function(x, x_hat)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_1.append(loss.cpu().detach().numpy())
        print(mean(stack(loss_1)))
        if epoch % 5 == 0:
            plt.plot(x.squeeze())
            plt.plot(x_hat.squeeze().cpu().detach().numpy())
            plt.show()
    return autoencoder


path = '../../data/raw/irradiance_data_NL_2007_2022.pkl'
df = read_pickle(path)[:1000]
GHI = df[['GHI']]
GHI = scale_train(GHI)
GHI = GHI['GHI']
GHI = DataLoader(GHIDataset(GHI))




ae = train(autoencoder, GHI, epochs=20)
