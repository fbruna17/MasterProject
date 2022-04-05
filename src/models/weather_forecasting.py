import torch.nn as nn
import torch.nn.functional as F
import torch
# %%
class WeatherForecasting(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(WeatherForecasting, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.lstm1 = nn.LSTM(input_size, hidden_size1)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2)
        self.linear1 = nn.Linear(hidden_size2 * 24, output_size)
        self.linear2 = nn.Linear(hidden_size2, output_size)




    # make a forward pass through the network and return the output
    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        y_hat = self.linear1(lstm_out.flatten(start_dim=1))
        # y_hat = self.linear2(y_hat)
        return y_hat



# %%

class LSTNet(nn.Module):
    def __init__(self,
                 features=30,
                 hidden_size_CNN=100,
                 memory=24,
                 hidden_size_RNN=48,
                 hidden_size_skip=6,
                 kernel_size=6,
                 skip=24,
                 highway_window=24,
                 dropout_p=0.2,
                 output_fun="tanh"):
        super(LSTNet, self).__init__()
        self.memory = memory
        self.n_features = features
        self.hidR = hidden_size_RNN
        self.hidC = hidden_size_CNN
        self.hidS = hidden_size_skip
        self.kernel_size = kernel_size
        self.skip = skip
        self.pt = (self.memory - self.kernel_size) / self.skip
        self.hw = highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.kernel_size, self.n_features))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=dropout_p)
        self.output_fun = output_fun
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.n_features)
        else:
            self.linear1 = nn.Linear(self.hidR, self.n_features)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (self.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (self.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.memory, self.n_features)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.memory)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res
