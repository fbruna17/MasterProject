import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import tensor, zeros, float32


class BayesianLSTM(nn.Module):

    def __init__(self, n_features, output_length, batch_size):
        super(BayesianLSTM, self).__init__()

        self.batch_size = batch_size  # user-defined

        self.hidden_size_1 = 16  # number of encoder cells (from paper)
        self.hidden_size_2 = 8  # number of decoder cells (from paper)
        self.stacked_layers = 2  # number of (stacked) LSTM layers for each stage
        self.dropout_probability = 0.1  # arbitrary value (the paper suggests that performance is generally stable across all ranges)

        self.lstm1 = nn.LSTM(n_features,
                             self.hidden_size_1,
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)

        self.fc = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        output = output[:, -1, :]  # take the last decoder cell's outputs
        y_pred = self.fc(output)
        return y_pred

    def init_hidden1(self, batch_size):
        hidden_state = Variable(zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        return hidden_state, cell_state

    def init_hidden2(self, batch_size):
        hidden_state = Variable(zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        cell_state = Variable(zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        return hidden_state, cell_state

    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)

    def predict(self, X):
        return self(tensor(X, dtype=float32)).view(-1).detach().numpy()


class PredNet(nn.Module):
    def __init__(self, input_size, output_length, hidden_layer_1=128, hidden_layer_2=64, hidden_layer_3=16):
        # default hidden layer sizes according to Uber-article
        super(PredNet, self).__init__()
        self.input_size = input_size
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.hidden_layer_3 = hidden_layer_3
        self.output_length = output_length

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_layer_1),
            nn.Tanh(),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.Tanh(),
            nn.Linear(hidden_layer_2, hidden_layer_3),
            nn.Tanh(),
            nn.Linear(hidden_layer_3, output_length)
        )

    def forward(self, x):
        return self.network(x)
