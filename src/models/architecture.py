import torch
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


class LSTMEncoder(nn.Module):
    def __init__(self, encoder_input, hidden_size, encoder_output, num_layers):
        super(LSTMEncoder, self).__init__()
        self.encoder_input = encoder_input
        self.hidden_size = hidden_size
        self.encoder_output = encoder_output
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(self.encoder_input, self.hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size, self.encoder_output, batch_first=True)

    def forward(self, x):
        out, (hs, cs) = self.lstm1(x)
        out, _ = self.lstm2(out)
        return out, (hs, cs)


class LSTMDecoder(nn.Module):
    def __init__(self, encoder_input, decoder_input, encoder_output, num_layers, horizon):
        super(LSTMDecoder, self).__init__()
        self.encoder_input = encoder_input
        self.encoder_output = encoder_output
        self.decoder_input = decoder_input
        self.horizon = horizon
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.decoder_input, self.encoder_output, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.encoder_output, self.horizon)

    def forward(self, x2, out):
        x_cat = torch.cat([x2, out], dim=1)
        lstm_out, _ = self.lstm(x_cat)
        out = self.linear(lstm_out)
        return out


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, encoder_input, encoder_output, decoder_input, hidden_size, num_layers, horizon=0):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder_input = encoder_input
        self.encoder_output = encoder_output
        self.decoder_input = decoder_input
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.encoder = LSTMEncoder(
            encoder_input=self.encoder_input,
            hidden_size=self.hidden_size,
            encoder_output=self.encoder_output,
            num_layers=self.num_layers
        )
        self.decoder = LSTMDecoder(
            encoder_input=self.encoder_input,
            decoder_input=self.decoder_input,
            encoder_output=self.encoder_output,
            num_layers=self.num_layers,
            horizon=self.horizon)

    def forward(self, x1, x2):
        out, (hs, cs) = self.encoder(x1)
        decoder_out = self.decoder(x2, out)
        return decoder_out
