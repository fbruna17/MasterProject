import torch.nn as nn


class SimpleLSTM:
    def __init__(self):
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=16,
                            num_layers=1,
                            batch_first=True)
        self.linear = nn.Linear(16, 1)

    def forward(self, x):
        lstm_out, other = self.lstm(x)  # x shape: (batch, memory, n_features = 3),
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred
