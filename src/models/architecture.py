from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import tensor, zeros, float32
from torch.nn.utils.rnn import PackedSequence


class LSTMEncoder(nn.Module):
    def __init__(self, encoder_input, hidden_size_1, hidden_size_2, encoder_output, num_layers, dropouto):
        super(LSTMEncoder, self).__init__()
        self.encoder_input = encoder_input
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.encoder_output = encoder_output
        self.num_layers = num_layers
        self.dropout = dropouto

        self.lsmt1 = VariationalLSTM(self.encoder_input, self.hidden_size_1, dropouto=self.dropout, batch_first=True)
        self.lsmt2 = VariationalLSTM(self.hidden_size_1, self.hidden_size_2, dropouto=self.dropout, batch_first=True)
        self.lsmt3 = VariationalLSTM(self.hidden_size_2, self.encoder_output, dropouto=self.dropout, batch_first=True)

    def forward(self, x):
        out, _ = self.lsmt1(x)
        out, _ = self.lsmt2(out)
        out, _ = self.lsmt3(out)
        return out


class LSTMDecoder(nn.Module):
    def __init__(self, decoder_input, hidden_size_1, hidden_size_2, num_layers, decoder_output, dropouto):
        super(LSTMDecoder, self).__init__()
        self.dropout = dropouto
        self.decoder_input = decoder_input
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.decoder_output = decoder_output
        self.num_layers = num_layers

        self.lstm1 = VariationalLSTM(self.decoder_input, self.hidden_size_1, self.num_layers, dropouto=self.dropout,
                                     batch_first=True)
        self.lstm2 = VariationalLSTM(self.hidden_size_1, self.hidden_size_2, self.num_layers, dropouto=self.dropout,
                                     batch_first=True)
        self.lstm3 = VariationalLSTM(self.hidden_size_2, self.decoder_output, self.num_layers, dropouto=self.dropout,
                                     batch_first=True)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        return out


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, encoder_input, hidden_size_1, hidden_size_2, encoder_output, decoder_input, decoder_hs_1,
                 decoder_hs_2, decoder_output, num_layers, dropout,
                 horizon=0):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder_input = encoder_input
        self.encoder_output = encoder_output
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.decoder_input = decoder_input
        self.decoder_hs_1 = decoder_hs_1
        self.decoder_hs_2 = decoder_hs_2
        self.decoder_output = decoder_output
        self.num_layers = num_layers
        self.horizon = horizon
        #
        self.encoder = LSTMEncoder(
            encoder_input=self.encoder_input,
            hidden_size_1=self.hidden_size_1,
            hidden_size_2=self.hidden_size_2,
            encoder_output=self.encoder_output,
            num_layers=self.num_layers,
            dropouto=dropout
        )
        self.decoder = LSTMDecoder(
            decoder_input=self.decoder_input,
            hidden_size_1=decoder_hs_1,
            hidden_size_2=decoder_hs_2,
            decoder_output=decoder_output,
            num_layers=num_layers,
            dropouto=dropout)

        self.fc = nn.Linear(624, 4)
        self.fc2 = nn.Linear(216, 4)

    def forward(self, x):
        encoder_out = self.encoder(x)

        x_auxiliary = x[:, -self.horizon:, [0]]
        decoder_input = torch.cat([encoder_out, x_auxiliary], dim=1)

        out = self.decoder(decoder_input)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        #out = self.fc2(out)
        return out


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """

    def __init__(self, dropout: float, batch_first: Optional[bool] = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x


class VariationalLSTM(nn.LSTM):
    def __init__(self, *args, dropouti: float = 0.,
                 dropoutw: float = 0., dropouto: float = 0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state


class EncoderPrediction(nn.Module):
    def __init__(self, pretrained_encoder: nn.Module):
        super(EncoderPrediction, self).__init__()

        self.encoder = pretrained_encoder.eval()


    def forward(self, x):
        x_input, external = x
        out = self.encoder(x_input)

        return out

class PredictionNet(nn.Module):
    def __init__(self, params, dropout):
        super(PredictionNet, self).__init__()

        self.params = params
        self.linear1 = nn.Linear(528, params['predict_hidden_1'])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(params['predict_hidden_1'], params['predict_hidden_2'])
        self.linear3 = nn.Linear(params['predict_hidden_2'], params['n_output_steps'])

    def forward(self, external, encoder_prediction):

        x_concat = torch.cat([encoder_prediction, external], dim=2)
        x_concat = x_concat.reshape(x_concat.shape[0], -1)

        out = self.linear1(x_concat)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear3(out)

        return out