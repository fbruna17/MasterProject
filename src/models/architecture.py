from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import tensor, zeros, float32
from torch.nn.utils.rnn import PackedSequence


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


class LSTMCoder(nn.Module):
    def __init__(self, **kwargs):
        super(LSTMCoder, self).__init__()
        self.input_size = kwargs.get('input_size', '')  # second parameter is a default value if key does not exist
        self.hs_1 = kwargs.get('hs_1', 10)
        self.hs_2 = kwargs.get('hs_2', 10)
        self.output_size = kwargs.get('output_size', 10)
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout = kwargs.get('dropout', 0.)
        self.batch_first = kwargs.get('batch_first', True)

        self.lstm_1 = VariationalLSTM(self.input_size, self.hs_1, dropouto=self.dropout, batch_first=self.batch_first)
        self.lstm_2 = VariationalLSTM(self.hs_1, self.hs_2, dropouto=self.dropout, batch_first=self.batch_first)
        self.lstm_3 = VariationalLSTM(self.hs_2, self.output_size, dropouto=self.dropout, batch_first=self.batch_first)

    def forward(self, x):
        out, _ = self.lstm_1(x)
        out, _ = self.lstm_2(out)
        out, _ = self.lstm_3(out)
        return out


class LSTMEncoder1(LSTMCoder):
    def __init__(self, **kwargs):
        super(LSTMEncoder1, self).__init__(**kwargs)


class LSTMDecoder1(LSTMCoder):
    def __init__(self, **kwargs):
        super(LSTMDecoder1, self).__init__(**kwargs)


class LSMTEncoderDecoder1(nn.Module):
    def __init__(self, encoder_params: dict, decoder_params: dict, fc_hidden_size: int, output_size: int, memory: int):
        super(LSMTEncoderDecoder1, self).__init__()
        self.encoder = LSTMEncoder1(**encoder_params)
        self.decoder = LSTMDecoder1(**decoder_params)
        self.fc_1 = nn.Linear(self.decoder.output_size * memory, fc_hidden_size)
        self.fc_2 = nn.Linear(fc_hidden_size, output_size)

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        fc1_out = self.fc_1(decoder_out.flatten(start_dim=1))
        out = self.fc_2(fc1_out)
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

        # 'lstm1': vd.LSTM(1, 2, dropouto=p, batch_first=True),
        # 'lstm2': vd.LSTM(2, 2, dropouto=p, batch_first=True),
        # 'lstm3': vd.LSTM(2, 1, dropouto=p, batch_first=True)

        self.lstm1 = VariationalLSTM(self.decoder_input, self.hidden_size_1, self.num_layers, dropouto=self.dropout)
        self.lstm2 = VariationalLSTM(self.hidden_size_1, self.hidden_size_2, self.num_layers, dropouto=self.dropout)
        self.lstm3 = VariationalLSTM(self.hidden_size_2, self.decoder_output, self.num_layers, dropouto=self.dropout)

        # self.lstm = VariationalLSTM(self.decoder_input, self.encoder_output, self.num_layers, dropouto=self.dropout,
        #                             batch_first=True)
        # self.linear = nn.Linear(self.encoder_output, self.horizon)

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

        self.fc = nn.Linear(decoder_output, 32)
        self.fc2 = nn.Linear(32, horizon)
        #
        # self.encoder = VariationalLSTM(
        #     input_size=self.encoder_input,
        #     hidden_size=self.hidden_size,
        #     num_layers=self.num_layers,
        # )

    # def forward(self, x1, x2):
    #     out, (hs, cs) = self.encoder(x1)
    #     decoder_out = self.decoder(x2, out)
    #     return decoder_out
    def forward(self, x):
        out = self.encoder(x)

        x_auxiliary = x[:, :, -self.horizon:]
        decoder_input = torch.cat([out, x_auxiliary], dim=2)

        out = self.decoder(decoder_input)
        out = self.fc(out)
        out = self.fc2(out)
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


class VDEncoder(nn.Module):
    def __init__(self, in_features, out_features, p):
        super(VDEncoder, self).__init__()
        self.model = nn.ModuleDict({
            'lstm1': VariationalLSTM(in_features, 32, dropouto=p, batch_first=True),
            'lstm2': VariationalLSTM(32, 8, dropouto=p, batch_first=True),
            'lstm3': VariationalLSTM(8, out_features, dropouto=p, batch_first=True),
            'relu': nn.ReLU()
        })

    def forward(self, x):
        out, _ = self.model['lstm1'](x)
        out, _ = self.model['lstm2'](out)
        out, _ = self.model['lstm3'](out)
        out = self.model['relu'](out)

        return out


class VDDecoder(nn.Module):
    def __init__(self, p):
        super(VDDecoder, self).__init__()
        self.model = nn.ModuleDict({
            'lstm1': VariationalLSTM(1, 2, dropouto=p, batch_first=True),
            'lstm2': VariationalLSTM(2, 2, dropouto=p, batch_first=True),
            'lstm3': VariationalLSTM(2, 1, dropouto=p, batch_first=True)
        })

    def forward(self, x):
        out, _ = self.model['lstm1'](x)
        out, _ = self.model['lstm2'](out)
        out, _ = self.model['lstm3'](out)

        return out


class VDEncoderDecoder(nn.Module):
    def __init__(self, in_features, input_steps, output_steps, p):
        super(VDEncoderDecoder, self).__init__()
        self.enc_in_features = in_features
        self.input_steps = input_steps  # t in the paper
        self.output_steps = output_steps  # f in the paper
        self.enc_out_features = 1
        self.traffic_col = 4
        self.p = p

        self.model = nn.ModuleDict({
            'encoder': VDEncoder(self.enc_in_features, self.enc_out_features, self.p),
            'decoder': VDDecoder(self.p),
            'fc1': nn.Linear(self.input_steps + self.output_steps, 32),
            'fc2': nn.Linear(32, self.output_steps)
        })

    def forward(self, x):
        out = self.model['encoder'](x)

        x_auxiliary = x[:, -self.output_steps:, [self.traffic_col]]
        decoder_input = torch.cat([out, x_auxiliary], dim=1)

        out = self.model['decoder'](decoder_input)
        out = self.model['fc1'](out.view(-1, self.input_steps + self.output_steps))
        out = self.model['fc2'](out)

        return out


class Predict(nn.Module):
    def __init__(self, params, p, pretrained_encoder: nn.Module):
        super(Predict, self).__init__()

        self.encoder = pretrained_encoder.eval()
        self.params = params
        self.model = nn.Sequential(
            nn.Linear(params['n_extracted_features'] + params['n_external_features'], params['predict_hidden_1']),
            nn.Dropout(p),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(params['predict_hidden_1'], params['predict_hidden_2']),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(params['predict_hidden_2'], params['n_output_steps'])
        )

    def forward(self, x):
        x_input, external = x
        extracted = self.encoder(x_input.permute(0,2,1)).view(-1, self.params['n_extracted_features'])
        x_concat = torch.cat([extracted, external], dim=-1)
        out = self.model(x_concat)
        return out


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hs_1, hs_2, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hs_1, batch_first=True)
        self.lstm2 = nn.LSTM(hs_1, hs_2)
        self.fc = nn.Linear(hs_2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1])
        return out

