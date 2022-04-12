import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import tensor, zeros, float32
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils import weight_norm
from entmax import entmax15, entmax_bisect


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
        self.horizon = output_size
        self.fc_1 = nn.Linear(self.decoder.output_size * (memory + self.horizon), output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        encoder_out = self.encoder(x)

        x_aux = x[:, -self.horizon:, [0]]

        decoder_input = torch.cat([encoder_out, x_aux], dim=1)
        decoder_out = self.decoder(decoder_input)
        out = self.fc_1(decoder_out.flatten(start_dim=1))
        out = self.relu(out)
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
    def __init__(self, encoder, params, dropout):
        super(PredictionNet, self).__init__()

        self.encoder = encoder
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


class PredNet(nn.Module):
    def __init__(self, encoder, params, dropout, n_univariate_featues):
        super(PredNet, self).__init__()
        self.encoder = encoder.eval()
        self.n_univariate_featues = n_univariate_featues

        self.fc1 = nn.Linear(params['input_size'], params['hs_1'])
        self.fc2 = nn.Linear(params['hs_1'], params['hs_2'])
        self.fc3 = nn.Linear(params['hs_2'], params['output'])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, x):
        encoder_input = x[:, :, : self.n_univariate_featues] # all batches, all memory, first 5 features
        encoder_output = self.encoder(encoder_input)

        pred_input = torch.cat([encoder_output, x], dim=2).flatten(start_dim=1) # Take all from x except GHI and take all output from encoder

        out = self.fc1(pred_input)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out


class PredNet2(nn.Module):
    def __init__(self, decoder, params, dropout):
        super(PredNet2, self).__init__()
        self.dropout = dropout
        self.params = params
        self.decoder = decoder.eval()
        self.fc1 = nn.Linear(params['input_size'], params['hs_1'])
        self.fc2 = nn.Linear(params['hs_1'], params['hs_2'])
        self.fc3 = nn.Linear(params['hs_2'], params['output'])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
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


class TemporalBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 kernel_size: int,
                 stride,
                 dilation,
                 padding: int,
                 dropout: float):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(input_size,
                                           output_size,
                                           (1, kernel_size),
                                           stride=stride,
                                           padding=0,
                                           dilation=dilation))
        self.pad = nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(output_size,
                                           output_size,
                                           (1, kernel_size),
                                           stride=stride,
                                           padding=0,
                                           dilation=dilation))
        self.net = nn.Sequential(self.pad,
                                 self.conv1,
                                 self.relu,
                                 self.dropout,
                                 self.pad,
                                 self.conv2,
                                 self.relu,
                                 self.dropout)
        self.downsample = nn.Conv1d(in_channels=input_size,
                                    out_channels=output_size,
                                    kernel_size=1) if input_size != output_size else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self,
                 input_size: int,
                 num_channels: list,
                 kernel_size: int = 2,
                 dropout: float = 0.2):
        super(TemporalConvolutionalNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            input_channels = input_size if i == 0 else num_channels[i-1]
            output_channels = num_channels[i]
            layers += [TemporalBlock(input_channels,
                                     output_channels,
                                     kernel_size,
                                     stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels: list, kernel_size, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvolutionalNetwork(input_size, num_channels, kernel_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.output_size = output_size
        self.out_layer = nn.Linear(num_channels[-1], self.output_size)

    def forward(self, x):
        return self.out_layer(self.dropout(self.tcn(x)[:, :, : -self.output_size]))


class TCNEncoder(nn.Module):
    def __init__(self,
                 encoder,
                 encoder_univariate_features,
                 sequence_length,
                 encoder_output_size,
                 past_covariates_size,
                 output_size,
                 kernel_size,
                 dropout):
        super(TCNEncoder, self).__init__()
        self.encoder_univariate_features = encoder_univariate_features
        self.encoder = encoder.eval()
        self.encoder_input_size = sequence_length
        self.tcn = TCNModel(input_size=encoder_output_size + past_covariates_size,
                            num_channels=[self.encoder_input_size] * int(np.ceil(np.log2(sequence_length/(kernel_size - 1) + 1))),
                            output_size=output_size,
                            kernel_size=kernel_size,
                            dropout=dropout)


    def forward(self, x):
        encoder_input = x[:, :, : self.encoder_univariate_features]  # all batches, all memory, first 5 features
        encoder_output = self.encoder.encoder(encoder_input)
        tcn_input = torch.concat((encoder_output, x), dim=2)
        tcn_output = self.tcn(tcn_input)
        return tcn_output

class TCNDecoder(nn.Module):
    def __init__(self,
                 encoder_output_size,
                 horizon,
                 kernel_size,
                 sequence_length,
                 dropout):
        super(TCNDecoder, self).__init__()
        self.horizon = horizon
        self.tcn = TCNModel(input_size=encoder_output_size + horizon,
                            num_channels=[sequence_length] * int(np.ceil(np.log2(sequence_length/(kernel_size - 1) + 1))),
                            output_size=horizon,
                            kernel_size=kernel_size,
                            dropout=dropout)

    def forward(self, x):
        return self.tcn(x)


class TCNEncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder, weather_predictor):
        super(TCNEncoderDecoderModel, self).__init__()
        self.weather_predictor = weather_predictor
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoder_out = self.encoder(x)
        weather_out = self.weather_predictor(x[:, :, 13 : ])
        decoder_in = torch.concat((encoder_out, weather_out.unsqueeze(-1)), 1)
        decoder_out = self.decoder(decoder_in)
        return decoder_out


class WeatherPredictor(nn.Module):
    def __init__(self,
                 tamb_net,
                 cloud_opacity_net,
                 dew_point_net,
                 pw_net,
                 pressure_net,
                 wind_vel_net,
                 as_feats=False):
        super(WeatherPredictor, self).__init__()
        self.tamb_net = tamb_net
        self.cloud_opacity_net = cloud_opacity_net
        self.dew_point_net = dew_point_net
        self.pw_net = pw_net
        self.pressure_net = pressure_net
        self.wind_vel_net = wind_vel_net
        self.as_feats = as_feats

    def forward(self, x):
        tamb = self.tamb_net(x)
        cloud_op = self.cloud_opacity_net(x)
        dewpoint = self.dew_point_net(x)
        pw = self.pw_net(x)
        pressure = self.pressure_net(x)
        wind_vel = self.wind_vel_net(x)
        return torch.concat((tamb, cloud_op, dewpoint, pw, pressure, wind_vel), dim=1) if not self.as_feats else torch.concat((tamb, cloud_op, dewpoint, pw, pressure, wind_vel), dim=2)


class PastCovariatesEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, encoder_out_feat,  output_size, encoder, univar_features):
        super(PastCovariatesEncoder, self).__init__()
        self.univar_features = univar_features
        self.encoder = encoder.eval()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=input_size + encoder_out_feat, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        encoder_input = x[:, :, : self.univar_features]
        encoder_out = self.encoder.encoder(encoder_input)
        past_covar_input = torch.concat((encoder_out, x), dim=-1)
        out, _ = self.lstm(past_covar_input)
        out = self.linear(out)
        return out

class FutureCovariatesDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, weather_features):
        super(FutureCovariatesDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weather_features = weather_features
        self.lstm = nn.LSTM(input_size=input_size + weather_features, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out)

