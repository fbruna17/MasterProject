from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import tensor, zeros, float32
from torch.nn.utils.rnn import PackedSequence

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

        self.lstm = nn.LSTM(dropout=0.0, num_layers=1, batch_first=True, hidden_size=self.hs_1, input_size=self.input_size)
        self.lstm1 = nn.LSTM(dropout=0.0, num_layers=1, batch_first=True, hidden_size=self.hs_2, input_size=self.hs_1)
        self.lstm_1 = VariationalLSTM(self.input_size, self.hs_1, dropouto=self.dropout, batch_first=self.batch_first)
        self.lstm_2 = VariationalLSTM(self.hs_1, self.hs_2, dropouto=self.dropout, batch_first=self.batch_first)
        self.lstm_3 = VariationalLSTM(self.hs_2, self.output_size, dropouto=self.dropout, batch_first=self.batch_first)

    def forward(self, x):
        out, _ = self.lstm_1(x)
        out, _ = self.lstm_2(out)
        #out, _ = self.lstm_2(out)
        #out, _ = self.lstm_3(out)
        return out


class LSTMEncoder1(LSTMCoder):
    def __init__(self, **kwargs):
        super(LSTMEncoder1, self).__init__(**kwargs)


class LSTMDecoder1(LSTMCoder):
    def __init__(self, **kwargs):
        super(LSTMDecoder1, self).__init__(**kwargs)


class WeatherEncoderDecoder(nn.Module):
    def __init__(self, encoder_params: dict, decoder_params: dict, output_size: int, memory: int):
        super(WeatherEncoderDecoder, self).__init__()
        self.encoder = LSTMEncoder1(**encoder_params)
        self.decoder = LSTMDecoder1(**decoder_params)
        self.memory = memory
        self.horizon = output_size
        self.fc_1 = nn.Linear(self.encoder.hs_2 * memory, output_size)
        self.fc_2 = nn.Linear((self.encoder.hs_2 * memory) // 2, output_size)
        self.relu = nn.SELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        encoder_out = self.encoder(x)
        #decoder_out = self.decoder(encoder_out)
        #out = decoder_out.flatten(start_dim=1)
        out = encoder_out.flatten(start_dim=1)
        out = self.fc_1(out)
        out = self.relu(out)
        return out


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class PastCovariatesEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PastCovariatesEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size)
        self.linear = TimeDistributed(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out


class WeatherCovariatesEncoder(nn.Module):
    def __init__(self, encoder, decoder, weather_pred):
        super(WeatherCovariatesEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.weather_pred = weather_pred

    def forward(self, x):
        encoder_out = self.encoder(x)
        weather_features = self.weather_pred(x)
        decoder_in = torch.cat((encoder_out, weather_features), dim=1)
        out = self.decoder(decoder_in)
        return out

class LSMTEncoderDecoder1(nn.Module):
    def __init__(self, encoder_params: dict, decoder_params: dict, fc_hidden_size: int, output_size: int, memory: int):
        super(LSMTEncoderDecoder1, self).__init__()
        self.encoder = LSTMEncoder1(**encoder_params)
        self.decoder = LSTMDecoder1(**decoder_params)
        self.horizon = output_size
        self.fc_1 = nn.Linear(self.encoder.hs_2 * memory, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        encoder_out = self.encoder(x)

        #x_aux = x[:, -self.horizon:, [0]]

        #decoder_input = torch.cat([encoder_out, x_aux], dim=1)
        #decoder_out = self.decoder(decoder_input)
        out = encoder_out.flatten(start_dim=1)
        out = self.fc_1(out)
        out = self.relu(out)
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
    def __init__(self,
                 *args,
                 dropouti: float = 0.,
                 dropoutw: float = 0.,
                 dropouto: float = 0.,
                 batch_first=True,
                 unit_forget_bias=True,
                 **kwargs):
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


class PredNet(nn.Module):
    def __init__(self, encoder, weather_pred, params, dropout, n_univariate_featues):
        super(PredNet, self).__init__()
        self.encoder = encoder.eval()
        self.cloud_op_encoder = weather_pred.cloud_opacity_net.encoder
        self.dew_point_encoder = weather_pred.dew_point_net.encoder
        self.pressure_net_encoder = weather_pred.pressure_net.encoder
        self.pw_net_encoder = weather_pred.pw_net.encoder
        self.tamb_net_encoder = weather_pred.tamb_net.encoder
        self.wind_vel_encoder = weather_pred.wind_vel_net.encoder

        self.weather_pred = weather_pred
        self.n_univariate_featues = n_univariate_featues

        self.fc1 = nn.Linear(params['input_size'], params['output'])
        self.fc2 = nn.Linear(params['hs_1'], params['output'])
        self.fc3 = nn.Linear(params['hs_2'], params['output'])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.SELU()

    def forward(self, x):
        encoder_input = x[:, :, :self.n_univariate_featues]  # all batches, all memory, first 5 features
        encoder_output = self.encoder(encoder_input)
        w_x = x[:, :, 1:25]
        weather_out = torch.concat(
            [self.cloud_op_encoder(w_x), self.dew_point_encoder(w_x), self.pressure_net_encoder(w_x),
             self.pw_net_encoder(w_x), self.tamb_net_encoder(w_x), self.wind_vel_encoder(w_x)], dim=2)
        pred_input = torch.cat([encoder_output, x, weather_out], dim=2).flatten(
            start_dim=1)  # Take all from x except GHI and take all output from encoder

        out = self.fc1(pred_input)
        out = self.dropout(out)
        out = self.relu(out)
        #out = self.dropout(out)
        #out = self.fc2(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        # out = self.fc3(out)
        # out = self.relu(out)
        return out

class SimplePredNet(nn.Module):
    def __init__(self, dropout, params):
        super(SimplePredNet, self).__init__()
        self.fc1 = nn.Linear(params['input_size'], params['hs_1'])
        self.fc2 = nn.Linear(params['hs_1'], params['hs_2'])
        self.fc3 = nn.Linear(params['hs_2'], params['output'])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.flatten(start_dim=1)
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


class WeatherPredictor(nn.Module):
    def __init__(self,
                 tamb_net,
                 cloud_opacity_net,
                 dew_point_net,
                 pw_net,
                 pressure_net,
                 wind_vel_net,
                 ):
        super(WeatherPredictor, self).__init__()
        self.tamb_net = tamb_net
        self.cloud_opacity_net = cloud_opacity_net
        self.dew_point_net = dew_point_net
        self.pw_net = pw_net
        self.pressure_net = pressure_net
        self.wind_vel_net = wind_vel_net

    def forward(self, x):
        tamb = self.tamb_net(x).unsqueeze(dim=2)
        cloud_op = self.cloud_opacity_net(x).unsqueeze(dim=2)
        dewpoint = self.dew_point_net(x).unsqueeze(dim=2)
        pw = self.pw_net(x).unsqueeze(dim=2)
        pressure = self.pressure_net(x).unsqueeze(dim=2)
        wind_vel = self.wind_vel_net(x).unsqueeze(dim=2)

        return torch.concat((tamb, cloud_op, dewpoint, pw, pressure, wind_vel), dim=2)
