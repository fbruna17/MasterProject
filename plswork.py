
# %%
from typing import Tuple, Sequence, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.logging import raise_if_not, get_logger
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel, \
    TorchParametricProbabilisticForecastingModel, TorchForecastingModel
from darts.utils.data import PastCovariatesShiftedDataset
from darts.utils.likelihood_models import Likelihood
from darts.utils.torch import random_method
from numpy.random.mtrand import RandomState
from torch import Tensor

from darts_utility import SolarFlare_No_Bigru, SolarFlare_No_Attention, SolarFlare
from src.models.archs import WhateverNet2

logger = get_logger(__name__)

import math
from typing import Optional, Tuple, Dict

import torch.nn.functional as F
import numpy as np
import torch
from torch import nn



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


class ResidualBlock(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_fn,
        weight_norm: bool,
        nr_blocks_below: int,
        num_layers: int,
        input_size: int,
        target_size: int,
    ):
        super(ResidualBlock, self).__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            input_dim,
            num_filters,
            kernel_size,
            dilation=(dilation_base ** nr_blocks_below),
        )
        self.conv2 = nn.Conv1d(
            num_filters,
            output_dim,
            kernel_size,
            dilation=(dilation_base ** nr_blocks_below),
        )
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.weight_norm(
                self.conv1
            ), nn.utils.weight_norm(self.conv2)

        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        residual = x

        # first step
        left_padding = (self.dilation_base ** self.nr_blocks_below) * (
            self.kernel_size - 1
        )
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(F.relu(self.conv1(x)))

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x)
        x = self.dropout_fn(x)

        # add residual
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x = x + residual

        return x


class TCNModule(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_chunk_length: int,
        kernel_size: int,
        num_filters: int,
        num_layers: Optional[int],
        dropout: float,
        target_size: int = 1,
        nr_params: int = 1,
        dilation_base: int = 2,
        weight_norm: bool = False,
    ):
        super(TCNModule, self).__init__()

        # Defining parameters
        self.input_size = input_size
        self.input_chunk_length = input_chunk_length
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.target_size = target_size
        self.nr_params = nr_params
        self.dilation_base = dilation_base
        self.dropout = nn.Dropout(p=dropout)

        # If num_layers is not passed, compute number of layers needed for full history coverage
        if num_layers is None and dilation_base > 1:
            num_layers = math.ceil(
                math.log(
                    (input_chunk_length - 1)
                    * (dilation_base - 1)
                    / (kernel_size - 1)
                    / 2
                    + 1,
                    dilation_base,
                )
            )

        elif num_layers is None:
            num_layers = math.ceil((input_chunk_length - 1) / (kernel_size - 1) / 2)

        self.num_layers = num_layers

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = ResidualBlock(
                num_filters,
                kernel_size,
                dilation_base,
                self.dropout,
                weight_norm,
                i,
                num_layers,
                self.input_size,
                target_size * nr_params,
            )
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)

    def forward(self, x):
        # data is of size (batch_size, input_chunk_length, input_size)
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        for res_block in self.res_blocks_list:
            x = res_block(x)
        x = x.transpose(1, 2)

        return x


class BiGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers: int = 1,
                 dropout: float = 0.1):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bidirectional=True,
                          num_layers=num_layers,
                          dropout=dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        return out


class GateAddNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        skip_size: int = None,
        trainable_add: bool = False,
        dropout: float = None
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class AddNorm(nn.Module):
    def __init__(
            self, input_size: int, skip_size: int = None, trainable_add: bool = True
    ):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(
                self.input_size, batch_first=True, trainable=False
            )

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(x + skip)
        return output


class TimeDistributedInterpolation(nn.Module):
    def __init__(
        self, output_size: int, batch_first: bool = False, trainable: bool = False
    ):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(
            x.unsqueeze(1), self.output_size, mode="linear", align_corners=True
        ).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None, bidirectional_input: bool = False):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size * 2 if bidirectional_input else input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


class ResampleNorm(nn.Module):
    def __init__(
        self, input_size: int, output_size: int = None, trainable_add: bool = True
    ):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size

        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(
                self.output_size, batch_first=True, trainable=False
            )

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        output = self.norm(x)
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(
                    p, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        input_embedding_flags: Dict[str, bool] = {},
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, GatedResidualNetwork] = {},
        prescalers: Dict[str, nn.Linear] = {},
    ):
        """
        Calcualte weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout
        self.context_size = context_size

        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    self.context_size,
                    residual=False,
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    residual=False,
                )

        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = ResampleNorm(
                    input_size, self.hidden_size
                )
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    output_size=self.hidden_size,
                    dropout=self.dropout,
                )
            if name in prescalers:  # reals need to be first scaled up
                self.prescalers[name] = prescalers[name]
            elif not self.input_embedding_flags.get(name, False):
                self.prescalers[name] = nn.Linear(1, input_size)

        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(
            size if name in self.input_embedding_flags else size
            for name, size in self.input_sizes.items()
        )

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: Dict[str, torch.Tensor], context: torch.Tensor = None):
        if self.num_inputs > 1:
            # transform single variables
            var_outputs = []
            weight_inputs = []
            for name in self.input_sizes.keys():
                # select embedding belonging to a single input
                variable_embedding = x[name]
                if name in self.prescalers:
                    variable_embedding = self.prescalers[name](variable_embedding)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)

            # calculate variable weights
            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            outputs = var_outputs * sparse_weights
            outputs = outputs.sum(dim=-1)
        else:  # for one input, do not perform variable selection but just encoding
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](
                variable_embedding
            )  # fast forward if only one variable
            if outputs.ndim == 3:  # -> batch size, time, hidden size, n_variables
                sparse_weights = torch.ones(
                    outputs.size(0), outputs.size(1), 1, 1, device=outputs.device
                )  #
            else:  # ndim == 2 -> batch size, hidden size, n_variables
                sparse_weights = torch.ones(
                    outputs.size(0), 1, 1, device=outputs.device
                )
        return outputs, sparse_weights


class OutputNetwork(nn.Module):
    def __init__(self, d_model, nr_parameters, hidden_size, horizon):
        super(OutputNetwork, self).__init__()
        self.attention = InterpretableMultiHeadAttention(n_head=2, d_model=d_model)
        self.fc = nn.Linear(hidden_size, nr_parameters)
        self.horizon = horizon

    def forward(self, x):
        out, _ = self.attention(q=x[:, -self.horizon:, :],
                               k=x,
                               v=x)
        out = self.fc(out)
        return out




class SolarFlare(TorchParametricProbabilisticForecastingModel, PastCovariatesTorchModel):
    @random_method
    def __init__(
            self,
            input_chunk_length: int,
            output_chunk_length: int,
            kernel_size: int = 3,
            hidden_size: int = 16,
            attention_head_size: int = 4,
            bigur_layers: int = 1,
            num_filters: int = 3,
            num_layers: Optional[int] = None,
            weight_norm: bool = False,
            dilation_base: int = 2,
            dropout: float = 0.2,
            likelihood: Optional[Likelihood] = None,
            random_state: Optional[Union[int, RandomState]] = None,
            **kwargs
            ):
        raise_if_not(
            kernel_size < input_chunk_length,
            "The kernel size must be strictly smaller than the input length.",
            logger,
        )
        raise_if_not(
            output_chunk_length < input_chunk_length,
            "The output length must be strictly smaller than the input length",
            logger,
        )

        kwargs["input_chunk_length"] = input_chunk_length
        kwargs["output_chunk_length"] = output_chunk_length

        super().__init__(likelihood=likelihood, **kwargs)

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.bigru_layers = bigur_layers
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.dilation_base = dilation_base
        self.dropout = dropout
        self.weight_norm = weight_norm


    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> PastCovariatesShiftedDataset:

        return PastCovariatesShiftedDataset(
            target_series=target,
            covariates=past_covariates,
            length=self.input_chunk_length,
            shift=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
        )


    def _create_model(self, train_sample: Tuple[Tensor]) -> torch.nn.Module:
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        tcn_params = {"num_filters": self.num_filters,
                      "dilation_base": self.dilation_base,
                      "kernel_size": self.kernel_size}
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
        return WhateverNet2(input_size=input_dim,
                            memory=self.input_chunk_length,
                            horizon=self.output_chunk_length,
                            target_size=output_dim,
                            nr_parameters=nr_params,
                            hidden_size=self.hidden_size,
                            bigru_layers=self.bigru_layers,
                            attention_head_size=self.attention_head_size,
                            tcn_params=tcn_params,
                            dropout=self.dropout)

    @random_method
    def _produce_predict_output(self, x):
        if self.likelihood:
            output = self.model(x)
            return self.likelihood.sample(output)
        else:
            return self.model(x).squeeze(dim=-1)


    @property
    def first_prediction_index(self) -> int:
        return -self.output_chunk_length


    @staticmethod
    def load_model(path: str) -> "TorchForecastingModel":
        """loads a model from a given file path. The file name should end with '.pth.tar'

        Parameters
        ----------
        path
            Path under which to save the model at its current state. The path should end with '.pth.tar'
        """

        raise_if_not(
            path.endswith(".pth.tar"),
            "The given path should end with '.pth.tar'.",
            logger,
        )

        with open(path, "rb") as fin:
            model = torch.load(fin, map_location=torch.device('cpu'))
        return model

# %%

path = 'irradiance_dataset2.csv'
df = pd.read_csv(path)[:10000]
df = df.drop(columns="Unnamed: 0")
memory = 24
horizon = 5
batch = 128

df.insert(0, 'GHI', df.pop('GHI'))

# Split data
target_ts = TimeSeries.from_series(df["GHI"]).astype(np.float32)
target_train, target_val = target_ts.split_after(0.8)
target_val, target_test = target_val.split_after(0.5)


past_covar_ts = TimeSeries.from_dataframe(df[df.columns.to_list()[1:]]).astype(np.float32)
past_covar_train, past_covar_val = past_covar_ts.split_after(0.8)
past_covar_val, past_covar_test = past_covar_val.split_after(0.5)

# Scale data
target_ts_scaler = Scaler()
target_train = target_ts_scaler.fit_transform(target_train)
target_val = target_ts_scaler.transform(target_val)
target_test = target_ts_scaler.transform(target_test)
series_transformed = target_ts_scaler.transform(target_ts)

covar_ts_scaler = Scaler()
covar_train = covar_ts_scaler.fit_transform(past_covar_train)
covar_val = covar_ts_scaler.transform(past_covar_val)
covar_test = covar_ts_scaler.transform(past_covar_test)

model = SolarFlare.load_model("SolarFuck123_no_attention.pth.tar")
model.device = torch.device("cpu")

pred = model.predict(5, series=target_val[9:129], past_covariates=past_covar_val[9:129], num_samples=600)
target_ts.slice_intersect(pred).plot(label="target")
pred.plot(label="forecast")
plt.show()

backcast = model.historical_forecasts(series=target_train,
                                      past_covariates=past_covar_train,
                                      num_samples=300,
                                      start=0.95,
                                      retrain=False,
                                      verbose=True)


def make_preds(n_preds=10):
    pred_temp = None
    for i in range(n_preds):
        pred = model.predict(5, series=target_val[(0 + i):(120 + i)], past_covariates=past_covar_val[(0 + i):(120 + i)], num_samples=300)
        if i == 0:
            pred_temp = pred
        else:
            pred_temp = pred_temp.append(pred)


    series_transformed.slice_intersect(pred_temp).plot(label="target")
    pred_temp.plot(label="forecast")
    plt.show()



