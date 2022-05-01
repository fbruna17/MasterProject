from typing import Tuple, Sequence, Optional, Union

import torch
from darts import TimeSeries
from darts.logging import raise_if_not, get_logger
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel, \
    TorchParametricProbabilisticForecastingModel
from darts.utils.data import PastCovariatesShiftedDataset
from darts.utils.likelihood_models import Likelihood
from darts.utils.torch import random_method
from numpy.random.mtrand import RandomState
from torch import Tensor

from src.models.archs import WhateverNet2

logger = get_logger(__name__)


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