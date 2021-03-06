from typing import Optional, Dict, List, Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_forecasting import QuantileLoss
from pytorch_lightning.utilities.types import STEP_OUTPUT
from src.models.layers import (TCNModule,
                               BiGRU, AddNorm,
                               OutputNetwork, GatedLinearUnit,
                               InterpretableMultiHeadAttention, VariableSelectionNetwork, GatedResidualNetwork)

class WhateverNet2(nn.Module):
    def __init__(self,
                 input_size: int,
                 memory: int,
                 target_size: int,
                 horizon: int,
                 hidden_size: int,
                 bigru_layers: int,
                 attention_head_size: int,
                 tcn_params: dict,
                 dropout: float = 0.1,
                 nr_parameters: int = 1):
        super(WhateverNet2, self).__init__()

        self.input_size = input_size
        self.horizon = horizon
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.memory = memory
        self.dropout = dropout
        self.nr_parameters = nr_parameters

        # ------------- HELPER VARS ------

        self.prescalers_linear = {
            name: nn.Linear(1, self.hidden_size) for name in self.reals
        }

        encoder_input_sizes = {
            name: self.hidden_size for name in self.encoder_variables
        }

        # -------------- START COVARIATE ENCODER --------------
        self.covariate_gate = VariableSelectionNetwork(input_sizes=encoder_input_sizes,
                                                       hidden_size=self.hidden_size,
                                                       input_embedding_flags={},
                                                       dropout=self.dropout,
                                                       context_size=self.hidden_size,
                                                       prescalers=self.prescalers_linear,
                                                       single_variable_grns={})

        self.bidirection_GRU = BiGRU(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=bigru_layers,
                                     dropout=dropout)

        self.post_bigru_gate = GatedLinearUnit(input_size=hidden_size,
                                               dropout=dropout,
                                               bidirectional_input=True)

        self.post_bigru_addnorm = AddNorm(input_size=hidden_size, trainable_add=True)

        # -------------- END COVARIATE ENCODER --------------

        # -------------- START LONG & SHORT-TERM CONTEXT ENCODER --------------
        self.pre_attention_grn = GatedResidualNetwork(input_size=hidden_size,
                                                      hidden_size=hidden_size,
                                                      output_size=hidden_size,
                                                      dropout=dropout)

        self.multihead_attention = InterpretableMultiHeadAttention(n_head=attention_head_size,
                                                                   d_model=hidden_size)

        self.post_attention_gate = GatedLinearUnit(input_size=hidden_size,
                                                   dropout=dropout)

        self.post_attention_addnorm = AddNorm(input_size=hidden_size, trainable_add=False)

        self.pre_tcn_grn = GatedResidualNetwork(input_size=hidden_size,
                                                hidden_size=hidden_size,
                                                output_size=hidden_size,
                                                dropout=dropout)

        self.tcn = TCNModule(input_size=hidden_size,
                             input_chunk_length=memory,
                             num_filters=tcn_params["num_filters"],
                             dilation_base=tcn_params["dilation_base"],
                             kernel_size=tcn_params["kernel_size"],
                             num_layers=None,
                             dropout=dropout,
                             nr_params=nr_parameters)

        self.post_tcn_gate = GatedLinearUnit(input_size=nr_parameters,
                                         hidden_size=hidden_size,
                                         dropout=None)

        self.post_tcn_addnorm = AddNorm(input_size=hidden_size, trainable_add=False)

        # -------------- END LONG & SHORT-TERM CONTEXT ENCODER --------------

        # -------------- START OUTPUT NETWORK --------------

        self.pre_output_network_grn = GatedResidualNetwork(input_size=hidden_size,
                                                           hidden_size=hidden_size,
                                                           output_size=hidden_size)

        self.pre_output_network_gate = GatedLinearUnit(input_size=hidden_size,
                                                       hidden_size=hidden_size,
                                                       dropout=None)

        self.pre_output_network_addnorm = AddNorm(input_size=hidden_size, trainable_add=False)

        self.output_network = nn.Linear(hidden_size, nr_parameters)
        # -------------- END OUTPUT NETWORK --------------

    @property
    def encoder_variables(self) -> List[str]:
        """
        List of all encoder variables in model (excluding static variables)
        """
        return [f"{i}" for i in range(self.input_size)]

    @property
    def reals(self) -> List[str]:
        """
        List of all continuous variables in model
        """
        return [f"{i}" for i in range(self.input_size)]

    def forward(self, x):
        # Input: (Batch, Memory, Features)
        batch_size = x.size(0)
        input_vectors = {name: x[..., idx].unsqueeze(-1) for idx, name in enumerate(self.encoder_variables)}
        embeddings_varying_encoder = {name: input_vectors[name][:, :self.memory] for name in self.encoder_variables}
        covariate_gate_out, _ = self.covariate_gate(x=embeddings_varying_encoder)
        # Output: (Batch, Memory, Hidden Size)

        bigru_out = self.bidirection_GRU(covariate_gate_out)
        bigru_out = self.post_bigru_gate(bigru_out)
        bigru_out = self.post_bigru_addnorm(bigru_out, skip=covariate_gate_out)

        # Input: (Batch, Memory, Hidden Size * 2)
        attn_input = self.pre_attention_grn(bigru_out)
        attn_output, _ = self.multihead_attention(q=attn_input,
                                                  k=attn_input,
                                                  v=attn_input)
        attn_output = self.post_attention_gate(attn_output)
        attn_output = self.post_attention_addnorm(attn_output, skip=attn_input)
        # Output (Batch, Memory, Hidden Size)

        # Input: (Batch, Memory, Hidden Size)
        tcn_input = self.pre_tcn_grn(attn_output)
        tcn_output = self.tcn(tcn_input)
        #tcn_output = self.post_tcn_gate(tcn_output)
        #tcn_output = self.post_tcn_addnorm(tcn_output, bigru_out)
        # Output: (Batch, Memory, 1)
        # Input: (Batch, Memory, 1)
        #o_network_input = self.pre_output_network_grn(tcn_output)
        #o_network_output = self.output_network(o_network_input)


        # Output: (Batch, Horizon, 1)
        #out = o_network_output.view(batch_size, self.memory, self.target_size, self.nr_parameters)
        return tcn_output.view(batch_size, self.memory, 1, self.nr_parameters)














