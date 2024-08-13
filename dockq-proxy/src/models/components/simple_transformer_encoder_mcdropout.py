import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple, Dict
# import ipdb

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from functools import partial

from .mc_dropout import ConsistentMCDropout
ConsistentMCDropout = partial(ConsistentMCDropout, p=0.1)

class SimpleTransformerModel(nn.Module):

    def __init__(
            self,
            ntoken: int = 28,
            d_model: int = 32,  # embedding dimension
            nhead: int = 2,
            d_hid: int = 32,
            nlayers: int = 2,
            dropout: float = 0.5,
            nout: int = 32,
            scalar_head: bool = True,
            scalar_linear_layers: int = 1,
            predict_on_last_token_only: bool = False,
            special_weight_init: bool = True,
    ):
        super().__init__()

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Watch out !! Double dropout
        self.pos_dropout = ConsistentMCDropout()
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.enc_dropout = ConsistentMCDropout()
        self.embedding = nn.Embedding(ntoken, d_model)
        self.emb_dropout = ConsistentMCDropout()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, nout)
        self.linear_dropout = ConsistentMCDropout()
        self.scalar_head = scalar_head
        self.predict_on_last_token_only = predict_on_last_token_only
        if scalar_head:
            # Might be useful to have linear -> linear, if we project in a different space size
            modules = []
            for _ in range(scalar_linear_layers - 1):
                modules.append(nn.Linear(nout, nout))
                modules.append(ConsistentMCDropout())
                modules.append(nn.ReLU())
            modules.append(nn.Linear(nout, 1))
            modules.append(ConsistentMCDropout())
            self.scalar_linear = nn.Sequential(*modules)

        if special_weight_init:
            self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(
            self,
            src: Tensor,
            src_mask: Tensor = None,
            src_key_padding_mask: Tensor = None,
            **kwargs
    ) -> Dict[str, Tensor]:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``
            src_key_padding_mask: Tensor, shape ``[batch_size, seq_len]`! type float!!
                (mask for different seq lengths 1 for valid 0 for padding)
        Returns:
            output Tensor of shape ``[seq_len, batch_size, nout]``
        """
        src = src.transpose(0, 1)  # Seq x batch x
        src_key_padding_mask = src_key_padding_mask.float()  # weird why it doesn't like bool

        src = self.embedding(src) * math.sqrt(self.d_model)
        # import ipdb; ipdb.set_trace()
        src = self.emb_dropout(src)
        src = self.pos_encoder(src)
        src = self.pos_dropout(src)
        output = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        output = self.enc_dropout(output)
        output = self.linear(output)
        output = self.linear_dropout(output)

        out_dict = {
            "features": output,
        }
        if self.scalar_head:
            if self.predict_on_last_token_only:
                last_tokens = output[
                    src_key_padding_mask.sum(1).long() - 1,
                    torch.arange(output.shape[1], device=output.device)
                ]
                feat_seq = last_tokens
            else:
                # mean on dim 0 with masking based on src_key_padding_mask
                masked_out = output * src_key_padding_mask.t().unsqueeze(-1)
                masked_sum = torch.sum(masked_out, dim=0)  # sum over seq
                feat_seq = masked_sum / torch.sum(src_key_padding_mask, dim=1).unsqueeze(-1)

            scalar = self.scalar_linear(feat_seq)
            out_dict["seq_scalar"] = scalar.squeeze(1)

        return out_dict


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":
    _ = SimpleTransformerModel()