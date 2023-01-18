import math
from typing import Optional, Tuple, Any
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from .setup import setup_module
from torch_geometric.utils import negative_sampling


class Discriminator(nn.Module):
    def __init__(self, n_hidden, n_out):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_out, n_out))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DGI(nn.Module):
    def __init__(self, 
                in_dim: int,
                num_hidden: int,
                num_layers: int,
                nhead: int,
                nhead_out: int,
                activation: str,
                feat_drop: float,
                attn_drop: float,
                negative_slope: float,
                residual: bool,
                norm: Optional[str],
                mask_rate: float = 0.3,
                encoder_type: str = "gat",
                decoder_type: str = "gat",
                loss_fn: str = "sce",
                drop_edge_rate: float = 0.0,
                replace_rate: float = 0.1,
                alpha_l: float = 2,
                concat_hidden: bool = False,
                decoder: Optional[Module] = None):
        super(DGI, self).__init__()

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )



        self.discriminator = Discriminator(num_hidden, num_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, features):
        positive = self.encoder(g, features) # , corrupt=False
        negative = self.encoder(g, features) # , corrupt=True
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2