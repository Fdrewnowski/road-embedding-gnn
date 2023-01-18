from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gat import GAT
from .gcn import GCN

from utils import create_norm




def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    # elif m_type == "dotgat":
    #     mod = DotGAT(
    #         in_dim=in_dim,
    #         num_hidden=num_hidden,
    #         out_dim=out_dim,
    #         num_layers=num_layers,
    #         nhead=nhead,
    #         nhead_out=nhead_out,
    #         concat_out=concat_out,
    #         activation=activation,
    #         feat_drop=dropout,
    #         attn_drop=attn_drop,
    #         residual=residual,
    #         norm=create_norm(norm),
    #         encoding=(enc_dec == "encoding"),
    #     )
    # elif m_type == "gin":
    #     mod = GIN(
    #         in_dim=in_dim,
    #         num_hidden=num_hidden,
    #         out_dim=out_dim,
    #         num_layers=num_layers,
    #         dropout=dropout,
    #         activation=activation,
    #         residual=residual,
    #         norm=norm,
    #         encoding=(enc_dec == "encoding"),
    #     )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


