import argparse
import logging
import os
import random
from functools import partial

import dgl
import numpy as np
import torch
import torch.nn as nn
import yaml
import wandb
from sklearn.metrics import f1_score as sk_f1_score, recall_score as sk_recall_score
from torch import optim as optim
from secret import WANDB_KEY
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def f1(y_pred, y_true) -> float:
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    return float(sk_f1_score(y_true.cpu(), preds.cpu()))


def recall(y_pred, y_true) -> float:
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    return float(sk_recall_score(y_true.cpu(), preds.cpu()))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2137])
    parser.add_argument("--dataset", type=str, default="bikeguessr")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--num_features", type=int, default=11,
                        help="number of features in dataset")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2,
                        help="`pow`inddex for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001,
                        help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float,
                        default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=False)

    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    parser.add_argument("--path", type=str,
                        default='./data_transformed/bikeguessr.bin')
    parser.add_argument("--eval_epoch", type=int, default=10)
    parser.add_argument("--eval_repeats", type=int, default=5)
    parser.add_argument("--transform", action="store_true")
    parser.add_argument("--targets", nargs='+', default=None)
    parser.add_argument("--wandb_key", type=str, default=None)

    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true",
                        default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)
    
    parser.add_argument("--full_pipline", action="store_true")

    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None) -> torch.optim.Optimizer:
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


# ------ logging ------

class TBLogger(object):
    def __init__(self, name: str, wandb_api_key: str = "", project: str = "road-embedding-gnn", entity="fdrewnowski", options={}):
        super(TBLogger, self).__init__()
        os.environ['WANDB_API_KEY'] = WANDB_KEY#wandb_api_key
        self.writer = wandb.init(
            project=project, entity=entity, name=name, config=options)
        assert self.writer is wandb.run
        #wandb.config = options

    def note(self, metrics, step):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()

    def watch_model(self, model):
        wandb.watch(model)


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(
            tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view(
            (-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


class ArgParser():
    def __init__(self,
        lr= 0.001,
        lr_f= 0.01,
        num_hidden= 128,
        out_dim= 64,
        num_heads= 4,
        num_layers= 2,
        weight_decay= 2e-4,
        weight_decay_f= 1e-4,
        max_epoch= 500,
        max_epoch_f= 50,
        mask_rate= 0.5,
        encoder= 'gat',
        decoder= 'gat',
        activation= 'prelu',
        in_drop= 0.2,
        attn_drop= 0.1,
        linear_prob= True,
        loss_fn= 'sce' ,
        drop_edge_rate= 0.0,
        optimizer= 'adam',
        replace_rate= 0.05 ,
        alpha_l= 3,
        norm= 'batchnorm') -> None:

        self.lr = lr
        self.lr_f = lr_f
        self.num_hidden = num_hidden
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.weight_decay = weight_decay
        self.weight_decay_f = weight_decay_f
        self.max_epoch = max_epoch
        self.max_epoch_f = max_epoch_f
        self.mask_rate = mask_rate
        self.encoder = encoder
        self.decoder = encoder
        self.activation = activation
        self.in_drop = in_drop
        self.attn_drop = attn_drop
        self.linear_prob = linear_prob
        self.loss_fn = loss_fn 
        self.drop_edge_rate = drop_edge_rate
        self.optimizer = optimizer
        self.replace_rate = replace_rate 
        self.alpha_l = alpha_l
        self.norm = norm
        
        self.scheduler = False
        self.num_out_heads = 1
        self.seeds = [2137]
        self.warmup_steps = -1
        self.num_features = 11
        self.residual = False
        self.negative_slope = 0.2
        self.eval_epoch = 10
        self.eval_repeats = 5
        self.targets = None
        self.pooling = "mean"
        self.deg4feat = False
        self.batch_size = 32
        self.concat_hidden = False