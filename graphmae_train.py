import copy
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from dgl.data.utils import load_graphs
from dgl.heterograph import DGLHeteroGraph
from tqdm import tqdm
from models.load_data import load_data, load_train_and_val_data
import pickle
from params import ENCODER, ACTIVATION, NUM_HIDDEN, NUM_OUT, NUM_LAYERS, IN_DROP, OPTIMIZER, WEIGHT_DECAY, LR

from models import GraphMAE, build_model
from utils import (TBLogger, build_args, create_optimizer,
                            get_current_lr, load_best_configs, set_random_seed, ArgParser)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def train_transductive(args, train_graphs, val_graphs, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    #device = 'cpu'
    logging.info("using device: {}".format(device))
    max_epoch = args.max_epoch

    optim_type = args.optimizer

    lr = args.lr
    weight_decay = args.weight_decay
    use_scheduler = args.scheduler

    num_features = train_graphs[0].ndata["feat"].shape[1]
    args.num_features = num_features

    options = {
    "architecture":"Graphmae",
    "encoder_type":args.encoder,
    "hidden_dim": args.num_hidden,
    "out_dim": args.out_dim,
    "layers": args.num_layers,
    "in_drop": args.in_drop,
    "dataset": dataset,
    "lr":args.lr,
    "lr_f": args.lr_f,
    "num_heads": args.num_heads,
    "weight_decay": args.weight_decay,
    "weight_decay_f": args.weight_decay_f,
    "max_epoch": args.max_epoch,
    "max_epoch_f": args.max_epoch_f,
    "mask_rate": args.mask_rate,
    "encoder": args.encoder,
    "activation": args.activation,
    "in_drop": args.in_drop,
    "attn_drop": args.attn_drop,
    "linear_prob": args.linear_prob,
    "loss_fn": args.loss_fn ,
    "drop_edge_rate": args.drop_edge_rate,
    "optimizer": args.optimizer,
    "replace_rate": args.replace_rate,
    "alpha_l": args.alpha_l,
    "norm": args.norm,
    "current_time": current_time
    }
    logger = TBLogger(name=f"{options['architecture']}_{current_time}", options=options)


    model = build_model(args, 'graphmae')
    print(model.eval())
    #model= nn.DataParallel(model)

    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)

    if use_scheduler:
        logging.info("Use schedular")

        def scheduler(epoch): return (
            1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
        # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=scheduler)
    else:
        scheduler = None
    X = [g.ndata['feat'] for g in train_graphs]

    train_stats, val_stats, best_representation = pretrain(args, model, train_graphs, X, optimizer, max_epoch, device, scheduler,
                    logger, val_graphs = val_graphs, experiment_time=current_time)

    if logger is not None:
        logger.finish()

    with open("./data/training_data/graphmae/graphmae_{}_{}_{}_{}_{}_{}.pkl".format(args.encoder,
                                                        args.num_hidden,
                                                        args.out_dim,
                                                        args.num_layers,
                                                        args.lr,
                                                        current_time), 'wb') as handle:
        pickle.dump([train_stats, val_stats, best_representation], handle, protocol=pickle.HIGHEST_PROTOCOL)


def pretrain(args,
             model: GraphMAE,
             graphs: List[DGLHeteroGraph],
             feats: List[torch.Tensor],
             optimizer: torch.optim.Optimizer,
             max_epoch: int,
             device: torch.device,
             scheduler: torch.optim.lr_scheduler.LambdaLR,
             logger: TBLogger = None,
             val_graphs = None,
             experiment_time = None) -> Tuple[GraphMAE, GraphMAE, Tuple[float]]:

    logging.info("start training..")
    epoch_iter = tqdm(range(max_epoch))
    best_loss = 1e10
    best_val_representations = []
    early_stopping = args.max_epoch_f
    early_stopping_counter = 0

    train_loss = [[] for i in range(0, len(train_graphs))]
    val_loss = [[] for i in range(0, len(val_graphs))]

    for epoch in epoch_iter:

        total_loss = 0
        #total_ap = 0
        # City-batch training
        for idx, (graph, feat) in enumerate(zip(graphs, feats)):

            g = graph.to(device)
            x = feat.to(device)
            model.train()

            loss = model(g, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            #total_ap += ap.item()

            #epoch_iter.set_description(
            #    f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
            train_loss[idx].append(loss.item())


        # Every eval_epoch city-batch evaluation of gmae
        total_val_loss = 0
        #total_val_ap = 0
        val_representations = []
        model.eval()
        epoch_iter.set_description(
            f"# Epoch {epoch}: evaluating: {loss.item():.4f}")
        with torch.no_grad():
            for idx, val_graph in enumerate(val_graphs):
                g = val_graph.to(device)
                x = val_graph.ndata['feat'].to(device)

                loss = model(g, x)
                total_val_loss += loss.item()
                #total_val_ap += ap.item()
                val_loss[idx].append(loss.item())
                val_representations.append(model.embed(g, x).cpu().detach().numpy())
        
        
        logging_dict = {}
        logging_dict['Loss/train'] = total_loss/len(graphs)
        #logging_dict['AP/train'] = total_ap/len(graphs)
        logging_dict['Loss/val'] = total_val_loss/len(val_graphs)
        #logging_dict['AP/val'] = total_val_ap/len(val_graphs)

        logger.note(logging_dict, step=epoch)
                

        if total_val_loss < best_loss:
            best_loss = total_val_loss
            early_stopping_counter = 0
            best_val_representations = val_representations
            try:
                torch.save(model.cpu().state_dict(),
                            "./data/models/graphmae/gmae_{}_{}_{}_{}_{}_{}.bin".format(args.encoder,
                                                        args.num_hidden,
                                                        args.out_dim,
                                                        args.num_layers,
                                                        args.lr,
                                                        experiment_time))
            except Exception as e:
                print(str(e))


        if early_stopping_counter >= early_stopping:
            break
        else:
            early_stopping_counter += 1

    return train_loss, val_loss, best_val_representations



if __name__ == '__main__':
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    logging.info(args)
    dataset = 'polish_cities'
    dataset_dir = './data/raw_data/'

    if not os.path.exists("./data/models/graphmae"):
        os.makedirs("./data/models/graphmae")
    if not os.path.exists("./data/training_data/graphmae"):
        os.makedirs("./data/training_data/graphmae")

    if args.full_pipline:
        train_graphs, val_graphs = load_train_and_val_data()
        for lr in LR:
            for layers in NUM_LAYERS:
                for encoder in ENCODER:
                    for num_out in NUM_OUT:
                        for num_hidden in NUM_HIDDEN:
                            try:
                                args_object = ArgParser(lr=lr,
                                                        num_hidden=num_hidden, 
                                                        out_dim=num_out,
                                                        num_layers=layers,
                                                        encoder=encoder,
                                                        lr_f=args.lr_f,
                                                        num_heads=args.num_heads,
                                                        weight_decay=args.weight_decay,
                                                        weight_decay_f=args.weight_decay_f,
                                                        max_epoch=args.max_epoch,
                                                        max_epoch_f=args.max_epoch_f,
                                                        mask_rate=args.mask_rate,
                                                        decoder=args.decoder,
                                                        activation=args.activation,
                                                        in_drop=args.in_drop,
                                                        attn_drop=args.attn_drop,
                                                        linear_prob=args.linear_prob,
                                                        loss_fn=args.loss_fn ,
                                                        drop_edge_rate=args.drop_edge_rate,
                                                        optimizer=args.optimizer,
                                                        replace_rate=args.replace_rate,
                                                        alpha_l=args.alpha_l,
                                                        norm=args.norm)
                                train_transductive(args_object, train_graphs, val_graphs, dataset)
                            except Exception as e:
                                print(str(e))
                                print("FAILED for model GraphMAE with {},{},{},{},{}".format(encoder,
                                                                                        layers,
                                                                                        num_out,
                                                                                        num_hidden,
                                                                                        lr))

    else:
        train_graphs, val_graphs = load_data(dataset_dir+dataset+".bin")
        train_transductive(args, train_graphs, val_graphs, dataset)
