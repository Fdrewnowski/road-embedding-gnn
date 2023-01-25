import argparse
import logging
import time
from models import build_model
import pickle

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from utils import build_args, load_best_configs, TBLogger, ArgParser
from models.load_data import load_data, load_train_and_val_data
from datetime import datetime
from tqdm import tqdm
from params import ENCODER, ACTIVATION, NUM_HIDDEN, NUM_OUT, NUM_LAYERS, IN_DROP, OPTIMIZER, WEIGHT_DECAY, LR
# train deep graph infomax
def train_dgi(model, optimizer, train_graphs, val_graphs, logger, args, experiment_time):
    
    train_loss = [[] for i in range(0, len(train_graphs))]
    val_losses = [[] for i in range(0, len(val_graphs))]
    best_val_representations = []

    best = 1e9
    early_stopping = args.max_epoch_f
    early_stopping_counter = 0

    for epoch in tqdm(range(args.max_epoch), total=args.max_epoch):
        total_loss = 0
        
        # model training
        for idx, train_graph in enumerate(train_graphs):
            g = train_graph.to(device)
            x = train_graph.ndata["feat"].to(device)

            # training
            model.train()
            loss = model(g, x)
            total_loss+=loss.item()

            train_loss[idx].append(loss.item())

            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()

        # model evaluation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for idx, val_graph in enumerate(val_graphs):
                g = val_graph.to(device)
                x = val_graph.ndata["feat"].to(device)

                val_loss = model(g, x)
                val_losses[idx].append(val_loss.item())
                total_val_loss += val_loss.item()

        # save best model based on total_val_loss
        if total_val_loss < best:
            best = total_val_loss
            early_stopping_counter = 0
            best_val_representations = []
            with torch.no_grad():
                for val_graph in val_graphs:
                    g = val_graph.to(device)
                    x = val_graph.ndata["feat"].to(device)

                    best_val_representations.append(model.encoder(g, x).cpu().detach().numpy())


            torch.save(model.cpu().state_dict(), 
                        "./data/models/dgi/dgi_{}_{}_{}_{}_{}_{}.bin".format(args.encoder,
                                                                args.num_hidden,
                                                                args.out_dim,
                                                                args.num_layers,
                                                                args.lr,
                                                                experiment_time)
                        )
        
        logging_dict = {}
        logging_dict['Loss/train'] = total_loss/len(train_graphs)
        logging_dict['Loss/val'] = total_val_loss/len(val_graphs)
        logger.note(logging_dict, step=epoch)

        if early_stopping_counter >= early_stopping:
            break
        else:
            early_stopping_counter += 1

    logger.finish()
    return train_loss, val_losses, best_val_representations


def setup_dgi_training(args, train_graphs, val_graph, dataset):
   
    current_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options = {
            "architecture":"DGI",
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
            "current_time": current_time,
            "device": device
    }

    logger = TBLogger(name="{}_{}".format(options['architecture'], current_time), entity="fdrewnowski", options=options)
    model = build_model(args, 'dgi')
    print(model.eval())
    #model= nn.DataParallel(model)
    model.to(device)
    dgi_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_stats, val_stats, best_representation = train_dgi(model, 
                                                            dgi_optimizer, 
                                                            train_graphs, 
                                                            val_graph, 
                                                            logger,
                                                            args,
                                                            current_time)

    with open("./data/training_data/dgi/dgi_{}_{}_{}_{}_{}_{}.pkl".format(args.encoder,
                                                            args.num_hidden,
                                                            args.out_dim,
                                                            args.num_layers,
                                                            args.lr,
                                                            current_time), 'wb') as handle:
        pickle.dump([train_stats, val_stats, best_representation], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    logging.info(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = 'polish_cities'
    directory = './data/raw_data/'
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
                                setup_dgi_training(args_object, train_graphs, val_graphs, dataset)
                            except Exception as e:
                                print(str(e))
                                print("FAILED for model DGI with {},{},{},{},{}".format(encoder,
                                                                                        layers,
                                                                                        num_out,
                                                                                        num_hidden,
                                                                                        lr))

    else:
        train_graphs, val_graphs = load_data(directory+dataset+".bin")
        setup_dgi_training(args, train_graphs, val_graphs, dataset)

    #error log file 

    