import argparse
import logging
import time
from models import build_model
import pickle

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from utils import build_args, load_best_configs, TBLogger
from models.load_data import load_data
from datetime import datetime
from tqdm import tqdm

# train deep graph infomax
def train_dgi(model, optimizer, train_graphs, val_graphs, logger, args, experiment_time):
    
    best = 1e9
    best_val_representations = []
    train_loss = [[] for i in range(0, len(train_graphs))]
    val_losses = [[] for i in range(0, len(val_graphs))]
    early_stopping = args.max_epoch_f
    early_stopping_counter = 0

    for epoch in tqdm(range(args.max_epoch), total=args.max_epoch):
        total_loss = 0
        
        # model training
        for idx, g in enumerate(train_graphs):
            features = g.ndata["feat"]

            # training
            model.train()
            loss = model(g, features)
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
                val_loss = model(val_graph, val_graph.ndata['feat'])
                val_losses[idx].append(val_loss.item())
                total_val_loss += val_loss.item()

        # save best model based on total_val_loss
        if total_val_loss < best:
            best = total_val_loss
            early_stopping_counter = 0
            best_val_representations = []
            with torch.no_grad():
                for val_graph in val_graphs:
                    best_val_representations.append(model.encoder(val_graph,
                                                                  val_graph.ndata['feat']).cpu()
                                                    )

            torch.save(model.cpu().state_dict(), 
                        "data/models/dgi_{}_{}_{}_{}_{}.bin".format(args.encoder,
                                                                args.num_hidden,
                                                                args.out_dim,
                                                                args.num_layers,
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
    
    options = {
            "architecture":"DGI",
            "encoder_type":args.encoder,
            "hidden_dim": args.num_hidden,
            "out_dim": args.out_dim,
            "layers": args.num_layers,
            "in_drop": args.in_drop,
            "optimizer": "adam"
    }

    logger = TBLogger(name="{}_{}".format(dataset, current_time), entity="fdrewnowski", options=options)
    model = build_model(args, 'dgi')
    print(model.eval())
    dgi_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_stats, val_stats, best_representation = train_dgi(model, 
                                                            dgi_optimizer, 
                                                            train_graphs, 
                                                            val_graph, 
                                                            logger,
                                                            args,
                                                            current_time)

    with open("./data/training_data/dgi_{}_{}_{}_{}_{}.pkl".format(args.encoder,
                                                            args.num_hidden,
                                                            args.out_dim,
                                                            args.num_layers,
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
    train_graphs, val_graphs = load_data(directory+dataset+".bin")


    
    setup_dgi_training(args, train_graphs, val_graphs, dataset)