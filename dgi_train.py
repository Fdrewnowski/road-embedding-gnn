import argparse
import logging
import time
from models import build_model

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from utils import build_args, load_best_configs, TBLogger
from data.load_data import load_data
from datetime import datetime
from tqdm import tqdm 
# train deep graph infomax
def train_dgi(model, optimizer, train_graphs, val_graphs, logger, epochs):
    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []
    best_DGI = []
    results_loss = []
    train_loss = [[],[],[],[],[],[],[],[],[],[]]

    result_val_loss = []
    best_representation = []
    experiment_time = datetime.now().strftime('%Y_%m_%d_%H_%M')

    for epoch in tqdm(range(epochs), total=epochs):
        if epoch >= 3:
            t0 = time.time()

        total_loss = 0
        

        for idx, g in enumerate(train_graphs):
            features = g.ndata["feat"]
            n_edges = g.number_of_edges()

            # add self loop
            #if args.self_loop:
            #    g = dgl.remove_self_loop(g)
            #    g = dgl.add_self_loop(g)
            n_edges = g.number_of_edges()  



            # training
            model.train()
            loss = model(g, features)
            total_loss+=loss

            train_loss[idx].append(loss.item())

            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = model(val_graphs, val_graphs.ndata['feat'])

        if val_loss < best:
            best = val_loss
            best_t = epoch
            with torch.no_grad():
                best_representation = model.encoder(val_graphs, val_graphs.ndata['feat'])
            
            torch.save(model.state_dict(), "data/models/dgi_{}.bin".format(experiment_time))

        if epoch >= 3:
            dur.append(time.time() - t0)

        # print(
        #     "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
        #     "ETputs(KTEPS) {:.2f}".format(
        #         epoch,
        #         np.mean(dur),
        #         total_loss.item(),
        #         n_edges / np.mean(dur) / 1000,
        #     )
        # )
        logging_dict = {}
        logging_dict['Loss/train'] = np.mean(total_loss.item())
        logging_dict['Loss/val'] = val_loss.item()
        logger.note(logging_dict, step=epoch)
        results_loss.append(total_loss.item())
        result_val_loss.append(val_loss.item())
    logger.finish()
    return train_loss, result_val_loss, best_representation

if __name__ == '__main__':
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    logging.info(args)
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'polish_cities'
    directory = './data/raw_data/'
    train_graphs, val_graph = load_data(directory+dataset+".bin")
    options = {
            "architecture":"DGI",
            "encoder_type":args.encoder,
            "hidden_space": args.num_hidden
    }
    logger = TBLogger(name="{}_{}".format(dataset, current_time), entity="fdrewnowski", options=options)
    model = build_model(args, 'dgi')
    print(model.eval())
    dgi_optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    train_stats, val_stats, best_representation = train_dgi(model, 
                                                            dgi_optimizer, 
                                                            train_graphs, 
                                                            val_graph, 
                                                            logger,
                                                            epochs=100)

    import pickle
    with open("./data/raw_data/dgi_{}.pkl".format(datetime.now().strftime('%Y_%m_%d_%H_%M')), 'wb') as handle:
        pickle.dump([train_stats, val_stats, best_representation], handle, protocol=pickle.HIGHEST_PROTOCOL)