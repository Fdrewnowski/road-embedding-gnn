
import logging
import datetime
import torch
import torch.nn as nn
from models import GraphMAE, build_model

from data.load_data import load_data
from tqdm import tqdm
from torch_geometric.utils import negative_sampling
from utils import (TBLogger, build_args, create_optimizer,
                            get_current_lr, load_best_configs, set_random_seed)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def train_gae(model, optimizer, train_graphs, val_graph,epochs=10):
    val_adj = val_graph.adjacency_matrix().coalesce().indices()
    val_features = val_graph.ndata['feat']

    train_loss = [[],[],[],[],[],[],[],[],[],[]]
    train_auc = [[],[],[],[],[],[],[],[],[],[]]
    train_ap = [[],[],[],[],[],[],[],[],[],[]]

    val_loss = []
    val_auc = []
    val_ap = []
    
    best_l = 1e9
    
    best_representation = None

    for epoch in tqdm(range(1, epochs + 1), total=epochs + 1):
        total_loss = 0
        for idx, graph in enumerate(train_graphs):
            adj = graph.adjacency_matrix().coalesce().indices()
            #adj = adj[graph.ndata['train']]
            features = graph.ndata['feat']
            model.train()
            optimizer.zero_grad()
            z = model.encode(graph, features)
            loss = model.recon_loss(z, adj)
            #if args.variational:
            #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            total_loss += loss
            

            model.eval()
            with torch.no_grad():
                z = model.encode(graph, features)
                neg_edge_index = negative_sampling(adj, z.size(0))
            auc, ap = model.test(z, adj,neg_edge_index)

            train_loss[idx].append(loss)
            train_auc[idx].append(auc)
            train_ap[idx].append(ap)

        model.eval()
        with torch.no_grad():
            z = model.encode(val_graph, val_features)
            val_neg_edge_index = negative_sampling(val_adj, z.size(0))
            loss = model.recon_loss(z, val_adj, val_neg_edge_index)
        auc, ap = model.test(z, val_adj, val_neg_edge_index)
        
        if loss < best_l:
            best_l = loss
            best_t = epoch
            best_representation = z
                            
            torch.save(model.state_dict(), "data/models/gae_{}.bin".format(datetime.now().strftime('%Y_%m_%d_%H_%M')))
            #else:
            #    cnt_wait += 1
        
        val_loss.append(loss.item())
        val_auc.append(auc)
        val_ap.append(ap)
    return (model, [train_loss, train_auc, train_ap], [val_loss, val_auc, val_ap], best_representation)

if __name__ == '__main__':
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    logging.info(args)

    train_graphs, val_graph = load_data('./data/raw_data/polish_cities.bin')
    model = build_model(args, 'gae')
    print(model.eval())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    last_model, train_stats, val_stats, best_representation = train_gae(model, optimizer, train_graphs, val_graph,epochs=100)

    import pickle
    with open('./data/raw_data/GAE.pkl', 'wb') as handle:
        pickle.dump([train_stats, val_stats, best_representation], handle, protocol=pickle.HIGHEST_PROTOCOL)