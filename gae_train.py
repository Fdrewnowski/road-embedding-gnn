
import logging
import datetime
import torch
from datetime import datetime
from models import GraphMAE, build_model
import pickle

from models.load_data import load_data
from tqdm import tqdm
from torch_geometric.utils import negative_sampling
from utils import (TBLogger, build_args, create_optimizer,
                            get_current_lr, load_best_configs, set_random_seed)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def train_gae(model, optimizer, train_graphs, val_graphs, logger, args, experiment_time):

    train_loss = [[] for i in range(0, len(train_graphs))]
    train_auc = [[] for i in range(0, len(train_graphs))]
    train_ap = [[] for i in range(0, len(train_graphs))]

    val_loss = [[] for i in range(0, len(val_graphs))]
    val_auc = [[] for i in range(0, len(val_graphs))]
    val_ap = [[] for i in range(0, len(val_graphs))]
    
    best_representation = []
    
    best_l = 1e9
    early_stopping = args.max_epoch_f
    early_stopping_counter = 0

    for epoch in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch):
        logging_dict = {}
        total_loss = 0
        total_auc = 0
        total_ap = 0
        total_val_loss = 0
        total_val_auc = 0
        total_val_ap = 0

        # train model
        for idx, graph in enumerate(train_graphs):
            adj = graph.adjacency_matrix().coalesce().indices()
            #adj = adj[graph.ndata['train']]
            features = graph.ndata['feat']
            model.train()
            optimizer.zero_grad()
            z = model.encode(graph, features)
            loss = model.recon_loss(z, adj)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # eval train set
            model.eval()
            with torch.no_grad():
                z = model.encode(graph, features)
                neg_edge_index = negative_sampling(adj, z.size(0))
            auc, ap = model.test(z, adj,neg_edge_index)

            total_auc += auc
            total_ap += ap

            train_loss[idx].append(loss.item())
            train_auc[idx].append(auc.item())
            train_ap[idx].append(ap.item())

        logging_dict['Loss/train'] = total_loss/len(train_graphs)
        logging_dict['AUC/train'] = total_auc/len(train_graphs)
        logging_dict['AP/train'] = total_ap/len(train_graphs)

        # evaluate validation set
        val_representations = []
        model.eval()
        with torch.no_grad():
            for idx, val_graph in enumerate(val_graphs):
                val_adj = val_graph.adjacency_matrix().coalesce().indices()
                val_features = val_graph.ndata['feat']

                z = model.encode(val_graph, val_features)
                val_neg_edge_index = negative_sampling(val_adj, z.size(0))
                loss = model.recon_loss(z, val_adj, val_neg_edge_index)
                total_val_loss += loss.item()
                auc, ap = model.test(z, val_adj, val_neg_edge_index)

                total_val_auc += auc.item()
                total_val_ap += ap.item()
                val_representations.append(z.cpu().detach().numpy())

                val_loss[idx].append(loss.item())
                val_auc[idx].append(auc.item())
                val_ap[idx].append(ap.item())

        logging_dict['Loss/val'] = total_val_loss/len(val_graphs)
        logging_dict['AUC/val'] = total_val_auc/len(val_graphs)
        logging_dict['AP/val'] = total_val_ap/len(val_graphs)
        logger.note(logging_dict, step=epoch)
        
        if total_val_loss < best_l:
            best_l = total_val_loss
            early_stopping_counter = 0
            best_representation = val_representations

            torch.save(model.cpu().state_dict(),
                        "data/models/gae_{}_{}_{}_{}_{}.bin".format(args.encoder,
                                                                args.num_hidden,
                                                                args.out_dim,
                                                                args.num_layers,
                                                                experiment_time)
                        )

        if early_stopping_counter >= early_stopping:
            break
        else:
            early_stopping_counter += 1

    logger.finish()
    return ([train_loss, train_auc, train_ap], [val_loss, val_auc, val_ap], best_representation)


def setup_gae_training(args, train_graphs, val_graph, dataset):

    current_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    options = {
            "architecture":"GAE",
            "encoder_type":args.encoder,
            "hidden_dim": args.num_hidden,
            "out_dim": args.out_dim,
            "layers": args.num_layers,
            "in_drop": args.in_drop,
            "optimizer": "adam",
            "dataset": dataset
    }

    logger = TBLogger(name="{}_{}".format(options['architecture'], current_time), entity="fdrewnowski", options=options)
    model = build_model(args, 'gae')
    print(model.eval())
    dgi_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_stats, val_stats, best_representation = train_gae(model, 
                                                            dgi_optimizer, 
                                                            train_graphs, 
                                                            val_graph, 
                                                            logger,
                                                            args,
                                                            current_time)

    with open("./data/training_data/gae_{}_{}_{}_{}_{}.pkl".format(args.encoder,
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

    # open data
    dataset = 'polish_cities'
    dataset_dir = './data/raw_data/'
    train_graphs, val_graphs = load_data(dataset_dir+dataset+'.bin')

    # init model
    setup_gae_training(args, train_graphs, val_graphs, dataset)