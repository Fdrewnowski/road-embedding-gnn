
import logging
import datetime
import torch
import torch.nn as nn
from datetime import datetime
from models import GraphMAE, build_model
import pickle
import os
from models.load_data import load_data, load_train_and_val_data
from tqdm import tqdm
from torch_geometric.utils import negative_sampling
from utils import (TBLogger, build_args, create_optimizer,
                            get_current_lr, load_best_configs, ArgParser)

from params import ENCODER, ACTIVATION, NUM_HIDDEN, NUM_OUT, NUM_LAYERS, IN_DROP, OPTIMIZER, WEIGHT_DECAY, LR

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def train_gae(model, optimizer, train_graphs, val_graphs, logger, args, experiment_time, device):

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
        for idx, train_graph in enumerate(train_graphs):

            g = train_graph.to(device)
            x = train_graph.ndata['feat'].to(device)

            adj = g.adjacency_matrix().coalesce().indices()
            #adj = adj[graph.ndata['train']]

            model.train()
            optimizer.zero_grad()
            z = model.encode(g, x)
            loss = model.recon_loss(z, adj)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # eval train set
            model.eval()
            with torch.no_grad():
                z = model.encode(g, x)
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
                g = val_graph.to(device)
                x = val_graph.ndata['feat'].to(device)
                val_adj = g.adjacency_matrix().coalesce().indices()

                z = model.encode(g, x)
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

            try:
                torch.save(model.cpu().state_dict(),
                            "./data/models/gae/gae_{}_{}_{}_{}_{}_{}.bin".format(args.encoder,
                                                                    args.num_hidden,
                                                                    args.out_dim,
                                                                    args.num_layers,
                                                                    args.lr,
                                                                    experiment_time)
                            )
            except Exception as e:
                print(str(e))

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

    logger = TBLogger(name="{}_{}".format(options['architecture'], current_time), entity="fdrewnowski", options=options)
    model = build_model(args, 'gae')
    print(model.eval())
    #model= nn.DataParallel(model)

    model.to(device)
    dgi_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_stats, val_stats, best_representation = train_gae(model, 
                                                            dgi_optimizer, 
                                                            train_graphs, 
                                                            val_graph, 
                                                            logger,
                                                            args,
                                                            current_time,
                                                            device)

    with open("./data/training_data/gae/gae_{}_{}_{}_{}_{}_{}.pkl".format(args.encoder,
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

    if not os.path.exists("./data/models/gae"):
        os.makedirs("./data/models/gae")
    if not os.path.exists("./data/training_data/gae"):
        os.makedirs("./data/training_data/gae")

    # open data
    dataset = 'polish_cities'
    dataset_dir = './data/raw_data/'
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
                                setup_gae_training(args_object, train_graphs, val_graphs, dataset)
                            except Exception as e:
                                print(str(e))
                                print("FAILED for model GAE with {},{},{},{},{}".format(encoder,
                                                                                        layers,
                                                                                        num_out,
                                                                                        num_hidden,
                                                                                        lr))

    else:
        train_graphs, val_graphs = load_data(dataset_dir+dataset+".bin")
        # init model
        setup_gae_training(args, train_graphs, val_graphs, dataset)