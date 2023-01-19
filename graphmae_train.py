import copy
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from dgl.data.utils import load_graphs
from dgl.heterograph import DGLHeteroGraph
from tqdm import tqdm
from data.load_data import load_data

#from bikeguessr_transform import DATA_OUTPUT, _sizeof_fmt, load_transform_dir_bikeguessr
#from graphmae.evaluation import (LogisticRegression, f1,
#                                 node_classification_evaluation, recall)
from models import GraphMAE, build_model
from utils import (TBLogger, build_args, create_optimizer,
                            get_current_lr, load_best_configs, set_random_seed)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def load_bikeguessr_dataset(filepath: str) -> Tuple[List[DGLHeteroGraph], Tuple[int, int]]:
    logging.info('load bikeguessr dataset')
    if filepath is None:
        filepath = str(Path(DATA_OUTPUT, 'bikeguessr.bin'))
    file = Path(filepath)

    logging.info('processing: ' + str(file.absolute()) +
                 ' size: ' + _sizeof_fmt(os.path.getsize(file)))
    graphs, _ = load_graphs(str(file))
    num_features, num_classes = [], []
    for i in range(len(graphs)):
        graphs[i] = graphs[i].remove_self_loop()
        graphs[i] = graphs[i].add_self_loop()
    num_features = graphs[i].ndata["feat"].shape[1]
    num_classes = 2

    return graphs, (num_features, num_classes)


def train_transductive(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    logging.info("using device: {}".format(device))
    seeds = args.seeds
    dataset_name = args.dataset
    dataset_path = args.path
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    eval_epoch = args.eval_epoch
    eval_repeats = args.eval_repeats

    optim_type = args.optimizer

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = True
    logs = args.logging
    use_scheduler = args.scheduler
    targets = args.targets


    train_graphs, val_graphs = load_data('./data/raw_data/polish_cities.bin')



    num_features = train_graphs[0].ndata["feat"].shape[1]
    num_classes = 2
    args.num_features = num_features

    for i, seed in enumerate(seeds):
        logging.info(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            # logger = TBLogger(
            #    name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
            options = {
            "architecture":"GraphMAE",
            "encoder_type":args.encoder,
            "hidden_space": args.num_hidden
            }
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            logger = TBLogger(name=f"{dataset_name}_{current_time}", options=options)
        else:
            logger = None

        model = build_model(args, 'graphmae')
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
        if load_model:
            logging.info("Loading Model ... ")
            model = torch.load("sgmae.model")
        else:
            gae_full, gae_early_stopping = \
                pretrain(model, train_graphs, X, optimizer, max_epoch, device, scheduler,
                         num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger,
                         eval_epoch, eval_repeats, val_graphs = val_graphs)
            model = gae_early_stopping
            if save_model:
                logging.info("Saving trained Graph Masked Auto Encoders...")
                torch.save(gae_full.cpu(), "./data/models/gmae_{}.model".format(current_time))
                #torch.save(gae_early_stopping, "./data/models/gmae_early_stopping_{}.model".format(current_time))
            # logging.info(
            #     f'early stopping f1 score on test (pretrain): {f1_early_stopping}')
            # logging.info(
            #     f'end of training f1 score on test (pretrain): {f1_full}')

        model = model.to(device)
        #logger.watch_model(model)

        model.eval()

        # if load_model:
        #     logging.info(f'early stopping f1 score on test: {f1_early_stopping}')
        #     logging.info(f'end of training f1 score on test: {f1_full}')

        # best_model = sgmae_full if f1_score_early_stopping > f1_score_eot[2] else sgmae_full_eot
        #best_clf = train_clf(model, graphs, X, num_classes, lr_f,
        #                     weight_decay_f, 200, device, logger)
        #full_model = SGMAE(model, best_clf)
        # full_model = model
        # if save_model:
        #     logging.info("Saving Model ...")
        #     torch.save(full_model.cpu(), "./data/models/gmae_{}.model".format(current_time))

        if logger is not None:
            logger.finish()

    return model


# def train_clf(
#         gmae: GraphMAE,
#         graphs: List[DGLHeteroGraph],
#         X: List[torch.Tensor],
#         num_classes: int,
#         lr_f: float,
#         weight_decay_f: float,
#         max_epoch_f: float,
#         device: torch.device,
#         logger: TBLogger = None,
#         mute: bool = False) -> Union[torch.nn.Module, None]:
#     gmae.eval()

#     with torch.no_grad():
#         for i in range(len(X)):
#             X[i] = gmae.embed(graphs[i].to(device), X[i].to(device))
#         in_feat = X[i].shape[1]
#     clf = LogisticRegression(in_feat, num_classes)

#     num_finetune_params = [p.numel()
#                            for p in clf.parameters() if p.requires_grad]
#     if not mute:
#         logging.info(
#             f"num parameters for finetuning: {sum(num_finetune_params)}")

#     clf.to(device)
#     optimizer_f = create_optimizer("adam", clf, lr_f, weight_decay_f)

#     # iterate
#     criterion = torch.nn.CrossEntropyLoss()
#     if not mute:
#         epoch_iter = tqdm(range(max_epoch_f))
#     else:
#         epoch_iter = range(max_epoch_f)

#     best_val_f1 = 0
#     best_clf_f1 = None
#     for epoch in epoch_iter:
#         e_f1_scores = [], [], []
#         e_recall_scores = [], [], []
#         e_losses = []
#         for graph, x in zip(graphs, X):
#             graph = graph.to(device)
#             x = x.to(device)

#             train_mask = graph.ndata["train_mask"]
#             val_mask = graph.ndata["val_mask"]
#             test_mask = graph.ndata["test_mask"]
#             labels = graph.ndata["label"]

#             clf.train()
#             out = clf(graph, x)
#             loss = criterion(out[train_mask], labels[train_mask])
#             optimizer_f.zero_grad()
#             loss.backward()
#             # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
#             optimizer_f.step()

#             with torch.no_grad():
#                 clf.eval()
#                 pred = clf(graph, x)
#                 # metrics
#                 e_f1_scores[0].append(f1(pred[test_mask], labels[test_mask]))
#                 e_f1_scores[1].append(f1(pred[val_mask], labels[val_mask]))
#                 e_f1_scores[2].append(f1(pred[train_mask], labels[train_mask]))

#                 e_recall_scores[0].append(
#                     recall(pred[test_mask], labels[test_mask]))
#                 e_recall_scores[1].append(
#                     recall(pred[val_mask], labels[val_mask]))
#                 e_recall_scores[2].append(
#                     recall(pred[train_mask], labels[train_mask]))

#                 e_losses.append(
#                     criterion(pred[train_mask], labels[train_mask]).cpu())

#         test_f1 = np.mean(e_f1_scores[0])
#         val_f1 = np.mean(e_f1_scores[1])
#         if not mute:
#             with torch.no_grad():
#                 np.mean(e_losses)
#                 epoch_iter.set_description(
#                     f"# Epoch: {epoch}, train_loss:{np.mean(e_losses): .4f}, val_f1:{val_f1}, test_f1:{test_f1: .4f}")

#         if val_f1 >= best_val_f1:
#             best_val_f1 = val_f1
#             best_val_f_epoch = epoch
#             best_clf_f1 = copy.deepcopy(clf)

#         if logger is not None:
#             logging_dict = {}
#             logging_dict['Clf/Loss/train'] = np.mean(e_losses)
#             logging_dict['Clf/F1/test'] = np.mean(e_f1_scores[0])
#             logging_dict['Clf/F1/val'] = np.mean(e_f1_scores[1])
#             logging_dict['Clf/F1/train'] = np.mean(e_f1_scores[2])
#             logger.note(logging_dict, step=epoch)

#     best_clf_f1.eval()
#     try:
#         with torch.no_grad():
#             test_f1_early_stopping = []
#             for graph, x in zip(graphs, X):
#                 pred = best_clf_f1(graph, x)
#                 test_f1_early_stopping.append(f1(pred[test_mask], labels[test_mask]))
#             test_f1_early_stopping = np.mean(test_f1_early_stopping)
#     except Exception as e:
#         logging.info(
#             f"Best ValF1: {best_val_f1:.4f} in epoch {best_val_f_epoch} --- ")
#         return best_clf_f1

#     if not mute:
#         logging.info(
#             f"--- TestF1: {test_f1:.4f}, early-stopping-TestF1: {test_f1_early_stopping:.4f}," +
#             f"Best ValF1: {best_val_f1:.4f} in epoch {best_val_f_epoch} --- ")

#     return best_clf_f1


def _is_same_model(model: GraphMAE, other_model: GraphMAE):
    try:
        for p1, p2 in zip(model.parameters(), other_model.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                return False
        return True
    except:
        return False


def pretrain(model: GraphMAE,
             graphs: List[DGLHeteroGraph],
             feats: List[torch.Tensor],
             optimizer: torch.optim.Optimizer,
             max_epoch: int,
             device: torch.device,
             scheduler: torch.optim.lr_scheduler.LambdaLR,
             num_classes: int,
             lr_f: float,
             weight_decay_f: float,
             max_epoch_f: int,
             linear_prob: bool,
             logger: TBLogger = None,
             eval_epoch: int = 10,
             eval_repeats: int = 5,
             val_graphs = None) -> Tuple[GraphMAE, GraphMAE, Tuple[float]]:
    logging.info("start training..")
    epoch_iter = tqdm(range(max_epoch))
    best_val_f1_score, best_test_f1_score = 0.0, 0.0
    best_gae = None
    best_loss = 1e10

    for epoch in epoch_iter:
        e_f1_test, e_f1_val, e_f1_train = [], [], []
        e_recall_test, e_recall_val, e_recall_train = [], [], []
        epoch_train_loss = []
        epoch_val_loss = []
        total_loss = 0
        # City-batch training
        for graph, feat in zip(graphs, feats):

            g = graph.to(device)
            x = feat.to(device)
            model.train()

            loss, _ = model(g, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

            epoch_iter.set_description(
                f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
            epoch_train_loss.append(loss.cpu().detach().numpy())

        # Every eval_epoch city-batch evaluation of gmae
        total_val_loss = 0
        model.eval()
        epoch_iter.set_description(
            f"# Epoch {epoch}: evaluating: {loss.item():.4f}")
        with torch.no_grad():
            #for val_g in val_graphs:
            loss, _ = model(val_graphs, val_graphs.ndata['feat'])
            epoch_val_loss.append(loss.cpu().detach().numpy())
            total_val_loss += loss.cpu().detach().numpy()
        if total_val_loss < best_loss:
            best_loss = total_val_loss
            torch.save(model.state_dict(), "./data/models/gmae.model")
                

            # for graph, feat in tqdm(zip(graphs, feats), total=len(graphs)):
            #     g = graph.to(device)
            #     x = feat.to(device)
            #     g_f1_test, g_f1_val, g_f1_train = [], [], []
            #     g_recall_test, g_recall_val, g_recall_train = [], [], []
            #     #logging.info(f'start evaluating gmae at epoch {epoch}')
            #     for _ in range(eval_repeats):
            #         _, scores = node_classification_evaluation(
            #             model, g, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)
            #         g_f1_test.append(scores['F1/Test'])
            #         g_f1_val.append(scores['F1/Val'])
            #         g_f1_train.append(scores['F1/Train'])
            #         g_recall_test.append(scores['Recall/Test'])
            #         g_recall_val.append(scores['Recall/Val'])
            #         g_recall_train.append(scores['Recall/Train'])
            #     e_f1_test.append(np.mean(g_f1_test))
            #     e_f1_val.append(np.mean(g_f1_val))
            #     e_f1_train.append(np.mean(g_f1_train))
            #     e_recall_test.append(np.mean(g_recall_test))
            #     e_recall_val.append(np.mean(g_recall_val))
            #     e_recall_train.append(np.mean(g_recall_train))
            #     if best_val_f1_score < np.mean(e_f1_val):
            #         best_val_f1_score = np.mean(e_f1_val)
            #         best_test_f1_score = np.mean(e_f1_test)
            #         torch.save(model, "sgmae_early_stopping.pt")
            #         best_gae = torch.load("sgmae_early_stopping.pt")

        if logger is not None:
            logging_dict = {}
            logging_dict['Loss/train'] = total_loss/len(graphs)
            logging_dict['Loss/val'] = total_val_loss

            # if (epoch + 1) % eval_epoch == 0:
            #     logging_dict['F1/test'] = np.mean(e_f1_test)
            #     logging_dict['F1/val'] = np.mean(e_f1_val)
            #     logging_dict['F1/train'] = np.mean(e_f1_train)
            #     logging_dict['Recall/test'] = np.mean(e_recall_test)
            #     logging_dict['Recall/val'] = np.mean(e_recall_val)
            #     logging_dict['Recall/train'] = np.mean(e_recall_train)
            logger.note(logging_dict, step=epoch)

    #early_stopping_f1_score = best_test_f1_score
    full_f1_score = np.mean(e_f1_test)

    if _is_same_model(model, best_gae):
        logging.warn('Best model and currently trained model are identical')

    return model, best_gae#, (early_stopping_f1_score, full_f1_score)


if __name__ == '__main__':
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    logging.info(args)
    train_transductive(args)
