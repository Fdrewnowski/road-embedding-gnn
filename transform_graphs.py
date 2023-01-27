import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import dgl
import osmnx as ox
import torch
from dgl.data.utils import save_graphs
from dgl.heterograph import DGLHeteroGraph
from networkx.classes.multidigraph import MultiDiGraph
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

from scipy.sparse import coo_matrix
import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


SELECTED_KEYS = ['oneway', 'lanes', 'highway', 'maxspeed',
                 'access', 'bridge', 'junction', 'width', 'service', 
                 'tunnel', 'label', 'idx', 'busway',
                 'cycleway', 'bicycle', 'surface', 'psv', 'sidewalk']
DEFAULT_VALUES = {'oneway': False, 'lanes': 2, 'highway': 11, 'maxspeed': 50,
                  'length': 0, 'access': 6, 'bridge': 0, 'junction': 0,
                  'width': 2.0, 'service': 0, 'tunnel': 0, 'surface': 13,
                  'psv': 0, 'sidewalk': 0, 'bicycle':0, 'cycleway': 0, 'busway':0}
HIGHWAY_CODING = {'highway': {'primary': 0, 'unclassified': 1, 'tertiary_link': 8, 'secondary': 3,
                              'residential': 4, 'track': 5, 'service': 6, 'trunk': 7, 'tertiary': 8,
                              'primary_link': 0, 'pedestrian': 10, 'path': 11, 'living_street': 12,
                              'trunk_link': 7, 'cycleway': 14, 'bridleway': 15, 'secondary_link': 3},
                  'access': {'customers': 0, 'delivery': 1, 'designated': 2, 'destination': 3,
                             'emergency': 4, 'military': 5, 'no': 6, 'permissive': 7, 'permit': 8, 'yes': 9},
                  'bridge': {'1': 1, 'viaduct': 1, 'yes': 1, 'trestle':1, 'movable': 1, 'cantilever':1, 'boardwalk':1, 'aqueduct':1},
                  'junction': {'yes': 1, 'roundabout': 2, 'y_junction': 3, 'circular':2},
                  'tunnel': {'yes': 1, 'building_passage': 2, 'passage': 3},
                  'service': {'alley': 1, 'bus': 2, 'drive-through': 3, 'driveway': 4,
                              'emergency_access': 5, 'ground': 6, 'parking_aisle': 7, 'spur': 8},
                   'bicycle':{ 'no': 0, 'dismount': 1, 'official': 2, 'yes': 3,
                                'permissive': 4, 'designated': 2, 'use_sidepath': 6},
                   'cycleway':{'no': 0, 'share_busway': 1, 'link': 2, 'shared_lane': 3,
                                'yes': 4, 'separate': 5, 'opposite_lane': 6, 'opposite': 6,
                                'lane': 7, 'crossing': 8, 'asl':9
                                },
                    'busway':{'no':0, 'lane':1, 'right':1, 'left':1, 'both':2, 'opposite_lane':1},
                   'surface': {'earth': 0, 'grass_paver': 1,  'fine_gravel': 2, 'concrete': 3,
                                'gravel': 4, 'ground': 5, 'unpaved': 6, 'grass': 7, 'pebblestone': 8,
                                'compacted': 9, 'wood': 10, 'concrete:plates': 11, 'metal': 12, 'unhewn_cobblestone': 23,
                                'paved': 14,  'rock': 15, 'dirt': 16, 'asphalt': 17, 'concrete:lanes': 11, 'sett;concrete': 19,
                                'sand': 20, 'paving_stones': 21, 'sett': 22, 'cobblestone': 23, 'cheapseal':18, 'mud':24, 'woodchips':25, 'cheapseal':26},
                   'psv': {'designated': 1, 'yes': 1, 'no':0},
                   'sidewalk': {'both':2, 'left':1, 'no':0, 'right':1, 'separate':1}}

CONTINOUS_KEYS = ['maxspeed', 'width', 'lanes']

CATEGORIES = [  [False, True], #oneway
                [0,1,3,4,5,6,7,8,10,11,12,14,15], #highway
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], #access
                [0, 1], #bridge
                [0,1,2,3], #junction
                [0,1,2,3,4,5,6,7,8], #service
                [0,1,2],#tunnel
                [0,1,2],#busway
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],#cycleway
                [0, 1, 2, 3, 4, 6],#bicycle
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26],#surface
                [0,1],#psv
                [0,1,2]#sidewalk
            ]
DATA_INPUT = 'data_raw'
DATA_OUTPUT = 'data_transformed'


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='bikeguessr_transform')
    data_to_transform = parser.add_mutually_exclusive_group(required=True)
    data_to_transform.add_argument('-a', '--all', action='store_true',
                                   help='Transform all graphml files from either default folder or specified path')
    data_to_transform.add_argument('-s', '--single', action='store_true',
                                   help='Transform single graphml files from either default folder or specified path')
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='Path where to look for graphml file/files')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Path where transformed data will be stored')
    parser.add_argument('-t', '--targets', nargs='+', default=None,
                        help='Path or paths of graphml files which should be ignored when using -a option so that train and test data are split')
    parser.add_argument('-tr', '--training', nargs='+', default=None,
                        help='Path to directory with training graphs')
    parser.add_argument('-val', '--validation', nargs='+', default=None,
                        help='Path to directory with validation graphs')
    
    return parser.parse_args()


def load_transform_dir_bikeguessr(directory: str = None, save: bool = True, output: str = None, targets: List[str] = None) -> List[DGLHeteroGraph]:
    logging.info('load bikeguessr directory')
    if directory is None:
        directory = DATA_INPUT
    found_files = list(Path(directory).glob('*.xml'))
    print(found_files)
    graphs = []
    for path in tqdm(found_files):
        if targets is not None:
            is_target = False
            for target in targets:
                if Path(target) == path:
                    logging.info('skipping target city' + target)
                    is_target = True
                    break
            if is_target:
                continue
        logging.info('processing: ' + str(path.stem) +
                     ' size: ' + _sizeof_fmt(os.path.getsize(path)))
        graph = load_transform_single_bikeguessr(path, False)
        graphs.append(graph)
    logging.info('merging bikeguessr graphs')
    if output is None:
        output = Path(DATA_OUTPUT, 'bikeguessr.bin')
    else:
        output = Path(output)
    if save:
        save_bikeguessr(output, graphs)
    logging.info('end load bikeguessr directory')
    return graphs


def load_transform_single_bikeguessr(path: str, save: bool = True, output: str = None) -> DGLHeteroGraph:
    logging.debug('load single bikeguessr')
    bikeguessr_linegraph = _load_transform_linegraph(path)
    bikeguessr_linegraph_with_masks, _ = _create_mask(
        bikeguessr_linegraph)
    if output is None:
        output = Path(DATA_OUTPUT, 'bikeguessr.bin')
    else:
        output = Path(output)
    if save:
        save_bikeguessr(output, bikeguessr_linegraph_with_masks)
    logging.debug('end load single bikeguessr')
    return bikeguessr_linegraph_with_masks


def save_bikeguessr(output: Path, graph: DGLHeteroGraph) -> None:
    logging.info('saving bikeguessr graph')
    save_graphs(str(output), graph)


def _load_transform_linegraph(path: str) -> DGLHeteroGraph:
    raw_graphml = ox.io.load_graphml(path)
    encoded_graphml = _encode_data(raw_graphml)
    return _convert_nx_to_dgl_as_linegraph(encoded_graphml)


def _create_mask(graph: DGLHeteroGraph, split_type: str = 'stratified') -> Tuple[DGLHeteroGraph, Tuple[int, int]]:
    num_nodes = graph.num_nodes()

    if split_type == 'stratified':
        split_idx = _get_stratified_split(graph.ndata['label'])
    elif split_type == 'random':
        split_idx = _get_random_split(num_nodes)
    else:
        raise NotImplementedError()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = _preprocess(graph)

    if not torch.is_tensor(train_idx):
        train_idx = torch.as_tensor(train_idx)
        val_idx = torch.as_tensor(val_idx)
        test_idx = torch.as_tensor(test_idx)

    feat = graph.ndata["feat"]
    #feat, scaler = _scale_feats(feat)
    graph.ndata["feat"] = feat

    train_mask = torch.full(
        (num_nodes,), False).index_fill_(0, train_idx, True)
    val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
    test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
    graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    num_features = graph.ndata["feat"].shape[1]
    num_classes = 2
    return graph, (num_features, num_classes)


def _encode_data(graph_nx: MultiDiGraph, selected_keys: List = SELECTED_KEYS, default_values: Dict = DEFAULT_VALUES, onehot_key: Dict = HIGHWAY_CODING) -> MultiDiGraph:
    graph_nx_copy = graph_nx.copy()
    for edge in graph_nx.edges():
        for connection in graph_nx[edge[0]][edge[1]].keys():
            graph_edge = graph_nx_copy[edge[0]][edge[1]][connection]
            for key in selected_keys:
                # decide if key exists if not create
                if key in graph_edge.keys():
                    # if value of edge key is a list take first element
                    if type(graph_edge[key]) == list:
                        graph_edge[key] = graph_edge[key][0]

                    if key in onehot_key.keys():
                        if graph_edge[key] in onehot_key[key].keys():
                            graph_edge[key] = onehot_key[key][graph_edge[key]]
                        else:
                            if key in default_values.keys():
                                graph_edge[key] = default_values[key]
                            else:
                                graph_edge[key] = 0

                    if type(graph_edge[key]) == str:
                        try:
                            graph_edge[key] = float(graph_edge[key])
                        except ValueError as e:
                            graph_edge[key] = 0.0

                else:
                    # create key with default values or set to 0
                    if key in default_values.keys():
                        graph_edge[key] = default_values[key]
                    else:
                        graph_edge[key] = 0
    return graph_nx_copy


def _get_all_key_and_unique_values(graph_nx: MultiDiGraph, selected_keys: Dict = SELECTED_KEYS) -> Dict:
    seen_values = {}
    if not selected_keys:
        selected_keys = ['oneway', 'lanes', 'highway', 'maxspeed',
                         'length', 'access', 'bridge', 'junction',
                         'width', 'service', 'tunnel', 'cycleway', 'bycycle']

    # get all values by selected key for each edge
    for edge in graph_nx.edges():
        for connection in graph_nx[edge[0]][edge[1]].keys():
            for key, val in graph_nx[edge[0]][edge[1]][connection].items():
                if key in selected_keys:
                    if key not in seen_values:
                        seen_values[key] = [val]
                    else:
                        if type(val) == list:
                            seen_values[key].extend(val)
                        else:
                            seen_values[key].extend([val])

    for key in seen_values.keys():
        seen_values[key] = set(seen_values[key])
    return seen_values


def _generate_cycle_label(graph_nx: MultiDiGraph, highway_coding: Dict = {}) -> MultiDiGraph:
    graph_nx_copy = graph_nx.copy()
    for edge in graph_nx.edges():
        for connection in graph_nx[edge[0]][edge[1]].keys():
            for key, val in graph_nx[edge[0]][edge[1]][connection].items():
                graph_edge = graph_nx_copy[edge[0]][edge[1]][connection]
                road_type = graph_edge['highway']
                if road_type == 14:
                    graph_edge['label'] = 1
                else:
                    graph_edge['label'] = 0
    return graph_nx_copy


def _generate_id(graph_nx: MultiDiGraph) -> MultiDiGraph:
    graph_nx_copy = graph_nx.copy()
    edge_id = 0
    for edge in graph_nx.edges():
        for connection in graph_nx[edge[0]][edge[1]].keys():
            for key, val in graph_nx[edge[0]][edge[1]][connection].items():
                graph_edge = graph_nx_copy[edge[0]][edge[1]][connection]
                graph_edge['idx'] = edge_id
        edge_id += 1

    return graph_nx_copy


def _convert_nx_to_dgl_as_linegraph(graph_nx: MultiDiGraph, selected_keys: List = SELECTED_KEYS) -> DGLHeteroGraph:
    sel_keys = selected_keys.copy()
    sel_keys.remove('label')
    sel_keys.remove('idx')

    graph_dgl = dgl.from_networkx(
        graph_nx, edge_attrs=(sel_keys + ['label', 'idx']))
    graph_dgl_line_graph = dgl.line_graph(graph_dgl)
    # populate linegraph with nodes

    continous_keys = ['maxspeed', 'width', 'lanes']
    features_to_line_graph = [_scale_feats_by_key(graph_dgl.edata[key], key) for key in continous_keys]
    continous_features = torch.stack(features_to_line_graph, axis=1)

    sel_keys.remove('maxspeed')
    sel_keys.remove('width')
    sel_keys.remove('lanes')

    d_features_to_line_graph = [graph_dgl.edata[key] for key in sel_keys]
    features_as_tensor = torch.stack(d_features_to_line_graph, axis=1)
    encoded_features = _one_hot_encoding(features_as_tensor)

    encoded_data = torch.cat((continous_features, encoded_features), 1)

    #raise Exception
    graph_dgl_line_graph.ndata['feat'] = encoded_data

    graph_dgl_line_graph.ndata['label'] = graph_dgl.edata['label'].type(
        torch.LongTensor)
    graph_dgl_line_graph.ndata['idx'] = graph_dgl.edata['idx']
    return graph_dgl_line_graph


def _get_random_split(number_of_nodes, train_size_coef=0.05, val_size_coef=0.18, test_size_coef=0.37):
    split_idx = {}
    train_size = int(number_of_nodes * train_size_coef)
    val_size = int(number_of_nodes * val_size_coef)
    test_size = int(number_of_nodes * test_size_coef)
    split_idx['train'] = random.sample(range(0, number_of_nodes), train_size)
    split_idx['train'].sort()
    split_idx['valid'] = random.sample(range(0, number_of_nodes), val_size)
    split_idx['valid'].sort()
    split_idx['test'] = random.sample(range(0, number_of_nodes), test_size)
    split_idx['test'].sort()

    return split_idx


def _get_stratified_split(labels, train_bicycle_coef=0.2, val_bicycle_coef=0.3, test_bicycle_coef=0.4):
    number_of_nodes = labels.shape[0]
    cycle_ids = ((labels == True).nonzero(as_tuple=True)[0]).tolist()
    number_of_cycle = len(cycle_ids)
    train_size = int(number_of_cycle * train_bicycle_coef)
    val_size = int(number_of_cycle * val_bicycle_coef)
    test_size = int(number_of_cycle * test_bicycle_coef)

    assert number_of_cycle > train_size
    assert number_of_cycle > val_size
    assert number_of_cycle > test_size

    split_idx = {}
    train_cycle_idx = random.sample(cycle_ids, train_size)
    train_noncycle_idx = _randome_sample_with_exceptions(
        number_of_nodes, train_size, cycle_ids)
    split_idx['train'] = train_cycle_idx + train_noncycle_idx
    split_idx['train'].sort()

    val_cycle_idx = random.sample(cycle_ids, val_size)
    val_noncycle_idx = _randome_sample_with_exceptions(
        number_of_nodes, val_size, cycle_ids)
    split_idx['valid'] = val_cycle_idx + val_noncycle_idx
    split_idx['valid'].sort()

    test_cycle_idx = random.sample(cycle_ids, test_size)
    test_noncycle_idx = _randome_sample_with_exceptions(
        number_of_nodes, test_size, cycle_ids)
    split_idx['test'] = test_cycle_idx + test_noncycle_idx
    split_idx['test'].sort()

    return split_idx


def _randome_sample_with_exceptions(max_range, size, exceptions):
    not_cycle = list(range(0, max_range))
    for elem in exceptions:
        not_cycle.remove(elem)
    return random.sample(not_cycle, size)


def _scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats, scaler

def _scale_feats_by_key(x, key):
    data_fit = None
    if key =='width':
        data_fit = [[0], [20]]
    elif key =='lanes':
        data_fit = [[0], [10]]
    elif key =='maxspeed':
        data_fit = [[0], [150]]

    scaler = StandardScaler()
    scaler.fit(data_fit)
    feats = scaler.transform(x.numpy().reshape(-1, 1)).reshape(-1)
    feats = torch.from_numpy(feats).float()
    return feats


def _preprocess(graph):
    feat = graph.ndata["feat"]
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph

def _one_hot_encoding(data):
    one_hot_enc = OneHotEncoder(categories=CATEGORIES, handle_unknown='ignore')


    encoded_data = one_hot_enc.fit_transform(data.numpy())
    
    encoded_data = coo_matrix(encoded_data)

    values = encoded_data.data
    indices = np.vstack((encoded_data.row, encoded_data.col))

    encoded_data_tensor = torch.sparse_coo_tensor(indices, values, encoded_data.shape)
    encoded_data_tensor = encoded_data_tensor.to_dense().float()

    return encoded_data_tensor


def _sizeof_fmt(num: int, suffix: str = "B") -> str:
    for unit in ["", "k", "M", "G", "T", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


if __name__ == "__main__":
    args = build_args()
    if args.all:
        load_transform_dir_bikeguessr(
            directory=args.path, output=args.output, targets=args.targets)
    if args.single:
        assert args.path is not None
        logging.info('processing single graph {}'.format(args.path))
        load_transform_single_bikeguessr(path=args.path, output=args.output)
