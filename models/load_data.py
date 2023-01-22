from dgl.data.utils import load_graphs


def load_data(graphs_dir):
    graphs = load_graphs(graphs_dir)[0]
    train_graphs = graphs[:-2]
    val_graph = graphs[-2:]
    return train_graphs, val_graph

def load_train_and_val_data():
    train_graph_dir = "data/data_train/training_graphs.bin"
    val_graph_dir = "data/data_val/validation_graphs.bin"

    training_graphs = load_graphs(train_graph_dir)[0]
    val_graphs = load_graphs(val_graph_dir)[0]

    return training_graphs, val_graphs


# def load_bikeguessr_dataset(filepath: str) -> Tuple[List[DGLHeteroGraph], Tuple[int, int]]:
#     logging.info('load bikeguessr dataset')
#     if filepath is None:
#         filepath = str(Path(DATA_OUTPUT, 'bikeguessr.bin'))
#     file = Path(filepath)

#     logging.info('processing: ' + str(file.absolute()) +
#                  ' size: ' + _sizeof_fmt(os.path.getsize(file)))
#     graphs, _ = load_graphs(str(file))
#     num_features, num_classes = [], []
#     for i in range(len(graphs)):
#         graphs[i] = graphs[i].remove_self_loop()
#         graphs[i] = graphs[i].add_self_loop()
#     num_features = graphs[i].ndata["feat"].shape[1]
#     num_classes = 2

#     return graphs, (num_features, num_classes)