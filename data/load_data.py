from dgl.data.utils import load_graphs


def load_data(graphs_dir):
    graphs = load_graphs(graphs_dir)[0]
    train_graphs = graphs[:9]
    val_graph = graphs[-1]
    return train_graphs, val_graph