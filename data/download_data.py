from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomNodeSplit

def load_data():
    num_val, num_test = 200, 500
    dataset = Planetoid(root=".", name="Cora", transform=RandomNodeSplit(num_val=num_val, num_test=num_test))
    return dataset, dataset[0]
