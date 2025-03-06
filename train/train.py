import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from models.gnn_model import GCN
from data.download_data import load_data

def train():
    cora, data = load_data()  # Adjusted to return both dataset and data object
    model = GCN(input_dim=data.num_features, hidden_dim=100, output_dim=cora.num_classes)  # Fix here
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()

    for epoch in range(100):  # Adjust training epochs
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item()}")

if __name__ == "__main__":
    train()
