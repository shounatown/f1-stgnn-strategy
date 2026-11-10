import torch
import torch.nn as nn
from torch_geometric.data import Data

from f1_strategy_ai.models.spatial_encoder import SpatialEncoder


class F1StrategySTGNN(nn.Module):
    #Spatio-temporal GNN for F1 race strategy prediction

    def __init__(self, node_features: int = 11, spatial_dim: int = 64, hidden_dim: int = 64, pred_dim: int = 1):

        super().__init__()
        self.spatial_encoder = SpatialEncoder(in_channels=node_features, out_channels=spatial_dim)
        self.skip = nn.Linear(node_features, pred_dim)

        self.lstm = nn.LSTM(
            input_size=spatial_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, pred_dim))

    def forward(self, graph_sequence: list[Data]) -> torch.Tensor:
        #forward: processes a sequence of race graphs and predicts the next step target
        spatial_embeddings = []

        for graph in graph_sequence:
            emb = self.spatial_encoder(graph.x, graph.edge_index)
            spatial_embeddings.append(emb)

        seq_tensor = torch.stack(spatial_embeddings, dim=1)
        lstm_out, _ = self.lstm(seq_tensor)
        last_hidden = lstm_out[:, -1, :]

        return self.head(last_hidden)+ self.skip(graph_sequence[-1].x)