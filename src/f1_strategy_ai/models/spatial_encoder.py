import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SpatialEncoder(nn.Module):
    #GAT-based spatial encoder for modeling on-track traffic interactions

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, out_channels: int = 64, heads: int = 4):

        super().__init__()

        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        #forward: computes normalized spatial node embeddings

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return self.norm(x)