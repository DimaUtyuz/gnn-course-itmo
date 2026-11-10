"""Слои для task1: scatter-агрегация из лекции 3 и практики 3."""

import torch
from torch import nn
from torch_geometric.utils import add_self_loops, degree


class MyGCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        n = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n)
        row, col = edge_index
        deg = degree(col, n, dtype=x.dtype).clamp(min=1)
        norm = deg[row].pow(-0.5) * deg[col].pow(-0.5)
        h = self.lin(x)
        msg = h[row] * norm.unsqueeze(-1)
        out = torch.zeros_like(h)
        out.index_add_(0, col, msg)
        return out


class MySAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        h = self.lin_neigh(x)
        out = torch.zeros_like(h)
        out.index_add_(0, col, h[row])
        deg = degree(col, x.size(0), dtype=x.dtype).clamp(min=1)
        return out / deg.unsqueeze(-1) + self.lin_self(x)
