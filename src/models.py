# src/models.py
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNImitator(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.input_layer = nn.Linear(in_channels, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_channels)
    def forward(self, x, current_node_idx, edge_index):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        current_node_embedding = x[current_node_idx]
        return self.output_layer(current_node_embedding)

class GNN_QNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_actions):
        super().__init__()
        self.input_layer = nn.Linear(in_channels, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_actions)
    def forward(self, x, edge_index):
        x = F.relu(self.input_layer(x))
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.fc(h)