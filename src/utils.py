# src/utils.py
import pickle
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
from . import config
from .models import GNNImitator, GNN_QNetwork
from .data_utils import get_weather_data_for_graph, create_node_features, get_trajectories_from_ais


def snap_to_grid(lat, lon, step):
    """Snaps latitude and longitude to the nearest grid point."""
    return (round(round(lat / step) * step, 4), round(round(lon / step) * step, 4))

def load_graph(path):
    """Loads the graph from a pickle file."""
    with open(path, "rb") as f:
        graph = pickle.load(f)
    print(f"Graph loaded from {path} with {len(graph.nodes())} nodes.")
    return graph

def get_edge_index(graph):
    """Converts NetworkX graph to PyTorch Geometric edge_index."""
    pyg_data = from_networkx(graph)
    return pyg_data.edge_index.to(config.DEVICE)

def load_imitation_model(path, node_features_dim, num_nodes):
    """Loads the pre-trained imitation learning model."""
    model = GNNImitator(node_features_dim, config.HIDDEN_DIM, num_nodes).to(config.DEVICE)
    model.load_state_dict(torch.load(path, map_location=config.DEVICE))
    print(f"Imitation model loaded from {path}.")
    return model

def load_rl_model(path, node_features_dim):
    """Loads the final reinforcement learning model."""
    model = GNN_QNetwork(node_features_dim, config.HIDDEN_DIM, config.MAX_NEIGHBORS).to(config.DEVICE)
    model.load_state_dict(torch.load(path, map_location=config.DEVICE))
    print(f"RL model loaded from {path}.")
    return model

def plot_routes(graph, historical_path, imitation_path, ai_path, start_node, end_node, output_path):
    """Generates and saves the 3-way route comparison plot."""
    print("Generating comparison plot...")
    plt.figure(figsize=(16, 16))
    
    # Plot all sea nodes
    xs, ys = zip(*graph.nodes())
    plt.scatter(ys, xs, s=8, color='lightblue', label='Sea Nodes')
    
    # Plot paths
    if historical_path:
        xs_hist, ys_hist = zip(*historical_path)
        plt.plot(ys_hist, xs_hist, marker='o', markersize=3, color='red', linestyle='--', label=f'Historical AIS Path ({len(historical_path)} steps)')
    if imitation_path:
        xs_imit, ys_imit = zip(*imitation_path)
        plt.plot(ys_imit, xs_imit, marker='x', markersize=5, color='blue', linestyle=':', label=f'Imitation Path ({len(imitation_path)} steps)')
    if ai_path:
        xs_ai, ys_ai = zip(*ai_path)
        plt.plot(ys_ai, xs_ai, marker='o', markersize=4, color='green', linestyle='-', label=f'Final AI (RL) Path ({len(ai_path)} steps)')
        
    # Plot start and end points
    plt.scatter(start_node[1], start_node[0], s=200, color='orange', edgecolors='black', zorder=5, label='Start')
    plt.scatter(end_node[1], end_node[0], s=200, color='purple', edgecolors='black', zorder=5, label='End')
    
    plt.title('Route Optimization: 3-Way Path Comparison (Weather-Aware GNN)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    plt.savefig(output_path)
    print(f"Plot saved as {output_path}")
    plt.show()

def load_training_data(graph_path, ais_path):
    """Loads all necessary data for training stages."""
    graph = load_graph(graph_path)
    weather_data = get_weather_data_for_graph(graph.nodes())
    node_features = create_node_features(graph, weather_data)
    trajectories = get_trajectories_from_ais(ais_path, set(graph.nodes()))
    return graph, weather_data, node_features, trajectories