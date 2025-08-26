# 4_run_inference.py
import pickle
import random
import torch
import os 
import sys

from src import config, utils, data_utils
from src.models import GNNImitator, GNN_QNetwork

def find_rl_path(model, graph, start_node, end_node, node_features, node_to_index, edge_index):
    model.eval()
    path = [start_node]
    current_node = start_node
    with torch.no_grad():
        for _ in range(config.MAX_STEPS_PER_EPISODE):
            if current_node == end_node: break
            current_node_idx = node_to_index[current_node]
            q_values = model(node_features, edge_index)[current_node_idx]
            neighbors = list(graph.neighbors(current_node))
            if not neighbors: break
            
            # Filter Q-values for valid neighbors only and choose the best
            valid_q_values = q_values[:len(neighbors)]
            best_valid_action = valid_q_values.argmax().item()
            next_node = neighbors[best_valid_action]
            
            path.append(next_node)
            current_node = next_node
    return path

def find_imitation_path(model, graph, start_node, end_node, node_features, node_to_index, edge_index):
    model.eval()
    path = [start_node]
    current_node = start_node
    with torch.no_grad():
        for _ in range(config.MAX_STEPS_PER_EPISODE):
            if current_node == end_node: break
            current_node_idx = torch.tensor([node_to_index[current_node]], device=config.DEVICE)
            predictions = model(node_features, current_node_idx, edge_index)
            neighbors = list(graph.neighbors(current_node))
            if not neighbors: break
            neighbor_indices = [node_to_index[n] for n in neighbors]
            neighbor_scores = predictions[0, neighbor_indices]
            best_neighbor_idx = neighbor_scores.argmax().item()
            next_node = neighbors[best_neighbor_idx]
            path.append(next_node)
            current_node = next_node
    return path

def main():
    print("--- Stage 4: Running Inference and Visualization ---")
    
    # Load data
    graph = utils.load_graph(config.GRAPH_PATH)
    weather_data = data_utils.get_weather_data_for_graph(graph.nodes())
    node_features = data_utils.create_node_features(graph, weather_data)
    trajectories = data_utils.get_trajectories_from_ais(config.CLEANED_AIS_PATH, set(graph.nodes()))
    
    # Setup
    node_list = list(graph.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}
    edge_index = utils.get_edge_index(graph)
    
    # Load Models
    rl_model = utils.load_rl_model(config.RL_MODEL_PATH, node_features.shape[1])
    imitation_model = utils.load_imitation_model(config.IMITATION_MODEL_PATH, node_features.shape[1], len(node_list))
    
    # Select a random historical journey to test against
    historical_trajectory = None
    while not historical_trajectory or len(historical_trajectory) < 20 or historical_trajectory[0] == historical_trajectory[-1]:
        historical_trajectory = random.choice(trajectories)
    
    start_node = historical_trajectory[0]
    end_node = historical_trajectory[-1]
    
    print(f"\nSelected Journey Start Node: {start_node}")
    print(f"Selected Journey End Node:   {end_node}")
    
    # Generate the three paths
    print("\nFinding path with final AI (RL) model...")
    ai_path = find_rl_path(rl_model, graph, start_node, end_node, node_features, node_to_index, edge_index)
    print(f"  -> AI (RL) Path found with {len(ai_path)} steps.")

    print("Finding path with Imitation model...")
    imitation_path = find_imitation_path(imitation_model, graph, start_node, end_node, node_features, node_to_index, edge_index)
    print(f"  -> Imitation Path found with {len(imitation_path)} steps.")
    print(f"  -> Historical Path took {len(historical_trajectory)} steps.")

    # Generate the plot
    utils.plot_routes(graph, historical_trajectory, imitation_path, ai_path, start_node, end_node, config.OUTPUT_PLOT_PATH)

if __name__ == "__main__":
    main()