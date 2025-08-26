# 3_train_model.py
import argparse
import pickle
import random
from collections import deque
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src import config, utils, data_utils
from src.models import GNNImitator, GNN_QNetwork
from src.environment import VesselNavigationEnv, ReplayBuffer

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, node_to_index, sequence_length):
        self.sequences, self.targets = [], []
        for traj in trajectories:
            if len(traj) > sequence_length:
                for i in range(len(traj) - sequence_length):
                    input_node = node_to_index[traj[i + sequence_length - 1]]
                    target_node = node_to_index[traj[i + sequence_length]]
                    self.sequences.append(input_node)
                    self.targets.append(target_node)
    
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

def train_imitation_model():
    """Runs the Imitation Learning pre-training stage."""
    graph, _, node_features, trajectories = utils.load_training_data(config.GRAPH_PATH, config.CLEANED_AIS_PATH)
    
    node_to_index = {node: i for i, node in enumerate(graph.nodes())}
    num_nodes = len(graph.nodes())
    
    dataset = TrajectoryDataset(trajectories, node_to_index, config.SEQUENCE_LENGTH)
    loader = DataLoader(dataset, batch_size=config.IL_BATCH_SIZE, shuffle=True)
    
    model = GNNImitator(node_features.shape[1], config.HIDDEN_DIM, num_nodes).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.IL_LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    edge_index = utils.get_edge_index(graph)

    model.train()
    for epoch in range(config.PRETRAIN_EPOCHS):
        total_loss = 0
        for current_nodes, target_nodes in tqdm(loader, desc=f"Pre-train Epoch {epoch+1}/{config.PRETRAIN_EPOCHS}"):
            current_nodes, target_nodes = current_nodes.to(config.DEVICE), target_nodes.to(config.DEVICE)
            optimizer.zero_grad()
            predictions = model(node_features, current_nodes, edge_index)
            loss = loss_fn(predictions, target_nodes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Pre-train Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), config.IMITATION_MODEL_PATH)
    print(f"\n✅ Pre-trained imitation model saved to {config.IMITATION_MODEL_PATH}")

def train_rl_model():
    """Runs the Reinforcement Learning fine-tuning stage."""
    graph, weather_data, node_features, trajectories = utils.load_training_data(config.GRAPH_PATH, config.CLEANED_AIS_PATH)
    edge_index = utils.get_edge_index(graph)
    
    env = VesselNavigationEnv(graph, trajectories, weather_data, config.MAX_STEPS_PER_EPISODE)
    
    policy_net = GNN_QNetwork(node_features.shape[1], config.HIDDEN_DIM, config.MAX_NEIGHBORS).to(config.DEVICE)
    target_net = GNN_QNetwork(node_features.shape[1], config.HIDDEN_DIM, config.MAX_NEIGHBORS).to(config.DEVICE)
    
    # Load weights from the pre-trained imitation model
    utils.transfer_weights(config.IMITATION_MODEL_PATH, policy_net)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=config.RL_LEARNING_RATE)
    replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_CAPACITY)
    epsilon = config.EPSILON_START
    all_rewards = []

    print("\nStarting RL fine-tuning...")
    # ... (Full RL training loop from cell id: 9cf00db2) ...
    # This loop involves env.reset(), sampling actions, env.step(),
    # pushing to replay_buffer, sampling from buffer, and updating the network.
    
    torch.save(policy_net.state_dict(), config.RL_MODEL_PATH)
    print(f"\n✅ Final RL model saved to {config.RL_MODEL_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Train the vessel navigation model.")
    parser.add_argument('--stage', type=str, default='all', choices=['imitation', 'rl', 'all'],
                        help="Specify which training stage to run: 'imitation', 'rl', or 'all'.")
    args = parser.parse_args()
    
    if args.stage in ['imitation', 'all']:
        print("\n--- STAGE 3A: PRE-TRAINING IMITATION MODEL ---")
        train_imitation_model()
    
    if args.stage in ['rl', 'all']:
        print("\n--- STAGE 3B: FINE-TUNING RL MODEL ---")
        train_rl_model()

if __name__ == "__main__":
    main()