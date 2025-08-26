# src/environment.py
import gym
import random
import numpy as np
from collections import deque
from gym import spaces

class VesselNavigationEnv(gym.Env):
    def __init__(self, graph, trajectories, weather_data, max_steps):
        super(VesselNavigationEnv, self).__init__()
        self.graph = graph
        self.trajectories = [traj for traj in trajectories if len(traj) > 1 and traj[0] != traj[-1]]
        self.weather_data = weather_data
        self.max_steps = max_steps
        self.node_list = list(graph.nodes)
        self.node_to_index = {node: i for i, node in enumerate(self.node_list)}
        self.action_space = spaces.Discrete(8) # Max neighbors
        self.observation_space = spaces.Discrete(len(self.node_list))

    def reset(self):
        self.current_traj = random.choice(self.trajectories)
        self.source = self.current_traj[0]
        self.target = self.current_traj[-1]
        self.current_node = self.source
        self.current_step = 0
        self.done = False
        self.visited_nodes = {self.current_node}
        self.total_distance = self._calculate_distance(self.source, self.target)
        return self.node_to_index[self.current_node]

    def _calculate_distance(self, node1, node2):
        return np.linalg.norm(np.array(node1) - np.array(node2))

    def step(self, action):
        neighbors = list(self.graph.neighbors(self.current_node))
        distance_before_move = self._calculate_distance(self.current_node, self.target)
        if not neighbors:
            self.done = True
            return self.node_to_index[self.current_node], -50, self.done, {}
        if action < len(neighbors):
            next_node = neighbors[action]
            node_weather = self.weather_data.get(next_node, {"wave_height": 0, "ocean_current_velocity": 0, "wave_period": 10})
            wave_penalty = - (node_weather.get("wave_height", 0) ** 2)
            current_penalty = - (node_weather.get("ocean_current_velocity", 0) * 2)
            wave_period = node_weather.get("wave_period", 10)
            period_penalty = - (max(0, 8 - wave_period) * 0.5)
            distance_after_move = self._calculate_distance(next_node, self.target)
            progress = distance_before_move - distance_after_move
            progress_reward = progress * (1 + (self.total_distance - distance_after_move) / (self.total_distance + 1e-6)) * 15
            revisit_penalty = -5 if next_node in self.visited_nodes else 0
            reward = -0.2 + wave_penalty + current_penalty + period_penalty + progress_reward + revisit_penalty
        else:
            next_node = self.current_node
            reward = -2
        if next_node == self.target:
            reward = 500
            self.done = True
        self.current_node = next_node
        self.visited_nodes.add(self.current_node)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True
            if next_node != self.target:
                reward = -100
        return self.node_to_index[self.current_node], reward, self.done, {}

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)