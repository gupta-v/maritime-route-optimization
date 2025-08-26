# src/config.py
import torch

# --- Core Paths ---
RAW_AIS_DIR = "data/raw_ais"
PROCESSED_DATA_DIR = "data/processed"
SHAPEFILE_DIR = "data/shapefile/ne_10m_land"
MODELS_DIR = "models"
OUTPUT_DIR = "output"

CLEANED_AIS_PATH = f"{PROCESSED_DATA_DIR}/cleaned_guam_ais.csv"
GRAPH_PATH = f"{PROCESSED_DATA_DIR}/sea_graph_guam.pkl"
IMITATION_MODEL_PATH = f"{MODELS_DIR}/gnn_imitator.pth"
RL_MODEL_PATH = f"{MODELS_DIR}/gnn_rl_agent.pth"
OUTPUT_PLOT_PATH = f"{OUTPUT_DIR}/route_comparison_plot.png"

# --- Graph & Grid Parameters ---
LAT_MIN, LAT_MAX = 13.0, 14.0
LON_MIN, LON_MAX = 144.0, 145.0
GRID_STEP = 0.05
LAND_SHAPEFILE = f"{SHAPEFILE_DIR}/ne_10m_land.shp"

# --- Model & Training Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 128
MAX_NEIGHBORS = 8  # Corresponds to RL action space

# Imitation Learning (IL)
PRETRAIN_EPOCHS = 50
IL_LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 10
IL_BATCH_SIZE = 128

# Reinforcement Learning (RL)
RL_EPISODES = 3000
RL_LEARNING_RATE = 1e-5
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 0.999
RL_BATCH_SIZE = 128
REPLAY_BUFFER_CAPACITY = 20000
TARGET_UPDATE_FREQ = 20
MAX_STEPS_PER_EPISODE = 1000