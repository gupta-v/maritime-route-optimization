# src/data_utils.py
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
from . import config, utils

def _fetch_weather_for_node(lat, lon, client):
    """Fetches key weather data for a single node."""
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["wave_height", "wave_period", "ocean_current_velocity"],
        "timezone": "GMT"
    }
    try:
        responses = client.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        return {
            "wave_height": hourly.Variables(0).ValuesAsNumpy()[0],
            "wave_period": hourly.Variables(1).ValuesAsNumpy()[0],
            "ocean_current_velocity": hourly.Variables(2).ValuesAsNumpy()[0],
        }
    except Exception:
        # Return default safe values if API fails
        return {"wave_height": 0, "wave_period": 10, "ocean_current_velocity": 0}

def get_weather_data_for_graph(graph_nodes):
    """Fetches weather data for all nodes in the graph."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo_client = openmeteo_requests.Client(session=retry_session)
    weather_data = {}
    for node in tqdm(graph_nodes, desc="Fetching Weather Data"):
        lat, lon = node
        weather_data[node] = _fetch_weather_for_node(lat, lon, openmeteo_client)
    return weather_data

def create_node_features(graph, weather_data):
    """Creates a scaled feature tensor for GNN input."""
    node_list = list(graph.nodes())
    feature_list = []
    for node in node_list:
        weather = weather_data.get(node, {"wave_height": 0, "wave_period": 10, "ocean_current_velocity": 0})
        # Feature vector: [latitude, longitude, wave_height, wave_period, ocean_current_velocity]
        features = [node[0], node[1], weather['wave_height'], weather['wave_period'], weather['ocean_current_velocity']]
        feature_list.append(features)

    scaler = StandardScaler()
    node_features_scaled = scaler.fit_transform(feature_list)
    node_features = torch.tensor(node_features_scaled, dtype=torch.float).to(config.DEVICE)
    
    print(f"Node feature tensor created with shape: {node_features.shape}")
    return node_features

def get_trajectories_from_ais(ais_path, graph_nodes_set):
    """Processes cleaned AIS data to extract valid vessel trajectories."""
    print("Loading cleaned AIS data and creating trajectories...")
    df_ais = pd.read_csv(ais_path)
    df_ais['BaseDateTime'] = pd.to_datetime(df_ais['BaseDateTime'])
    df_ais['grid_node'] = df_ais.apply(
        lambda row: utils.snap_to_grid(row['LAT'], row['LON'], config.GRID_STEP),
        axis=1
    )
    
    all_trajectories = []
    for mmsi, group in tqdm(df_ais.groupby('MMSI'), desc="Splitting Trajectories"):
        group = group.sort_values('BaseDateTime')
        # Split trajectory if there's a gap of more than 6 hours
        time_diffs = group['BaseDateTime'].diff().dt.total_seconds().fillna(0)
        split_indices = np.where(time_diffs > 6 * 3600)[0]
        
        last_split = 0
        for index in split_indices:
            trajectory = group.iloc[last_split:index]['grid_node'].tolist()
            if len(trajectory) > 10:  # Minimum length for a valid trajectory
                all_trajectories.append(trajectory)
            last_split = index
            
        final_trajectory = group.iloc[last_split:]['grid_node'].tolist()
        if len(final_trajectory) > 10:
            all_trajectories.append(final_trajectory)

    # Filter trajectories to ensure all nodes exist in the graph
    trajectories = [
        traj for traj in all_trajectories if all(node in graph_nodes_set for node in traj)
    ]
    print(f"Generated {len(trajectories)} valid trajectories from AIS data.")
    return trajectories