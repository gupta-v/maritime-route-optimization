# src/graph_builder.py
import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
from joblib import Parallel, delayed

def create_sea_graph(shapefile_path, lat_min, lat_max, lon_min, lon_max, step):
    """
    Generates a graph of sea nodes based on a land shapefile, excluding land areas.
    """
    print("Loading land shapefile...")
    land = gpd.read_file(shapefile_path)
    land.crs = "EPSG:4326"
    land = land.to_crs(epsg=4326)
    land_union = land.geometry.union_all()

    print("Generating grid and filtering for sea points...")
    lats = np.arange(lat_min, lat_max + step, step)
    lons = np.arange(lon_min, lon_max + step, step)
    grid_points = [(round(lat, 4), round(lon, 4)) for lat in lats for lon in lons]

    def is_sea_point(latlon):
        # shapely uses (lon, lat) format
        return not land_union.contains(Point(latlon[1], latlon[0]))

    # Use joblib for parallel processing to speed up point checking
    sea_flags = Parallel(n_jobs=-1)(delayed(is_sea_point)(pt) for pt in tqdm(grid_points, desc="Filtering Sea Nodes"))
    sea_points = [pt for pt, is_sea in zip(grid_points, sea_flags) if is_sea]

    print("Building graph connections...")
    G = nx.Graph()
    # Add all valid sea points as nodes first
    for lat, lon in sea_points:
        G.add_node((lat, lon))

    # Connect nodes to their valid neighbors (including diagonals)
    for lat, lon in tqdm(sea_points, desc="Connecting Graph Nodes"):
        for dlat in [-step, 0, step]:
            for dlon in [-step, 0, step]:
                if dlat == 0 and dlon == 0:
                    continue
                neighbor = (round(lat + dlat, 4), round(lon + dlon, 4))
                if neighbor in G: # Check if the neighbor is a valid sea node
                    G.add_edge((lat, lon), neighbor)

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G