# 1_create_graph.py
import pickle
import os
from src import config
from src import graph_builder

def main():
    """
    Runs the graph generation process and saves the output.
    """
    print("--- Stage 1: Creating Sea Graph ---")
    
    if not os.path.exists(config.SHAPEFILE_DIR):
        print(f"Error: Land shapefile directory not found at {config.SHAPEFILE_DIR}")
        print("Please download and place the Natural Earth 10m land shapefile there.")
        return

    sea_graph = graph_builder.create_sea_graph(
        shapefile_path=config.LAND_SHAPEFILE,
        lat_min=config.LAT_MIN, lat_max=config.LAT_MAX,
        lon_min=config.LON_MIN, lon_max=config.LON_MAX,
        step=config.GRID_STEP
    )

    with open(config.GRAPH_PATH, "wb") as f:
        pickle.dump(sea_graph, f)
    print(f"\n✅ Graph successfully created and saved to {config.GRAPH_PATH}")

if __name__ == "__main__":
    main()