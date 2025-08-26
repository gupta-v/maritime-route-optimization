# 2_prepare_data.py
import pandas as pd
import os
from glob import glob
from src import config

def clean_and_filter_ais(raw_folder_path, output_path, lat_min, lat_max, lon_min, lon_max):
    """
    Reads all CSVs from a directory, cleans them, filters by region,
    and saves to a single file.
    """
    csv_files = sorted(glob(os.path.join(raw_folder_path, "**/*.csv"), recursive=True))
    if not csv_files:
        print(f"❌ Error: No CSV files found in '{raw_folder_path}'.")
        print("Please place your raw daily AIS CSV files in that directory.")
        return

    first_write = True
    chunk_size = 100_000

    for file in csv_files:
        print(f"📂 Processing: {file}")
        try:
            for chunk in pd.read_csv(file, chunksize=chunk_size):
                chunk.dropna(subset=["LAT", "LON", "BaseDateTime", "MMSI"], inplace=True)
                chunk["BaseDateTime"] = pd.to_datetime(chunk["BaseDateTime"], errors="coerce")
                chunk.dropna(subset=["BaseDateTime"], inplace=True)
                
                chunk_filtered = chunk[
                    (chunk["LAT"] >= lat_min) & (chunk["LAT"] <= lat_max) &
                    (chunk["LON"] >= lon_min) & (chunk["LON"] <= lon_max)
                ]

                if not chunk_filtered.empty:
                    chunk_filtered.to_csv(output_path, mode='a', header=first_write, index=False)
                    first_write = False
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
    
    print(f"\n✅ Done! Cleaned data saved to: {output_path}")

def main():
    print("--- Stage 2: Preparing and Cleaning AIS Data ---")
    clean_and_filter_ais(
        raw_folder_path=config.RAW_AIS_DIR,
        output_path=config.CLEANED_AIS_PATH,
        lat_min=config.LAT_MIN, lat_max=config.LAT_MAX,
        lon_min=config.LON_MIN, lon_max=config.LON_MAX
    )

if __name__ == "__main__":
    main()