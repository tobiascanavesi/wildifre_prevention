import os
import re
import glob
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta

# Directories
BASE_DIR = os.path.abspath(".")  # or ".." if you need the parent dir
MODIS_TIF_DIR = os.path.join(BASE_DIR, "data", "raw", "modis_tif")  # Input TIFs
GRID_PATH     = os.path.join(BASE_DIR, "data", "raw", "ca_grid_10km.shp")
OUTPUT_CSV    = os.path.join(BASE_DIR, "data", "processed", "modis_ndvi_10km.csv")

def parse_date_from_filename(filename):
    """
    Extract the date from a MODIS TIF filename.
    Looks for a substring like: 'doyYYYYDDD' 
     - YYYY = 4-digit year
     - DDD  = day of year
    
    Example:
      MYD13A3.061__1_km_monthly_NDVI_doy2019032_aid0001.tif
    => 'doy2019032' => year=2019, day_of_year=32 => 2019-02-01
    """
    match = re.search(r"doy(\d{4})(\d{3})", filename)
    if match:
        year_str, doy_str = match.groups()
        year = int(year_str)
        day_of_year = int(doy_str)
        # Convert to datetime
        date_val = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        return date_val
    return None

def compute_ndvi_per_cell(grid_gdf, tif_path):
    """
    For each 10 km grid cell, sample the NDVI pixel at the centroid.
    Returns a DataFrame: [cell_id, ndvi].
    """
    with rasterio.open(tif_path) as src:
        ndvi_data = src.read(1)  # first band
        ndvi_transform = src.transform

        records = []
        for _, row in grid_gdf.iterrows():
            cell_id = row["cell_id"]
            centroid = row.geometry.centroid

            # Convert centroid (lon, lat) => (row, col)
            row_col = ~ndvi_transform * (centroid.x, centroid.y)
            row_i, col_j = map(int, map(round, row_col))

            # Check if indices are within raster bounds
            if (0 <= row_i < ndvi_data.shape[0]) and (0 <= col_j < ndvi_data.shape[1]):
                cell_ndvi = ndvi_data[row_i, col_j]

                # Negative values often indicate fill, so treat as NaN
                if cell_ndvi < 0:
                    cell_ndvi = np.nan
                else:
                    # Apply scale factor for MYD13A3: NDVI = raw * 0.0001
                    cell_ndvi *= 0.0001
            else:
                cell_ndvi = np.nan

            records.append((cell_id, cell_ndvi))

    df_out = pd.DataFrame(records, columns=["cell_id", "ndvi"])
    return df_out

def process_modis_ndvi():
    # 1) Load the 10 km grid
    if not os.path.exists(GRID_PATH):
        print(f"[Error] Grid file not found: {GRID_PATH}")
        return
    
    grid_gdf = gpd.read_file(GRID_PATH)
    print(f"[Info] Loaded grid with {len(grid_gdf)} cells (CRS={grid_gdf.crs}).")

    # 2) Find TIF files. We'll ONLY process those that contain 'NDVI' in the filename.
    #    If your real NDVI TIFs have a different naming pattern, adjust accordingly.
    tif_candidates = sorted(glob.glob(os.path.join(MODIS_TIF_DIR, "*.tif")))
    tif_files = [f for f in tif_candidates if "ndvi" in os.path.basename(f).lower()]

    print(f"[Info] Found {len(tif_candidates)} total TIFs; "
          f"{len(tif_files)} appear to be NDVI in {MODIS_TIF_DIR}.")

    if not tif_files:
        print("[Error] No NDVI TIF files found (none have 'NDVI' in the filename). Exiting.")
        return

    all_records = []

    # 3) Parse date & compute NDVI for each TIF
    for tif_file in tif_files:
        filename = os.path.basename(tif_file)
        date_val = parse_date_from_filename(filename)
        if not date_val:
            print(f"[Warning] Could not parse date from {filename}. Skipping.")
            continue

        print(f"[Processing] {filename} => date={date_val.date()}")
        df_ndvi = compute_ndvi_per_cell(grid_gdf, tif_file)
        df_ndvi["date"] = date_val.date()  # store as date
        all_records.append(df_ndvi)

    if not all_records:
        print("[Error] No valid NDVI records to combine. Exiting.")
        return

    # 4) Concatenate all results
    df_all = pd.concat(all_records, ignore_index=True)
    print(f"[Info] Combined NDVI records: {df_all.shape}")

    # 5) Create output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 6) Save final CSV
    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"[Success] Saved NDVI data (with scale factor) to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_modis_ndvi()
