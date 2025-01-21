import os
import re
import glob
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import box

# Directories
BASE_DIR = os.path.abspath(".")  
MODIS_TIF_DIR = os.path.join(BASE_DIR, "data", "raw", "modis_tif")  
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

def compute_ndvi_per_cell(grid_gdf, tif_path, clip_to_raster=True):
    """
    For each polygon in `grid_gdf`, sample the NDVI pixel at its centroid
    from the given GeoTIFF. Return a DataFrame with columns:
      ['cell_id', 'ndvi'].
    
    Steps:
      1) Reproject the grid to match the raster CRS (if different).
      2) (Optional) Clip grid polygons to the raster bounding box.
      3) For each polygon's centroid, compute (row, col) and sample the raster.
      4) Treat negative or large fill values as NaN.
      5) Apply 0.0001 scale factor to valid pixels.
    """

    with rasterio.open(tif_path) as src:
        # 1) Ensure the grid is in the same CRS as the raster
        if grid_gdf.crs != src.crs:
            print(f"[compute_ndvi_per_cell] Reprojecting grid from {grid_gdf.crs} to {src.crs}")
            grid_gdf = grid_gdf.to_crs(src.crs)

        # 2) Optionally clip polygons to the raster bounding box
        if clip_to_raster:
            left, bottom, right, top = src.bounds
            raster_bounds_poly = box(left, bottom, right, top)
            # Clip polygons to the raster extent
            clipped_gdf = gpd.clip(grid_gdf, raster_bounds_poly)
            print(f"[compute_ndvi_per_cell] Clipped grid from {len(grid_gdf)} to {len(clipped_gdf)} polygons based on raster bounds.")
        else:
            clipped_gdf = grid_gdf

        # 3) Read the first band (assume NDVI)
        ndvi_data = src.read(1)
        transform = src.transform

        records = []
        for _, row in clipped_gdf.iterrows():
            cell_id = row["cell_id"]
            centroid = row.geometry.centroid

            # Convert (x, y) => (row, col)
            # Here, x=Easting, y=Northing in the raster CRS (likely meters).
            row_col = ~transform * (centroid.x, centroid.y)
            row_i, col_j = map(int, map(round, row_col))

            # 4) Check if in range
            if (0 <= row_i < ndvi_data.shape[0]) and (0 <= col_j < ndvi_data.shape[1]):
                raw_val = ndvi_data[row_i, col_j]
                # Common fill values: negative, 32767, etc.
                if (raw_val < 0) or (raw_val == 32767):
                    ndvi_val = np.nan
                else:
                    # 5) Apply scale factor for MYD13A3
                    ndvi_val = raw_val * 0.0001
            else:
                ndvi_val = np.nan

            records.append((cell_id, ndvi_val))

    # Convert to DataFrame
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
