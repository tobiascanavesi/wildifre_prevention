import os
import glob
from datetime import datetime, timedelta
import rasterio
from rasterio import shutil
import pandas as pd
import geopandas as gpd
import numpy as np

# Directories
BASE_DIR = os.path.abspath(".")
HDF_DIR = os.path.join(BASE_DIR, "data", "raw", "modis_hdf")
TIFF_DIR = os.path.join(BASE_DIR, "data", "processed", "ndvi_tiffs")
GRID_PATH = os.path.join(BASE_DIR, "data", "raw", "ca_grid_10km.shp")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "modis_ndvi_10km.csv")

os.makedirs(TIFF_DIR, exist_ok=True)

def extract_ndvi(hdf_file, output_dir):
    """
    Extract MYD13A3 '1 km monthly NDVI' subdataset and save to GeoTIFF.
    """
    try:
        # Parse date from filename (e.g., MYD13A3.A2023244...)
        filename = os.path.basename(hdf_file)
        date_code = filename.split(".")[1][1:]  # e.g. "2023244"
        year, day_of_year = int(date_code[:4]), int(date_code[4:])
        date_val = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

        # Make GeoTIFF name
        tiff_name = f"MYD13A3_{date_val.year}_{date_val.month:02d}_{date_val.day:02d}.tif"
        tiff_path = os.path.join(output_dir, tiff_name)
        
        with rasterio.open(hdf_file) as src:
            print(f"Available subdatasets in {hdf_file}: {src.subdatasets}")


        # Must match exactly what gdalinfo shows
        subdataset_str = (
            f'HDF4_EOS:EOS_GRID:"{hdf_file}":MOD_Grid_monthly_1km_VI:"1 km monthly NDVI"'
        )

        with rasterio.open(subdataset_str) as src:
            shutil.copy(src, tiff_path)

        print(f"Extracted NDVI to {tiff_path}")
        return tiff_path, date_val

    except Exception as e:
        print(f"Error processing {hdf_file}: {e}")
        return None, None

def compute_ndvi_per_cell(grid_gdf, ndvi_path):
    """
    Compute NDVI at each grid cell's centroid.
    """
    with rasterio.open(ndvi_path) as src:
        ndvi_data = src.read(1)
        ndvi_transform = src.transform

        records = []
        for _, row in grid_gdf.iterrows():
            cell_id = row["cell_id"]
            centroid = row.geometry.centroid
            row_col = ~ndvi_transform * (centroid.x, centroid.y)
            row_i, col_j = map(int, map(round, row_col))

            if 0 <= row_i < ndvi_data.shape[0] and 0 <= col_j < ndvi_data.shape[1]:
                cell_ndvi = ndvi_data[row_i, col_j]
                # Replace negative fill with NaN if needed
                if cell_ndvi < 0:
                    cell_ndvi = np.nan
            else:
                cell_ndvi = np.nan

            records.append((cell_id, cell_ndvi))

    return pd.DataFrame(records, columns=["cell_id", "ndvi"])

def process_modis_ndvi():
    # 1) Load your shapefile
    grid_gdf = gpd.read_file(GRID_PATH)
    print(f"Loaded grid with {len(grid_gdf)} cells.")

    # 2) Find .hdf files
    hdf_files = sorted(glob.glob(os.path.join(HDF_DIR, "*.hdf")))
    print(f"Found {len(hdf_files)} HDF files.")

    all_dfs = []

    # 3) Extract NDVI & compute
    for hdf_file in hdf_files:
        tiff_path, date_val = extract_ndvi(hdf_file, TIFF_DIR)

        if tiff_path and date_val:
            df_ndvi = compute_ndvi_per_cell(grid_gdf, tiff_path)
            df_ndvi["date"] = date_val
            all_dfs.append(df_ndvi)

    # 4) Combine & save
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        print(f"Combined NDVI records: {df_all.shape}")
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df_all.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved NDVI data to {OUTPUT_CSV}")
    else:
        print("No records to process.")

if __name__ == "__main__":
    process_modis_ndvi()
