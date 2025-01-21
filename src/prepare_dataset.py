# src/prepare_dataset.py

import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime

# Define base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define data directories
NOAA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
USFS_FOD_PATH = os.path.join(NOAA_PROCESSED_DIR, "usfs_fod_processed_20250116_183345.csv") 
NDVI_PATH = os.path.join(NOAA_PROCESSED_DIR, "modis_ndvi_10km.csv")  # Adjusted to the correct filename
MERGED_OUTPUT_PATH = os.path.join(NOAA_PROCESSED_DIR, "merged_weather_fire_ndvi.csv")

def combine_noaa():
    """Combine multiple NOAA processed CSVs into one dataframe."""
    pattern = os.path.join(NOAA_PROCESSED_DIR, "noaa_cdo_processed_*.csv")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} NOAA processed files to combine.")
    
    df_list = []
    for f in files:
        df_temp = pd.read_csv(f)
        # Ensure 'date' column exists
        if 'date' not in df_temp.columns:
            print(f"[Warning] 'date' column not found in {f}. Skipping.")
            continue
        # Convert 'date' to datetime
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_list.append(df_temp)
    
    if not df_list:
        print("[Error] No valid NOAA data to combine.")
        return pd.DataFrame()
    
    df_noaa = pd.concat(df_list, ignore_index=True)
    # Drop duplicates in case of overlap
    df_noaa.drop_duplicates(subset='date', inplace=True)
    df_noaa.sort_values(by='date', inplace=True)
    df_noaa.reset_index(drop=True, inplace=True)
    print(f"Combined NOAA shape: {df_noaa.shape}")
    return df_noaa

def load_usfs_fod(grid_gdf_projected):
    """
    Load the processed USFS Fire Occurrence CSV and map fires to grid cells.
    Assumes USFS data contains 'LATITUDE' and 'LONGITUDE' columns.
    """
    df_usfs = pd.read_csv(USFS_FOD_PATH)
    # Ensure required columns exist
    required_cols = ['DISCOVERY_DATE', 'LATITUDE', 'LONGITUDE']
    for col in required_cols:
        if col not in df_usfs.columns:
            print(f"[Error] Required column '{col}' not found in USFS data.")
            return pd.DataFrame()
    
    # Convert DISCOVERY_DATE to datetime
    df_usfs['DISCOVERY_DATE'] = pd.to_datetime(df_usfs['DISCOVERY_DATE'], errors='coerce').dt.date
    # Drop rows with invalid dates or coordinates
    df_usfs.dropna(subset=['DISCOVERY_DATE', 'LATITUDE', 'LONGITUDE'], inplace=True)
    
    # Convert to GeoDataFrame with Point geometries
    gdf_usfs = gpd.GeoDataFrame(
        df_usfs,
        geometry=gpd.points_from_xy(df_usfs.LONGITUDE, df_usfs.LATITUDE),
        crs='EPSG:4326'  # Assuming lat/lon; adjust if different
    )
    
    # Reproject USFS GeoDataFrame to match grid's projected CRS
    gdf_usfs_projected = gdf_usfs.to_crs(grid_gdf_projected.crs)
    
    # Spatial join: assign each fire to a grid cell
    # Using 'predicate' instead of 'op' as per GeoPandas 0.10.0+
    gdf_fire_joined = gpd.sjoin(gdf_usfs_projected, grid_gdf_projected, how='left', predicate='within')
    # 'cell_id' from grid_gdf_projected is now part of gdf_fire_joined
    gdf_fire_joined = gdf_fire_joined.drop(columns=['index_right'])
    
    # Drop fires that don't fall within any grid cell
    gdf_fire_joined.dropna(subset=['cell_id'], inplace=True)
    
    # Convert 'cell_id' to integer if necessary
    gdf_fire_joined['cell_id'] = gdf_fire_joined['cell_id'].astype(int)
    
    # Create (cell_id, date) pairs with fire occurrence
    df_fire = gdf_fire_joined.groupby(['cell_id', 'DISCOVERY_DATE']).size().reset_index(name='fire_occurred')
    # Convert to binary flag
    df_fire['fire_occurred'] = 1
    # Rename for consistency
    df_fire.rename(columns={'DISCOVERY_DATE': 'date'}, inplace=True)
    # Convert 'date' to datetime
    df_fire['date'] = pd.to_datetime(df_fire['date'])
    
    print(f"Total fire events mapped to grid cells: {df_fire.shape[0]}")
    return df_fire

def load_ndvi():
    """
    Load the NDVI CSV which has at least ['cell_id', 'ndvi', 'date'] columns.
    """
    df_ndvi = pd.read_csv(NDVI_PATH)
    # Ensure required columns exist
    required_cols = ['cell_id', 'ndvi', 'date']
    for col in required_cols:
        if col not in df_ndvi.columns:
            print(f"[Error] Required column '{col}' not found in NDVI data.")
            return pd.DataFrame()
    
    # Convert 'date' to datetime
    df_ndvi['date'] = pd.to_datetime(df_ndvi['date'])
    
    return df_ndvi

def create_complete_grid(grid_gdf_projected, df_noaa):
    """
    Create a DataFrame with all combinations of 'cell_id' and 'date'.
    """
    cell_ids = grid_gdf_projected['cell_id'].unique()
    dates = df_noaa['date'].unique()
    df_grid = pd.MultiIndex.from_product([cell_ids, dates], names=['cell_id', 'date']).to_frame(index=False)
    print(f"Created complete grid with {df_grid.shape[0]} rows.")
    return df_grid

def merge_data(df_grid, df_noaa, df_ndvi, df_fire):
    """
    Merge NOAA weather, NDVI, and fire occurrence data into a single DataFrame.
    
    Parameters:
        df_grid (DataFrame): Complete (cell_id, date) grid.
        df_noaa (DataFrame): NOAA weather data.
        df_ndvi (DataFrame): NDVI data with (cell_id, date, ndvi).
        df_fire (DataFrame): Fire occurrence data with (cell_id, date, fire_occurred).
    
    Returns:
        df_final (DataFrame): Merged dataset ready for modeling.
    """
    # 1) Merge NOAA weather data into grid
    df_combined = pd.merge(df_grid, df_noaa, on='date', how='left')
    print(f"Merged NOAA weather data: {df_combined.shape}")
    
    # 2) Merge NDVI data into combined data
    df_combined = pd.merge(df_combined, df_ndvi, on=['cell_id', 'date'], how='left')
    print(f"Merged NDVI data: {df_combined.shape}")
    
    # 3) Merge fire occurrence data into combined data
    df_combined = pd.merge(df_combined, df_fire, on=['cell_id', 'date'], how='left')
    print(f"Merged Fire Occurrence data: {df_combined.shape}")
    
    # 4) Fill NaN in 'fire_occurred' with 0 (no fire)
    df_combined['fire_occurred'] = df_combined['fire_occurred'].fillna(0).astype(int)
    
    # 5) Handle missing NDVI or weather data as needed
    # Example: Forward fill NDVI within each cell
    df_combined.sort_values(['cell_id', 'date'], inplace=True)
    df_combined['ndvi'] = df_combined.groupby('cell_id')['ndvi'].ffill()
    
    # Optionally, drop rows with any remaining NaN
    df_final = df_combined.dropna()
    print(f"Final merged dataset shape: {df_final.shape}")
    
    return df_final

def main():
    # 1) Combine NOAA weather data
    df_noaa = combine_noaa()
    if df_noaa.empty:
        print("[Error] No NOAA data available. Exiting.")
        return
    
    # 2) Load grid shapefile
    grid_shapefile_path = os.path.join(BASE_DIR, "data", "raw", "ca_grid_10km.shp")
    if not os.path.exists(grid_shapefile_path):
        print(f"[Error] Grid shapefile not found at {grid_shapefile_path}. Exiting.")
        return
    
    grid_gdf = gpd.read_file(grid_shapefile_path)
    print(f"[Info] Loaded grid with {len(grid_gdf)} cells (CRS={grid_gdf.crs}).")
    
    # 3) Reproject grid to a projected CRS (California Albers)
    projected_crs = 'EPSG:3310'  # California Albers
    grid_gdf_projected = grid_gdf.to_crs(projected_crs)
    print(f"[Info] Reprojected grid to {projected_crs}.")
    
    # 4) Compute centroids in projected CRS
    # Ensure that centroids are computed correctly by setting a projected CRS
    grid_gdf_projected['centroid'] = grid_gdf_projected.geometry.centroid
    grid_gdf_projected['centroid_x'] = grid_gdf_projected.centroid.x
    grid_gdf_projected['centroid_y'] = grid_gdf_projected.centroid.y
    
    # 5) Reproject centroids back to geographic CRS (EPSG:4326) for plotting
    # Create a separate GeoDataFrame for centroids
    gdf_centroids = grid_gdf_projected.set_geometry('centroid').copy()
    gdf_centroids = gdf_centroids.to_crs('EPSG:4326')
    
    # Extract LONGITUDE and LATITUDE from reprojected centroids
    grid_gdf_projected['centroid_lon'] = gdf_centroids.geometry.x
    grid_gdf_projected['centroid_lat'] = gdf_centroids.geometry.y
    
    # 6) Load USFS Fire Occurrence data and map to grid cells
    df_fire = load_usfs_fod(grid_gdf_projected)
    if df_fire.empty:
        print("[Warning] No fire events mapped to grid cells.")
    
    # 7) Load NDVI data
    df_ndvi = load_ndvi()
    if df_ndvi.empty:
        print("[Error] No NDVI data available. Exiting.")
        return
    
    # 8) Create complete (cell_id, date) grid
    df_grid = create_complete_grid(grid_gdf_projected, df_noaa)
    
    # 9) Merge all data
    df_final = merge_data(df_grid, df_noaa, df_ndvi, df_fire)
    
    # 10) Save final dataset
    df_final.to_csv(MERGED_OUTPUT_PATH, index=False)
    print(f"[Success] Saved merged dataset to {MERGED_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
