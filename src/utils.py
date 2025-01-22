# src/utils.py

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import joblib
import os
import streamlit as st

# Suppress warnings globally in utilities
import warnings
warnings.filterwarnings("ignore")

@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    return df

@st.cache_data
def load_grid(shapefile_path):
    grid_gdf = gpd.read_file(shapefile_path)
    return grid_gdf

@st.cache_data
@st.cache_data
def load_grid_centroids(_grid_gdf):
    """
    Compute centroids in projected CRS and reproject them to geographic CRS.

    Parameters:
        _grid_gdf (GeoDataFrame): Projected GeoDataFrame of grid cells.

    Returns:
        GeoDataFrame: DataFrame with 'cell_id', 'centroid_lon', 'centroid_lat'.
    """
    # Compute centroids in projected CRS (EPSG:3310)
    centroids_projected = _grid_gdf.geometry.centroid

    # Create a GeoDataFrame with centroids
    gdf_centroids = gpd.GeoDataFrame(
        _grid_gdf[['cell_id']].copy(),
        geometry=centroids_projected,
        crs=_grid_gdf.crs
    )

    # Reproject centroids to geographic CRS (EPSG:4326)
    gdf_centroids = gdf_centroids.to_crs(epsg=4326)

    # Extract longitude and latitude
    gdf_centroids['centroid_lon'] = gdf_centroids.geometry.x
    gdf_centroids['centroid_lat'] = gdf_centroids.geometry.y

    # Return only necessary columns
    return gdf_centroids[['cell_id', 'centroid_lon', 'centroid_lat']]

def load_models(model_path='models/', scaler_path='models/scaler.joblib'):
    """
    Load trained models and scaler from the specified directory.

    Parameters:
        model_path (str): Path to the directory containing model files.
        scaler_path (str): Path to the scaler file.

    Returns:
        dict: Dictionary of models keyed by cluster.
        StandardScaler: Loaded scaler object.
    """
    models = {}
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    for file in os.listdir(model_path):
        if file.startswith('LightGBM_cluster_') and file.endswith('.joblib'):
            try:
                cluster = int(file.split('_')[-1].split('.joblib')[0])
                models[cluster] = joblib.load(os.path.join(model_path, file))
            except ValueError:
                st.warning(f"Could not parse cluster number from {file}. Skipping this file.")
    return models, scaler
