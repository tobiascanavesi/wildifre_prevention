# src/pages/Prediction.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from shapely.geometry import Polygon
import joblib
import requests
from datetime import datetime, timedelta
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set page title
st.set_page_config(
    page_title="🔥 Wildfire Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔥 Wildfire Prediction for Next 7 Days")

# Sidebar for User Inputs
st.sidebar.header("Prediction Settings")

# Function to load grid centroids
@st.cache_data
def load_grid_centroids(shapefile_path):
    grid_gdf = gpd.read_file(shapefile_path)
    centroids_projected = grid_gdf.geometry.centroid
    gdf_centroids = gpd.GeoDataFrame(
        grid_gdf[['cell_id']].copy(),
        geometry=centroids_projected,
        crs=grid_gdf.crs
    )
    gdf_centroids = gdf_centroids.to_crs(epsg=4326)
    gdf_centroids['centroid_lon'] = gdf_centroids.geometry.x
    gdf_centroids['centroid_lat'] = gdf_centroids.geometry.y
    return gdf_centroids[['cell_id', 'centroid_lon', 'centroid_lat']]

# Load grid centroids
GRID_SHP_PATH = "data/raw/ca_grid_10km.shp"
grid_centroids = load_grid_centroids(GRID_SHP_PATH)

# Model and Scaler Loading
def load_models(model_path='models/', scaler_path='models/scaler.joblib'):
    import os
    models = {}
    scaler = joblib.load(scaler_path)
    for file in os.listdir(model_path):
        if file.startswith('LightGBM_cluster_') and file.endswith('.joblib'):
            cluster = int(file.split('_')[-1].split('.')[0])
            models[cluster] = joblib.load(f'{model_path}/{file}')
    return models, scaler

models, scaler = load_models()

# Function to fetch NOAA API data
def fetch_noaa_forecast(api_token, location_id, start_date, end_date):
    url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'
    headers = {'token': api_token}
    params = {
        'datasetid': 'GHCND',
        'locationid': location_id,
        'startdate': start_date,
        'enddate': end_date,
        'limit': 1000
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return None

# Function to prepare features
def prepare_features(forecast_df, latest_ndvi_df, scaler, clustering_labels):
    # Feature Engineering similar to training
    forecast_df['day_of_year'] = forecast_df['date'].dt.dayofyear
    forecast_df['day_of_month'] = forecast_df['date'].dt.day
    forecast_df['month'] = forecast_df['date'].dt.month_name()
    forecast_df['day_of_week'] = forecast_df['date'].dt.day_name()
    
    def get_season(month):
        if month in ['December', 'January', 'February']:
            return 'Winter'
        elif month in ['March', 'April', 'May']:
            return 'Spring'
        elif month in ['June', 'July', 'August']:
            return 'Summer'
        else:
            return 'Fall'
    
    forecast_df['season'] = forecast_df['month'].apply(get_season)
    
    # Moving averages
    window_sizes = [7, 15, 30, 60, 90, 180, 360]
    for window in window_sizes:
        forecast_df[f'precip_ma_{window}d'] = forecast_df.groupby('cell_id')['precip_in'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        forecast_df[f'tmax_ma_{window}d'] = forecast_df.groupby('cell_id')['tmax_F'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        forecast_df[f'tmin_ma_{window}d'] = forecast_df.groupby('cell_id')['tmin_F'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        forecast_df[f'ndvi_ma_{window}d'] = forecast_df.groupby('cell_id')['ndvi_latest'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    
    # NDVI lag feature (assuming latest_ndvi is the same as ndvi_latest)
    forecast_df['ndvi_lag_1d'] = forecast_df['ndvi_latest'].shift(1)
    
    # Previous fires - assuming data is up-to-date
    forecast_df['fires_last_7d'] = forecast_df.groupby('cell_id')['fire_occurred'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    
    # Handle missing values
    forecast_df = forecast_df.groupby('cell_id').apply(lambda group: group.fillna(method='ffill')).reset_index(drop=True)
    forecast_df = forecast_df.fillna(forecast_df.mean())
    
    # One-Hot Encoding
    categorical_cols = ['month', 'day_of_week', 'season']
    for col in categorical_cols:
        forecast_df[col] = forecast_df[col].fillna(forecast_df[col].mode()[0])
    
    forecast_encoded = pd.get_dummies(forecast_df, columns=categorical_cols, drop_first=True)
    
    # Align with training features
    # Load the same feature columns used during training
    # For simplicity, assume that feature_cols are known or stored
    
    # Scaling
    numerical_cols = [col for col in forecast_encoded.columns if forecast_encoded[col].dtype in ['int64', 'float64'] and col not in ['fire_occurred', 'cell_id', 'date']]
    forecast_encoded[numerical_cols] = scaler.transform(forecast_encoded[numerical_cols])
    
    return forecast_encoded

# Prediction Function
def make_predictions(forecast_encoded, models):
    predictions = []
    for idx, row in forecast_encoded.iterrows():
        cell_id = row['cell_id']
        cluster = row['cluster']
        model = models.get(cluster, None)
        if model:
            # Drop non-feature columns
            features = row.drop(['fire_occurred', 'cell_id', 'date', 'cluster']).values.reshape(1, -1)
            proba = model.predict_proba(features)[0][1]
            pred = model.predict(features)[0]
            predictions.append({
                'cell_id': cell_id,
                'date': row['date'],
                'fire_probability': proba,
                'fire_prediction': pred
            })
    return pd.DataFrame(predictions)

# Visualization Function
def visualize_predictions(predictions, grid_centroids):
    merged = pd.merge(predictions, grid_centroids, on='cell_id', how='left')
    
    # Create a map
    m = folium.Map(location=[36.7783, -119.4179], zoom_start=6)
    
    # Add circle markers
    for _, row in merged.iterrows():
        folium.CircleMarker(
            location=[row['centroid_lat'], row['centroid_lon']],
            radius=5,
            popup=f"Cell ID: {row['cell_id']}<br>Date: {row['date'].date()}<br>Fire Probability: {row['fire_probability']:.2f}",
            color='red' if row['fire_prediction'] == 1 else 'blue',
            fill=True,
            fill_color='red' if row['fire_prediction'] == 1 else 'blue'
        ).add_to(m)
    
    st_map = st_folium(m, width=700, height=500)
    return st_map

# Main Prediction Workflow
def main():
    # User inputs
    st.sidebar.header("Prediction Inputs")
    
    # Select grid cells for prediction
    select_all_cells = st.sidebar.checkbox("Select All Grid Cells for Prediction", value=True)
    if select_all_cells:
        selected_prediction_cells = grid_centroids['cell_id'].unique().tolist()
        st.sidebar.write(f"**All {len(selected_prediction_cells)} Grid Cells Selected for Prediction**")
    else:
        selected_prediction_cells = st.sidebar.multiselect(
            "Select Grid Cell IDs for Prediction",
            options=grid_centroids['cell_id'].unique(),
            default=None
        )
    
    # NOAA API Token Input
    api_token = st.sidebar.text_input("Enter NOAA API Token", type="password")
    
    # NOAA Location ID
    location_id = st.sidebar.text_input("Enter NOAA Location ID", value="CITY:US360019")  # Example: New York
    
    # Prediction Button
    if st.sidebar.button("Run Prediction"):
        if not api_token:
            st.error("Please enter your NOAA API Token.")
            return
        if not selected_prediction_cells:
            st.error("Please select at least one grid cell for prediction.")
            return
        
        # Fetch NOAA forecast data
        today = datetime.today()
        start_date = today.strftime('%Y-%m-%d')
        end_date = (today + timedelta(days=7)).strftime('%Y-%m-%d')
        
        forecast_data = fetch_noaa_forecast(api_token, location_id, start_date, end_date)
        
        if forecast_data:
            # Convert to DataFrame
            forecast_df = pd.DataFrame(forecast_data)
            # Ensure 'cell_id' is in forecast_df if needed
            # For demonstration, assume forecast_df has 'cell_id', 'date', 'precip_in', 'tmax_F', 'tmin_F'
            
            # Mock NDVI data - Replace with actual NDVI fetching logic
            # Here, we use the latest available NDVI
            latest_ndvi_df = pd.read_csv("data/processed/latest_ndvi.csv")  # Ensure this file exists
            
            # Merge forecast with latest NDVI
            forecast_df = pd.merge(forecast_df, latest_ndvi_df, on='cell_id', how='left')
            
            # Prepare features
            forecast_encoded = prepare_features(forecast_df, latest_ndvi_df, scaler, clustering_labels=None)
            
            # Make predictions
            predictions = make_predictions(forecast_encoded, models)
            
            # Visualize predictions
            st.markdown("### 🔥 Predicted Fire Risk Map")
            visualize_predictions(predictions, grid_centroids)
            
            # Display predictions table
            st.markdown("### 📄 Predictions Table")
            st.dataframe(predictions)
        else:
            st.error("Failed to fetch NOAA forecast data.")

if __name__ == "__main__":
    main()
