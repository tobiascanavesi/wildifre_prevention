# src/pages/Prediction.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from shapely.geometry import Polygon
import joblib
import os
import logging
import numpy as np
from datetime import datetime

# Import shared utilities
from utils import load_grid_centroids, load_grid, load_models

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Title of the Page
st.title("🔥 Wildfire Prediction for Next 7 Days")

# Load spatial data
GRID_SHP_PATH = os.path.join("data", "raw", "ca_grid_10km.shp")
grid_gdf = load_grid(GRID_SHP_PATH)
grid_centroids = load_grid_centroids(grid_gdf)
grid_centroids['cell_id'] = grid_centroids['cell_id'].astype(str)

# Load models
try:
    models, scaler = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

def map_selection_interface():
    """Render map interface for spatial selection"""
    st.sidebar.header("Spatial Selection")
    
    m = folium.Map(location=[36.7783, -119.4179], zoom_start=6)
    draw = Draw(
        export=True,
        position='topleft',
        draw_options={'polygon': True, 'rectangle': True, 'circle': False}
    )
    draw.add_to(m)
    
    map_output = st_folium(m, width=700, height=500) or {}
    selected_cells = []
    
    # Check if drawings exist and are polygons
    if map_output.get('all_drawings'):
        for drawing in map_output['all_drawings']:
            if isinstance(drawing, dict) and drawing.get('geometry', {}).get('type') == 'Polygon':
                try:
                    coords = drawing['geometry']['coordinates'][0]
                    polygon = Polygon(coords)
                    
                    gdf_poly = gpd.GeoDataFrame(
                        geometry=[polygon], 
                        crs="EPSG:4326"
                    ).to_crs(grid_gdf.crs)
                    
                    selected = gpd.sjoin(
                        grid_gdf.to_crs(grid_gdf.crs),
                        gdf_poly,
                        predicate='within'
                    )
                    if not selected.empty:
                        selected_cells.extend(selected['cell_id'].astype(str).tolist())
                except Exception as e:
                    logging.error(f"Error processing drawing: {str(e)}")
                    continue
    
    return list(set(selected_cells))

def station_selection_interface():
    """Render station selection interface"""
    merged_path = os.path.join("data", "processed", "merged_future_dataset.csv")
    if not os.path.exists(merged_path):
        return []
    
    stations = pd.read_csv(merged_path, usecols=['station_id', 'station_name'])
    stations = stations.drop_duplicates()
    
    st.sidebar.header("Weather Data Selection")
    selected_stations = st.sidebar.multiselect(
        "Select Weather Stations",
        options=stations['station_name'].unique(),
        help="Select stations to use for weather data"
    )
    
    station_ids = stations[stations['station_name'].isin(selected_stations)]['station_id'].unique()
    return list(station_ids)

def prepare_features(forecast_df, scaler):
    """Prepare features for prediction"""
    logging.info("Starting feature preparation...")
    
    # Feature Engineering
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
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
        forecast_df[f'precip_ma_{window}d'] = (
            forecast_df.groupby('cell_id')['forecast_precip_in']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        forecast_df[f'tmax_ma_{window}d'] = (
            forecast_df.groupby('cell_id')['forecast_tmax_F']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        forecast_df[f'tmin_ma_{window}d'] = (
            forecast_df.groupby('cell_id')['forecast_tmin_F']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        forecast_df[f'ndvi_ma_{window}d'] = (
            forecast_df.groupby('cell_id')['ndvi_latest']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    # NDVI lag feature
    forecast_df['ndvi_lag_1d'] = forecast_df.groupby('cell_id')['ndvi_latest'].shift(1)

    # Handle missing values
    forecast_df.fillna(method='ffill', inplace=True)
    forecast_df.fillna(method='bfill', inplace=True)

    # Rename forecast columns
    forecast_df.rename(columns={
        'forecast_precip_in': 'precip_in',
        'forecast_tmax_F': 'tmax_F',
        'forecast_tmin_F': 'tmin_F',
        'ndvi_latest': 'ndvi'
    }, inplace=True)

    # One-Hot Encoding
    forecast_encoded = pd.get_dummies(forecast_df, columns=['month', 'day_of_week', 'season'], drop_first=True)

    # Load feature names
    feature_names_path = os.path.join('models', 'feature_names.txt')
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f]

    # Add missing features
    for feature in feature_names:
        if feature not in forecast_encoded.columns:
            forecast_encoded[feature] = 0

    # Drop extra features
    extra_features = set(forecast_encoded.columns) - set(feature_names) - {'cell_id', 'date', 'cluster'}
    if extra_features:
        logging.warning(f"Dropping extra features: {extra_features}")
        forecast_encoded = forecast_encoded.drop(columns=list(extra_features))

    # Maintain column order
    forecast_encoded = forecast_encoded[feature_names + ['cell_id', 'date', 'cluster']]

    # Load numerical features
    numerical_features_path = os.path.join('models', 'numerical_features.txt')
    with open(numerical_features_path, 'r') as f:
        numerical_cols = [line.strip() for line in f]
    # transform cell_id to string
    
    # Scale features
    try:
        forecast_encoded[numerical_cols] = scaler.transform(forecast_encoded[numerical_cols])
    except Exception as e:
        logging.error(f"Scaling error: {e}")
        raise

    logging.info("Feature preparation completed.")
    return forecast_encoded

def make_predictions(forecast_encoded, models):
    """Make predictions using cluster-specific models"""
    predictions = []
    
    if not {'cell_id', 'date', 'cluster'}.issubset(forecast_encoded.columns):
        raise ValueError("Missing required columns in input data")
    
    for idx, row in forecast_encoded.iterrows():
        try:
            cluster = int(row['cluster'])
            model = models.get(cluster, None)
            
            if not model:
                continue
                
            features = row.drop(['cell_id', 'date', 'cluster']).values.reshape(1, -1)
            proba = model.predict_proba(features)[0][1]
            pred = model.predict(features)[0]
            
            predictions.append({
                'cell_id': row['cell_id'],
                'date': row['date'],
                'fire_probability': proba,
                'fire_prediction': pred,
                'cluster': cluster
            })
            
        except Exception as e:
            logging.error(f"Prediction error for row {idx}: {str(e)}")
            continue
    
    if not predictions:
        raise ValueError("No predictions generated - check input data and models")
    
    return pd.DataFrame(predictions)

def visualize_predictions(predictions, grid_centroids):
    """Visualize predictions on an interactive map"""
    # Convert cell_id to string in both DataFrames
    predictions['cell_id'] = predictions['cell_id'].astype(str)
    grid_centroids['cell_id'] = grid_centroids['cell_id'].astype(str)
    
    merged = pd.merge(predictions, grid_centroids, on='cell_id', how='left')
    
    # Rest of the function remains the same...
    # Handle missing coordinates
    if merged[['centroid_lat', 'centroid_lon']].isnull().sum().sum() > 0:
        st.warning("Some predictions missing location data")
        merged = merged.dropna(subset=['centroid_lat', 'centroid_lon'])
    
    # Create base map
    m = folium.Map(
        location=[merged['centroid_lat'].mean(), merged['centroid_lon'].mean()],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Add heatmap
    heat_data = [[row['centroid_lat'], row['centroid_lon'], row['fire_probability']] 
                for _, row in merged.iterrows()]
    folium.plugins.HeatMap(
        heat_data,
        name='Fire Risk',
        min_opacity=0.2,
        radius=20,
        blur=15,
        max_zoom=12
    ).add_to(m)
    
    # Add markers
    marker_cluster = folium.plugins.MarkerCluster(name="Individual Cells")
    for _, row in merged.iterrows():
        color = 'red' if row['fire_prediction'] == 1 else 'blue'
        popup = folium.Popup(
            f"<b>Cell ID:</b> {row['cell_id']}<br>"
            f"<b>Date:</b> {row['date'].date()}<br>"
            f"<b>Risk:</b> {row['fire_probability']:.2%}",
            max_width=250
        )
        folium.CircleMarker(
            location=[row['centroid_lat'], row['centroid_lon']],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            popup=popup,
            tooltip=f"Risk: {row['fire_probability']:.2%}"
        ).add_to(marker_cluster)
    marker_cluster.add_to(m)
    
    folium.LayerControl().add_to(m)
    st_folium(m, width=1200, height=600)

def main():
    # Selection interface
    st.sidebar.header("Prediction Settings")
    
    # Selection method
    selection_method = st.sidebar.radio(
        "Grid Selection Method",
        ("Map Selection", "Manual Input")
    )
    
    selected_cells = []
    if selection_method == "Map Selection":
        selected_cells = map_selection_interface()
    else:
        selected_cells = st.sidebar.multiselect(
            "Select Grid Cells",
            options=grid_centroids['cell_id'].unique(),
            help="Select grid cells for prediction"
        )
    
    # Station selection
    selected_station_ids = station_selection_interface()
    
    # Prediction execution
    if st.sidebar.button("Run Prediction"):
        if not selected_cells:
            st.error("Please select at least one grid cell")
            return
        if not selected_station_ids:
            st.error("Please select at least one weather station")
            return
        
        try:
            # Load data
            merged_path = os.path.join("data", "processed", "merged_future_dataset.csv")
            merged_df = pd.read_csv(merged_path, parse_dates=['date'])
            
            # Clean duplicate columns
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            merged_df['cell_id'] = merged_df['cell_id'].astype(str).str.strip()
            merged_df['station_id'] = merged_df['station_id'].str.strip() 
            print(merged_df.head())
            selected_cells = [str(cell) for cell in selected_cells]

            # Filter data
            filtered = merged_df[
                (merged_df['cell_id'].isin(selected_cells)) &
                (merged_df['station_id'].isin(selected_station_ids))
            ]
            print(filtered.head())
            print(selected_cells, selected_station_ids)
            print(merged_df.dtypes)
            if filtered.empty:
                st.warning("No data matching selected criteria")
                return
            
            # Process predictions
            with st.spinner("Generating predictions..."):
                forecast_encoded = prepare_features(filtered, scaler)
                print(forecast_encoded, forecast_encoded.dtypes)
                predictions = make_predictions(forecast_encoded, models)
                
                if predictions.empty:
                    st.warning("No predictions generated")
                    return
                
                # Display results
                st.header("Wildfire Prediction Results")
                visualize_predictions(predictions, grid_centroids)
                
                # Statistics
                st.subheader("Prediction Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High Risk Cells", 
                             len(predictions[predictions['fire_prediction'] == 1]))
                    st.metric("Average Probability", 
                             f"{predictions['fire_probability'].mean():.2%}")
                with col2:
                    st.write("### Probability Distribution")
                    st.bar_chart(predictions['fire_probability'].value_counts(bins=10))
                
                # Raw data
                st.subheader("Detailed Predictions")
                st.dataframe(
                    predictions.sort_values('fire_probability', ascending=False),
                    height=400
                )
            st.session_state.predictions = predictions
            st.session_state.last_successful_run = datetime.now()
            st.session_state.show_predictions = True
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.session_state.show_predictions = False
            logging.error(f"Prediction error: {str(e)}")
            
    # Add this after the prediction button block (outside the if statement)
    if st.session_state.get('show_predictions', False):
        # Create a container for predictions
        prediction_container = st.container()
        
        with prediction_container:
            st.header("Wildfire Prediction Results")
            
            # Add a clear button
            if st.button("Clear Previous Predictions"):
                st.session_state.show_predictions = False
                st.session_state.predictions = None
                st.experimental_rerun()
            
            # Show debug info
            debug_expander = st.expander("Debug Information")
            with debug_expander:
                st.write("**Last successful prediction run:**", 
                        st.session_state.get('last_successful_run', 'Never'))
                st.write("**Prediction data shape:**", 
                        st.session_state.predictions.shape)
                st.write("**Sample predictions:**")
                st.write(st.session_state.predictions.head(3))
            
            # Visualizations
            try:
                visualize_predictions(st.session_state.predictions, grid_centroids)
                
                # Statistics
                st.subheader("Prediction Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High Risk Cells", 
                            len(st.session_state.predictions[
                                st.session_state.predictions['fire_prediction'] == 1
                            ]))
                    st.metric("Average Probability", 
                            f"{st.session_state.predictions['fire_probability'].mean():.2%}")
                with col2:
                    st.write("### Probability Distribution")
                    st.bar_chart(st.session_state.predictions['fire_probability']
                                .value_counts(bins=10, sort=False))
                
                # Raw data
                st.subheader("Detailed Predictions")
                st.dataframe(
                    st.session_state.predictions.sort_values('fire_probability', ascending=False),
                    height=400
                )
                
            except Exception as e:
                st.error(f"Error displaying results: {str(e)}")
                logging.error(f"Visualization error: {str(e)}")

if __name__ == "__main__":
    main()