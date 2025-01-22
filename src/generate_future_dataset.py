# src/generate_future_dataset.py

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import asyncio
import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry
from geopy.distance import geodesic

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generate_future_dataset.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env
load_dotenv()

# Constants
AVERAGE_PRECIP_IN = 0.1  # Average precipitation in inches when there's a chance
MODIS_NDVI_CSV_PATH = os.path.join("data", "processed", "modis_ndvi_10km.csv")
LATEST_NDVI_CSV_PATH = os.path.join("data", "processed", "latest_ndvi.csv")
STATIONS_CSV_PATH = os.path.join("data", "raw", "stations.csv")
CELL_CENTROIDS_CSV_PATH = os.path.join("data", "raw", "cell_centroids.csv")  # New
CLUSTERING_LABELS_PATH = os.path.join("data", "processed", "clustering_labels.csv")  # Cluster assignments
FORECAST_DATA_DIR = os.path.join("data", "processed", "forecast_data")
OUTPUT_CSV_PATH = os.path.join("data", "processed", "merged_future_dataset.csv")

# NOAA NWS API Configuration
NWS_API_BASE_URL = "https://api.weather.gov"
USER_AGENT = "wildfire-prevention-app (tobiascanavesi@gmail.com)"  # Ensure this is accurate

# Maximum number of concurrent requests
MAX_CONCURRENT_REQUESTS = 5

# Semaphore for rate limiting
SEM = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def process_modis_ndvi(modis_csv_path, latest_ndvi_csv_path):
    """
    Processes the MODIS NDVI data to extract the latest non-null NDVI value for each cell.

    Parameters:
        modis_csv_path (str): Path to the modis_ndvi_10km.csv file.
        latest_ndvi_csv_path (str): Path where latest_ndvi.csv will be saved.
    """
    try:
        logging.info(f"Loading MODIS NDVI data from {modis_csv_path}")
        # Read the CSV assuming it has headers
        ndvi_df = pd.read_csv(modis_csv_path)
        
        # Ensure the necessary columns are present
        required_columns = {'cell_id', 'ndvi', 'date'}
        if not required_columns.issubset(ndvi_df.columns):
            logging.error(f"CSV columns {ndvi_df.columns.tolist()} do not match the required columns {required_columns}")
            return None
        
        # Convert 'date' to datetime
        ndvi_df['date'] = pd.to_datetime(ndvi_df['date'], errors='coerce')
        
        # Drop rows with invalid dates
        invalid_dates = ndvi_df['date'].isnull().sum()
        if invalid_dates > 0:
            logging.warning(f"Dropping {invalid_dates} rows with invalid dates.")
            ndvi_df = ndvi_df.dropna(subset=['date'])
        
        # Sort by date to ensure the latest records come first
        ndvi_df.sort_values(by=['cell_id', 'date'], ascending=[True, False], inplace=True)
        
        # Drop duplicates, keeping the first (latest) non-null ndvi_value
        latest_ndvi_df = ndvi_df.drop_duplicates(subset=['cell_id'], keep='first')
        
        # Drop rows where ndvi is null
        latest_ndvi_df = latest_ndvi_df.dropna(subset=['ndvi'])
        
        # Select required columns and rename 'ndvi' to 'ndvi_latest'
        latest_ndvi_df = latest_ndvi_df[['cell_id', 'ndvi']]
        latest_ndvi_df.rename(columns={'ndvi': 'ndvi_latest'}, inplace=True)
        
        # Save to CSV
        os.makedirs(os.path.dirname(latest_ndvi_csv_path), exist_ok=True)
        latest_ndvi_df.to_csv(latest_ndvi_csv_path, index=False)
        logging.info(f"Latest NDVI data saved to {latest_ndvi_csv_path}")
        
        return latest_ndvi_df

    except Exception as e:
        logging.error(f"Error processing MODIS NDVI data: {e}")
        return None

def celsius_to_fahrenheit(celsius):
    """Converts Celsius to Fahrenheit."""
    return celsius * 9/5 + 32

def is_valid_coordinate(lat, lon):
    """Validates latitude and longitude values."""
    return -90 <= lat <= 90 and -180 <= lon <= 180

async def fetch_forecast(session, latitude, longitude):
    """
    Fetches the 7-day weather forecast for a given latitude and longitude using the NWS API.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session for making requests.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.

    Returns:
        pd.DataFrame: DataFrame containing 'date', 'tmax_F', 'tmin_F', 'precip_in' for each day.
    """
    async with SEM:
        try:
            if not is_valid_coordinate(latitude, longitude):
                logging.warning(f"Invalid coordinates ({latitude}, {longitude}). Skipping.")
                return None

            # Step 1: Get the forecast URL from the NWS API
            point_url = f"{NWS_API_BASE_URL}/points/{latitude},{longitude}"
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "application/geo+json"
            }
            async with session.get(point_url, headers=headers) as response:
                if response.status != 200:
                    logging.error(f"Error fetching point data for ({latitude}, {longitude}): HTTP {response.status}")
                    return None
                point_data = await response.json()

            forecast_url = point_data.get('properties', {}).get('forecast')
            if not forecast_url:
                logging.error(f"No forecast URL found for ({latitude}, {longitude}).")
                return None

            # Step 2: Fetch the forecast data
            async with session.get(forecast_url, headers=headers) as response:
                if response.status != 200:
                    logging.error(f"Error fetching forecast data for ({latitude}, {longitude}): HTTP {response.status}")
                    return None
                forecast_data = await response.json()

            periods = forecast_data.get('properties', {}).get('periods', [])
            if not periods:
                logging.error(f"No forecast periods found for ({latitude}, {longitude}).")
                return None

            # Organize data by day
            daily_forecast = {}
            for period in periods:
                start_time = period.get('startTime')
                is_daytime = period.get('isDaytime')
                temperature = period.get('temperature')
                temperature_unit = period.get('temperatureUnit')
                precip = period.get('probabilityOfPrecipitation', {}).get('value')  # Percentage

                if not start_time:
                    logging.warning(f"Missing 'startTime' in forecast period for ({latitude}, {longitude}). Skipping.")
                    continue

                try:
                    date = pd.to_datetime(start_time).date()
                except Exception as e:
                    logging.warning(f"Invalid 'startTime' format '{start_time}' for ({latitude}, {longitude}): {e}. Skipping.")
                    continue

                if temperature_unit != 'F':
                    try:
                        temperature = celsius_to_fahrenheit(float(temperature))
                    except Exception as e:
                        logging.warning(f"Invalid temperature value '{temperature}' for ({latitude}, {longitude}): {e}. Skipping.")
                        continue
                else:
                    try:
                        temperature = float(temperature)
                    except Exception as e:
                        logging.warning(f"Invalid temperature value '{temperature}' for ({latitude}, {longitude}): {e}. Skipping.")
                        continue

                # Initialize the day in daily_forecast
                if date not in daily_forecast:
                    daily_forecast[date] = {'tmax_F': None, 'tmin_F': None, 'precip_in': 0.0}

                # Update max and min temperatures
                if is_daytime:
                    if (daily_forecast[date]['tmax_F'] is None) or (temperature > daily_forecast[date]['tmax_F']):
                        daily_forecast[date]['tmax_F'] = temperature
                else:
                    if (daily_forecast[date]['tmin_F'] is None) or (temperature < daily_forecast[date]['tmin_F']):
                        daily_forecast[date]['tmin_F'] = temperature

                # Update precipitation
                if precip is not None and precip > 0:
                    # Assuming precip_in is the expected precipitation; adjust if necessary
                    daily_forecast[date]['precip_in'] += precip / 100 * 0.1  # Example: 10% chance translates to 0.1 inches

            if not daily_forecast:
                logging.error(f"No valid forecast data processed for ({latitude}, {longitude}).")
                return None

            # Convert daily_forecast to DataFrame
            forecast_list = []
            for date, data in daily_forecast.items():
                forecast_list.append({
                    'date': date,
                    'tmax_F': data['tmax_F'] if data['tmax_F'] is not None else np.nan,
                    'tmin_F': data['tmin_F'] if data['tmin_F'] is not None else np.nan,
                    'precip_in': round(data['precip_in'], 2)
                })

            forecast_df = pd.DataFrame(forecast_list)

            # Handle missing temperatures by imputing with average
            if forecast_df['tmax_F'].isnull().any():
                mean_tmax = forecast_df['tmax_F'].mean()
                forecast_df['tmax_F'].fillna(mean_tmax, inplace=True)
                logging.warning(f"Imputed missing 'tmax_F' with mean value {mean_tmax:.2f} for ({latitude}, {longitude}).")
            if forecast_df['tmin_F'].isnull().any():
                mean_tmin = forecast_df['tmin_F'].mean()
                forecast_df['tmin_F'].fillna(mean_tmin, inplace=True)
                logging.warning(f"Imputed missing 'tmin_F' with mean value {mean_tmin:.2f} for ({latitude}, {longitude}).")

            return forecast_df

        except Exception as e:
            logging.error(f"Unexpected error while fetching forecast for ({latitude}, {longitude}): {e}")
            return None

async def fetch_all_forecasts(stations_df, cell_centroids_df):
    """
    Fetches forecast data for all stations asynchronously and maps each cell to its nearest station.

    Parameters:
        stations_df (DataFrame): DataFrame containing 'station_id', 'station_name', 'latitude', 'longitude'.
        cell_centroids_df (DataFrame): DataFrame containing 'cell_id', 'centroid_lat', 'centroid_lon'.

    Returns:
        DataFrame: Merged DataFrame containing 'cell_id', 'station_id', 'station_name', 'date', 'tmax_F', 'tmin_F', 'precip_in'.
    """
    try:
        # Step 1: Assign each cell to the nearest station
        logging.info("Assigning each cell to the nearest weather station.")
        def assign_nearest_station(row):
            cell_coords = (row['centroid_lat'], row['centroid_lon'])
            min_distance = float('inf')
            nearest_station_id = None
            nearest_station_name = None
            for _, station in stations_df.iterrows():
                station_coords = (station['latitude'], station['longitude'])
                distance = geodesic(cell_coords, station_coords).kilometers
                if distance < min_distance:
                    min_distance = distance
                    nearest_station_id = station['station_id']
                    nearest_station_name = station['station_name']
            return pd.Series([nearest_station_id, nearest_station_name])

        cell_centroids_df[['nearest_station_id', 'nearest_station_name']] = cell_centroids_df.apply(assign_nearest_station, axis=1)
        logging.info("Assigned stations to all cells.")

        # Step 2: Fetch forecasts for each unique station
        unique_stations = stations_df[['station_id', 'station_name', 'latitude', 'longitude']].drop_duplicates()
        retry_options = ExponentialRetry(attempts=3, statuses={500, 502, 503, 504})
        async with RetryClient(
            raise_for_status=False,
            retry_options=retry_options,
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
        ) as session:
            tasks = []
            for _, station in unique_stations.iterrows():
                station_id = station['station_id']
                station_name = station['station_name']
                lat = station['latitude']
                lon = station['longitude']
                task = asyncio.create_task(fetch_forecast(session, lat, lon))
                tasks.append((station_id, station_name, task))

            results = {}
            for station_id, station_name, task in tasks:
                forecast_df = await task
                if forecast_df is not None:
                    forecast_df['station_id'] = station_id
                    forecast_df['station_name'] = station_name
                    results[station_id] = forecast_df
                else:
                    logging.warning(f"No forecast data retrieved for station {station_name} ({station_id}). Skipping.")

            if not results:
                logging.error("No forecast data was fetched for any station.")
                return pd.DataFrame()

            # Step 3: Combine all forecast data
            all_forecasts = []
            for station_id, df in results.items():
                df_copy = df.copy()
                df_copy['station_id'] = station_id
                df_copy['station_name'] = results[station_id]['station_name'].iloc[0] if not df_copy.empty else None
                all_forecasts.append(df_copy)
            merged_forecast_df = pd.concat(all_forecasts, ignore_index=True)
            logging.info(f"Fetched forecast data for {len(results)} stations.")

        # Step 4: Merge forecast data back to cell_centroids_df
        logging.info("Merging forecast data back to cell centroids.")
        # Melt forecast_df to have one row per cell per date
        # First, merge cell_centroids_df with forecast_df based on nearest_station_id
        merged_df = pd.merge(
            cell_centroids_df,
            merged_forecast_df,
            left_on='nearest_station_id',
            right_on='station_id',
            how='left',
            suffixes=('_cell', '_forecast')
        )

        # Drop unnecessary columns
        merged_df.drop(columns=['station_id'], inplace=True)

        # Rename columns for clarity
        merged_df.rename(columns={
            'nearest_station_id': 'station_id',
            'nearest_station_name': 'station_name',
            'tmax_F': 'forecast_tmax_F',
            'tmin_F': 'forecast_tmin_F',
            'precip_in': 'forecast_precip_in'
        }, inplace=True)

        logging.info("Merged forecast data with cell centroids.")

        return merged_df

    except Exception as e:
        logging.error(f"Exception occurred during forecast fetching and merging: {e}")
        return pd.DataFrame()

async def fetch_all_forecasts_async(stations_df, cell_centroids_df):
    """
    Wrapper to run the asynchronous forecast fetching and merging.

    Parameters:
        stations_df (DataFrame): DataFrame containing station details.
        cell_centroids_df (DataFrame): DataFrame containing cell centroid details.

    Returns:
        DataFrame: Merged forecast and cell data.
    """
    return await fetch_all_forecasts(stations_df, cell_centroids_df)

def save_forecast_data(merged_df, forecast_data_dir):
    """
    Saves the merged forecast data to the specified directory with a timestamp.

    Parameters:
        merged_df (DataFrame): DataFrame containing merged forecast and cell data.
        forecast_data_dir (str): Directory where the forecast data will be saved.
    """
    try:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"forecast_merged_{timestamp}.csv"
        filepath = os.path.join(forecast_data_dir, filename)
        os.makedirs(forecast_data_dir, exist_ok=True)
        merged_df.to_csv(filepath, index=False)
        logging.info(f"Saved merged forecast data to {filepath}")
    except Exception as e:
        logging.error(f"Error saving merged forecast data: {e}")

def load_latest_ndvi(ndvi_csv_path):
    """
    Loads the latest NDVI data.

    Parameters:
        ndvi_csv_path (str): Path to the latest_ndvi.csv file.

    Returns:
        DataFrame: DataFrame containing 'cell_id' and 'ndvi_latest'.
    """
    try:
        logging.info(f"Loading latest NDVI data from {ndvi_csv_path}")
        ndvi_df = pd.read_csv(ndvi_csv_path)
        if not {'cell_id', 'ndvi_latest'}.issubset(ndvi_df.columns):
            logging.error("Latest NDVI CSV must contain 'cell_id' and 'ndvi_latest' columns.")
            return None
        return ndvi_df
    except Exception as e:
        logging.error(f"Error loading latest NDVI data: {e}")
        return None

def merge_data(merged_forecast_df, ndvi_df, clustering_labels_df):
    """
    Merges forecast data with NDVI data and cluster assignments based on 'cell_id'.

    Parameters:
        merged_forecast_df (DataFrame): DataFrame containing merged forecast and cell data.
        ndvi_df (DataFrame): Latest NDVI data.
        clustering_labels_df (DataFrame): Cluster assignments for each cell.

    Returns:
        DataFrame: Final merged DataFrame ready for prediction.
    """
    try:
        logging.info("Merging forecast data with NDVI data.")
        # Merge on 'cell_id'
        final_df = pd.merge(
            merged_forecast_df,
            ndvi_df,
            on='cell_id',
            how='left'
        )

        # Handle missing NDVI values
        missing_ndvi = final_df['ndvi_latest'].isnull().sum()
        if missing_ndvi > 0:
            mean_ndvi = ndvi_df['ndvi_latest'].mean()
            final_df['ndvi_latest'].fillna(mean_ndvi, inplace=True)
            logging.warning(f"Filled {missing_ndvi} missing NDVI values with mean NDVI ({mean_ndvi:.4f}).")

        # Merge cluster assignments
        logging.info("Merging cluster assignments into the dataset.")
        final_df = pd.merge(
            final_df,
            clustering_labels_df,
            on='cell_id',
            how='left'
        )

        # Handle missing cluster assignments
        missing_clusters = final_df['cluster'].isnull().sum()
        if missing_clusters > 0:
            logging.warning(f"Filled {missing_clusters} missing cluster assignments with default cluster '0'.")
            final_df['cluster'].fillna(0, inplace=True)  # Assign to a default cluster if necessary
            final_df['cluster'] = final_df['cluster'].astype(int)

        # Add 'fire_occurred' column as NaN for future predictions
        final_df['fire_occurred'] = np.nan

        # Reorder and select relevant columns
        final_df = final_df[['cell_id', 'station_id', 'station_name', 'date', 'forecast_tmax_F', 'forecast_tmin_F', 'forecast_precip_in', 'ndvi_latest', 'fire_occurred', 'cluster']]

        return final_df

    except Exception as e:
        logging.error(f"Error merging forecast, NDVI, and cluster data: {e}")
        return None

def main():
    try:
        # Step 1: Process MODIS NDVI data
        latest_ndvi_df = process_modis_ndvi(MODIS_NDVI_CSV_PATH, LATEST_NDVI_CSV_PATH)
        if latest_ndvi_df is None:
            logging.error("Failed to process MODIS NDVI data. Exiting.")
            return

        # Step 2: Load NOAA Weather Stations
        if not os.path.exists(STATIONS_CSV_PATH):
            logging.error(f"Stations CSV file not found at {STATIONS_CSV_PATH}. Please create it with station details.")
            return

        stations_df = pd.read_csv(STATIONS_CSV_PATH)
        required_columns = {'station_id', 'station_name', 'latitude', 'longitude'}
        if not required_columns.issubset(stations_df.columns):
            logging.error(f"Stations CSV must contain the following columns: {required_columns}")
            return

        # Step 3: Load Cell Centroids
        if not os.path.exists(CELL_CENTROIDS_CSV_PATH):
            logging.error(f"Cell Centroids CSV file not found at {CELL_CENTROIDS_CSV_PATH}. Please create it with cell centroid details.")
            return

        cell_centroids_df = pd.read_csv(CELL_CENTROIDS_CSV_PATH)
        required_columns_centroids = {'cell_id', 'centroid_lat', 'centroid_lon'}
        if not required_columns_centroids.issubset(cell_centroids_df.columns):
            logging.error(f"Cell Centroids CSV must contain the following columns: {required_columns_centroids}")
            return

        # Step 4: Fetch Forecast Data and Assign to Cells
        logging.info("Fetching and assigning forecast data to cells.")
        merged_forecast_df = asyncio.run(fetch_all_forecasts(stations_df, cell_centroids_df))
        if merged_forecast_df.empty:
            logging.error("No forecast data was fetched or merged. Exiting.")
            return

        # Step 5: Save Merged Forecast Data
        save_forecast_data(merged_forecast_df, FORECAST_DATA_DIR)

        # Step 6: Load Latest NDVI Data
        ndvi_df = load_latest_ndvi(LATEST_NDVI_CSV_PATH)
        if ndvi_df is None:
            logging.error("Failed to load latest NDVI data. Exiting.")
            return

        # Step 7: Load Cluster Assignments
        if not os.path.exists(CLUSTERING_LABELS_PATH):
            logging.error(f"Clustering Labels CSV file not found at {CLUSTERING_LABELS_PATH}. Please run `model_training.py` first.")
            return

        clustering_labels_df = pd.read_csv(CLUSTERING_LABELS_PATH)
        required_columns_clustering = {'cell_id', 'cluster'}
        if not required_columns_clustering.issubset(clustering_labels_df.columns):
            logging.error(f"Clustering Labels CSV must contain the following columns: {required_columns_clustering}")
            return

        # Step 8: Merge Forecast, NDVI, and Cluster Data
        final_df = merge_data(merged_forecast_df, ndvi_df, clustering_labels_df)
        if final_df is None:
            logging.error("Failed to merge forecast, NDVI, and cluster data. Exiting.")
            return

        # Step 9: Save the Final Merged Dataset
        try:
            os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
            final_df.to_csv(OUTPUT_CSV_PATH, index=False)
            logging.info(f"Merged future dataset saved to {OUTPUT_CSV_PATH}")
        except Exception as e:
            logging.error(f"Error saving merged dataset: {e}")

    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}")

if __name__ == "__main__":
    main()
