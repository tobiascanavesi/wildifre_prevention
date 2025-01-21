import os
import requests
import pandas as pd
from datetime import datetime, timedelta

CDO_TOKEN = "FBYGDxCnrDBxxUXdXqAbPRFUYqMBWzqK"

# NOAA CDO specifics
CDO_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
DATASET_ID = "GHCND"  # daily data
# We need to think you to integrate each station's data with the fire data, in the sense that each station will show different 
# information and that could help us to have a better prediction of the fire risk.
# To see the station : https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00023234/detail
# Los angeles station is GHCND:USW00093134
STATION_ID = "GHCND:USW00023234"  # Example: SFO Airport
START_DATE = "2015-01-01"
END_DATE = "2015-12-31"

# Local directories (relative to the script)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def fetch_noaa_cdo_data(dataset_id, station_id, start_date, end_date, limit=1000):
    """
    Fetch daily climate data from NOAA CDO for a given station and date range.
    Returns a DataFrame with columns [date, datatype, station, value].
    """
    headers = {
        "token": CDO_TOKEN
    }
    params = {
        "datasetid": dataset_id,
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "limit": limit,
        "units": "standard",
        "datatypeid": ["TMAX", "TMIN", "PRCP"],  # daily max/min temp, precipitation
        "includemetadata": "false"
    }
    response = requests.get(CDO_URL, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    
    if "results" not in data:
        print("No data found. Check station ID, date range, or token.")
        return pd.DataFrame()
    
    return pd.DataFrame(data["results"])

def main():
    # 1. Fetch data
    df_raw = fetch_noaa_cdo_data(
        dataset_id=DATASET_ID,
        station_id=STATION_ID,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    if df_raw.empty:
        print("No data retrieved. Exiting.")
        return
    
    # 2. Save RAW data to data/raw with a timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    raw_filename = f"noaa_cdo_raw_{timestamp}.csv"
    raw_path = os.path.join(RAW_DIR, raw_filename)
    
    # Make sure raw directory exists
    os.makedirs(RAW_DIR, exist_ok=True)

    df_raw.to_csv(raw_path, index=False)
    print(f"[Local] Saved raw NOAA CDO data to {raw_path}")
    
    # 3. Process: Pivot TMAX, TMIN, PRCP into columns, rename them
    df_raw['date'] = pd.to_datetime(df_raw['date']).dt.date
    df_pivot = df_raw.pivot_table(
        index='date',
        columns='datatype',
        values='value'
    ).reset_index()

    # Rename columns
    df_pivot.rename(columns={
        'TMAX': 'tmax_F',
        'TMIN': 'tmin_F',
        'PRCP': 'precip_in'
    }, inplace=True)
    
    # 4. Save processed data to data/processed
    processed_filename = f"noaa_cdo_processed_{timestamp}.csv"
    processed_path = os.path.join(PROCESSED_DIR, processed_filename)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df_pivot.to_csv(processed_path, index=False)
    print(f"[Local] Saved processed NOAA CDO data to {processed_path}")

if __name__ == "__main__":
    main()
