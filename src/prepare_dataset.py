# src/prepare_dataset.py

import os
import glob
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOAA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
USFS_FOD_PATH = os.path.join(BASE_DIR, "data", "processed", "usfs_fod_processed_20250116_183345.csv") 
NDVI_PATH = os.path.join(BASE_DIR, "data", "processed", "modis_ndvi.csv")  # <-- Adjust filename as needed

def combine_noaa():
    """Combine multiple NOAA processed CSVs into one dataframe."""
    pattern = os.path.join(NOAA_PROCESSED_DIR, "noaa_cdo_processed_*.csv")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} NOAA processed files to combine.")

    df_list = []
    for f in files:
        df_temp = pd.read_csv(f)
        # Expect columns: ['date','tmax_F','tmin_F','precip_in'] etc.
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_list.append(df_temp)
    
    df_noaa = pd.concat(df_list, ignore_index=True)
    # Drop duplicates in case of overlap
    df_noaa.drop_duplicates(subset='date', inplace=True)
    df_noaa.sort_values(by='date', inplace=True)
    df_noaa.reset_index(drop=True, inplace=True)
    print(f"Combined NOAA shape: {df_noaa.shape}")
    return df_noaa

def load_usfs_fod():
    """
    Load the processed USFS Fire Occurrence CSV.
    We assume it has a 'DISCOVERY_DATE' or something similar
    to indicate the date the fire was discovered.
    """
    df_usfs = pd.read_csv(USFS_FOD_PATH)
    # Example columns: [FIRE_YEAR, DISCOVERY_DATE, ...]
    if 'DISCOVERY_DATE' in df_usfs.columns:
        df_usfs['DISCOVERY_DATE'] = pd.to_datetime(df_usfs['DISCOVERY_DATE'], errors='coerce').dt.date
    return df_usfs

def load_ndvi():
    """
    Load the NDVI CSV which has at least ['date','ndvi'] columns.
    Adjust NDVI_PATH or columns as needed.
    """
    df_ndvi = pd.read_csv(NDVI_PATH)
    df_ndvi['date'] = pd.to_datetime(df_ndvi['date'])
    return df_ndvi

def merge_data(df_noaa, df_usfs, df_ndvi):
    """
    Merge weather + NDVI + fire occurrence on date.
    
    1) Merge NOAA & NDVI
    2) Build daily 'fire_occurred' from USFS
    3) Merge fire occurrence
    """

    # 1) Merge NOAA & NDVI on 'date'
    #    We'll keep all NOAA dates; NDVI might have fewer or different dates.
    df_combined = pd.merge(df_noaa, df_ndvi, on='date', how='left')
    # If NDVI is missing on some days, decide if you want to fill with a default:
    # df_combined['ndvi'] = df_combined['ndvi'].fillna(method='ffill') # Example forward fill

    # 2) Create daily fire_occurred from USFS
    df_usfs['fire_occurred'] = 1
    daily_fire = df_usfs.groupby('DISCOVERY_DATE')['fire_occurred'].max().reset_index()
    daily_fire.rename(columns={'DISCOVERY_DATE': 'date'}, inplace=True)
    daily_fire['date'] = pd.to_datetime(daily_fire['date'])

    # 3) Merge fire occurrence with combined (NOAA + NDVI)
    df_merged = pd.merge(df_combined, daily_fire, on='date', how='left')
    df_merged['fire_occurred'] = df_merged['fire_occurred'].fillna(0).astype(int)

    return df_merged

def main():
    df_noaa = combine_noaa()
    df_usfs = load_usfs_fod()
    df_ndvi = load_ndvi()

    df_merged = merge_data(df_noaa, df_usfs, df_ndvi)

    print(f"Merged dataset shape: {df_merged.shape}")
    print(df_merged.head(10))

    # Optionally save final dataset
    out_path = os.path.join(NOAA_PROCESSED_DIR, "merged_weather_fire_ndvi.csv")
    df_merged.to_csv(out_path, index=False)
    print(f"Saved merged dataset to {out_path}")

if __name__ == "__main__":
    main()
