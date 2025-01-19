# src/ingest_usfs_fod.py

import os
import pandas as pd
from datetime import datetime

# Define base directory relative to this script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

RAW_FILENAME = "usfs_fod_total.csv"  # The CSV you already have in data/raw

def main():
    """
    Reads 'usfs_fod_total.csv' from data/raw, applies optional processing,
    and saves a cleaned CSV to data/processed with a timestamped filename.
    """
    # 1. Load the raw CSV
    raw_path = os.path.join(RAW_DIR, RAW_FILENAME)
    if not os.path.exists(raw_path):
        print(f"[Error] {raw_path} does not exist. Please place 'usfs_fod_total.csv' in data/raw.")
        return

    df_raw = pd.read_csv(raw_path)
    print(f"[Info] Loaded USFS FOD CSV: {len(df_raw)} rows")

    # 2. Optional: Basic data inspection or filtering
    #    - For example, keep only columns of interest
    #    - Or filter for a certain date range, or state == CA
    #      (Though your dataset might already be CA-only)
    #    - Below is just an example:
    # Keep only state == CA
    df_raw = df_raw[df_raw["STATE"] == "CA"] if "STATE" in df_raw else df_raw
    # Let's suppose we only want to keep a few core columns for modeling:
    # (Adjust as needed; here we pick date-related columns + location + cause + size)
    columns_to_keep = [
        "FOD_ID", "FIRE_YEAR", "DISCOVERY_DATE", "CONT_DATE",
        "NWCG_CAUSE_CLASSIFICATION", "NWCG_GENERAL_CAUSE",
        "FIRE_SIZE", "FIRE_SIZE_CLASS", "LATITUDE", "LONGITUDE", 
        "STATE", "COUNTY", "FIPS_CODE", "FIPS_NAME"
    ]

    # Check which of these columns are actually in the CSV
    existing_cols = [col for col in columns_to_keep if col in df_raw.columns]
    df_processed = df_raw[existing_cols].copy()

    # 3. Optional: Convert date columns to datetime, if needed
    #    E.g. "DISCOVERY_DATE" in format "YYYY/MM/DD HH:MM:SS+00"? 
    #    Let's parse if it exists:
    if "DISCOVERY_DATE" in df_processed.columns:
        df_processed["DISCOVERY_DATE"] = pd.to_datetime(df_processed["DISCOVERY_DATE"], errors="coerce")
    if "CONT_DATE" in df_processed.columns:
        df_processed["CONT_DATE"] = pd.to_datetime(df_processed["CONT_DATE"], errors="coerce")

    # 4. (Optional) Filter rows if you only want certain years or non-null coords, etc.
    #    E.g., keep rows with FIRE_YEAR >= 2000
    df_processed = df_processed[df_processed["FIRE_YEAR"] >= 2000] if "FIRE_YEAR" in df_processed else df_processed

    # 5. Save the processed CSV to data/processed with a timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    processed_filename = f"usfs_fod_processed_{timestamp}.csv"
    processed_path = os.path.join(PROCESSED_DIR, processed_filename)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_processed.to_csv(processed_path, index=False)
    print(f"[Info] Processed USFS FOD data saved to {processed_path}")

if __name__ == "__main__":
    main()
