"""
data_loader.py
Handles data loading, timestamp normalization, and merging of generation and weather sensor datasets.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

DEFAULT_DATA_DIR = Path(r"D:\solar panel underperformance analyser\data\raw")


def load_raw_data(plant_num: int = 1, data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw generation and weather CSV files for a given plant (1 or 2).
    """
    gen_file = data_dir / f"Plant_{plant_num}_Generation_Data.csv"
    weather_file = data_dir / f"Plant_{plant_num}_Weather_Sensor_Data.csv"

    if not gen_file.exists() or not weather_file.exists():
        raise FileNotFoundError(f"Missing data files for Plant {plant_num} in {data_dir}")

    # Plant 1 generation uses dayfirst=True ('%d-%m-%Y %H:%M')
    # Plant 2 and weather sensor files use ISO '%Y-%m-%d %H:%M:%S'
    if plant_num == 1:
        gen_df = pd.read_csv(gen_file)
        gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], dayfirst=True)
    else:
        gen_df = pd.read_csv(gen_file)
        gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'])

    weather_df = pd.read_csv(weather_file)
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])

    return gen_df, weather_df


def merge_and_clean_data(plant_num: int = 1, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Merges generation and weather sensor data on DATE_TIME.
    Adds time-derived features, calculates basic metrics, and tags night vs daylight slots.
    """
    gen_df, weather_df = load_raw_data(plant_num=plant_num, data_dir=data_dir)

    # Weather sensor data has a single sensor key per plant; drop SOURCE_KEY to avoid collision
    weather_clean = weather_df.drop(columns=['PLANT_ID', 'SOURCE_KEY'], errors='ignore')

    # Merge on DATE_TIME
    merged = pd.merge(gen_df, weather_clean, on='DATE_TIME', how='inner')

    # Add temporal features
    merged['DATE'] = merged['DATE_TIME'].dt.date
    merged['HOUR'] = merged['DATE_TIME'].dt.hour
    merged['MINUTE'] = merged['DATE_TIME'].dt.minute
    merged['TIME_SLOT'] = merged['DATE_TIME'].dt.strftime('%H:%M')
    merged['DAY_OF_WEEK'] = merged['DATE_TIME'].dt.dayofweek

    # Clean negative or erroneous sensor readings
    merged['IRRADIATION'] = merged['IRRADIATION'].clip(lower=0.0)
    merged['AC_POWER'] = merged['AC_POWER'].clip(lower=0.0)
    merged['DC_POWER'] = merged['DC_POWER'].clip(lower=0.0)

    # Compute temperature differential
    merged['TEMP_DIFF'] = merged['MODULE_TEMPERATURE'] - merged['AMBIENT_TEMPERATURE']

    # Solar flag: Daylight when irradiation > 0.005 kW/m² or AC_POWER > 0
    merged['IS_DAYLIGHT'] = (merged['IRRADIATION'] > 0.005) | (merged['AC_POWER'] > 0)

    # Sort deterministically
    merged = merged.sort_values(by=['SOURCE_KEY', 'DATE_TIME']).reset_index(drop=True)

    return merged


if __name__ == "__main__":
    for p in [1, 2]:
        df = merge_and_clean_data(plant_num=p)
        print(f"Plant {p} merged shape: {df.shape}")
        print(f"Plant {p} unique inverters: {df['SOURCE_KEY'].nunique()}")
        print(f"Plant {p} daylight rows: {df['IS_DAYLIGHT'].sum()} / {len(df)}")
