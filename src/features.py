"""
features.py
Feature engineering for solar expected power regression modeling.
"""

import numpy as np
import pandas as pd


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches dataset with solar, meteorological, and cyclical time features.
    """
    df = df.copy()

    # Continuous time of day (0.00 to 23.75)
    solar_time = df['HOUR'] + (df['MINUTE'] / 60.0)
    df['SOLAR_TIME'] = solar_time

    # Cyclical solar angle approximations
    df['SIN_TIME'] = np.sin(2 * np.pi * solar_time / 24.0)
    df['COS_TIME'] = np.cos(2 * np.pi * solar_time / 24.0)

    # Temperature differential and ratio
    df['TEMP_DIFF'] = df['MODULE_TEMPERATURE'] - df['AMBIENT_TEMPERATURE']
    df['TEMP_RATIO'] = df['MODULE_TEMPERATURE'] / np.maximum(df['AMBIENT_TEMPERATURE'], 1e-3)

    # Interaction terms
    df['IRRAD_X_MODTEMP'] = df['IRRADIATION'] * df['MODULE_TEMPERATURE']
    df['IRRAD_X_AMBTEMP'] = df['IRRADIATION'] * df['AMBIENT_TEMPERATURE']

    # Irradiation squared (accounts for non-linear saturation curves)
    df['IRRADIATION_SQ'] = df['IRRADIATION'] ** 2

    return df


FEATURE_COLUMNS = [
    'IRRADIATION',
    'MODULE_TEMPERATURE',
    'AMBIENT_TEMPERATURE',
    'TEMP_DIFF',
    'SOLAR_TIME',
    'SIN_TIME',
    'COS_TIME',
    'IRRAD_X_MODTEMP',
    'IRRAD_X_AMBTEMP',
    'IRRADIATION_SQ'
]

TARGET_COLUMN = 'AC_POWER'
