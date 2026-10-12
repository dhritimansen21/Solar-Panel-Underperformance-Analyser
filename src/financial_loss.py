"""
financial_loss.py
Quantifies energy losses (kWh/MWh) and financial losses in Indian Rupees (INR)
due to solar panel soiling, dust accumulation, and inverter outages.
Computes Return on Investment (ROI) for maintenance actions (panel cleaning, technical inspection).
Dynamically auto-detects sample interval (5-min, 15-min, 30-min, 60-min).
"""

import numpy as np
import pandas as pd


DEFAULT_TARIFF_INR_PER_KWH = 4.50  # Average commercial/industrial solar PPA tariff in India
DEFAULT_CLEANING_COST_PER_INV = 500.0  # Estimated cost to clean a commercial inverter string block


def detect_time_interval_hours(df: pd.DataFrame) -> float:
    """
    Detects the median time interval in hours from the dataset's timestamps.
    Supports 5m, 10m, 15m, 30m, 60m data streams.
    """
    if 'DATE_TIME' not in df.columns or len(df) < 2:
        return 0.25

    # Sample successive unique timestamps
    unique_ts = pd.to_datetime(df['DATE_TIME']).drop_duplicates().sort_values()
    if len(unique_ts) < 2:
        return 0.25

    diffs = unique_ts.diff().dt.total_seconds().dropna() / 3600.0
    median_diff = float(diffs.median())

    # Sanity bounds: 1 minute (0.016h) to 3 hours (3.0h)
    if 0.015 <= median_diff <= 3.0:
        return median_diff

    return 0.25


def compute_financial_losses(
    df: pd.DataFrame,
    tariff_inr: float = DEFAULT_TARIFF_INR_PER_KWH,
    interval_hours: float | None = None
) -> pd.DataFrame:
    """
    Computes energy deficit (kWh) and monetary loss (INR) using detected sampling interval.
    """
    df = df.copy()

    if interval_hours is None:
        interval_hours = detect_time_interval_hours(df)

    df['INTERVAL_HOURS'] = interval_hours

    # Energy gap = delta kW * interval hours
    power_gap_kw = np.maximum(df['EXPECTED_AC_POWER'] - df['AC_POWER'], 0.0)
    df['ENERGY_LOSS_KWH'] = power_gap_kw * interval_hours
    df['INR_LOSS'] = df['ENERGY_LOSS_KWH'] * tariff_inr

    return df


def get_plant_loss_summary(
    df: pd.DataFrame,
    tariff_inr: float = DEFAULT_TARIFF_INR_PER_KWH,
    cleaning_cost: float = DEFAULT_CLEANING_COST_PER_INV
) -> dict:
    """
    Returns plant-level financial and energy impact totals.
    """
    interval_hours = float(df['INTERVAL_HOURS'].iloc[0]) if 'INTERVAL_HOURS' in df.columns else detect_time_interval_hours(df)

    total_loss_kwh = float(df['ENERGY_LOSS_KWH'].sum())
    total_loss_mwh = total_loss_kwh / 1000.0
    total_loss_inr = float(df['INR_LOSS'].sum())

    total_actual_kwh = float(df['AC_POWER'].sum() * interval_hours)
    total_expected_kwh = float(df['EXPECTED_AC_POWER'].sum() * interval_hours)
    overall_plant_pr = (total_actual_kwh / total_expected_kwh) if total_expected_kwh > 0 else 1.0

    num_days = max(1, df['DATE'].nunique())
    daily_avg_loss_inr = total_loss_inr / num_days

    return {
        "total_energy_loss_kwh": round(total_loss_kwh, 1),
        "total_energy_loss_mwh": round(total_loss_mwh, 2),
        "total_revenue_loss_inr": round(total_loss_inr, 2),
        "daily_avg_loss_inr": round(daily_avg_loss_inr, 2),
        "total_actual_mwh": round(total_actual_kwh / 1000.0, 2),
        "total_expected_mwh": round(total_expected_kwh / 1000.0, 2),
        "overall_plant_pr": round(overall_plant_pr, 3),
        "num_days_analyzed": num_days,
        "tariff_rate_applied": tariff_inr,
        "sampling_interval_min": round(interval_hours * 60)
    }


def get_inverter_loss_table(
    df: pd.DataFrame,
    scorecard_df: pd.DataFrame,
    tariff_inr: float = DEFAULT_TARIFF_INR_PER_KWH,
    cleaning_cost: float = DEFAULT_CLEANING_COST_PER_INV
) -> pd.DataFrame:
    """
    Generates inverter-by-inverter financial loss table with maintenance ROI.
    """
    interval_hours = float(df['INTERVAL_HOURS'].iloc[0]) if 'INTERVAL_HOURS' in df.columns else detect_time_interval_hours(df)

    inv_loss = df.groupby('SOURCE_KEY').agg(
        TOTAL_LOSS_KWH=('ENERGY_LOSS_KWH', 'sum'),
        TOTAL_LOSS_INR=('INR_LOSS', 'sum'),
        ACTUAL_KWH=('AC_POWER', lambda p: (p * interval_hours).sum()),
        EXPECTED_KWH=('EXPECTED_AC_POWER', lambda p: (p * interval_hours).sum())
    ).reset_index()

    merged = pd.merge(scorecard_df, inv_loss, on='SOURCE_KEY', how='left')

    # Calculate cleaning ROI
    merged['CLEANING_COST_INR'] = cleaning_cost
    merged['NET_CLEANING_ROI_INR'] = merged['TOTAL_LOSS_INR'] - cleaning_cost
    merged['ROI_RATIO'] = (merged['TOTAL_LOSS_INR'] / np.maximum(cleaning_cost, 1.0))

    # Sort by total revenue lost descending
    merged = merged.sort_values(by='TOTAL_LOSS_INR', ascending=False).reset_index(drop=True)
    merged['RANK'] = merged.index + 1

    return merged


def get_daily_loss_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups data by date to track daily revenue lost and energy generation gap.
    """
    interval_hours = float(df['INTERVAL_HOURS'].iloc[0]) if 'INTERVAL_HOURS' in df.columns else detect_time_interval_hours(df)

    daily = df.groupby('DATE').agg(
        TOTAL_LOSS_INR=('INR_LOSS', 'sum'),
        TOTAL_LOSS_KWH=('ENERGY_LOSS_KWH', 'sum'),
        ACTUAL_KWH=('AC_POWER', lambda p: (p * interval_hours).sum()),
        EXPECTED_KWH=('EXPECTED_AC_POWER', lambda p: (p * interval_hours).sum()),
        PEAK_IRRADIATION=('IRRADIATION', 'max'),
        AVG_MODULE_TEMP=('MODULE_TEMPERATURE', 'mean')
    ).reset_index()

    daily['DAILY_PR'] = (daily['ACTUAL_KWH'] / np.maximum(daily['EXPECTED_KWH'], 1.0)).clip(0.0, 1.2)
    daily['DATE_STR'] = daily['DATE'].astype(str)

    return daily
