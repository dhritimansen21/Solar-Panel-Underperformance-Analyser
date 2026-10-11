"""
financial_loss.py
Quantifies energy losses (kWh/MWh) and financial losses in Indian Rupees (INR)
due to solar panel soiling, dust accumulation, and inverter outages.
Computes Return on Investment (ROI) for maintenance actions (panel cleaning, technical inspection).
"""

import numpy as np
import pandas as pd


DEFAULT_TARIFF_INR_PER_KWH = 4.50  # Average commercial/industrial solar PPA tariff in India
DEFAULT_CLEANING_COST_PER_INV = 500.0  # Estimated cost to clean a commercial inverter string block


def compute_financial_losses(
    df: pd.DataFrame,
    tariff_inr: float = DEFAULT_TARIFF_INR_PER_KWH
) -> pd.DataFrame:
    """
    Computes energy deficit (kWh) and monetary loss (INR) for each 15-minute slot.
    """
    df = df.copy()

    # 15-minute generation window = 0.25 hours
    power_gap_kw = np.maximum(df['EXPECTED_AC_POWER'] - df['AC_POWER'], 0.0)
    df['ENERGY_LOSS_KWH'] = power_gap_kw * 0.25
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
    total_loss_kwh = float(df['ENERGY_LOSS_KWH'].sum())
    total_loss_mwh = total_loss_kwh / 1000.0
    total_loss_inr = float(df['INR_LOSS'].sum())

    total_actual_kwh = float(df['AC_POWER'].sum() * 0.25)
    total_expected_kwh = float(df['EXPECTED_AC_POWER'].sum() * 0.25)
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
        "tariff_rate_applied": tariff_inr
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
    inv_loss = df.groupby('SOURCE_KEY').agg(
        TOTAL_LOSS_KWH=('ENERGY_LOSS_KWH', 'sum'),
        TOTAL_LOSS_INR=('INR_LOSS', 'sum'),
        ACTUAL_KWH=('AC_POWER', lambda p: (p * 0.25).sum()),
        EXPECTED_KWH=('EXPECTED_AC_POWER', lambda p: (p * 0.25).sum())
    ).reset_index()

    merged = pd.merge(scorecard_df, inv_loss, on='SOURCE_KEY', how='left')

    # Calculate cleaning ROI
    # For inverters suffering from dust/soiling or heavy degradation,
    # cleaning can recover the recoverable soiling losses
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
    daily = df.groupby('DATE').agg(
        TOTAL_LOSS_INR=('INR_LOSS', 'sum'),
        TOTAL_LOSS_KWH=('ENERGY_LOSS_KWH', 'sum'),
        ACTUAL_KWH=('AC_POWER', lambda p: (p * 0.25).sum()),
        EXPECTED_KWH=('EXPECTED_AC_POWER', lambda p: (p * 0.25).sum()),
        PEAK_IRRADIATION=('IRRADIATION', 'max'),
        AVG_MODULE_TEMP=('MODULE_TEMPERATURE', 'mean')
    ).reset_index()

    daily['DAILY_PR'] = (daily['ACTUAL_KWH'] / np.maximum(daily['EXPECTED_KWH'], 1.0)).clip(0.0, 1.2)
    daily['DATE_STR'] = daily['DATE'].astype(str)

    return daily
