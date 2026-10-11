"""
detector.py
Underperformance and anomaly detection engine for solar inverters.
Categorizes operational degradation (dust/soiling) vs electrical/hardware tripping.
"""

import numpy as np
import pandas as pd


def detect_anomalies(
    df: pd.DataFrame,
    pr_threshold_warn: float = 0.88,
    pr_threshold_severe: float = 0.60,
    min_irradiation: float = 0.05,
    offline_threshold_kw: float = 5.0
) -> pd.DataFrame:
    """
    Computes performance ratio (PR) and flags operational status for each 15-minute slot.
    """
    df = df.copy()

    # Performance Ratio: Actual / Expected (only meaningful during daylight)
    expected_safe = np.maximum(df['EXPECTED_AC_POWER'], 1.0)
    df['PERFORMANCE_RATIO'] = df['AC_POWER'] / expected_safe

    # Default category
    df['STATUS'] = 'NIGHT_INACTIVE'

    daylight_mask = df['IRRADIATION'] >= min_irradiation

    # 1. Critical Inverter Trip / Hardware Disconnect (Sun is high, but power near 0)
    offline_mask = daylight_mask & (df['EXPECTED_AC_POWER'] > 40.0) & (df['AC_POWER'] < offline_threshold_kw)
    df.loc[offline_mask, 'STATUS'] = 'OFFLINE_TRIPPED'

    # 2. Severe Fault / String Drop
    severe_mask = daylight_mask & (~offline_mask) & (df['PERFORMANCE_RATIO'] < pr_threshold_severe)
    df.loc[severe_mask, 'STATUS'] = 'SEVERE_FAULT'

    # 3. Soiling / Dust Degradation (operating, but deficit between 12% and 40%)
    soiling_mask = (
        daylight_mask
        & (~offline_mask)
        & (~severe_mask)
        & (df['PERFORMANCE_RATIO'] < pr_threshold_warn)
    )
    df.loc[soiling_mask, 'STATUS'] = 'SOILING_DUST'

    # 4. Optimal Operation
    optimal_mask = (
        daylight_mask
        & (~offline_mask)
        & (~severe_mask)
        & (~soiling_mask)
    )
    df.loc[optimal_mask, 'STATUS'] = 'OPTIMAL'

    return df


def generate_inverter_scorecard(
    df: pd.DataFrame,
    min_irradiation: float = 0.05
) -> pd.DataFrame:
    """
    Aggregates inverter health metrics, diagnostic classification, and maintenance urgency.
    """
    daylight = df[df['IRRADIATION'] >= min_irradiation].copy()

    # Map friendly inverter names (INV-01, INV-02...) alongside unique SOURCE_KEY
    unique_inverters = sorted(df['SOURCE_KEY'].unique())
    inv_map = {k: f"INV-{i+1:02d}" for i, k in enumerate(unique_inverters)}

    scorecard_rows = []
    for source_key, group in daylight.groupby('SOURCE_KEY'):
        total_slots = len(group)
        if total_slots == 0:
            continue

        mean_pr = float(group['PERFORMANCE_RATIO'].mean())
        median_pr = float(group['PERFORMANCE_RATIO'].median())
        offline_slots = int((group['STATUS'] == 'OFFLINE_TRIPPED').sum())
        severe_slots = int((group['STATUS'] == 'SEVERE_FAULT').sum())
        soiling_slots = int((group['STATUS'] == 'SOILING_DUST').sum())
        optimal_slots = int((group['STATUS'] == 'OPTIMAL').sum())

        offline_hours = offline_slots * 0.25
        soiling_hours = soiling_slots * 0.25
        severe_hours = severe_slots * 0.25

        # Underperformance deficit percentage
        underperf_pct = max(0.0, (1.0 - mean_pr) * 100.0)

        # Health score (0 to 100)
        health_score = max(0.0, min(100.0, (mean_pr * 100.0) - (offline_hours * 0.5)))

        # Diagnostic recommendation
        if offline_hours > 5.0 or (offline_slots / total_slots > 0.03):
            category = "Critical Fault"
            recommendation = "Inspect Breaker / Inverter Hardware Immediately (High Outage Hours)"
            badge_color = "red"
        elif underperf_pct > 15.0 or (severe_slots / total_slots > 0.10):
            category = "Heavy Degradation"
            recommendation = "Schedule Urgent Panel Wash & String Inspection"
            badge_color = "orange"
        elif underperf_pct > 5.0 or (soiling_slots / total_slots > 0.20):
            category = "Moderate Soiling"
            recommendation = "Panel Cleaning Recommended (Dust Accumulation)"
            badge_color = "yellow"
        else:
            category = "Optimal Health"
            recommendation = "Normal Operation (Maintain Routine Schedule)"
            badge_color = "green"

        scorecard_rows.append({
            'SOURCE_KEY': source_key,
            'INVERTER_NAME': inv_map.get(source_key, source_key),
            'HEALTH_SCORE': round(health_score, 1),
            'MEAN_PR': round(mean_pr, 3),
            'UNDERPERF_PCT': round(underperf_pct, 1),
            'OFFLINE_HOURS': round(offline_hours, 1),
            'SOILING_HOURS': round(soiling_hours, 1),
            'SEVERE_HOURS': round(severe_hours, 1),
            'DIAGNOSIS': category,
            'ACTION': recommendation,
            'BADGE_COLOR': badge_color,
            'TOTAL_DAYLIGHT_HOURS': round(total_slots * 0.25, 1)
        })

    scorecard_df = pd.DataFrame(scorecard_rows)
    # Sort worst health first
    scorecard_df = scorecard_df.sort_values(by=['HEALTH_SCORE', 'MEAN_PR']).reset_index(drop=True)
    return scorecard_df
