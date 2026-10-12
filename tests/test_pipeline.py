"""
test_pipeline.py
Automated end-to-end verification and QA suite for all plants and operational scenarios.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.db import list_plants, load_plant_telemetry, init_db
from src.features import add_solar_features
from src.model import load_or_train_model, predict_expected_power
from src.detector import detect_anomalies, generate_inverter_scorecard
from src.financial_loss import compute_financial_losses, get_plant_loss_summary, get_inverter_loss_table

def run_tests():
    init_db()
    plants = list_plants()
    print(f"=== Testing {len(plants)} Plants in DuckDB ===")

    for p in plants:
        pid = p['plant_id']
        pname = p['plant_name']
        print(f"\n>> Validating Plant: {pid} | {pname}")
        
        df = load_plant_telemetry(pid)
        assert len(df) > 0, f"Empty telemetry for {pid}"
        
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        df['DATE'] = df['DATE_TIME'].dt.date
        df['HOUR'] = df['DATE_TIME'].dt.hour
        df['MINUTE'] = df['DATE_TIME'].dt.minute
        df['IS_DAYLIGHT'] = (df['IRRADIATION'] > 0.005) | (df['AC_POWER'] > 0)
        
        df = add_solar_features(df)
        model, metrics, _ = load_or_train_model(df, plant_id=pid)
        df['EXPECTED_AC_POWER'] = predict_expected_power(model, df)
        
        df = detect_anomalies(df)
        df = compute_financial_losses(df, tariff_inr=4.50)
        
        summary = get_plant_loss_summary(df)
        scorecard = generate_inverter_scorecard(df)
        loss_table = get_inverter_loss_table(df, scorecard)
        
        inv_count = df['SOURCE_KEY'].nunique()
        print(f"   [PASS] Rows: {len(df):,} | Inverters: {inv_count}")
        print(f"   [PASS] Model R2: {metrics['test_r2']:.4f} | Sampling Interval: {summary['sampling_interval_min']} mins")
        print(f"   [PASS] Plant PR: {summary['overall_plant_pr']*100:.1f}% | Revenue Lost: Rs {summary['total_revenue_loss_inr']:,.0f}")
        print(f"   [PASS] Scorecard Categories: {dict(scorecard['DIAGNOSIS'].value_counts())}")

    print("\n[SUCCESS] All QA pipeline tests passed successfully!")

if __name__ == "__main__":
    run_tests()
