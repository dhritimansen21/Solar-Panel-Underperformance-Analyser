"""
generate_sample_datasets.py
Generates 5 realistic synthetic solar plant telemetry datasets representing distinct
real-world operational failure and soiling scenarios across Rajasthan and Western India.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "sample_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_weather_profile(dates: pd.DatetimeIndex, base_amb_temp: float = 32.0, peak_irr: float = 1.05) -> pd.DataFrame:
    """Simulates 15-minute ambient temp, module temp, and solar irradiation with realistic diurnal curves."""
    hours = dates.hour + dates.minute / 60.0
    
    # Sun elevation curve (sunrise ~05:45, sunset ~19:00, peak ~12:30)
    day_mask = (hours >= 5.75) & (hours <= 19.0)
    solar_factor = np.sin((hours - 5.75) / (19.0 - 5.75) * np.pi)
    solar_factor = np.where(day_mask, np.maximum(0.0, solar_factor), 0.0)
    
    # Introduce random cloud noise on select days
    day_indices = (dates - dates[0]).days
    cloud_noise = 1.0 - 0.25 * np.sin(day_indices * 0.7) * (np.random.RandomState(42).rand(len(dates)) > 0.85)
    
    irradiation = peak_irr * (solar_factor ** 1.3) * cloud_noise
    irradiation = np.where(day_mask, np.clip(irradiation, 0.0, 1.25), 0.0)
    
    # Ambient temperature curve (minimum at 05:00, peak at 14:30)
    temp_diurnal = np.sin((hours - 8.5) / 24.0 * 2 * np.pi)
    ambient_temp = base_amb_temp + (8.5 * temp_diurnal) + np.random.normal(0, 0.4, len(dates))
    
    # Module temperature (ambient + irradiation * thermal heating coefficient ~28°C/kW/m²)
    module_temp = ambient_temp + (irradiation * 29.0) + np.random.normal(0, 0.6, len(dates))
    
    return pd.DataFrame({
        'DATE_TIME': dates,
        'AMBIENT_TEMPERATURE': np.round(ambient_temp, 2),
        'MODULE_TEMPERATURE': np.round(module_temp, 2),
        'IRRADIATION': np.round(irradiation, 4)
    })


def compute_healthy_ac_power(irradiation: np.ndarray, module_temp: np.ndarray, rated_kw: float = 850.0) -> np.ndarray:
    """Computes nominal AC output accounting for silicon thermal degradation (-0.4%/°C above 25°C)."""
    temp_derate = 1.0 - 0.004 * np.maximum(0.0, module_temp - 25.0)
    inverter_eff = 0.98
    power = rated_kw * irradiation * temp_derate * inverter_eff
    return np.clip(power, 0.0, rated_kw)


# ---------------------------------------------------------------------------------
# DATASET 1: Rooftop Jaipur Commercial (30 Days - Progressive Dust/Soiling Accumulation)
# ---------------------------------------------------------------------------------
def generate_dataset_1():
    print("Generating Dataset 1: Rooftop Jaipur (Dust & Soiling Scenario)...")
    dates = pd.date_range("2024-05-01 00:00", "2024-05-30 23:45", freq="15min")
    weather = simulate_weather_profile(dates, base_amb_temp=34.0, peak_irr=1.08)
    
    inverters = [f"INV_JP_{i+1:02d}" for i in range(8)]
    records = []
    
    for inv in inverters:
        inv_weather = weather.copy()
        nominal_ac = compute_healthy_ac_power(inv_weather['IRRADIATION'].values, inv_weather['MODULE_TEMPERATURE'].values, rated_kw=800.0)
        days = (dates - dates[0]).days.values
        
        # Inverter 03 and 07 accumulate severe unwashed dust (soiling drops by ~0.8% per day)
        if inv in ["INV_JP_03", "INV_JP_07"]:
            soiling_factor = np.maximum(0.68, 0.96 - (days * 0.009))
        else:
            soiling_factor = np.random.uniform(0.93, 0.97, len(dates))
            
        ac_power = np.round(nominal_ac * soiling_factor + np.random.normal(0, 1.5, len(dates)), 2)
        ac_power = np.where(inv_weather['IRRADIATION'] < 0.005, 0.0, np.clip(ac_power, 0.0, 800.0))
        
        inv_weather['SOURCE_KEY'] = inv
        inv_weather['AC_POWER'] = ac_power
        inv_weather['DC_POWER'] = np.round(ac_power * 1.025, 2)
        records.append(inv_weather)
        
    df1 = pd.concat(records).sort_values(['SOURCE_KEY', 'DATE_TIME']).reset_index(drop=True)
    out_file = OUTPUT_DIR / "1_Rooftop_Jaipur_Soiling_30D.csv"
    df1.to_csv(out_file, index=False)
    print(f" -> Saved {out_file.name} ({len(df1):,} rows)")


# ---------------------------------------------------------------------------------
# DATASET 2: Solar Farm Bhadla Desert (45 Days - Thermal Tripping & Hardware Outages)
# ---------------------------------------------------------------------------------
def generate_dataset_2():
    print("Generating Dataset 2: Bhadla Solar Park (Thermal Tripping Scenario)...")
    dates = pd.date_range("2024-04-15 00:00", "2024-05-29 23:45", freq="15min")
    weather = simulate_weather_profile(dates, base_amb_temp=38.0, peak_irr=1.15)
    
    inverters = [f"BHD_INVERTER_{i+1:02d}" for i in range(10)]
    records = []
    
    for inv in inverters:
        inv_weather = weather.copy()
        nominal_ac = compute_healthy_ac_power(inv_weather['IRRADIATION'].values, inv_weather['MODULE_TEMPERATURE'].values, rated_kw=1000.0)
        hours = dates.hour + dates.minute / 60.0
        days = (dates - dates[0]).days.values
        
        ac_power = nominal_ac * 0.96
        
        # Inverter 02 experiences midday thermal breaker tripping during peak hours
        if inv == "BHD_INVERTER_02":
            tripped_days = (days % 3 == 0) # trips every 3rd day
            noon_peak = (hours >= 11.5) & (hours <= 14.0) & (inv_weather['IRRADIATION'] > 0.70)
            ac_power[tripped_days & noon_peak] = 0.0
            
        # Inverter 08 has a continuous 5-day hardware outage (days 18 to 23)
        if inv == "BHD_INVERTER_08":
            outage_mask = (days >= 18) & (days <= 23)
            ac_power[outage_mask] = 0.0
            
        ac_power = np.where(inv_weather['IRRADIATION'] < 0.005, 0.0, np.clip(ac_power, 0.0, 1000.0))
        
        inv_weather['SOURCE_KEY'] = inv
        inv_weather['AC_POWER'] = np.round(ac_power, 2)
        inv_weather['DC_POWER'] = np.round(ac_power * 1.02, 2)
        records.append(inv_weather)
        
    df2 = pd.concat(records).sort_values(['SOURCE_KEY', 'DATE_TIME']).reset_index(drop=True)
    out_file = OUTPUT_DIR / "2_Bhadla_Desert_Tripping_45D.csv"
    df2.to_csv(out_file, index=False)
    print(f" -> Saved {out_file.name} ({len(df2):,} rows)")


# ---------------------------------------------------------------------------------
# DATASET 3: AgriPV Jodhpur (21 Days - Dual File Upload Test with W/m² irradiance)
# ---------------------------------------------------------------------------------
def generate_dataset_3():
    print("Generating Dataset 3: AgriPV Jodhpur (Dual-File Upload Scenario)...")
    dates = pd.date_range("2024-06-01 00:00", "2024-06-21 23:45", freq="15min")
    weather = simulate_weather_profile(dates, base_amb_temp=35.0, peak_irr=1.02)
    
    # 1. Weather file in W/m² to test auto-scaling
    weather_export = pd.DataFrame({
        'timestamp': dates,
        'ghi_flux_w_m2': np.round(weather['IRRADIATION'] * 1000.0, 1),
        'air_temperature_c': weather['AMBIENT_TEMPERATURE'],
        'pv_surface_temp_c': weather['MODULE_TEMPERATURE']
    })
    weather_file = OUTPUT_DIR / "3_AgriPV_Jodhpur_Weather_Sensor.csv"
    weather_export.to_csv(weather_file, index=False)
    
    # 2. Inverter generation file
    inverters = [f"AGRI_INV_{i+1:02d}" for i in range(6)]
    gen_records = []
    
    for inv in inverters:
        nominal_ac = compute_healthy_ac_power(weather['IRRADIATION'].values, weather['MODULE_TEMPERATURE'].values, rated_kw=650.0)
        # Moderate agricultural crop dust on INV_04
        efficiency = 0.78 if inv == "AGRI_INV_04" else 0.94
        ac_power = np.round(nominal_ac * efficiency, 2)
        ac_power = np.where(weather['IRRADIATION'] < 0.005, 0.0, np.clip(ac_power, 0.0, 650.0))
        
        gen_df = pd.DataFrame({
            'timestamp': dates,
            'inverter_serial': inv,
            'active_generation_kw': ac_power
        })
        gen_records.append(gen_df)
        
    gen_file = OUTPUT_DIR / "3_AgriPV_Jodhpur_Generation_Stream.csv"
    pd.concat(gen_records).sort_values(['inverter_serial', 'timestamp']).to_csv(gen_file, index=False)
    print(f" -> Saved {weather_file.name} & {gen_file.name}")


# ---------------------------------------------------------------------------------
# DATASET 4: Industrial Rooftop Bhiwadi (25 Days - String Fuse Blow & Obstruction Shading)
# ---------------------------------------------------------------------------------
def generate_dataset_4():
    print("Generating Dataset 4: Bhiwadi Industrial (String Fault & Shading)...")
    dates = pd.date_range("2024-05-10 00:00", "2024-06-03 23:45", freq="15min")
    weather = simulate_weather_profile(dates, base_amb_temp=33.0, peak_irr=1.04)
    
    inverters = [f"BHIWADI_STRING_{i+1}" for i in range(6)]
    records = []
    
    for inv in inverters:
        inv_weather = weather.copy()
        nominal_ac = compute_healthy_ac_power(inv_weather['IRRADIATION'].values, inv_weather['MODULE_TEMPERATURE'].values, rated_kw=500.0)
        hours = dates.hour + dates.minute / 60.0
        
        ac_power = nominal_ac * 0.95
        
        # String 4 has 1 of 3 strings blown (constant 33% power reduction throughout)
        if inv == "BHIWADI_STRING_4":
            ac_power = nominal_ac * 0.64
            
        # String 1 has afternoon structural shadow from a cooling tower (drops after 14:15)
        if inv == "BHIWADI_STRING_1":
            afternoon_shade = (hours >= 14.25) & (hours <= 17.5)
            ac_power[afternoon_shade] = ac_power[afternoon_shade] * 0.40
            
        ac_power = np.where(inv_weather['IRRADIATION'] < 0.005, 0.0, np.clip(ac_power, 0.0, 500.0))
        
        inv_weather['SOURCE_KEY'] = inv
        inv_weather['AC_POWER'] = np.round(ac_power, 2)
        inv_weather['DC_POWER'] = np.round(ac_power * 1.03, 2)
        records.append(inv_weather)
        
    df4 = pd.concat(records).sort_values(['SOURCE_KEY', 'DATE_TIME']).reset_index(drop=True)
    out_file = OUTPUT_DIR / "4_Bhiwadi_Industrial_Faults_25D.csv"
    df4.to_csv(out_file, index=False)
    print(f" -> Saved {out_file.name} ({len(df4):,} rows)")


# ---------------------------------------------------------------------------------
# DATASET 5: Clean Benchmark Neemrana (30 Days - Optimal Reference Plant)
# ---------------------------------------------------------------------------------
def generate_dataset_5():
    print("Generating Dataset 5: Neemrana Clean Benchmark (Healthy Reference)...")
    dates = pd.date_range("2024-05-01 00:00", "2024-05-30 23:45", freq="15min")
    weather = simulate_weather_profile(dates, base_amb_temp=31.0, peak_irr=1.06)
    
    inverters = [f"NEEMRANA_INV_{i+1:02d}" for i in range(8)]
    records = []
    
    for inv in inverters:
        inv_weather = weather.copy()
        nominal_ac = compute_healthy_ac_power(inv_weather['IRRADIATION'].values, inv_weather['MODULE_TEMPERATURE'].values, rated_kw=900.0)
        
        # Robotic cleaning plant: high performance factor between 96% and 99%
        health_factor = np.random.uniform(0.96, 0.99, len(dates))
        ac_power = np.round(nominal_ac * health_factor + np.random.normal(0, 1.0, len(dates)), 2)
        ac_power = np.where(inv_weather['IRRADIATION'] < 0.005, 0.0, np.clip(ac_power, 0.0, 900.0))
        
        inv_weather['SOURCE_KEY'] = inv
        inv_weather['AC_POWER'] = ac_power
        inv_weather['DC_POWER'] = np.round(ac_power * 1.02, 2)
        records.append(inv_weather)
        
    df5 = pd.concat(records).sort_values(['SOURCE_KEY', 'DATE_TIME']).reset_index(drop=True)
    out_file = OUTPUT_DIR / "5_Neemrana_Clean_Benchmark_30D.csv"
    df5.to_csv(out_file, index=False)
    print(f" -> Saved {out_file.name} ({len(df5):,} rows)")


if __name__ == "__main__":
    generate_dataset_1()
    generate_dataset_2()
    generate_dataset_3()
    generate_dataset_4()
    generate_dataset_5()
    print("\n[SUCCESS] All 5 synthetic solar datasets successfully generated in:", OUTPUT_DIR)
