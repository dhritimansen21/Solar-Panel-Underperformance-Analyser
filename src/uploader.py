"""
uploader.py
Handles custom solar dataset uploads, intelligent column mapping, unit normalization,
and ingestion into DuckDB.
"""

import io
import re
import pandas as pd
import numpy as np
from src.db import save_plant_telemetry


STANDARD_COLUMNS = {
    'DATE_TIME': ['date_time', 'datetime', 'date', 'timestamp', 'time', 'ts'],
    'SOURCE_KEY': ['source_key', 'inverter', 'inverter_id', 'inv_id', 'inverter_serial', 'inverter_name', 'device_id', 'string_id', 'device', 'inv'],
    'AC_POWER': ['ac_power', 'power', 'active_power', 'pac', 'generation_kw', 'power_kw', 'yield_kw', 'p_ac', 'ac_kw', 'active_generation_kw'],
    'IRRADIATION': ['irradiation', 'irradiance', 'solar_radiation', 'ghi', 'poa', 'solar_flux', 'rad', 'sun_irrad', 'ghi_flux_w_m2'],
    'MODULE_TEMPERATURE': ['module_temperature', 'mod_temp', 'panel_temp', 'cell_temp', 'module_temp', 't_mod', 'temp_panel', 'pv_surface_temp_c'],
    'AMBIENT_TEMPERATURE': ['ambient_temperature', 'amb_temp', 'air_temp', 'temp_ambient', 't_amb', 'temp_air', 'air_temperature_c']
}


def parse_flexible_datetime(series: pd.Series) -> pd.Series:
    """
    Intelligently parses timestamps whether they are ISO (YYYY-MM-DD) or dayfirst (DD-MM-YYYY).
    Prevents corrupting ISO dates into NaTs when day > 12.
    """
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return pd.to_datetime(series, errors='coerce')

    sample = str(clean_series.iloc[0]).strip()
    # If sample starts with 4-digit year followed by dash or slash (YYYY-MM-DD)
    if len(sample) >= 4 and sample[:4].isdigit() and ('-' in sample[:5] or '/' in sample[:5]):
        return pd.to_datetime(series, format='mixed', errors='coerce')
    else:
        return pd.to_datetime(series, format='mixed', dayfirst=True, errors='coerce')


def find_best_column_match(columns: list[str], patterns: list[str]) -> str | None:
    """
    Finds the best matching column name from a list of candidates based on pattern priority.
    """
    normalized_cols = {re.sub(r'[^a-zA-Z0-9]', '_', c.strip().lower()): c for c in columns}
    # 1. Exact match pass
    for pat in patterns:
        for norm_c, original_c in normalized_cols.items():
            if norm_c == pat:
                return original_c
    # 2. Prefix / suffix / substring pass
    for pat in patterns:
        for norm_c, original_c in normalized_cols.items():
            if norm_c.startswith(pat) or norm_c.endswith(pat) or pat in norm_c:
                return original_c
    return None


def guess_column_mapping(df_columns: list[str]) -> dict[str, str | None]:
    """
    Intelligently guesses standard column associations from raw column headers.
    """
    mapping = {}
    for std_col, patterns in STANDARD_COLUMNS.items():
        mapping[std_col] = find_best_column_match(df_columns, patterns)
    return mapping


def process_and_ingest_uploaded_df(
    df: pd.DataFrame,
    plant_id: str,
    plant_name: str,
    col_mapping: dict[str, str]
) -> tuple[bool, str, int]:
    """
    Applies column mappings, parses dates, scales units, and saves to DuckDB.
    Returns: (success, message, row_count)
    """
    required = ['DATE_TIME', 'SOURCE_KEY', 'AC_POWER', 'IRRADIATION', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE']
    missing = [col for col in required if not col_mapping.get(col)]
    if missing:
        return False, f"Missing required column mappings for: {', '.join(missing)}", 0

    try:
        standard_df = pd.DataFrame()
        for std_col in required:
            raw_col = col_mapping[std_col]
            if raw_col not in df.columns:
                return False, f"Mapped column '{raw_col}' not found in uploaded file.", 0
            standard_df[std_col] = df[raw_col]

        # Parse timestamps flexibly
        standard_df['DATE_TIME'] = parse_flexible_datetime(standard_df['DATE_TIME'])
        standard_df = standard_df.dropna(subset=['DATE_TIME'])

        if len(standard_df) == 0:
            return False, "Could not parse any valid timestamps from the selected date/time column.", 0

        # Numeric conversions
        numeric_cols = ['AC_POWER', 'IRRADIATION', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE']
        for c in numeric_cols:
            standard_df[c] = pd.to_numeric(standard_df[c], errors='coerce').fillna(0.0)

        # Scale irradiation: If maximum irradiation > 20, it is likely in W/m² instead of kW/m²
        if standard_df['IRRADIATION'].max() > 20.0:
            standard_df['IRRADIATION'] = standard_df['IRRADIATION'] / 1000.0

        standard_df['SOURCE_KEY'] = standard_df['SOURCE_KEY'].astype(str)

        # Deduplicate to prevent DB primary key conflicts
        standard_df = standard_df.drop_duplicates(subset=['DATE_TIME', 'SOURCE_KEY'])

        # Optional extra fields
        standard_df['DC_POWER'] = standard_df['AC_POWER'] * 1.02
        standard_df['DAILY_YIELD'] = 0.0
        standard_df['TOTAL_YIELD'] = 0.0

        # Save to DuckDB
        save_plant_telemetry(plant_id, plant_name, standard_df, source_type="uploaded")

        return True, f"Successfully imported {len(standard_df):,} records for '{plant_name}' into DuckDB.", len(standard_df)
    except Exception as e:
        return False, f"Error processing file: {str(e)}", 0


def process_dual_files_upload(
    gen_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    plant_id: str,
    plant_name: str,
    gen_time_col: str,
    weather_time_col: str,
    inv_col: str,
    power_col: str,
    irrad_col: str,
    mod_temp_col: str,
    amb_temp_col: str
) -> tuple[bool, str, int]:
    """
    Merges separate generation and weather files on timestamps and ingests into DuckDB.
    """
    try:
        gen_df = gen_df.copy()
        weather_df = weather_df.copy()

        # Standardize and validate timestamps
        gen_df['NORM_DT'] = parse_flexible_datetime(gen_df[gen_time_col])
        weather_df['NORM_DT'] = parse_flexible_datetime(weather_df[weather_time_col])

        gen_df = gen_df.dropna(subset=['NORM_DT'])
        weather_df = weather_df.dropna(subset=['NORM_DT'])

        if len(gen_df) == 0 or len(weather_df) == 0:
            return False, "Could not parse valid timestamps from one or both files.", 0

        # Build clean sub-frames prior to merge to avoid column name collisions (e.g. SOURCE_KEY or PLANT_ID)
        g_sub = pd.DataFrame({
            'NORM_DT': gen_df['NORM_DT'],
            'SOURCE_KEY': gen_df[inv_col].astype(str),
            'AC_POWER': pd.to_numeric(gen_df[power_col], errors='coerce').fillna(0.0),
        })

        if 'DC_POWER' in gen_df.columns:
            g_sub['DC_POWER'] = pd.to_numeric(gen_df['DC_POWER'], errors='coerce').fillna(0.0)
        else:
            g_sub['DC_POWER'] = g_sub['AC_POWER'] * 1.02

        if 'DAILY_YIELD' in gen_df.columns:
            g_sub['DAILY_YIELD'] = pd.to_numeric(gen_df['DAILY_YIELD'], errors='coerce').fillna(0.0)
        else:
            g_sub['DAILY_YIELD'] = 0.0

        if 'TOTAL_YIELD' in gen_df.columns:
            g_sub['TOTAL_YIELD'] = pd.to_numeric(gen_df['TOTAL_YIELD'], errors='coerce').fillna(0.0)
        else:
            g_sub['TOTAL_YIELD'] = 0.0

        w_sub = pd.DataFrame({
            'NORM_DT': weather_df['NORM_DT'],
            'IRRADIATION': pd.to_numeric(weather_df[irrad_col], errors='coerce').fillna(0.0),
            'MODULE_TEMPERATURE': pd.to_numeric(weather_df[mod_temp_col], errors='coerce').fillna(0.0),
            'AMBIENT_TEMPERATURE': pd.to_numeric(weather_df[amb_temp_col], errors='coerce').fillna(0.0),
        })
        # Deduplicate weather by timestamp (sensor readings at plant level)
        w_sub = w_sub.drop_duplicates(subset=['NORM_DT'])

        merged = pd.merge(g_sub, w_sub, on='NORM_DT', how='inner')
        if len(merged) == 0:
            return False, "No timestamp matches found between generation and weather files.", 0

        merged['DATE_TIME'] = merged['NORM_DT']
        if merged['IRRADIATION'].max() > 20.0:
            merged['IRRADIATION'] = merged['IRRADIATION'] / 1000.0

        merged = merged.drop_duplicates(subset=['DATE_TIME', 'SOURCE_KEY'])

        cols_to_save = ['DATE_TIME', 'SOURCE_KEY', 'AC_POWER', 'DC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD', 'IRRADIATION', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE']
        final_df = merged[cols_to_save].copy()

        save_plant_telemetry(plant_id, plant_name, final_df, source_type="uploaded")
        return True, f"Merged and imported {len(final_df):,} records for '{plant_name}' into DuckDB.", len(final_df)
    except Exception as e:
        return False, f"Failed dual file ingestion: {str(e)}", 0
