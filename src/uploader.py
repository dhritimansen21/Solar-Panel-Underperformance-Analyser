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
    'SOURCE_KEY': ['source_key', 'inverter', 'inverter_id', 'inv_id', 'inverter_name', 'device_id', 'string_id', 'device'],
    'AC_POWER': ['ac_power', 'power', 'active_power', 'pac', 'generation_kw', 'power_kw', 'yield_kw', 'p_ac', 'ac_kw'],
    'IRRADIATION': ['irradiation', 'irradiance', 'solar_radiation', 'ghi', 'poa', 'solar_flux', 'rad', 'sun_irrad'],
    'MODULE_TEMPERATURE': ['module_temperature', 'mod_temp', 'panel_temp', 'cell_temp', 'module_temp', 't_mod', 'temp_panel'],
    'AMBIENT_TEMPERATURE': ['ambient_temperature', 'amb_temp', 'air_temp', 'temp_ambient', 't_amb', 'temp_air']
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


def guess_column_mapping(df_columns: list[str]) -> dict[str, str | None]:
    """
    Intelligently guesses standard column associations from raw column headers.
    """
    mapping = {}
    normalized_cols = {re.sub(r'[^a-zA-Z0-9]', '_', c.strip().lower()): c for c in df_columns}

    for std_col, patterns in STANDARD_COLUMNS.items():
        matched_raw = None
        for pattern in patterns:
            # Exact or substring match
            for norm_c, original_c in normalized_cols.items():
                if pattern == norm_c or norm_c.startswith(pattern) or norm_c.endswith(pattern):
                    matched_raw = original_c
                    break
            if matched_raw:
                break
        mapping[std_col] = matched_raw

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

        gen_df['NORM_DT'] = parse_flexible_datetime(gen_df[gen_time_col])
        weather_df['NORM_DT'] = parse_flexible_datetime(weather_df[weather_time_col])

        gen_df = gen_df.dropna(subset=['NORM_DT'])
        weather_df = weather_df.dropna(subset=['NORM_DT'])

        # Drop duplicate weather timestamps if weather is plant-level
        weather_dedup = weather_df.drop_duplicates(subset=['NORM_DT'])

        merged = pd.merge(gen_df, weather_dedup, on='NORM_DT', how='inner')
        if len(merged) == 0:
            return False, "No timestamp matches found between generation and weather files.", 0

        merged_clean = pd.DataFrame({
            'DATE_TIME': merged['NORM_DT'],
            'SOURCE_KEY': merged[inv_col].astype(str),
            'AC_POWER': pd.to_numeric(merged[power_col], errors='coerce').fillna(0.0),
            'IRRADIATION': pd.to_numeric(merged[irrad_col], errors='coerce').fillna(0.0),
            'MODULE_TEMPERATURE': pd.to_numeric(merged[mod_temp_col], errors='coerce').fillna(0.0),
            'AMBIENT_TEMPERATURE': pd.to_numeric(merged[amb_temp_col], errors='coerce').fillna(0.0),
        })

        if merged_clean['IRRADIATION'].max() > 20.0:
            merged_clean['IRRADIATION'] = merged_clean['IRRADIATION'] / 1000.0

        merged_clean = merged_clean.drop_duplicates(subset=['DATE_TIME', 'SOURCE_KEY'])

        merged_clean['DC_POWER'] = merged_clean['AC_POWER'] * 1.02
        merged_clean['DAILY_YIELD'] = 0.0
        merged_clean['TOTAL_YIELD'] = 0.0

        save_plant_telemetry(plant_id, plant_name, merged_clean, source_type="uploaded")
        return True, f"Merged and imported {len(merged_clean):,} records for '{plant_name}' into DuckDB.", len(merged_clean)
    except Exception as e:
        return False, f"Failed dual file ingestion: {str(e)}", 0
