"""
db.py
Embedded DuckDB storage engine for solar plant telemetry, uploaded datasets, and audit logs.
Provides persistent zero-setup local database capabilities.
"""

from pathlib import Path
from datetime import datetime
import duckdb
import pandas as pd

DB_PATH = Path(r"D:\solar panel underperformance analyser\data\solar_analytics.duckdb")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    """Returns a DuckDB connection to the local database file."""
    return duckdb.connect(str(DB_PATH))


def init_db():
    """Initializes schema tables if they do not exist."""
    con = get_connection()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS plants (
                plant_id VARCHAR PRIMARY KEY,
                plant_name VARCHAR,
                source_type VARCHAR, -- 'kaggle' or 'uploaded'
                total_inverters INTEGER,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                created_at TIMESTAMP
            );
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                plant_id VARCHAR,
                date_time TIMESTAMP,
                source_key VARCHAR,
                ac_power DOUBLE,
                dc_power DOUBLE,
                daily_yield DOUBLE,
                total_yield DOUBLE,
                ambient_temp DOUBLE,
                module_temp DOUBLE,
                irradiation DOUBLE,
                PRIMARY KEY (plant_id, date_time, source_key)
            );
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                audit_id VARCHAR PRIMARY KEY,
                plant_id VARCHAR,
                run_timestamp TIMESTAMP,
                tariff_rate DOUBLE,
                cleaning_cost DOUBLE,
                total_loss_inr DOUBLE,
                total_loss_kwh DOUBLE,
                plant_pr DOUBLE,
                critical_inverter_count INTEGER,
                soiling_inverter_count INTEGER
            );
        """)
    finally:
        con.close()


def save_plant_telemetry(
    plant_id: str,
    plant_name: str,
    df: pd.DataFrame,
    source_type: str = "uploaded"
) -> bool:
    """
    Saves or updates plant metadata and telemetry records in DuckDB.
    """
    con = get_connection()
    try:
        total_inverters = int(df['SOURCE_KEY'].nunique())
        start_date = pd.to_datetime(df['DATE_TIME'].min())
        end_date = pd.to_datetime(df['DATE_TIME'].max())
        created_at = datetime.now()

        # Insert or replace plant record
        con.execute("""
            INSERT OR REPLACE INTO plants (plant_id, plant_name, source_type, total_inverters, start_date, end_date, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (plant_id, plant_name, source_type, total_inverters, start_date, end_date, created_at))

        # Format dataframe for telemetry insert
        telemetry_df = pd.DataFrame({
            'plant_id': plant_id,
            'date_time': pd.to_datetime(df['DATE_TIME']),
            'source_key': df['SOURCE_KEY'].astype(str),
            'ac_power': df['AC_POWER'].astype(float),
            'dc_power': df.get('DC_POWER', 0.0).astype(float),
            'daily_yield': df.get('DAILY_YIELD', 0.0).astype(float),
            'total_yield': df.get('TOTAL_YIELD', 0.0).astype(float),
            'ambient_temp': df['AMBIENT_TEMPERATURE'].astype(float),
            'module_temp': df['MODULE_TEMPERATURE'].astype(float),
            'irradiation': df['IRRADIATION'].astype(float),
        })

        # Remove existing telemetry for this plant to allow clean updates
        con.execute("DELETE FROM telemetry WHERE plant_id = ?", (plant_id,))
        con.register('df_to_insert', telemetry_df)
        con.execute("INSERT INTO telemetry SELECT * FROM df_to_insert")
        con.unregister('df_to_insert')
        return True
    finally:
        con.close()


def load_plant_telemetry(plant_id: str) -> pd.DataFrame:
    """
    Retrieves full telemetry dataframe from DuckDB for a given plant_id.
    """
    con = get_connection()
    try:
        query = """
            SELECT 
                date_time AS DATE_TIME,
                source_key AS SOURCE_KEY,
                ac_power AS AC_POWER,
                dc_power AS DC_POWER,
                daily_yield AS DAILY_YIELD,
                total_yield AS TOTAL_YIELD,
                ambient_temp AS AMBIENT_TEMPERATURE,
                module_temp AS MODULE_TEMPERATURE,
                irradiation AS IRRADIATION
            FROM telemetry
            WHERE plant_id = ?
            ORDER BY source_key, date_time
        """
        df = con.execute(query, (plant_id,)).fetchdf()
        return df
    finally:
        con.close()


def list_plants() -> list[dict]:
    """
    Returns list of all plants saved in the database.
    """
    con = get_connection()
    try:
        rows = con.execute("""
            SELECT plant_id, plant_name, source_type, total_inverters, start_date, end_date, created_at
            FROM plants
            ORDER BY created_at DESC
        """).fetchdf()
        return rows.to_dict(orient='records')
    finally:
        con.close()


def delete_plant(plant_id: str):
    """Deletes plant and its associated telemetry and audit logs."""
    con = get_connection()
    try:
        con.execute("DELETE FROM plants WHERE plant_id = ?", (plant_id,))
        con.execute("DELETE FROM telemetry WHERE plant_id = ?", (plant_id,))
        con.execute("DELETE FROM audit_logs WHERE plant_id = ?", (plant_id,))
    finally:
        con.close()


def save_audit_log(audit_data: dict):
    """Saves an analysis audit run to the database."""
    con = get_connection()
    try:
        con.execute("""
            INSERT OR REPLACE INTO audit_logs 
            (audit_id, plant_id, run_timestamp, tariff_rate, cleaning_cost, total_loss_inr, total_loss_kwh, plant_pr, critical_inverter_count, soiling_inverter_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            audit_data['audit_id'],
            audit_data['plant_id'],
            audit_data['run_timestamp'],
            audit_data['tariff_rate'],
            audit_data['cleaning_cost'],
            audit_data['total_loss_inr'],
            audit_data['total_loss_kwh'],
            audit_data['plant_pr'],
            audit_data['critical_inverter_count'],
            audit_data['soiling_inverter_count']
        ))
    finally:
        con.close()


def get_audit_history(plant_id: str) -> list[dict]:
    """Returns past audits for a given plant."""
    con = get_connection()
    try:
        rows = con.execute("""
            SELECT audit_id, run_timestamp, tariff_rate, cleaning_cost, total_loss_inr, total_loss_kwh, plant_pr, critical_inverter_count, soiling_inverter_count
            FROM audit_logs
            WHERE plant_id = ?
            ORDER BY run_timestamp DESC
            LIMIT 10
        """, (plant_id,)).fetchdf()
        return rows.to_dict(orient='records')
    finally:
        con.close()
