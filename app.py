"""
app.py
SOLAR HEALTH OS — Intelligent Solar Inverter Anomaly & Loss Analytics Platform
Bespoke Industrial UI with Embedded DuckDB Storage & Custom Telemetry Ingestion.
"""

import sys
from pathlib import Path
from datetime import datetime

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.db import (
    init_db,
    list_plants,
    load_plant_telemetry,
    save_audit_log,
    get_audit_history,
    delete_plant,
    seed_default_plants_if_needed
)
from src.uploader import (
    guess_column_mapping,
    find_best_column_match,
    process_and_ingest_uploaded_df,
    process_dual_files_upload,
    STANDARD_COLUMNS
)
from src.features import add_solar_features
from src.model import load_or_train_model, predict_expected_power
from src.detector import detect_anomalies, generate_inverter_scorecard
from src.financial_loss import (
    compute_financial_losses,
    get_plant_loss_summary,
    get_inverter_loss_table,
    get_daily_loss_trend
)

# Page Setup
st.set_page_config(
    page_title="HelioPulse | Solar Anomaly & Revenue Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize embedded database tables
init_db()

# Bespoke Cleantech CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Background and Layout Overhaul */
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0d1527 0%, #080c16 100%);
        color: #f1f5f9;
    }

    /* Header Brand Banner */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 24px;
        background: rgba(19, 27, 46, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        margin-bottom: 24px;
    }
    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .brand-tag {
        font-size: 0.75rem;
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        border: 1px solid rgba(245, 158, 11, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Modern KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #111827;
        padding: 20px 22px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .kpi-loss::before { background: linear-gradient(90deg, #ef4444, #f87171); }
    .kpi-mwh::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .kpi-pr::before { background: linear-gradient(90deg, #0ea5e9, #38bdf8); }
    .kpi-health::before { background: linear-gradient(90deg, #10b981, #34d399); }

    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94a3b8;
    }
    .kpi-number {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 6px 0 2px 0;
        letter-spacing: -0.5px;
    }
    .kpi-desc {
        font-size: 0.8rem;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Industrial Action Tickets */
    .ticket-card {
        background: #131b2e;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .ticket-card:hover {
        border-color: rgba(245, 158, 11, 0.4);
    }
    .ticket-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .ticket-badge {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-soiling {
        background: rgba(245, 158, 11, 0.2);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-optimal {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .ticket-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 14px;
        background: rgba(0, 0, 0, 0.2);
        padding: 12px 14px;
        border-radius: 8px;
    }
    .ticket-stat-title {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
    }
    .ticket-stat-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 2px;
    }
    .ticket-action-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.88rem;
    }

    /* Database status pill */
    .db-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .db-dot {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 6px #10b981;
    }

    /* Style enhancements for dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# Top Navigation & Brand Bar
st.markdown("""
<div class="brand-bar">
    <div>
        <div class="brand-title">HELIOPULSE OS</div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 2px;">
            Commercial Solar Underperformance & Rupee (₹) Loss Diagnostic Platform
        </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
        <div class="db-status">
            <span class="db-dot"></span> DuckDB Storage Active
        </div>
        <span class="brand-tag">v2.1 Industrial Edition</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.markdown("### 🎛️ Data Stream & Source")

# Fetch available plants from DuckDB
available_plants = list_plants()
if len(available_plants) == 0:
    seed_default_plants_if_needed()
    available_plants = list_plants()

plant_options = {p['plant_id']: p['plant_name'] for p in available_plants}

plant_mode = st.sidebar.radio(
    "Active Data Source:",
    options=["database_plant", "upload_custom"],
    format_func=lambda x: "📂 Managed Solar Plants (DuckDB)" if x == "database_plant" else "📤 Ingest Custom Plant Dataset"
)

# Financial and operational parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Financial Parameters")
tariff_inr = st.sidebar.number_input(
    "Power Purchase Agreement (PPA) Tariff (₹ / kWh):",
    min_value=1.00,
    max_value=15.00,
    value=4.50,
    step=0.25,
    help="Contracted feed-in tariff or commercial grid unit rate in Indian Rupees."
)

cleaning_cost_inr = st.sidebar.number_input(
    "Estimated Cleaning Cost per String/Block (₹):",
    min_value=50.0,
    max_value=3000.0,
    value=500.0,
    step=50.0,
    help="Direct cost (labor + RO water + robotics) to wash an inverter's photovoltaic module array."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Anomaly Tuning")
pr_warn_thresh = st.sidebar.slider(
    "Soiling Deficit Trigger (PR < x):",
    min_value=0.70,
    max_value=0.95,
    value=0.88,
    step=0.01,
    help="Performance ratio threshold below which degradation is categorized as soiling/dust accumulation."
)

severe_thresh = st.sidebar.slider(
    "Critical Outage Deficit Trigger (PR < x):",
    min_value=0.30,
    max_value=0.75,
    value=0.60,
    step=0.05,
    help="Performance ratio threshold indicating string disconnection or severe equipment fault."
)


# -------------------------------------------------------------
# MODE A: UPLOAD CUSTOM DATASET TO DUCKDB
# -------------------------------------------------------------
if plant_mode == "upload_custom":
    st.subheader("📤 Custom Solar Telemetry Ingestion Hub")
    st.markdown(
        "Upload raw operational CSV files from your solar SCADA, IoT dataloggers, or monitoring portals. "
        "The system auto-detects column schemas, standardizes weather units, and persists data into the **embedded DuckDB** storage."
    )

    # Quick Inject Sample Expander
    sample_dir = PROJECT_ROOT / "data" / "sample_datasets"
    if sample_dir.exists():
        with st.expander("🧪 Quick-Inject One of 5 Pre-Generated Scenarios into DuckDB", expanded=False):
            st.markdown(
                "Five synthetic datasets have been generated with distinct real-world failure patterns. "
                "Select one below to inject it directly into DuckDB with one click, or upload your own files below."
            )
            sample_options = {
                "1_Rooftop_Jaipur_Soiling_30D.csv": ("sample_jaipur_soiling", "1. Rooftop Jaipur (30 Days - Progressive Dust/Soiling Accumulation)"),
                "2_Bhadla_Desert_Tripping_45D.csv": ("sample_bhadla_tripping", "2. Bhadla Desert Solar Park (45 Days - Midday Thermal Tripping & Outages)"),
                "3_AgriPV_Jodhpur_DualFile": ("sample_agripv_jodhpur", "3. AgriPV Jodhpur (21 Days - Dual-File Merge & Crop Dusting)"),
                "4_Bhiwadi_Industrial_Faults_25D.csv": ("sample_bhiwadi_faults", "4. Bhiwadi Industrial (25 Days - String Fuse Blow & Obstruction Shading)"),
                "5_Neemrana_Clean_Benchmark_30D.csv": ("sample_neemrana_clean", "5. Neemrana Benchmark (30 Days - Optimal Robotic Cleaned Reference)")
            }
            chosen_sample = st.selectbox(
                "Choose Scenario:",
                options=list(sample_options.keys()),
                format_func=lambda k: sample_options[k][1]
            )
            col_inj1, col_inj2 = st.columns([2, 3])
            with col_inj1:
                if st.button("⚡ Inject Scenario into Database", type="primary"):
                    with st.spinner("Ingesting into DuckDB..."):
                        s_pid, s_pname = sample_options[chosen_sample]
                        if chosen_sample == "3_AgriPV_Jodhpur_DualFile":
                            # Merge dual files
                            g_path = sample_dir / "3_AgriPV_Jodhpur_Generation_Stream.csv"
                            w_path = sample_dir / "3_AgriPV_Jodhpur_Weather_Sensor.csv"
                            gdf = pd.read_csv(g_path)
                            wdf = pd.read_csv(w_path)
                            ok, msg, cnt = process_dual_files_upload(
                                gdf, wdf, s_pid, s_pname,
                                gen_time_col='timestamp', weather_time_col='timestamp',
                                inv_col='inverter_serial', power_col='active_generation_kw',
                                irrad_col='ghi_flux_w_m2', mod_temp_col='pv_surface_temp_c',
                                amb_temp_col='air_temperature_c'
                            )
                        else:
                            s_path = sample_dir / chosen_sample
                            s_df = pd.read_csv(s_path)
                            s_mapping = guess_column_mapping(list(s_df.columns))
                            ok, msg, cnt = process_and_ingest_uploaded_df(s_df, s_pid, s_pname, s_mapping)

                        if ok:
                            st.success(f"🎉 {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
            with col_inj2:
                st.caption(f"📁 Source: `data/sample_datasets/` &bull; Click to inject and run ML modeling immediately.")

    # Kaggle Auto-Restore Expander
    with st.expander("🔄 Auto-Restore Official Kaggle Datasets (Plant 1 & Plant 2)", expanded=True):
        st.markdown(
            "If you are deploying in a fresh cloud container (e.g. Streamlit Cloud) or your local database was reset, "
            "click below to download and restore the official Kaggle benchmark datasets into DuckDB automatically."
        )
        col_res1, col_res2 = st.columns([2, 3])
        with col_res1:
            if st.button("🚀 Restore Kaggle Plant 1 & Plant 2", type="secondary"):
                with st.spinner("Downloading and restoring Kaggle datasets into DuckDB..."):
                    seed_default_plants_if_needed(force_reseed=True)
                    st.success("Successfully restored Kaggle Plant 1 and Plant 2!")
                    st.rerun()
        with col_res2:
            st.caption("Downloads official Kaggle files and pre-loads both plants into DuckDB with all telemetry tables.")

    upload_type = st.radio(
        "Manual Upload Mode:",
        options=["single_unified", "dual_files"],
        format_func=lambda x: "Single Unified Telemetry CSV (Weather + Generation combined)" if x == "single_unified" else "Separate Generation & Weather Sensor CSVs",
        horizontal=True
    )

    new_plant_id = st.text_input("New Plant Identifier (e.g., rooftop_jaipur_01):", value=f"plant_{datetime.now().strftime('%Y%m%d_%H%M')}")
    new_plant_name = st.text_input("Solar Farm / Facility Name:", value="Rooftop Commercial Plant Rajasthan")

    if upload_type == "single_unified":
        uploaded_file = st.file_uploader("Upload CSV / Excel Telemetry File:", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".xlsx"):
                    raw_df = pd.read_excel(uploaded_file)
                else:
                    raw_df = pd.read_csv(uploaded_file)

                st.success(f"Loaded file with {len(raw_df):,} rows and columns: {', '.join(raw_df.columns)}")
                st.markdown("#### Intelligent Column Mapping Matrix")
                st.caption("Confirm or adjust column assignments:")

                guessed = guess_column_mapping(list(raw_df.columns))
                col_options = ["-- None --"] + list(raw_df.columns)

                c1, c2, c3 = st.columns(3)
                mapped_cols = {}
                with c1:
                    mapped_cols['DATE_TIME'] = st.selectbox("Date / Timestamp Column *", options=col_options, index=col_options.index(guessed.get('DATE_TIME')) if guessed.get('DATE_TIME') in col_options else 0)
                    mapped_cols['SOURCE_KEY'] = st.selectbox("Inverter Identifier Column *", options=col_options, index=col_options.index(guessed.get('SOURCE_KEY')) if guessed.get('SOURCE_KEY') in col_options else 0)
                with c2:
                    mapped_cols['AC_POWER'] = st.selectbox("AC Active Power (kW) *", options=col_options, index=col_options.index(guessed.get('AC_POWER')) if guessed.get('AC_POWER') in col_options else 0)
                    mapped_cols['IRRADIATION'] = st.selectbox("Solar Irradiation (kW/m² or W/m²) *", options=col_options, index=col_options.index(guessed.get('IRRADIATION')) if guessed.get('IRRADIATION') in col_options else 0)
                with c3:
                    mapped_cols['MODULE_TEMPERATURE'] = st.selectbox("Module Temperature (°C) *", options=col_options, index=col_options.index(guessed.get('MODULE_TEMPERATURE')) if guessed.get('MODULE_TEMPERATURE') in col_options else 0)
                    mapped_cols['AMBIENT_TEMPERATURE'] = st.selectbox("Ambient Temperature (°C) *", options=col_options, index=col_options.index(guessed.get('AMBIENT_TEMPERATURE')) if guessed.get('AMBIENT_TEMPERATURE') in col_options else 0)

                # Replace '-- None --' with None
                clean_mapping = {k: (v if v != "-- None --" else None) for k, v in mapped_cols.items()}

                missing_essentials = [k for k in ['IRRADIATION', 'MODULE_TEMPERATURE'] if not clean_mapping.get(k)]
                if missing_essentials:
                    st.warning(
                        "💡 **Notice for Kaggle Datasets:** The standard Kaggle dataset splits telemetry into separate "
                        "Generation (`Plant_X_Generation_Data.csv`) and Weather (`Plant_X_Weather_Sensor_Data.csv`) files. "
                        "If you are uploading Kaggle data, select **'Separate Generation & Weather Sensor CSVs'** above!"
                    )

                st.markdown("#### Sample Preview (First 5 Rows)")
                st.dataframe(raw_df.head(5), use_container_width=True)

                if st.button("🚀 Ingest Dataset into DuckDB & Run Analysis", type="primary"):
                    with st.spinner("Processing schema, normalizing units, and indexing into DuckDB..."):
                        ok, msg, count = process_and_ingest_uploaded_df(
                            raw_df,
                            plant_id=new_plant_id.strip(),
                            plant_name=new_plant_name.strip(),
                            col_mapping=clean_mapping
                        )
                        if ok:
                            st.success(f"🎉 {msg}")
                            st.info("You can now select this plant under 'Managed Solar Plants' in the sidebar!")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
            except Exception as e:
                st.error(f"Failed to read file: {str(e)}")

    else:
        st.markdown("##### Upload Two Files (Generation & Weather)")
        c_up1, c_up2 = st.columns(2)
        with c_up1:
            gen_f = st.file_uploader("1. Inverter Generation CSV (e.g., Plant_1_Generation_Data.csv):", type=["csv"])
        with c_up2:
            weather_f = st.file_uploader("2. Weather Sensor CSV (e.g., Plant_1_Weather_Sensor_Data.csv):", type=["csv"])

        if gen_f and weather_f:
            gdf = pd.read_csv(gen_f)
            wdf = pd.read_csv(weather_f)

            g_cols = list(gdf.columns)
            w_cols = list(wdf.columns)

            def get_col_index(columns, patterns, fallback=0):
                match = find_best_column_match(columns, patterns)
                if match and match in columns:
                    return columns.index(match)
                return fallback

            # Auto-detect generation columns
            g_dt_idx = get_col_index(g_cols, STANDARD_COLUMNS['DATE_TIME'])
            g_inv_idx = get_col_index(g_cols, STANDARD_COLUMNS['SOURCE_KEY'])
            g_ac_idx = get_col_index(g_cols, STANDARD_COLUMNS['AC_POWER'])

            # Auto-detect weather columns
            w_dt_idx = get_col_index(w_cols, STANDARD_COLUMNS['DATE_TIME'])
            w_irr_idx = get_col_index(w_cols, STANDARD_COLUMNS['IRRADIATION'])
            w_mod_idx = get_col_index(w_cols, STANDARD_COLUMNS['MODULE_TEMPERATURE'])
            w_amb_idx = get_col_index(w_cols, STANDARD_COLUMNS['AMBIENT_TEMPERATURE'])

            st.success(f"Loaded Generation file ({len(gdf):,} rows) and Weather file ({len(wdf):,} rows).")
            st.markdown("#### Intelligent Dual-File Column Auto-Mapping")
            st.caption("Fields below have been automatically matched with high-priority schema patterns:")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Generation File Mapping:**")
                g_dt = st.selectbox("Timestamp Column (Gen) *", options=g_cols, index=g_dt_idx)
                g_inv = st.selectbox("Inverter Identifier Column *", options=g_cols, index=g_inv_idx)
                g_ac = st.selectbox("AC Active Power (kW) *", options=g_cols, index=g_ac_idx)
            with c2:
                st.markdown("**Weather Sensor Mapping:**")
                w_dt = st.selectbox("Timestamp Column (Weather) *", options=w_cols, index=w_dt_idx)
                w_irr = st.selectbox("Solar Irradiation Column *", options=w_cols, index=w_irr_idx)
                w_mod = st.selectbox("Module Temperature (°C) *", options=w_cols, index=w_mod_idx)
                w_amb = st.selectbox("Ambient Temperature (°C) *", options=w_cols, index=w_amb_idx)

            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Generation Sample (Top 3 Rows):**")
                st.dataframe(gdf.head(3), use_container_width=True)
            with p2:
                st.markdown("**Weather Sample (Top 3 Rows):**")
                st.dataframe(wdf.head(3), use_container_width=True)

            if st.button("🚀 Merge & Ingest into DuckDB", type="primary"):
                with st.spinner("Aligning timestamps, normalizing units, and indexing into DuckDB..."):
                    ok, msg, count = process_dual_files_upload(
                        gdf, wdf,
                        plant_id=new_plant_id.strip(),
                        plant_name=new_plant_name.strip(),
                        gen_time_col=g_dt,
                        weather_time_col=w_dt,
                        inv_col=g_inv,
                        power_col=g_ac,
                        irrad_col=w_irr,
                        mod_temp_col=w_mod,
                        amb_temp_col=w_amb
                    )
                    if ok:
                        st.success(f"🎉 {msg}")
                        st.info("You can now select this plant under 'Managed Solar Plants' in the sidebar!")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

    st.stop()


# -------------------------------------------------------------
# MODE B: ACTIVE PLANT ANALYTICS (DUCKDB DRIVEN)
# -------------------------------------------------------------
selected_plant_id = st.sidebar.selectbox(
    "Select Target Solar Plant:",
    options=list(plant_options.keys()),
    format_func=lambda pid: plant_options.get(pid, pid)
)

# Optional delete button for uploaded plants
current_plant_info = next((p for p in available_plants if p['plant_id'] == selected_plant_id), None)
if current_plant_info and current_plant_info.get('source_type') == 'uploaded':
    if st.sidebar.button(f"🗑️ Remove '{current_plant_info['plant_name']}' from DB"):
        delete_plant(selected_plant_id)
        st.sidebar.success("Plant removed from database.")
        st.rerun()


@st.cache_data(show_spinner=False)
def load_and_model_plant(plant_id: str):
    """Loads telemetry directly from DuckDB and executes ML prediction pipeline."""
    df_raw = load_plant_telemetry(plant_id)
    if len(df_raw) == 0:
        return None, None, None

    # Derive temporal metadata
    df_raw['DATE_TIME'] = pd.to_datetime(df_raw['DATE_TIME'])
    df_raw['DATE'] = df_raw['DATE_TIME'].dt.date
    df_raw['HOUR'] = df_raw['DATE_TIME'].dt.hour
    df_raw['MINUTE'] = df_raw['DATE_TIME'].dt.minute
    df_raw['IS_DAYLIGHT'] = (df_raw['IRRADIATION'] > 0.005) | (df_raw['AC_POWER'] > 0)

    # Feature engineering
    featured_df = add_solar_features(df_raw)

    # Train/load plant-specific regression baseline
    model, metrics, feat_importance = load_or_train_model(featured_df, plant_id=plant_id)

    featured_df['EXPECTED_AC_POWER'] = predict_expected_power(model, featured_df)
    return featured_df, metrics, feat_importance


with st.spinner("Executing telemetry query from DuckDB and calculating power residuals..."):
    telemetry_df, model_metrics, feat_importance = load_and_model_plant(selected_plant_id)

if telemetry_df is None or len(telemetry_df) == 0:
    st.error(f"No records found in DuckDB for plant: {selected_plant_id}")
    st.stop()


# Date Range Filtering
min_d = telemetry_df['DATE'].min()
max_d = telemetry_df['DATE'].max()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Temporal Window")
date_range = st.sidebar.date_input(
    "Filter Audit Period:",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    active_df = telemetry_df[(telemetry_df['DATE'] >= start_d) & (telemetry_df['DATE'] <= end_d)].copy()
elif isinstance(date_range, tuple) and len(date_range) == 1:
    single_d = date_range[0]
    active_df = telemetry_df[telemetry_df['DATE'] == single_d].copy()
else:
    active_df = telemetry_df.copy()

if len(active_df) == 0:
    st.warning("Selected date range returned 0 records. Displaying full date range.")
    active_df = telemetry_df.copy()


# Execute Anomaly & Financial Calculations
active_df = detect_anomalies(
    active_df,
    pr_threshold_warn=pr_warn_thresh,
    pr_threshold_severe=severe_thresh
)
active_df = compute_financial_losses(active_df, tariff_inr=tariff_inr)

plant_summary = get_plant_loss_summary(active_df, tariff_inr=tariff_inr, cleaning_cost=cleaning_cost_inr)
scorecard = generate_inverter_scorecard(active_df)
loss_table = get_inverter_loss_table(active_df, scorecard, tariff_inr=tariff_inr, cleaning_cost=cleaning_cost_inr)
daily_trend = get_daily_loss_trend(active_df)

critical_inverters = loss_table[loss_table['DIAGNOSIS'] == 'Critical Fault']
soiling_inverters = loss_table[loss_table['DIAGNOSIS'].isin(['Moderate Soiling', 'Heavy Degradation'])]


# Top Industrial KPI Grid
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card kpi-loss">
        <div class="kpi-label">Cumulative Revenue Lost</div>
        <div class="kpi-number">₹{plant_summary['total_revenue_loss_inr']:,.0f}</div>
        <div class="kpi-desc">
            <span>Avg. ₹{plant_summary['daily_avg_loss_inr']:,.0f} / day</span> &bull; <span>{plant_summary['num_days_analyzed']} days</span>
        </div>
    </div>
    <div class="kpi-card kpi-mwh">
        <div class="kpi-label">Unharvested Generation Gap</div>
        <div class="kpi-number">{plant_summary['total_energy_loss_mwh']:,.1f} <span style="font-size: 1.1rem; font-weight: 500;">MWh</span></div>
        <div class="kpi-desc">
            <span>{plant_summary['total_energy_loss_kwh']:,.0f} kWh shortfall</span>
        </div>
    </div>
    <div class="kpi-card kpi-pr">
        <div class="kpi-label">Plant Performance Ratio</div>
        <div class="kpi-number">{plant_summary['overall_plant_pr'] * 100:.1f}%</div>
        <div class="kpi-desc">
            <span>Delivered: {plant_summary['total_actual_mwh']:,.0f} MWh</span> &bull; <span>Expected: {plant_summary['total_expected_mwh']:,.0f} MWh</span>
        </div>
    </div>
    <div class="kpi-card kpi-health">
        <div class="kpi-label">Operational Action Required</div>
        <div class="kpi-number">{len(critical_inverters) + len(soiling_inverters)} <span style="font-size: 1.1rem; font-weight: 500;">/ {len(loss_table)} Inverters</span></div>
        <div class="kpi-desc">
            <span style="color: #fca5a5;">⚡ {len(critical_inverters)} Tripped</span> &bull; <span style="color: #fcd34d;">🧹 {len(soiling_inverters)} Soiled</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Application Navigation Tabs
tab_action, tab_leaderboard, tab_curves, tab_ml, tab_audit = st.tabs([
    "⚡ Actionable Maintenance Center (USP)",
    "🏆 Inverter Leaderboard & Financials",
    "📈 Dynamic Power & Weather Analytics",
    "🧠 LightGBM Residual Model",
    "💾 DuckDB Audits & Historical Snapshots"
])


# -------------------------------------------------------------
# TAB 1: USP ACTION CENTER
# -------------------------------------------------------------
with tab_action:
    st.subheader("Field Maintenance Dispatch & ROI Optimizer")
    st.markdown(
        "Commercial solar teams cannot act on raw percentage deviations alone. "
        "The engine classifies root causes and maps every fault to a **concrete maintenance intervention with verified financial ROI**."
    )

    col_queue, col_summary = st.columns([3, 2])

    with col_queue:
        st.markdown("#### 🚨 Priority Action Queue")

        flagged = loss_table[loss_table['DIAGNOSIS'] != 'Optimal Health']

        if len(flagged) == 0:
            st.success("✨ All inverter blocks are operating at peak health and nominal capacity!")
        else:
            for _, row in flagged.iterrows():
                is_crit = (row['DIAGNOSIS'] == 'Critical Fault')
                badge_class = "badge-critical" if is_crit else "badge-soiling"
                icon = "⚡" if is_crit else "🧹"

                st.markdown(f"""
                <div class="ticket-card">
                    <div class="ticket-header">
                        <div>
                            <span style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">
                                {icon} {row['INVERTER_NAME']} <span style="color: #64748b; font-weight: 400; font-size: 0.85rem;">({row['SOURCE_KEY'][:12]}...)</span>
                            </span>
                        </div>
                        <span class="ticket-badge {badge_class}">{row['DIAGNOSIS']}</span>
                    </div>
                    <div class="ticket-grid">
                        <div>
                            <div class="ticket-stat-title">Revenue Lost</div>
                            <div class="ticket-stat-val" style="color: #f87171;">₹{row['TOTAL_LOSS_INR']:,.0f}</div>
                        </div>
                        <div>
                            <div class="ticket-stat-title">Underperformance</div>
                            <div class="ticket-stat-val">{row['UNDERPERF_PCT']:.1f}%</div>
                        </div>
                        <div>
                            <div class="ticket-stat-title">Downtime Hours</div>
                            <div class="ticket-stat-val">{row['OFFLINE_HOURS']:.1f} hrs</div>
                        </div>
                        <div>
                            <div class="ticket-stat-title">Health Score</div>
                            <div class="ticket-stat-val">{row['HEALTH_SCORE']:.1f} / 100</div>
                        </div>
                    </div>
                    <div class="ticket-action-bar">
                        <div style="color: #fbbf24; font-weight: 600;">
                            👉 Directive: {row['ACTION']}
                        </div>
                        <div style="color: #94a3b8; font-size: 0.82rem;">
                            Cleaning Cost: ₹{row['CLEANING_COST_INR']:,.0f} &bull; <b style="color: #34d399;">Net ROI: ₹{max(0.0, row['NET_CLEANING_ROI_INR']):,.0f} ({row['ROI_RATIO']:.0f}x)</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_summary:
        st.markdown("#### 📊 Financial Attribution")
        pie_data = loss_table.groupby('DIAGNOSIS')['TOTAL_LOSS_INR'].sum().reset_index()
        fig_pie = px.pie(
            pie_data,
            names='DIAGNOSIS',
            values='TOTAL_LOSS_INR',
            hole=0.55,
            color='DIAGNOSIS',
            color_discrete_map={
                'Critical Fault': '#ef4444',
                'Heavy Degradation': '#f97316',
                'Moderate Soiling': '#f59e0b',
                'Optimal Health': '#10b981'
            }
        )
        fig_pie.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            height=280
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Cleaning Economics Card
        soiling_losses = loss_table[loss_table['DIAGNOSIS'].isin(['Moderate Soiling', 'Heavy Degradation'])]['TOTAL_LOSS_INR'].sum()
        soiling_blocks = len(loss_table[loss_table['DIAGNOSIS'].isin(['Moderate Soiling', 'Heavy Degradation'])])
        total_wash_cost = soiling_blocks * cleaning_cost_inr

        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 18px 20px; margin-top: 16px;">
            <div style="font-weight: 700; color: #34d399; font-size: 1rem; margin-bottom: 8px;">
                💡 Panel Cleaning Business Case
            </div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                &bull; Total Revenue Lost to Soiling: <b style="color: #f1f5f9;">₹{soiling_losses:,.0f}</b><br>
                &bull; Cost to Wash {soiling_blocks} Identified Inverter Strings: <b style="color: #f1f5f9;">₹{total_wash_cost:,.0f}</b><br>
                &bull; Net Operating Profit Unlocked: <b style="color: #34d399; font-size: 1.05rem;">₹{max(0.0, soiling_losses - total_wash_cost):,.0f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 2: INVERTER LEADERBOARD
# -------------------------------------------------------------
with tab_leaderboard:
    st.subheader("Inverter Performance & Financial Audit")
    st.markdown("All inverters indexed and ranked by commercial revenue deficit.")

    fig_rank = px.bar(
        loss_table,
        x='TOTAL_LOSS_INR',
        y='INVERTER_NAME',
        orientation='h',
        color='DIAGNOSIS',
        color_discrete_map={
            'Critical Fault': '#ef4444',
            'Heavy Degradation': '#f97316',
            'Moderate Soiling': '#f59e0b',
            'Optimal Health': '#10b981'
        },
        labels={'TOTAL_LOSS_INR': 'Revenue Lost (₹)', 'INVERTER_NAME': 'Inverter'},
        hover_data=['UNDERPERF_PCT', 'OFFLINE_HOURS', 'HEALTH_SCORE']
    )
    fig_rank.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis={'categoryorder': 'total ascending'},
        height=580,
        margin=dict(l=10, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("#### Complete Diagnostic Audit Table")
    audit_cols = [
        'RANK', 'INVERTER_NAME', 'TOTAL_LOSS_INR', 'UNDERPERF_PCT',
        'OFFLINE_HOURS', 'SOILING_HOURS', 'HEALTH_SCORE', 'DIAGNOSIS', 'ACTION'
    ]
    st.dataframe(
        loss_table[audit_cols].style.format({
            'TOTAL_LOSS_INR': '₹{:,.0f}',
            'UNDERPERF_PCT': '{:.1f}%',
            'OFFLINE_HOURS': '{:.1f}h',
            'SOILING_HOURS': '{:.1f}h',
            'HEALTH_SCORE': '{:.1f}'
        }),
        use_container_width=True
    )

    report_csv = loss_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Export Formal Engineering Audit Report (CSV)",
        data=report_csv,
        file_name=f"{selected_plant_id}_Maintenance_Audit.csv",
        mime="text/csv"
    )


# -------------------------------------------------------------
# TAB 3: DYNAMIC POWER & WEATHER CURVES
# -------------------------------------------------------------
with tab_curves:
    st.subheader("15-Minute Generation vs Expected Capacity Curves")
    st.markdown("Investigate specific daily power curves with the unharvested energy gap highlighted.")

    inv_select_list = ["Entire Solar Farm (Aggregated)"] + list(loss_table['INVERTER_NAME'].values)
    chosen_inv = st.selectbox("Inspect Inverter Unit:", options=inv_select_list)

    chosen_date = st.date_input(
        "Select Calendar Day to Audit:",
        value=min_d,
        min_value=min_d,
        max_value=max_d
    )

    day_subset = active_df[active_df['DATE'] == chosen_date].copy()

    if chosen_inv != "Entire Solar Farm (Aggregated)":
        target_sk = loss_table[loss_table['INVERTER_NAME'] == chosen_inv]['SOURCE_KEY'].iloc[0]
        plot_data = day_subset[day_subset['SOURCE_KEY'] == target_sk].sort_values('DATE_TIME')
    else:
        plot_data = day_subset.groupby('DATE_TIME').agg({
            'AC_POWER': 'sum',
            'EXPECTED_AC_POWER': 'sum',
            'IRRADIATION': 'mean',
            'MODULE_TEMPERATURE': 'mean'
        }).reset_index().sort_values('DATE_TIME')

    if len(plot_data) == 0:
        st.warning(f"No telemetry data points available for {chosen_date}.")
    else:
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=plot_data['DATE_TIME'],
            y=plot_data['EXPECTED_AC_POWER'],
            mode='lines',
            name='Expected AC Capacity (Model)',
            line=dict(color='#38bdf8', width=2.5, dash='dash')
        ))
        fig_curve.add_trace(go.Scatter(
            x=plot_data['DATE_TIME'],
            y=plot_data['AC_POWER'],
            mode='lines+markers',
            name='Actual AC Generation (Delivered)',
            line=dict(color='#10b981', width=3),
            fill='tonexty',
            fillcolor='rgba(239, 68, 68, 0.18)'
        ))
        fig_curve.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f"Power Curve: {chosen_inv} on {chosen_date}",
            xaxis_title="Timestamp",
            yaxis_title="Active Power (kW)",
            hovermode="x unified",
            height=440
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        w1, w2 = st.columns(2)
        with w1:
            fig_irr = px.area(
                plot_data, x='DATE_TIME', y='IRRADIATION',
                title="Solar Irradiance Feed (kW/m²)",
                color_discrete_sequence=['#fbbf24']
            )
            fig_irr.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=260)
            st.plotly_chart(fig_irr, use_container_width=True)
        with w2:
            fig_temp = px.line(
                plot_data, x='DATE_TIME', y='MODULE_TEMPERATURE',
                title="Module Surface Temperature (°C)",
                color_discrete_sequence=['#f87171']
            )
            fig_temp.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=260)
            st.plotly_chart(fig_temp, use_container_width=True)


# -------------------------------------------------------------
# TAB 4: LIGHTGBM MODEL & EXPLAINABILITY
# -------------------------------------------------------------
with tab_ml:
    st.subheader("Machine Learning Baseline Architecture")
    st.markdown(
        "A gradient-boosted regression tree (**LightGBM**) models the non-linear relationship between solar geometry, "
        "atmospheric irradiance, thermal panel excitation, and delivered AC power."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Regression Fit (R² Score)", f"{model_metrics['test_r2']:.4f}")
    m2.metric("Mean Absolute Error (MAE)", f"{model_metrics['test_mae']:.2f} kW")
    m3.metric("Root Mean Squared Error (RMSE)", f"{model_metrics['test_rmse']:.2f} kW")

    st.markdown("#### Feature Importance Spectrum")
    fig_imp = px.bar(
        feat_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='plasma'
    )
    fig_imp.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis={'categoryorder': 'total ascending'},
        height=380
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# -------------------------------------------------------------
# TAB 5: DUCKDB AUDITS & PERSISTENCE
# -------------------------------------------------------------
with tab_audit:
    st.subheader("💾 DuckDB Audit Snapshots & Run History")
    st.markdown(
        "Save analytical snapshots directly into the local DuckDB database to track revenue loss "
        "trends across multiple maintenance cycles."
    )

    col_save, col_hist = st.columns([1, 2])

    with col_save:
        st.markdown("#### Snapshot Current Run")
        if st.button("💾 Commit Audit Snapshot to Database", type="primary"):
            audit_id = f"audit_{selected_plant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            audit_entry = {
                'audit_id': audit_id,
                'plant_id': selected_plant_id,
                'run_timestamp': datetime.now(),
                'tariff_rate': float(tariff_inr),
                'cleaning_cost': float(cleaning_cost_inr),
                'total_loss_inr': float(plant_summary['total_revenue_loss_inr']),
                'total_loss_kwh': float(plant_summary['total_energy_loss_kwh']),
                'plant_pr': float(plant_summary['overall_plant_pr']),
                'critical_inverter_count': int(len(critical_inverters)),
                'soiling_inverter_count': int(len(soiling_inverters))
            }
            save_audit_log(audit_entry)
            st.success(f"Audit snapshot committed as `{audit_id}`!")

    with col_hist:
        st.markdown("#### Historical Audit Records")
        history = get_audit_history(selected_plant_id)
        if len(history) == 0:
            st.info("No audit snapshots recorded yet for this plant. Click 'Commit Audit Snapshot' to record one.")
        else:
            hist_df = pd.DataFrame(history)
            st.dataframe(
                hist_df[['audit_id', 'run_timestamp', 'tariff_rate', 'total_loss_inr', 'total_loss_kwh', 'plant_pr', 'critical_inverter_count']].style.format({
                    'tariff_rate': '₹{:.2f}',
                    'total_loss_inr': '₹{:,.0f}',
                    'total_loss_kwh': '{:,.0f} kWh',
                    'plant_pr': '{:.2%}'
                }),
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
    "HelioPulse OS &bull; Rajasthan Solar Farm Telemetry & Loss Analytics &bull; Embedded DuckDB Engine &bull; LightGBM ML Pipeline"
    "</div>",
    unsafe_allow_html=True
)
