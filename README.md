# ☀️ HelioPulse OS — Solar Inverter Anomaly & Revenue Loss Analytics Platform

An end-to-end Machine Learning, DuckDB-powered Telemetry Engine, and Financial Analytics system designed for solar farm and rooftop operators in sunny, dust-prone regions (such as Rajasthan, India).

The system continuously forecasts **expected solar AC power** from meteorological sensor feeds (solar irradiation, ambient and module temperatures), identifies **inverter-level underperformance and electrical tripping anomalies**, and translates technical deficits directly into **rupee (₹) revenue losses and prioritized maintenance action cards with calculated ROI**.

---

## 🌟 Unique Selling Proposition (USP)

Most industrial solar analytics dashboards show raw kilowatt-hour plots or generic percentage efficiency drops. This tool answers the commercial question that plant managers care about:

> **"Inverter 11 is underperforming by 8.9%, costing ₹1,09,297 this month. Cleaning the panel string costs ₹500, delivering a net ROI of +₹1,08,797 (200x ROI)."**

It bridges the gap between predictive ML modeling and concrete on-ground operations (washing panels vs inspecting tripped circuit breakers).

---

## 📁 Project Structure

```
solar panel underperformance analyser/
├── data/
│   ├── raw/                           # Raw Kaggle CSVs (Plant 1 & 2 Generation + Weather)
│   ├── sample_datasets/               # 5 Pre-generated synthetic test scenarios
│   │   ├── 1_Rooftop_Jaipur_Soiling_30D.csv
│   │   ├── 2_Bhadla_Desert_Tripping_45D.csv
│   │   ├── 3_AgriPV_Jodhpur_Generation_Stream.csv
│   │   ├── 3_AgriPV_Jodhpur_Weather_Sensor.csv
│   │   ├── 4_Bhiwadi_Industrial_Faults_25D.csv
│   │   └── 5_Neemrana_Clean_Benchmark_30D.csv
│   └── solar_analytics.duckdb         # Embedded zero-config SQL database
├── models/
│   ├── expected_power_plant_1.joblib  # Plant-specific trained LightGBM models
│   └── expected_power_plant_2.joblib
├── src/
│   ├── db.py                          # Embedded DuckDB engine for telemetry & audit logs
│   ├── uploader.py                    # Intelligent column mapping, dual-file merger, date parser
│   ├── data_loader.py                 # Timestamp parsing, daylight tagging, data merging
│   ├── features.py                    # Solar angles, temperature differentials, interactions
│   ├── model.py                       # Plant-adaptive LightGBM regressor training & inference
│   ├── detector.py                    # Adaptive PR calculation, tripping vs soiling classifier
│   └── financial_loss.py              # Auto-interval sampling, kWh to INR (₹) loss engine & ROI
├── tests/
│   └── test_pipeline.py               # Automated end-to-end QA verification suite
├── scripts/
│   └── generate_sample_datasets.py    # Synthetic real-world dataset generator
├── app.py                             # Bespoke Industrial Streamlit Web Dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the App

1. Ensure requirements are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run automated verification suite:
   ```bash
   python tests/test_pipeline.py
   ```

3. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`.

---

## 🧠 Key Technical Highlights

1. **Plant-Specific Dynamic LightGBM Regressors**:
   - Automatically adapts to any inverter scale (from 10 kW residential systems to 100 MW utility plants).
   - Dynamically learns healthy nominal capacity ($R^2 \ge 0.97 - 0.99$).
   - Models are cached specifically per plant (`expected_power_{plant_id}.joblib`).

2. **Adaptive Interval Detection**:
   - Detects the sampling interval (5m, 10m, 15m, 30m, 60m) from telemetry timestamps instead of hardcoding time intervals.

3. **Fault Disambiguation**:
   - **Critical Inverter Trip / Breaker Drop**: Sun is high but power output near zero.
   - **Soiling & Dust Accumulation**: Consistent operational deficit ($60\% \le PR < 88\%$).
   - **Severe String Fault / Disconnection**: Persistent $PR < 60\%$.
   - **Optimal Operation**: $PR \ge 88\%$.

4. **Embedded DuckDB Storage**:
   - Zero background services required. Telemetry, plants, and audit snapshots are persisted in `data/solar_analytics.duckdb`.
