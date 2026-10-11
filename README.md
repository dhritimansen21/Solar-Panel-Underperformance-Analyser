# ☀️ Solar Panel Underperformance & Revenue Loss Detector

An end-to-end Machine Learning and Financial Analytics system designed for solar farm and rooftop operators in sunny, dust-prone regions (such as Rajasthan, India).

The system continuously forecasts **expected solar AC power** from meteorological sensor feeds (solar irradiation, ambient and module temperatures), identifies **inverter-level underperformance and electrical tripping anomalies**, and translates technical deficits directly into **rupee (₹) revenue losses and prioritized maintenance action cards**.

---

## 🌟 Unique Selling Proposition (USP)

Most industrial solar analytics dashboards show raw kilowatt-hour plots or generic percentage efficiency drops. This tool answers the commercial question that plant managers care about:

> **"Inverter 11 is underperforming by 8.9%, costing ₹1,09,297 this month. Cleaning the panel string costs ₹500, delivering a net ROI of +₹1,08,797."**

It bridges the gap between predictive ML modeling and concrete on-ground operations (washing panels vs inspecting tripped circuit breakers).

---

## 📁 Project Structure

```
solar panel underperformance analyser/
├── data/
│   ├── raw/                           # Raw Kaggle CSVs (Plant 1 & 2 Generation + Weather)
│   └── processed/                     # Cleaned, merged feature datasets
├── models/
│   ├── expected_power_plant_1.joblib  # Trained LightGBM regressor for Plant 1
│   └── expected_power_plant_2.joblib  # Trained LightGBM regressor for Plant 2
├── src/
│   ├── data_loader.py                 # Timestamp parsing, daylight tagging, data merging
│   ├── features.py                    # Solar angles, temperature differentials, interactions
│   ├── model.py                       # LightGBM regressor training, evaluation, inference
│   ├── detector.py                    # PR calculation, anomaly classification (tripping vs soiling)
│   └── financial_loss.py              # Energy deficit (kWh) to INR (₹) loss engine & ROI
├── app.py                             # Interactive Streamlit Web Dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the App

1. Ensure requirements are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501`.

---

## 🧠 Methodology & ML Pipeline

1. **Weather-Driven Expected Power Model**:
   - Uses a **LightGBM Regressor** trained on operational daylight periods to establish nominal capacity.
   - Inputs: Solar Irradiation ($kW/m^2$), Module Temperature (°C), Ambient Temperature (°C), Temperature Differential ($\Delta T$), Cyclical Solar Hour ($\sin / \cos$), and non-linear saturation terms.
   - **Performance**: Test $R^2 \approx 0.978$, MAE $\approx 30.68\text{ kW}$ on Plant 1.

2. **Fault Disambiguation**:
   - **Critical Inverter Trip / Breaker Drop**: Sun is high ($>0.05\text{ kW/m}^2$) but power output $< 5\text{ kW}$.
   - **Severe String / Electrical Deficit**: Performance Ratio ($PR < 60\%$).
   - **Soiling & Dust Accumulation**: Consistent deficit ($60\% \le PR < 88\%$).
   - **Optimal Health**: $PR \ge 88\%$.

3. **Financial Quantification**:
   $$\text{Energy Deficit (kWh)} = \max(\text{Expected AC Power} - \text{Actual AC Power}, 0) \times 0.25\text{ hr}$$
   $$\text{Financial Loss (₹)} = \text{Energy Deficit (kWh)} \times \text{Tariff Rate (₹/kWh)}$$
   $$\text{Maintenance Net ROI (₹)} = \text{Recoverable Soiling Loss} - \text{Cleaning Cost}$$

---

## 📊 Dataset & Limitations

- **Source**: Kaggle *"Solar Power Generation Data"* (two Indian solar power plants with 22 inverters each, 15-minute readings).
- **Observation Window**: ~34 days (May 15, 2020 – June 17, 2020).
- **Limitation & Future Scope**: Multi-month seasonal dust accumulation and monsoon wash cycles represent valuable future work; this project focuses on inverter-level fault detection and operational loss quantification.
