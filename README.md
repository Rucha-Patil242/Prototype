# Polar Energy Manager

AI-based energy management system for polar research stations: load forecasting, renewable integration and fuel optimization.

**SIH Problem Statement ID:** 61
**Team:** <team name> | <member names>
**Live demo:** <Streamlit link>
**Demo video:** <video link>

## Problem
Polar stations depend on diesel that is costly and hard to ship. Solar and wind are unpredictable, extreme cold raises heating load, and batteries lose capacity in the cold.

## Our solution
A web dashboard that forecasts station load and renewable output, then plans how to supply the load with the least diesel.

- **AI forecasting:** Gradient Boosting predicts load and solar + wind output, tested on unseen data (MAE, RMSE, MAPE).
- **Look-ahead dispatch:** every hour it reads the next 12 hours of forecast. Renewables come first, then the battery (cold-derated, 20% reserve), then diesel in short, efficient runs (never below 30% load).
- **Explainable recommendations:** the best source mix for any hour, with plain-language reasons, plus a 24-hour schedule (CSV download).
- **Comparison:** diesel-only vs reactive rules vs AI forecast-driven, with fuel saved, renewable share and CO₂ avoided.
- **Polar scenarios:** summer, polar night, blizzard, extreme cold.

## Workflow
User input (scenario, station, solar/wind/battery size, CSV or live weather) → data sources → processing → AI forecast → look-ahead optimizer → results and reports → recommended energy mix.

## Data
- **Simulated polar scenarios** (built in).
- **NASA POWER** hourly CSV (columns `YEAR, MO, DY, HR, ALLSKY_SFC_SW_DWN, T2M, WS10M`), uploaded in the sidebar.
- **Open-Meteo** live weather and 48-hour forecast for Bharati or Maitri (button in the sidebar).
- **Station CSV** with columns `timestamp, temp, wind_speed, sun, solar, wind, load`.

## Tech stack
Python, pandas, NumPy, scikit-learn, Plotly, Streamlit, GitHub, Streamlit Community Cloud, NASA POWER, Open-Meteo.

## Files
- `app.py`: dashboard
- `engine.py`: data handling, forecasting and dispatch logic
- `requirements.txt`: libraries

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
- This is a simulation prototype. Solar output, wind output and station load are modelled from weather, because real station meter data is not public.
- The look-ahead dispatch is rule-based, not a mathematical optimum.
- Live weather is real, but the station itself is not yet connected to live sensors.
- Results depend on the scenario and settings, and are shown live in the dashboard.

## Future scope
SCADA/IoT integration (MQTT/Modbus), MILP optimization (PuLP), LSTM forecasting, reinforcement learning control, digital twin, and extension to Ladakh and island microgrids.
