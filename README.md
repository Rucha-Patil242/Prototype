# Polar Energy Manager

AI-based energy management system for polar research stations.

## Problem
Polar stations depend on diesel that is costly and difficult to ship.
Solar and wind are variable, and extreme cold increases heating load
and reduces battery capacity.

## Our solution
- Forecasts load and renewable generation with Gradient Boosting (AI)
- Dispatches solar, wind, battery and diesel to minimise fuel use
- Derates battery capacity at low temperature
- Compares against a diesel-only baseline (fuel saved, renewable share, CO2 avoided)
- Polar scenarios: summer, polar night, blizzard, extreme cold
- Accepts real NASA POWER weather data or station CSV data

## Live demo
<paste your streamlit app link>

## How to run locally
pip install -r requirements.txt
streamlit run app.py

## Files
- app.py: dashboard
- engine.py: data, forecasting and optimisation logic
- requirements.txt: libraries

## Data note
Demo uses simulated data and real NASA POWER weather for Bharati station.
Station load and generation are modelled. The system is ready for real SCADA data.

## Team
<team name>: <member names>
SIH Problem Statement ID: <id>
