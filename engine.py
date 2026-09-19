"""
engine.py - the "brain" of the Polar Energy Manager.
Three parts:
  1. make_data()   -> creates realistic synthetic polar weather + load data
  2. forecast()    -> AI (Gradient Boosting) predicts load and renewable power
  3. dispatch()    -> decides how much solar / wind / battery / diesel to use each hour
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

SCENARIOS = {
    # name: (average temperature C, description)
    "Polar summer (24h daylight)": -5,
    "Polar night (no sun)": -25,
    "Blizzard (low sun, extreme wind)": -20,
    "Extreme cold snap": -40,
}


# ---------------------------------------------------------------- 1. DATA
def make_data(scenario, days=30, solar_kw=600, wind_kw=400, seed=42):
    """Hourly synthetic data. Every number here can be replaced by real station data later."""
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    hour = idx.hour.values
    hours = np.arange(n)

    # Temperature: scenario average + daily swing + noise
    base_temp = SCENARIOS[scenario]
    temp = base_temp + 4 * np.sin((hour - 15) / 24 * 2 * np.pi) + rng.normal(0, 2, n)

    # Wind speed (m/s): slow weather cycles + noise
    wind_speed = 8 + 3 * np.sin(hours / 50) + rng.normal(0, 2, n)
    if scenario.startswith("Blizzard"):
        wind_speed = wind_speed + 12  # storm winds
    wind_speed = np.clip(wind_speed, 0, None)

    # Sun (0..1)
    day_curve = np.clip(np.sin((hour - 6) / 24 * 2 * np.pi), 0, 1)
    if scenario.startswith("Polar summer"):
        sun = 0.45 + 0.4 * np.sin((hour - 6) / 24 * 2 * np.pi)  # sun never sets
    elif scenario.startswith("Polar night"):
        sun = np.zeros(n)
    elif scenario.startswith("Blizzard"):
        sun = 0.1 * day_curve  # snow clouds block sun
    else:
        sun = 0.5 * day_curve
    sun = np.clip(sun * rng.uniform(0.7, 1.0, n), 0, 1)

    # Power from renewables
    solar = solar_kw * sun
    cf = np.clip((wind_speed - 3) / (12 - 3), 0, 1) ** 3  # simple turbine curve
    cf[wind_speed > 25] = 0  # turbine shuts down in storm (cut-out)
    wind = wind_kw * cf

    # Load = base + heating that grows as it gets colder
    base = 450 * (1 + 0.08 * np.sin((hour - 9) / 24 * 2 * np.pi))
    heating = 9.0 * np.clip(10 - temp, 0, None)
    load = base + heating + rng.normal(0, 25, n)

    return pd.DataFrame(
        {"temp": temp, "wind_speed": wind_speed, "sun": sun,
         "solar": solar, "wind": wind, "load": load, "hour": hour},
        index=idx,
    )


# ---------------------------------------------------------- 2. FORECASTING
def forecast(df):
    """Train on first 80% of data, predict last 20%. Returns predictions + accuracy."""
    d = df.copy()
    d["ren"] = d["solar"] + d["wind"]
    d["load_lag1"] = d["load"].shift(1)
    d["load_lag24"] = d["load"].shift(24)
    d["ren_lag1"] = d["ren"].shift(1)
    d = d.dropna()

    split = int(len(d) * 0.8)
    train, test = d.iloc[:split], d.iloc[split:]

    load_feats = ["hour", "temp", "wind_speed", "load_lag1", "load_lag24"]
    ren_feats = ["hour", "temp", "wind_speed", "sun", "ren_lag1"]

    m_load = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0)
    m_ren = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0)
    m_load.fit(train[load_feats], train["load"])
    m_ren.fit(train[ren_feats], train["ren"])

    out = pd.DataFrame(index=test.index)
    out["load_actual"] = test["load"]
    out["load_pred"] = m_load.predict(test[load_feats])
    out["ren_actual"] = test["ren"]
    out["ren_pred"] = np.clip(m_ren.predict(test[ren_feats]), 0, None)

    def rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    metrics = {
        "load_mae": mean_absolute_error(out.load_actual, out.load_pred),
        "load_rmse": rmse(out.load_actual, out.load_pred),
        "load_mape": float(np.mean(np.abs(out.load_actual - out.load_pred) / out.load_actual) * 100),
        "ren_mae": mean_absolute_error(out.ren_actual, out.ren_pred),
        "ren_rmse": rmse(out.ren_actual, out.ren_pred),
    }
    return out, metrics


# ---------------------------------------------------------- 3. DISPATCH
DIESEL_RATED_KW = 1200
DIESEL_MIN_FRAC = 0.30      # generators are inefficient below 30% load
CO2_KG_PER_LITRE = 2.68


def fuel_litres(p_kw):
    """Diesel genset fuel curve: fixed cost of running + cost per kW."""
    return np.where(p_kw > 0, 0.084 * DIESEL_RATED_KW + 0.246 * p_kw, 0.0)


def battery_derate(temp):
    """Batteries lose usable capacity in the cold (1.0 at 0C, ~0.68 at -40C)."""
    return np.clip(1 + 0.008 * temp, 0.5, 1.0)


def dispatch_baseline(df):
    """Old way: diesel generators supply everything. Renewables are ignored."""
    p = np.clip(df["load"].values, 0, DIESEL_RATED_KW)
    return fuel_litres(p), p


def dispatch_optimized(df, battery_kwh=2000):
    """Smart way: renewables first -> battery -> diesel (at minimum load if it must run)."""
    n = len(df)
    load, solar, wind = df["load"].values, df["solar"].values, df["wind"].values
    ren = solar + wind
    cap_eff = battery_kwh * battery_derate(df["temp"].values)
    eff = 0.92
    soc_min, soc_max = 0.20, 0.95

    soc = 0.6 * battery_kwh
    res = {k: np.zeros(n) for k in
           ["solar_used", "wind_used", "batt_dis", "diesel_load", "diesel", "soc", "unmet", "curtailed"]}

    for i in range(n):
        cap = cap_eff[i]
        max_p = 0.5 * cap                       # battery power limit
        soc = min(soc, cap * soc_max)           # cold shrinks capacity
        net = load[i] - ren[i]

        if net <= 0:  # more renewables than needed -> charge battery
            surplus = -net
            room = max(0.0, cap * soc_max - soc)
            ch = min(surplus, max_p, room / eff)
            soc += ch * eff
            res["curtailed"][i] = surplus - ch
            used = load[i]
            share = used / ren[i] if ren[i] > 0 else 0
            res["solar_used"][i] = solar[i] * share
            res["wind_used"][i] = wind[i] * share
        else:         # deficit -> battery first, then diesel
            res["solar_used"][i] = solar[i]
            res["wind_used"][i] = wind[i]
            avail = max(0.0, soc - cap * soc_min)
            dis = min(net, max_p, avail * eff)
            soc -= dis / eff
            res["batt_dis"][i] = dis
            rem = net - dis
            if rem > 0:
                d = min(max(rem, DIESEL_MIN_FRAC * DIESEL_RATED_KW), DIESEL_RATED_KW)
                res["diesel"][i] = d
                res["diesel_load"][i] = min(rem, d)
                res["unmet"][i] = max(0.0, rem - d)
                extra = d - rem  # spare diesel power tops up battery
                if extra > 0:
                    room = max(0.0, cap * soc_max - soc)
                    ch = min(extra, max_p, room / eff)
                    soc += ch * eff
        res["soc"][i] = soc / cap * 100

    out = pd.DataFrame(res, index=df.index)
    out["fuel_l"] = fuel_litres(out["diesel"].values)
    return out


def kpis(df, base_fuel, opt):
    total_load = df["load"].sum()
    base_l, opt_l = float(base_fuel.sum()), float(opt["fuel_l"].sum())
    return {
        "base_l": base_l,
        "opt_l": opt_l,
        "saved_l": base_l - opt_l,
        "saved_pct": (base_l - opt_l) / base_l * 100 if base_l > 0 else 0,
        "ren_share": max(0.0, (1 - opt["diesel_load"].sum() / total_load) * 100),
        "co2_t": (base_l - opt_l) * CO2_KG_PER_LITRE / 1000,
        "unmet_kwh": float(opt["unmet"].sum()),
    }

def load_real_data(file):
    df = pd.read_csv(file, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index().resample("h").mean().interpolate()
    df["hour"] = df.index.hour
    return df
