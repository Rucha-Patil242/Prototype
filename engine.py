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

    load_feats = ["hour", "temp", "wind_speed", "load_lag24"]
    ren_feats = ["hour", "temp", "wind_speed", "sun"]

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
           ["solar_used", "wind_used", "batt_dis", "batt_chg", "diesel_load", "diesel", "soc", "unmet", "curtailed"]}

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
            res["batt_chg"][i] = ch
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
                give = min(dis, max(0.0, d - rem)) if dis > 0 else 0.0   # spare diesel power replaces battery output
                if give > 0:
                    soc += give / eff
                    dis -= give
                    rem += give
                    res["batt_dis"][i] = dis
                res["diesel"][i] = d
                res["diesel_load"][i] = min(rem, d)
                res["unmet"][i] = max(0.0, rem - d)
                extra = d - rem  # spare diesel power tops up battery
                if extra > 0:
                    room = max(0.0, cap * soc_max - soc)
                    ch = min(extra, max_p, room / eff)
                    soc += ch * eff
                    res["batt_chg"][i] = ch
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


def load_real_data(file, solar_kw=600, wind_kw=400):
    """Accepts EITHER a NASA POWER hourly CSV OR our own CSV
    (timestamp,temp,wind_speed,sun,solar,wind,load)."""
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    start = next((k for k, l in enumerate(lines) if l.startswith("YEAR")), None)

    if start is not None:                       # ---- NASA POWER format
        import io
        d = pd.read_csv(io.StringIO("\n".join(lines[start:])))
        d = d.replace(-999, np.nan).replace(-999.0, np.nan)
        d.index = pd.to_datetime(dict(year=d.YEAR, month=d.MO, day=d.DY, hour=d.HR))
        df = pd.DataFrame(index=d.index)
        df["temp"] = d["T2M"].interpolate().bfill().ffill()
        df["wind_speed"] = d["WS10M"].interpolate().bfill().ffill()
        irr = d["ALLSKY_SFC_SW_DWN"] if "ALLSKY_SFC_SW_DWN" in d else 0
        df["sun"] = (irr / 1000.0).fillna(0).clip(0, 1)   # missing sun -> 0
        # modelled station values from the REAL weather
        rng = np.random.default_rng(1)
        cf = np.clip((df["wind_speed"] - 3) / 9, 0, 1) ** 3
        cf[df["wind_speed"] > 25] = 0
        h = df.index.hour.values
        df["solar"] = solar_kw * df["sun"]
        df["wind"] = wind_kw * cf
        base = 450 * (1 + 0.08 * np.sin((h - 9) / 24 * 2 * np.pi))
        df["load"] = base + 9.0 * np.clip(10 - df["temp"], 0, None) + rng.normal(0, 25, len(df))
    else:                                       # ---- our own CSV format
        import io
        df = pd.read_csv(io.StringIO(text), parse_dates=["timestamp"], index_col="timestamp")
        df = df.sort_index().resample("h").mean().interpolate()

    df["hour"] = df.index.hour
    if len(df) < 120:
        raise ValueError(f"Only {len(df)} hourly rows found. Need at least 120 (5 days); "
                         "download 2-4 weeks of data.")
    return df


# ------------------------------------------------ LIVE WEATHER (Open-Meteo)
STATIONS = {
    "Bharati (Antarctica)": (-69.4, 76.2),
    "Maitri (Antarctica)": (-70.8, 11.7),
}


def parse_open_meteo(js, solar_kw=600, wind_kw=400):
    """Turn Open-Meteo JSON into the table the rest of the app uses.
    Weather is REAL; solar, wind power and load are modelled from it."""
    h = js["hourly"]
    idx = pd.to_datetime(h["time"])
    temp = pd.Series(h["temperature_2m"], index=idx, dtype="float").interpolate().bfill().ffill()
    ws = pd.Series(h["wind_speed_10m"], index=idx, dtype="float").interpolate().bfill().ffill()
    sw = pd.Series(h["shortwave_radiation"], index=idx, dtype="float").fillna(0)

    df = pd.DataFrame(index=idx)
    df["temp"] = temp
    df["wind_speed"] = ws
    df["sun"] = (sw / 1000.0).clip(0, 1)
    cf = np.clip((ws - 3) / 9, 0, 1) ** 3
    cf[ws > 25] = 0
    hour = idx.hour.values
    rng = np.random.default_rng(1)
    df["solar"] = solar_kw * df["sun"]
    df["wind"] = wind_kw * cf
    base = 450 * (1 + 0.08 * np.sin((hour - 9) / 24 * 2 * np.pi))
    df["load"] = base + 9.0 * np.clip(10 - temp, 0, None) + rng.normal(0, 25, len(df))
    df["hour"] = hour
    if len(df) < 120:
        raise ValueError("Live weather returned too little data.")
    return df


def fetch_live_weather(lat, lon, solar_kw=600, wind_kw=400, past_days=14, forecast_days=2):
    """Downloads recent + forecast hourly weather (free, no API key)."""
    import json
    import urllib.request
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           "&hourly=temperature_2m,wind_speed_10m,shortwave_radiation"
           f"&past_days={past_days}&forecast_days={forecast_days}"
           "&wind_speed_unit=ms&timezone=UTC")
    with urllib.request.urlopen(url, timeout=20) as r:
        js = json.loads(r.read().decode("utf-8"))
    return parse_open_meteo(js, solar_kw, wind_kw)


# ------------------------------------------------ RECOMMENDATION (what to use NOW)
def mode_label(o):
    """Short text: which sources are supplying the load in this hour."""
    parts = []
    if o["solar_used"] > 5: parts.append("Solar")
    if o["wind_used"] > 5: parts.append("Wind")
    if o["batt_dis"] > 5: parts.append("Battery")
    if o["diesel_load"] > 5: parts.append("Diesel")
    return " + ".join(parts) if parts else "-"


def recommendation(d, o):
    """d = weather/load row, o = optimizer row. Returns (headline, list of reasons)."""
    mix = mode_label(o)
    diesel_txt = f"Diesel ON at {o['diesel']:.0f} kW" if o["diesel"] > 0 else "Diesel OFF"
    headline = f"Use: {mix}  |  {diesel_txt}"

    r = []
    if d["sun"] < 0.02:
        r.append("No usable sunlight, so solar is about 0 kW.")
    else:
        r.append(f"Sunlight available, so solar supplies {o['solar_used']:.0f} kW.")
    ws = d["wind_speed"]
    if ws < 3:
        r.append(f"Wind is only {ws:.1f} m/s (below the 3 m/s start-up speed), so wind is about 0 kW.")
    elif ws > 25:
        r.append(f"Wind is {ws:.1f} m/s (above 25 m/s), so turbines shut down for safety.")
    else:
        r.append(f"Wind is {ws:.1f} m/s, so wind supplies {o['wind_used']:.0f} kW.")
    if d["temp"] < -15:
        r.append(f"Very cold ({d['temp']:.0f} C): heating demand is high and battery capacity "
                 f"is reduced to about {float(battery_derate(d['temp']))*100:.0f}%.")
    if o["batt_dis"] > 5:
        r.append(f"Battery is discharging {o['batt_dis']:.0f} kW (charge level {o['soc']:.0f}%).")
    elif o["batt_chg"] > 5:
        r.append(f"Battery is charging {o['batt_chg']:.0f} kW from spare power (charge level {o['soc']:.0f}%).")
    else:
        r.append(f"Battery is idle (charge level {o['soc']:.0f}%).")
    if o["diesel"] > 0 and o["batt_chg"] > 5 and o.get("need_kwh", 0) > 0:
        r.append(f"Forecast shows about {o['need_kwh']:.0f} kWh of shortfall in the next 12 h, so diesel "
                 "runs harder now to fill the battery and can then switch OFF and coast.")
    if o["diesel"] > 0:
        r.append(f"Renewables and battery cannot cover the load ({d['load']:.0f} kW), so diesel "
                 f"runs at {o['diesel']:.0f} kW (never below 30% of rating).")
    else:
        r.append("Renewables and battery cover the whole load, so diesel stays OFF and saves fuel.")
    return headline, r


# ------------------------------------------ FORECAST-DRIVEN DISPATCH (AI + optimizer)
def forecast_all(df, folds=4):
    """Out-of-sample forecast for EVERY hour (blocked cross-validation).
    Each block is predicted by a model that never saw that block."""
    d = df.copy()
    d["ren"] = d["solar"] + d["wind"]
    d["load_lag24"] = d["load"].shift(24).bfill()
    lf = ["hour", "temp", "wind_speed", "load_lag24"]
    rf = ["hour", "temp", "wind_speed", "sun"]
    n = len(d)
    edges = np.linspace(0, n, folds + 1).astype(int)
    load_pred, ren_pred = np.zeros(n), np.zeros(n)
    for f in range(folds):
        a, b = edges[f], edges[f + 1]
        tr = np.r_[0:a, b:n]
        ml = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
        mr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
        ml.fit(d.iloc[tr][lf], d.iloc[tr]["load"])
        mr.fit(d.iloc[tr][rf], d.iloc[tr]["ren"])
        load_pred[a:b] = ml.predict(d.iloc[a:b][lf])
        ren_pred[a:b] = np.clip(mr.predict(d.iloc[a:b][rf]), 0, None)
    return pd.DataFrame({"load_pred": load_pred, "ren_pred": ren_pred}, index=df.index)


def dispatch_forecast_driven(df, fc, battery_kwh=2000, horizon=12):
    """Look-ahead dispatch. Each hour the controller reads the forecast for the next
    `horizon` hours, works out how much battery energy will be needed, and when diesel
    MUST run it runs harder (efficient) to fill the battery, then switches off and coasts."""
    n = len(df)
    load, solar, wind = df["load"].values, df["solar"].values, df["wind"].values
    ren = solar + wind
    net_f = fc["load_pred"].values - fc["ren_pred"].values   # forecast deficit (kW)
    cap_eff = battery_kwh * battery_derate(df["temp"].values)
    eff, soc_min, soc_max = 0.92, 0.20, 0.95
    min_load = DIESEL_MIN_FRAC * DIESEL_RATED_KW

    soc = 0.6 * battery_kwh
    res = {k: np.zeros(n) for k in ["solar_used", "wind_used", "batt_dis", "batt_chg",
                                    "diesel_load", "diesel", "soc", "unmet", "curtailed", "need_kwh"]}
    for i in range(n):
        cap = cap_eff[i]
        max_p = 0.5 * cap
        soc = min(soc, cap * soc_max)
        net = load[i] - ren[i]

        # --- look-ahead: peak cumulative forecast deficit over the next hours
        fut = net_f[i + 1: i + 1 + horizon]
        need = max(0.0, float(np.cumsum(fut).max())) if len(fut) else 0.0
        res["need_kwh"][i] = need

        if net <= 0:                                     # surplus renewables -> charge
            surplus = -net
            room = max(0.0, cap * soc_max - soc)
            ch = min(surplus, max_p, room / eff)
            soc += ch * eff
            res["batt_chg"][i] = ch
            res["curtailed"][i] = surplus - ch
            share = load[i] / ren[i] if ren[i] > 0 else 0
            res["solar_used"][i] = solar[i] * share
            res["wind_used"][i] = wind[i] * share
        else:                                            # deficit
            res["solar_used"][i] = solar[i]
            res["wind_used"][i] = wind[i]
            avail = max(0.0, soc - cap * soc_min)
            dis = min(net, max_p, avail * eff)
            soc -= dis / eff
            res["batt_dis"][i] = dis
            rem = net - dis
            if rem > 0:                                  # diesel has to run
                # Only pre-charge if the battery could later carry the load by itself
                # (otherwise diesel runs anyway and charging just wastes energy in losses)
                mean_def = float(np.mean(np.maximum(fut, 0))) if len(fut) else 0.0
                can_bridge = mean_def <= 0.9 * max_p
                target = min(cap * soc_max, cap * soc_min + 1.1 * need / eff)
                want_ch = min(max_p, max(0.0, (target - soc) / eff)) if can_bridge else 0.0
                if want_ch > 0 and dis > 0:      # never discharge and charge in the same hour
                    soc += dis / eff
                    res["batt_dis"][i] = 0.0
                    rem, dis = net, 0.0
                    want_ch = min(max_p, max(0.0, (target - soc) / eff))
                d = min(max(rem + want_ch, min_load), DIESEL_RATED_KW)
                give = min(dis, max(0.0, d - rem)) if dis > 0 else 0.0   # spare diesel power replaces battery output
                if give > 0:
                    soc += give / eff
                    dis -= give
                    rem += give
                    res["batt_dis"][i] = dis
                res["diesel"][i] = d
                res["diesel_load"][i] = min(rem, d)
                res["unmet"][i] = max(0.0, rem - d)
                extra = d - rem
                if extra > 0:
                    room = max(0.0, cap * soc_max - soc)
                    ch = min(extra, max_p, room / eff)
                    soc += ch * eff
                    res["batt_chg"][i] = ch
        res["soc"][i] = soc / cap * 100

    out = pd.DataFrame(res, index=df.index)
    out["fuel_l"] = fuel_litres(out["diesel"].values)
    return out
