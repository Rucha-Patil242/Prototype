"""
app.py - Polar Energy Manager dashboard.
Run with:  streamlit run app.py

Page layout (top to bottom):
  1. What this page is + 3 steps
  2. THE ANSWER: what the station should run at the chosen hour
  3. Impact: fuel saved, renewable share, CO2, forecast error
  4. Details in tabs: power mix, fuel comparison, forecast, 24 h plan
"""
import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import engine as E

st.set_page_config(page_title="Polar Energy Manager", page_icon="🧊", layout="wide")

COL = {"solar": "#F2C94C", "wind": "#56CCF2", "batt": "#27AE60", "diesel": "#EB5757", "load": "#0B2545"}
NAVY, MUTED, GREEN, ORANGE = "#0B2545", "#5B6B7B", "#1F8A5B", "#C4571A"

st.markdown("""<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.pem-title {font-size: 2.1rem; font-weight: 800; margin: 0; line-height: 1.2;}
.pem-sub {font-size: 1.08rem; opacity: .85; margin: .25rem 0 .7rem 0;}
.pem-steps {display: flex; flex-wrap: wrap; gap: 10px; margin: .2rem 0 .9rem 0;}
.pem-step {background: #DCEBF7; color: #0B2545; border-radius: 999px; padding: 6px 14px; font-weight: 600; font-size: .92rem;}
.pem-badge {display: inline-block; background: #F1F6FB; color: #0B2545; border: 1px solid #C9D6E3; border-radius: 8px; padding: 4px 10px; font-size: .88rem; margin-bottom: .6rem;}
.hero {border-radius: 16px; padding: 18px 24px; color: #fff; margin-bottom: 12px;}
.hero-tag {font-size: .85rem; letter-spacing: .06em; text-transform: uppercase; opacity: .92;}
.hero-main {font-size: 2.1rem; font-weight: 800; line-height: 1.2; margin: 4px 0;}
.hero-sub {font-size: 1.15rem; font-weight: 700;}
.hero-plain {font-size: 1.02rem; opacity: .96; margin-top: 6px;}
.tiles {display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;}
.tile {flex: 1 1 150px; background: #fff; border: 1px solid #C9D6E3; border-top-width: 6px; border-radius: 12px; padding: 10px 12px; text-align: center; color: #1B2A3A;}
.tile-name {font-size: .9rem; font-weight: 700; color: #5B6B7B;}
.tile-val {font-size: 1.6rem; font-weight: 800; color: #0B2545;}
.tile-status {font-size: .86rem; font-weight: 600;}
.flow-title {font-weight: 700; margin: 6px 0 4px 0;}
.flow {display: flex; height: 32px; border-radius: 8px; overflow: hidden; border: 1px solid #C9D6E3; background: #eee;}
.flow div {display: flex; align-items: center; justify-content: center; color: #0B2545; font-weight: 700; font-size: .82rem; white-space: nowrap; overflow: hidden;}
.flow-legend {font-size: .86rem; margin: 4px 0 12px 0;}
.dot {display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 4px 0 10px;}
.why {background: #F1F6FB; border-radius: 12px; padding: 10px 18px 4px 18px; color: #1B2A3A; margin-bottom: 14px;}
.why b {color: #0B2545;}
.kpi {background: #fff; border: 1px solid #C9D6E3; border-radius: 12px; padding: 12px 14px; color: #1B2A3A;}
.kpi-name {font-size: .9rem; font-weight: 700; color: #5B6B7B;}
.kpi-val {font-size: 1.9rem; font-weight: 800; color: #0B2545; line-height: 1.15;}
.kpi-cap {font-size: .86rem; color: #1F8A5B; font-weight: 600;}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ Sidebar
st.sidebar.markdown("## 🧊 Controls")
st.sidebar.markdown("**1 · Choose a situation**")
scenario = st.sidebar.selectbox(
    "Polar condition", list(E.SCENARIOS.keys()),
    help="Simulated weather for this situation. Ignored if you upload a file or use live weather.")
st.sidebar.markdown("**2 · Size the station**")
solar_kw = st.sidebar.slider("Solar capacity (kW)", 0, 1500, 600, 50, help="Peak power of the solar panels")
wind_kw = st.sidebar.slider("Wind capacity (kW)", 0, 1500, 400, 50, help="Rated power of the wind turbines")
batt_kwh = st.sidebar.slider("Battery size (kWh)", 500, 6000, 2000, 250, help="Energy the battery can store")
st.sidebar.markdown("**3 · Or use real weather**")
station = st.sidebar.selectbox("Station", list(E.STATIONS.keys()))
if st.sidebar.button("🌐 Fetch live weather"):
    try:
        lat, lon = E.STATIONS[station]
        st.session_state["live_df"] = E.fetch_live_weather(lat, lon, solar_kw, wind_kw)
        st.session_state["live_name"] = station
        st.session_state["live_time"] = datetime.datetime.now(datetime.timezone.utc).strftime("%d %b %H:%M UTC")
    except Exception as err:
        st.sidebar.error(f"Could not fetch live weather: {err}. Use the scenarios or a CSV instead.")
use_live = False
if "live_df" in st.session_state:
    use_live = st.sidebar.checkbox("Use the live weather data", value=True)
uploaded = st.sidebar.file_uploader("Upload weather/station CSV (optional)", type="csv",
                                    help="NASA POWER hourly CSV, or a CSV with: timestamp, temp, wind_speed, sun, solar, wind, load")
st.sidebar.caption("An uploaded file overrides live weather, which overrides the scenario.")


# ------------------------------------------------------------------ Compute
@st.cache_data
def get_df(scenario, solar_kw, wind_kw, csv_bytes=None):
    import io
    if csv_bytes is not None:
        return E.load_real_data(io.BytesIO(csv_bytes), solar_kw, wind_kw)
    return E.make_data(scenario, solar_kw=solar_kw, wind_kw=wind_kw)


@st.cache_data
def compute(df, batt_kwh):
    fc, metrics = E.forecast(df)                              # accuracy on unseen test data
    fc_all = E.forecast_all(df)                               # out-of-sample forecast for every hour
    base_fuel, base_p = E.dispatch_baseline(df)               # 1) diesel only
    opt_reactive = E.dispatch_optimized(df, batt_kwh)         # 2) reactive rules (no forecast)
    opt = E.dispatch_forecast_driven(df, fc_all, batt_kwh)    # 3) AI forecast-driven
    return fc, metrics, base_fuel, opt, opt_reactive


live_mode = bool(use_live and not uploaded)
try:
    if uploaded:
        df = get_df(scenario, solar_kw, wind_kw, uploaded.getvalue())
        source = f"📄 Uploaded file · {len(df) // 24} days ({df.index[0].date()} to {df.index[-1].date()})"
    elif use_live:
        df = st.session_state["live_df"]
        source = (f"🌐 Live weather · {st.session_state['live_name']} · fetched {st.session_state['live_time']} "
                  "(weather is real; solar, wind power and load are modelled from it)")
    else:
        df = get_df(scenario, solar_kw, wind_kw)
        source = f"🧪 Simulated scenario · {scenario}"
    fc, metrics, base_fuel, opt, opt_reactive = compute(df, batt_kwh)
except Exception as err:
    st.error(f"Could not process the data: {err}")
    st.stop()
k = E.kpis(df, base_fuel, opt)
days = max(1, len(df) // 24)

# ------------------------------------------------------------------ 1. Header
st.markdown('<p class="pem-title">🧊 Polar Energy Manager</p>'
            '<p class="pem-sub">Tells a polar station operator <b>which power source to run each hour</b> '
            'so that the least diesel is burned.</p>'
            '<div class="pem-steps"><span class="pem-step">① Pick a situation on the left</span>'
            '<span class="pem-step">② See what to run at this hour</span>'
            '<span class="pem-step">③ Check the fuel saved</span></div>'
            f'<div class="pem-badge">Data shown: {source}</div>', unsafe_allow_html=True)

if k["unmet_kwh"] > 0:
    st.warning(f"⚠️ In this situation the generators cannot cover the whole load ({k['unmet_kwh']:,.0f} kWh unmet). "
               "Add capacity or storage.")

# ------------------------------------------------------------------ 2. THE ANSWER
if live_mode:
    now_ts = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    pos = int(df.index.get_indexer([now_ts], method="nearest")[0])
else:
    pos = max(0, len(df) - 24)
lo, hi = max(0, pos - 24), min(len(df) - 1, pos + 47)

hero_box = st.container()
slider_box = st.container()
with slider_box:
    i = st.select_slider("🕒 Drag to see what the system recommends at another hour",
                         options=list(range(lo, hi + 1)), value=pos,
                         format_func=lambda j: df.index[j].strftime("%d %b, %H:%M") + " UTC")

d, o = df.iloc[i], opt.iloc[i]
diesel_on = bool(o["diesel"] > 0)
mix = E.mode_label(o)
tag = ("Current hour" if (live_mode and i == pos) else "Selected hour") + " · " + df.index[i].strftime("%d %b %Y, %H:%M") + " UTC"
if diesel_on:
    hero_color, sub = ORANGE, f"⛽ Diesel ON at {o['diesel']:.0f} kW"
    plain = f"The station needs {d['load']:.0f} kW at this hour. Best plan: {mix}."
    if o["batt_chg"] > 5:
        plain += " Diesel runs a little harder to charge the battery, so it can switch OFF later."
else:
    hero_color, sub = GREEN, "✅ Diesel OFF — saving fuel"
    plain = f"The station needs {d['load']:.0f} kW at this hour. Best plan: {mix}. No diesel needed."


def tile(name, val, status, color, status_color=MUTED):
    return (f'<div class="tile" style="border-top-color:{color}"><div class="tile-name">{name}</div>'
            f'<div class="tile-val">{val}</div><div class="tile-status" style="color:{status_color}">{status}</div></div>')


if d["sun"] < 0.02:
    solar_status = "🌑 no sunlight"
else:
    solar_status = "☀️ producing" if o["solar_used"] > 5 else "idle"
if d["wind_speed"] > 25:
    wind_status = "🌪️ shut down (storm)"
elif d["wind_speed"] < 3:
    wind_status = "too calm"
else:
    wind_status = "💨 producing" if o["wind_used"] > 5 else "idle"
net_b = o["batt_dis"] - o["batt_chg"]
if o["batt_dis"] > 5:
    batt_status = f"🔋 discharging · {o['soc']:.0f}% full"
elif o["batt_chg"] > 5:
    batt_status = f"⚡ charging · {o['soc']:.0f}% full"
else:
    batt_status = f"idle · {o['soc']:.0f}% full"

tiles = "".join([
    tile("☀️ Solar", f"{o['solar_used']:.0f} kW", solar_status, COL["solar"]),
    tile("💨 Wind", f"{o['wind_used']:.0f} kW", wind_status, COL["wind"]),
    tile("🔋 Battery", f"{net_b:+.0f} kW", batt_status, COL["batt"]),
    tile("⛽ Diesel", f"{o['diesel']:.0f} kW", "ON" if diesel_on else "OFF",
         COL["diesel"], ORANGE if diesel_on else GREEN),
    tile("🏠 Station demand", f"{d['load']:.0f} kW", "what the station needs", COL["load"]),
])

parts = [("Solar", o["solar_used"], COL["solar"]), ("Wind", o["wind_used"], COL["wind"]),
         ("Battery", o["batt_dis"], COL["batt"]), ("Diesel", o["diesel_load"], COL["diesel"])]
total = sum(p[1] for p in parts)
flow = ""
if total > 0:
    segs = ""
    for name, v, c in parts:
        if v > 0:
            pct = v / total * 100
            segs += f'<div style="width:{pct:.1f}%;background:{c}">{name + " " + format(pct, ".0f") + "%" if pct > 14 else ""}</div>'
    legend = "".join(f'<span class="dot" style="background:{c}"></span>{n}' for n, v, c in parts)
    flow = (f'<div class="flow-title">How the station\'s demand is met at this hour</div>'
            f'<div class="flow">{segs}</div><div class="flow-legend">{legend}</div>')

_, reasons = E.recommendation(d, o)
why = '<div class="why"><b>Why?</b><ul>' + "".join(f"<li>{r}</li>" for r in reasons) + "</ul></div>"

with hero_box:
    st.markdown(
        f'<div class="hero" style="background:{hero_color}"><div class="hero-tag">{tag}</div>'
        f'<div class="hero-main">RUN: {mix}</div><div class="hero-sub">{sub}</div>'
        f'<div class="hero-plain">{plain}</div></div>'
        f'<div class="tiles">{tiles}</div>{flow}{why}', unsafe_allow_html=True)

# ------------------------------------------------------------------ 3. Impact
st.markdown(f"### 📊 Impact over {days} days of data")
kpi_cards = [
    ("⛽ Diesel saved", f"{k['saved_pct']:.1f}%", f"vs diesel only · {k['saved_l']:,.0f} litres"),
    ("🌬️ Demand met without diesel", f"{k['ren_share']:.1f}%", "from solar, wind and battery"),
    ("🌍 CO₂ avoided", f"{k['co2_t']:.1f} t", "compared with diesel only"),
    ("🔮 Forecast error", f"{metrics['load_mape']:.1f}%", f"average, on unseen data (MAE {metrics['load_mae']:.0f} kW)"),
]
for col, (name, val, cap) in zip(st.columns(4), kpi_cards):
    col.markdown(f'<div class="kpi"><div class="kpi-name">{name}</div><div class="kpi-val">{val}</div>'
                 f'<div class="kpi-cap">{cap}</div></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ 4. Details
st.markdown("### 🔎 Details")
WINDOW = 72
tab_mix, tab_fuel, tab_fc, tab_plan = st.tabs(
    ["⚡ Power mix", "⛽ Fuel: AI vs alternatives", "🔮 Forecast accuracy", "🗓️ 24-hour plan"])

with tab_mix:
    st.caption("Each colour shows which source supplied the station over time. The dotted line is the station's demand. "
               "The vertical line marks the hour selected above." + (" In live mode the last 48 h are forecast." if live_mode else ""))
    oo = opt.iloc[-WINDOW:]
    fig = go.Figure()
    for col_name, name, color in [("solar_used", "Solar", COL["solar"]), ("wind_used", "Wind", COL["wind"]),
                                  ("batt_dis", "Battery", COL["batt"]), ("diesel_load", "Diesel", COL["diesel"])]:
        fig.add_scatter(x=oo.index, y=oo[col_name], name=name, stackgroup="one", line=dict(width=0.5), fillcolor=color)
    fig.add_scatter(x=oo.index, y=df["load"].iloc[-WINDOW:], name="Station demand", line=dict(color=COL["load"], dash="dot"))
    if df.index[i] in oo.index:
        ts = df.index[i].strftime("%Y-%m-%d %H:%M:%S")
        fig.add_shape(type="line", x0=ts, x1=ts, y0=0, y1=1, yref="paper", line=dict(color=NAVY, width=2, dash="dash"))
    fig.update_layout(title="Who supplies the station (last 72 h)", yaxis_title="Power (kW)", height=400,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_scatter(x=oo.index, y=oo.soc, name="Battery charge %", fill="tozeroy", line=dict(color=COL["batt"]))
    fig2.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Minimum reserve 20%")
    fig2.update_layout(title="Battery charge level (usable capacity shrinks in the cold)",
                       yaxis_title="Charge (%)", yaxis_range=[0, 100], height=300)
    st.plotly_chart(fig2, use_container_width=True)

with tab_fuel:
    st.caption("The same station, run three different ways. Shorter bar = less diesel burned.")
    reactive_l, ai_l = float(opt_reactive["fuel_l"].sum()), k["opt_l"]
    extra_pct = (reactive_l - ai_l) / reactive_l * 100 if reactive_l > 0 else 0
    fig = go.Figure(go.Bar(
        x=["1. Diesel only", "2. Simple rules (no forecast)", "3. AI forecast-driven"],
        y=[k["base_l"], reactive_l, ai_l], marker_color=["#eb5757", "#f2c94c", "#27ae60"],
        text=[f"{k['base_l']:,.0f} L", f"{reactive_l:,.0f} L", f"{ai_l:,.0f} L"], textposition="auto"))
    fig.update_layout(title=f"Diesel burned over {days} days", yaxis_title="Litres", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"AI forecast-driven dispatch saves **{k['saved_l']:,.0f} litres** ({k['saved_pct']:.1f}%) "
               f"vs diesel only and avoids **{k['co2_t']:.1f} t CO₂**.")
    st.info(f"The AI forecast alone is worth **{extra_pct:.1f}% less diesel** than simple rules ({reactive_l - ai_l:,.0f} litres).")
    st.caption("Forecasts for each hour come from a model that never saw that hour. Small differences in final "
               "battery charge between strategies are not adjusted for.")

with tab_fc:
    st.caption("Solid line = what actually happened. Dashed line = what the AI predicted (on data it had not seen).")
    f = fc.iloc[-WINDOW:]
    fig = go.Figure()
    fig.add_scatter(x=f.index, y=f.load_actual, name="Demand (actual)", line=dict(color="#1f77b4"))
    fig.add_scatter(x=f.index, y=f.load_pred, name="Demand (AI forecast)", line=dict(color="#1f77b4", dash="dash"))
    fig.add_scatter(x=f.index, y=f.ren_actual, name="Solar + wind (actual)", line=dict(color="#2ca02c"))
    fig.add_scatter(x=f.index, y=f.ren_pred, name="Solar + wind (AI forecast)", line=dict(color="#2ca02c", dash="dash"))
    fig.update_layout(title="AI forecast vs actual (last 72 h)", yaxis_title="Power (kW)", height=420,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Demand:** average error {metrics['load_mae']:.1f} kW ({metrics['load_mape']:.1f}%), RMSE {metrics['load_rmse']:.1f} kW  |  "
             f"**Solar + wind:** average error {metrics['ren_mae']:.1f} kW, RMSE {metrics['ren_rmse']:.1f} kW")

with tab_plan:
    st.caption("What to run in each of the next 24 hours (from the hour marked as current or selected in the live view).")
    end = min(len(df), pos + 24)
    sched = pd.DataFrame({
        "Time (UTC)": df.index[pos:end].strftime("%d %b %H:%M"),
        "Run": opt.iloc[pos:end].apply(E.mode_label, axis=1).values,
        "Diesel": ["ON" if v > 0 else "off" for v in opt["diesel"].iloc[pos:end]],
        "Demand (kW)": df["load"].iloc[pos:end].round(0).values,
        "Solar (kW)": opt["solar_used"].iloc[pos:end].round(0).values,
        "Wind (kW)": opt["wind_used"].iloc[pos:end].round(0).values,
        "Battery (kW, + = discharging)": (opt["batt_dis"] - opt["batt_chg"]).iloc[pos:end].round(0).values,
        "Diesel (kW)": opt["diesel"].iloc[pos:end].round(0).values,
        "Battery charge %": opt["soc"].iloc[pos:end].round(0).values,
        "Temp (C)": df["temp"].iloc[pos:end].round(1).values,
        "Wind (m/s)": df["wind_speed"].iloc[pos:end].round(1).values,
    })
    st.dataframe(sched, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download plan (CSV)", sched.to_csv(index=False), "dispatch_plan.csv", "text/csv")
    st.caption("Future hours use forecast weather; station demand is modelled from temperature. "
               "With real SCADA data, this plan would drive generator and battery setpoints.")

with st.expander("ℹ️ How it works"):
    st.markdown("""
1. **Data**: hourly temperature, wind, sunlight and station demand (simulated, NASA POWER, or live Open-Meteo weather).
2. **Forecast**: Gradient Boosting predicts demand and solar + wind output from the weather and yesterday's demand.
3. **Look-ahead dispatch**: every hour the controller reads the next 12 hours of forecast. It uses solar and wind first,
   then the battery (capacity reduced in the cold, 20% kept in reserve), then diesel. When diesel *must* run it runs
   harder, because generators are more efficient at high load, to fill the battery, then switches OFF.
4. **Compare**: diesel only vs simple rules vs AI forecast-driven, so the value of the forecast is measured.
""")
st.caption("Simulation prototype: solar output, wind output and demand are modelled from weather. "
           "Real station data can be uploaded as a CSV.")
