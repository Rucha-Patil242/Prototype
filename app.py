"""
app.py - the dashboard (your working prototype).
Run with:  streamlit run app.py
"""
import streamlit as st
import plotly.graph_objects as go
import engine as E

st.set_page_config(page_title="Polar Energy Manager", page_icon="🧊", layout="wide")
st.title("🧊 AI Energy Management System for Polar Stations")
st.caption("Load forecasting + renewable integration + fuel optimization  |  "
           "Demo runs on simulated polar data (plug-in ready for real station data)")

# ------------------------------------------------------------ Sidebar controls
st.sidebar.header("⚙️ Scenario")
scenario = st.sidebar.selectbox("Polar condition", list(E.SCENARIOS.keys()))
solar_kw = st.sidebar.slider("Solar capacity (kW)", 0, 1500, 600, 50)
wind_kw = st.sidebar.slider("Wind capacity (kW)", 0, 1500, 400, 50)
batt_kwh = st.sidebar.slider("Battery size (kWh)", 500, 6000, 2000, 250)
uploaded = st.sidebar.file_uploader("Upload real data (CSV, optional)", type="csv")
st.sidebar.info("Change the scenario and watch fuel savings change live.")

# ------------------------------------------------------------------- Compute
@st.cache_data
def run(scenario, solar_kw, wind_kw, batt_kwh, csv_bytes=None):
    import io
    if csv_bytes is not None:
        df = E.load_real_data(io.BytesIO(csv_bytes), solar_kw, wind_kw)
    else:
        df = E.make_data(scenario, solar_kw=solar_kw, wind_kw=wind_kw)
    fc, metrics = E.forecast(df)
    base_fuel, base_p = E.dispatch_baseline(df)
    opt = E.dispatch_optimized(df, batt_kwh)
    return df, fc, metrics, base_fuel, opt

try:
    df, fc, metrics, base_fuel, opt = run(
        scenario, solar_kw, wind_kw, batt_kwh,
        uploaded.getvalue() if uploaded else None)
except Exception as err:
    st.error(f"Could not use the uploaded file: {err}")
    st.stop()
if uploaded:
    st.success(f"Using uploaded data: {len(df)} hours ({len(df)//24} days), "
               f"{df.index[0].date()} to {df.index[-1].date()}")
k = E.kpis(df, base_fuel, opt)

# ---------------------------------------------------------------------- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("⛽ Fuel saved", f"{k['saved_pct']:.1f}%", f"{k['saved_l']:,.0f} litres saved")
c2.metric("🌬️ Renewable share", f"{k['ren_share']:.1f}%")
c3.metric("🌍 CO₂ avoided", f"{k['co2_t']:.1f} tonnes")
c4.metric("🔮 Load forecast error", f"{metrics['load_mape']:.1f}% MAPE",
          f"MAE {metrics['load_mae']:.0f} kW", delta_color="off")

if k["unmet_kwh"] > 0:
    st.warning(f"⚠️ Generators can't cover the load in this scenario: {k['unmet_kwh']:,.0f} kWh unmet. "
               "Add more capacity/storage.")

WINDOW = 72  # show last 72 hours in charts
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "⚡ Power mix & battery", "⛽ Baseline vs AI"])

# --------------------------------------------------------------- Tab 1: Forecast
with tab1:
    f = fc.iloc[-WINDOW:]
    fig = go.Figure()
    fig.add_scatter(x=f.index, y=f.load_actual, name="Load (actual)", line=dict(color="#1f77b4"))
    fig.add_scatter(x=f.index, y=f.load_pred, name="Load (AI forecast)", line=dict(color="#1f77b4", dash="dash"))
    fig.add_scatter(x=f.index, y=f.ren_actual, name="Renewables (actual)", line=dict(color="#2ca02c"))
    fig.add_scatter(x=f.index, y=f.ren_pred, name="Renewables (AI forecast)", line=dict(color="#2ca02c", dash="dash"))
    fig.update_layout(title="AI forecast vs actual (last 72 h, unseen test data)",
                      yaxis_title="Power (kW)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Load:** MAE {metrics['load_mae']:.1f} kW, RMSE {metrics['load_rmse']:.1f} kW  |  "
             f"**Renewables:** MAE {metrics['ren_mae']:.1f} kW, RMSE {metrics['ren_rmse']:.1f} kW")

# ------------------------------------------------------------ Tab 2: Power mix
with tab2:
    o = opt.iloc[-WINDOW:]
    fig = go.Figure()
    for col, name, color in [("solar_used", "Solar", "#f2c94c"), ("wind_used", "Wind", "#56ccf2"),
                             ("batt_dis", "Battery", "#27ae60"), ("diesel_load", "Diesel", "#eb5757")]:
        fig.add_scatter(x=o.index, y=o[col], name=name, stackgroup="one", line=dict(width=0.5), fillcolor=color)
    fig.add_scatter(x=o.index, y=df["load"].iloc[-WINDOW:], name="Load", line=dict(color="black", dash="dot"))
    fig.update_layout(title="Optimized power mix (last 72 h)", yaxis_title="Power (kW)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_scatter(x=o.index, y=o.soc, name="Battery SOC %", fill="tozeroy", line=dict(color="#27ae60"))
    fig2.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Min reserve 20%")
    fig2.update_layout(title="Battery state of charge (capacity reduced by cold)",
                       yaxis_title="SOC (%)", yaxis_range=[0, 100], height=300)
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------- Tab 3: Baseline vs AI
with tab3:
    fig = go.Figure(go.Bar(x=["Diesel-only (baseline)", "AI-optimized"],
                           y=[k["base_l"], k["opt_l"]],
                           marker_color=["#eb5757", "#27ae60"],
                           text=[f"{k['base_l']:,.0f} L", f"{k['opt_l']:,.0f} L"], textposition="auto"))
    fig.update_layout(title="Diesel consumed over the whole period", yaxis_title="Litres", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"AI dispatch saves **{k['saved_l']:,.0f} litres** ({k['saved_pct']:.1f}%) "
               f"and avoids **{k['co2_t']:.1f} t CO₂** in this scenario.")

with st.expander("ℹ️ How it works"):
    st.markdown("""
1. **Data**: hourly temperature, wind, sun, and load (simulated; replace with real SCADA/weather data).
2. **Forecast**: Gradient Boosting model predicts load and renewable output from weather + recent history.
3. **Optimize**: each hour, use renewables first, then battery (capacity reduced in cold), then diesel.
   Generators never run below 30% load (inefficient).
4. **Compare**: against a diesel-only baseline to show fuel and CO₂ savings.
""")
