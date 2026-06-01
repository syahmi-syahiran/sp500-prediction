import datetime
import json
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import httpx

# Adjust path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Streamlit App Configurations
st.set_page_config(
    page_title="S&P 500 MLOps Hub",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Glassmorphic Styling & Font family
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;900&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif;
        background: radial-gradient(circle at 50% 50%, #0c112b 0%, #030514 100%) !important;
        color: #f1f5f9;
    }
    
    /* Hide default streamlit background decoration */
    [data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #ff416c, #ff4b2b) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 35, 0.6) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom Premium Glassmorphic Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 18px !important;
        padding: 24px !important;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.45) !important;
        margin-bottom: 20px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 50px 0 rgba(0, 242, 254, 0.12) !important;
        border-color: rgba(0, 242, 254, 0.25) !important;
    }
    
    .glow-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 242, 254, 0.15);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 900;
        margin-top: 10px;
        letter-spacing: -1px;
    }
    
    .glow-cyan {
        color: #00f2fe;
        text-shadow: 0 0 12px rgba(0, 242, 254, 0.5);
    }
    
    .glow-pink {
        color: #ff416c;
        text-shadow: 0 0 12px rgba(ff, 65, 108, 0.5);
    }
    
    .glow-green {
        color: #00ffd2;
        text-shadow: 0 0 12px rgba(0, 255, 210, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Helper to fetch DB data
def load_db_data():
    conn = config.get_db_connection()
    try:
        df_prices = pd.read_sql_query("SELECT * FROM raw_prices ORDER BY date DESC LIMIT 60", conn)
        df_preds = pd.read_sql_query("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 60", conn)
        df_actuals = pd.read_sql_query("SELECT * FROM actuals ORDER BY date DESC LIMIT 60", conn)
        return df_prices, df_preds, df_actuals
    except Exception as e:
        st.error(f"Error loading database records: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    finally:
        conn.close()

# Layout
st.markdown("<h1 class='glow-header'>🔮 S&P 500 MLOps Control Hub</h1>", unsafe_allow_html=True)
st.write("Production-grade price prediction, real-time feature drift analysis, and model monitoring.")
st.markdown("<br>", unsafe_allow_html=True)

# Load data
df_prices, df_preds, df_actuals = load_db_data()

# Sidebar Setup info
with st.sidebar:
    st.markdown("### 🧬 Systems health")
    
    # Simple check on FastAPI
    api_online = False
    try:
        resp = httpx.get(config.API_URL, timeout=2.0)
        if resp.status_code == 200:
            api_online = True
    except:
        pass
        
    if api_online:
        st.success("🟢 FastAPI: Serving")
    else:
        st.error("🔴 FastAPI: Offline / Sleeping")
        
    st.info(f"💾 Database: {config.DB_TYPE.upper()}")
    
    # Load model info
    model_ver = "None"
    if api_online:
        try:
            model_ver = resp.json().get("model_version", "v1.0")
        except:
            pass
    st.info(f"🤖 Model: {model_ver}")
    
    st.markdown("---")
    st.markdown("### 📊 Metrics Summary")
    if not df_actuals.empty:
        # Calculate MAE, Directional Accuracy
        mae = df_actuals['absolute_error'].mean()
        acc = df_actuals['direction_correct'].mean() * 100
        st.metric(label="30-day Rolling MAE", value=f"${mae:.2f}")
        st.metric(label="Directional Accuracy", value=f"{acc:.1f}%")
    else:
        st.write("No validation logs in DB yet.")

# Main Dashboard View Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Market Predictor", "📈 Performance Monitoring", "🧬 Feature Drift", "⚙️ MLOps Pipelines"])

# --- TAB 1: MARKET PREDICTOR ---
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🔮 Tomorrow's Price Forecast")
        
        # Fetch latest prediction
        latest_pred = None
        if not df_preds.empty:
            latest_pred = df_preds.iloc[0]
            
        if latest_pred is not None:
            pred_price = latest_pred['predicted_price']
            target_date = latest_pred['target_date']
            pred_date = latest_pred['prediction_date']
            
            # Fetch latest actual close to show direction
            latest_close = 0.0
            if not df_prices.empty:
                latest_close = df_prices.iloc[0]['close']
                
            direction = "🔺 UP" if pred_price > latest_close else "🔻 DOWN"
            glow_class = "glow-green" if pred_price > latest_close else "glow-pink"
            
            st.markdown(f"<p style='margin-bottom:0;'>Target Date: <strong>{target_date}</strong> (forecasted on {pred_date})</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value glow-cyan'>${pred_price:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<h4>Predicted Trend: <span class='{glow_class}'>{direction}</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<p>Latest Close Price: <strong>${latest_close:.2f}</strong></p>", unsafe_allow_html=True)
        else:
            st.write("No predictions generated yet. Run initial pipelines first!")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🎛️ Interactive Prediction Simulator")
        st.write("Override feature values below to simulate a real-time prediction model response:")
        
        # Load latest features as placeholder defaults
        latest_feats = {
            "sma_5": 510.0, "sma_20": 505.0, "sma_50": 498.0, "rsi_14": 55.0,
            "macd": 1.2, "macd_signal": 1.0, "bb_upper": 515.0, "bb_lower": 495.0,
            "volume_change_pct": 0.02, "vix": 14.5, "day_of_week": 2, "month": 6
        }
        
        # Load active defaults from latest row in SQLite if available
        conn = config.get_db_connection()
        try:
            df_feat = pd.read_sql_query("SELECT * FROM features ORDER BY date DESC LIMIT 1", conn)
            if not df_feat.empty:
                for col in latest_feats:
                    latest_feats[col] = float(df_feat.iloc[0][col])
        except:
            pass
        finally:
            conn.close()
            
        sim_sma5 = st.slider("SMA 5", float(latest_feats['sma_5']*0.8), float(latest_feats['sma_5']*1.2), float(latest_feats['sma_5']))
        sim_rsi = st.slider("RSI (14-day)", 10.0, 90.0, float(latest_feats['rsi_14']))
        sim_vix = st.slider("CBOE VIX Index", 9.0, 50.0, float(latest_feats['vix']))
        sim_vol = st.slider("Volume Change %", -0.5, 0.5, float(latest_feats['volume_change_pct']))
        
        if st.button("🔮 Simulate prediction"):
            payload = {
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "target_date": (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                "sma_5": sim_sma5,
                "sma_20": latest_feats['sma_20'],
                "sma_50": latest_feats['sma_50'],
                "rsi_14": sim_rsi,
                "macd": latest_feats['macd'],
                "macd_signal": latest_feats['macd_signal'],
                "bb_upper": latest_feats['bb_upper'],
                "bb_lower": latest_feats['bb_lower'],
                "volume_change_pct": sim_vol,
                "vix": sim_vix,
                "day_of_week": datetime.datetime.now().weekday(),
                "month": datetime.datetime.now().month
            }
            
            try:
                resp = httpx.post(f"{config.API_URL}/predict", json=payload, timeout=5.0)
                if resp.status_code == 200:
                    st.success(f"Simulated price prediction: **${resp.json()['predicted_price']:.2f}**")
                else:
                    st.error("FastAPI Prediction failed. Is FastAPI running locally?")
            except Exception as ex:
                st.error(f"Simulator error calling FastAPI: {ex}")
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: PERFORMANCE MONITORING ---
with tab2:
    if not df_actuals.empty:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Forecasted vs Actual price (30-day)")
        
        # Chronological sort
        df_actuals_sorted = df_actuals.iloc[::-1].copy()
        
        # Draw interactive dual line chart
        chart_data = df_actuals_sorted[['date', 'actual_price', 'predicted_price']].set_index('date')
        st.line_chart(chart_data)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Sub charts
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Rolling Absolute prediction error")
            st.bar_chart(df_actuals_sorted.set_index('date')['absolute_error'])
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### 🎯 Directional Accuracy timeline")
            # Draw correct vs incorrect direction indicator
            direction_chart = df_actuals_sorted[['date', 'direction_correct']].copy()
            direction_chart['direction_correct'] = direction_chart['direction_correct'].apply(lambda x: "CORRECT" if x == 1 else "FAIL")
            st.dataframe(direction_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No predictions evaluated yet. Run the daily orchestration pipeline to match predictions vs actual close prices!")

# --- TAB 3: FEATURE DRIFT ---
with tab3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🧬 Evidently AI Data Drift report")
    st.write("Below is the embedded live Feature and Target Drift Report generated by Evidently AI:")
    
    drift_path = "monitoring/drift_report.html"
    if os.path.exists(drift_path):
        with open(drift_path, "r", encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
    else:
        st.info("Evidently AI Drift Report has not been generated yet. It is generated automatically during the daily pipeline orchestration.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 4: MLOps PIPELINES ---
with tab4:
    col_x, col_y = st.columns([1, 1])
    
    with col_x:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Retrain Machine Learning Model")
        st.write("Kick off a complete model retraining flow. This will load the raw S&P 500 history, calculate engineering features, train a new XGBoost Regressor model, log metrics to MLflow, and automatically hot-swap the model in the API.")
        
        if st.button("🚀 Trigger Model Retraining"):
            with st.spinner("Retraining model in progress..."):
                try:
                    resp = httpx.post(f"{config.API_URL}/retrain", timeout=10.0)
                    if resp.status_code == 200:
                        st.success("Model retraining triggered successfully! Hot-reload complete.")
                        st.balloons()
                    else:
                        st.error("FastAPI endpoint retrain failed.")
                except Exception as ex:
                    st.error(f"Failed to communicate with FastAPI: {ex}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_y:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📅 Automated workflow scheduling")
        st.write("GitHub Actions daily cron runs are scheduled weekdays at **6:05 PM ET** (23:05 UTC) after market close:")
        
        st.code("""
on:
  schedule:
    - cron: '5 23 * * 1-5' # weekdays only
        """, language="yaml")
        st.write("This runs `pipelines/daily_pipeline.py` which ingests close prices, evaluates the yesterday forecast, computes feature drift, and monitors rolling errors.")
        st.markdown("</div>", unsafe_allow_html=True)
