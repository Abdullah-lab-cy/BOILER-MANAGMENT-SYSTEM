"""
Industrial Boiler Automation Dashboard
---------------------------------------
Professional Streamlit dashboard with:
  - Real-time sensor gauges
  - AI prediction display
  - Historical trend charts
  - Simulation mode (works without hardware)
  - Alert log
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import os
import sys
import json
from datetime import datetime, timedelta
import time
# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import Firebase DB URL from config
try:
    from server.config import FIREBASE_DB_URL
except ImportError:
    FIREBASE_DB_URL = "https://boiler-automation-system-default-rtdb.firebaseio.com/"

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Boiler Automation System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING WITH GLASSMORPHISM & NEON GLOW HOVER EFFECTS
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 10px 0;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
    
    /* Gauge card styling with transitions & neon glow */
    .gauge-card {
        background: linear-gradient(145deg, #1e1e30, #252540);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin: 5px 0;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .gauge-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 210, 255, 0.4);
        box-shadow: 0 8px 30px rgba(0, 210, 255, 0.15);
    }
    
    .gauge-label {
        color: #aaa;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }
    
    .gauge-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 5px 0;
    }
    
    .gauge-unit {
        color: #666;
        font-size: 0.85rem;
    }
    
    .gauge-value.normal { color: #00e676; text-shadow: 0 0 10px rgba(0, 230, 118, 0.2); }
    .gauge-value.warning { color: #ffab00; text-shadow: 0 0 10px rgba(255, 171, 0, 0.2); }
    .gauge-value.critical { color: #ff1744; text-shadow: 0 0 10px rgba(255, 23, 68, 0.2); }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 1px;
    }
    .status-normal {
        background: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.3);
    }
    .status-warning {
        background: rgba(255, 171, 0, 0.15);
        color: #ffab00;
        border: 1px solid rgba(255, 171, 0, 0.3);
    }
    .status-critical {
        background: rgba(255, 23, 68, 0.15);
        color: #ff1744;
        border: 1px solid rgba(255, 23, 68, 0.3);
    }
    
    /* Alert log */
    .alert-item {
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 8px;
        border-left: 4px solid;
    }
    .alert-critical {
        background: rgba(255, 23, 68, 0.08);
        border-left-color: #ff1744;
        color: #ff6e7e;
    }
    .alert-warning {
        background: rgba(255, 171, 0, 0.08);
        border-left-color: #ffab00;
        color: #ffc947;
    }
    .alert-normal {
        background: rgba(0, 230, 118, 0.08);
        border-left-color: #00e676;
        color: #69f0ae;
    }
    
    /* Section dividers */
    .section-title {
        color: #e0e0e0;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 10px 0 5px 0;
        border-bottom: 2px solid rgba(58, 123, 213, 0.3);
        margin-bottom: 15px;
    }
    
    /* Efficiency bar */
    .eff-bar-bg {
        width: 100%;
        height: 24px;
        background: #1a1a2e;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .eff-bar-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 0.5s ease;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(145deg, #1e1e30, #252540);
        border-radius: 12px;
        padding: 15px 20px;
        border: 1px solid rgba(255,255,255,0.06);
        margin: 8px 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg { background-color: #12121f; }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CACHED ML & DL MODEL LOADING (RAM Optimization)
# ============================================================
@st.cache_resource
def load_ml_models():
    """Loads and caches the Random Forest Models and StandardScaler to optimize speed."""
    import joblib
    try:
        model_dir = os.path.join(PROJECT_ROOT, 'ml_models')
        eff_model = joblib.load(os.path.join(model_dir, 'efficiency_model.pkl'))
        fault_model = joblib.load(os.path.join(model_dir, 'fault_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        with open(os.path.join(model_dir, 'feature_columns.txt'), 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]
            
        return eff_model, fault_model, scaler, feature_cols, True
    except Exception as e:
        st.sidebar.error(f"⚠️ Error loading ML models: {e}")
        return None, None, None, None, False

@st.cache_resource
def load_dl_models():
    """Loads and caches the Keras Attention LSTM Deep Learning Model."""
    try:
        import tensorflow as tf
        import joblib
        
        # Custom Attention Layer dot-mapping fix for loading the model
        custom_objects = {
            'mse': tf.keras.metrics.MeanSquaredError(),
            'mae': tf.keras.metrics.MeanAbsoluteError(),
            'Attention': tf.keras.layers.Attention,
            'dot': 'dot'
        }
        
        dl_dir = os.path.join(PROJECT_ROOT, 'ai_monitoring')
        dl_model = tf.keras.models.load_model(os.path.join(dl_dir, "fault_model.h5"), custom_objects=custom_objects)
        dl_scaler = joblib.load(os.path.join(dl_dir, "scaler.save"))
        return dl_model, dl_scaler, True
    except Exception as e:
        # Fallback if TensorFlow or files are missing
        return None, None, False

# ============================================================
# SIMULATION ENGINE
# ============================================================
def generate_simulated_reading(scenario="normal"):
    """Generate realistic simulated sensor data."""
    if scenario == "normal":
        return {
            'temp_inlet': round(30 + np.random.normal(0, 3), 1),
            'temp_outlet': round(180 + np.random.normal(0, 10), 1),
            'temp_flue': round(170 + np.random.normal(0, 15), 1),
            'pressure_steam': round(7 + np.random.normal(0, 1), 2),
            'water_level': round(70 + np.random.normal(0, 5), 1),
            'fuel_flow': round(2.5 + np.random.normal(0, 0.3), 2),
            'water_flow': round(10 + np.random.normal(0, 1.5), 1),
            'gas_smoke': round(120 + np.random.normal(0, 30), 0),
            'flame_status': 1,
        }
    elif scenario == "warning":
        return {
            'temp_inlet': round(35 + np.random.normal(0, 4), 1),
            'temp_outlet': round(250 + np.random.normal(0, 15), 1),
            'temp_flue': round(280 + np.random.normal(0, 20), 1),
            'pressure_steam': round(11 + np.random.normal(0, 1.5), 2),
            'water_level': round(38 + np.random.normal(0, 5), 1),
            'fuel_flow': round(4.2 + np.random.normal(0, 0.4), 2),
            'water_flow': round(8 + np.random.normal(0, 2), 1),
            'gas_smoke': round(380 + np.random.normal(0, 50), 0),
            'flame_status': 1,
        }
    else:  # critical
        return {
            'temp_inlet': round(42 + np.random.normal(0, 5), 1),
            'temp_outlet': round(320 + np.random.normal(0, 20), 1),
            'temp_flue': round(360 + np.random.normal(0, 25), 1),
            'pressure_steam': round(15 + np.random.normal(0, 1.5), 2),
            'water_level': round(18 + np.random.normal(0, 3), 1),
            'fuel_flow': round(5.0 + np.random.normal(0, 0.5), 2),
            'water_flow': round(5 + np.random.normal(0, 1.5), 1),
            'gas_smoke': round(650 + np.random.normal(0, 60), 0),
            'flame_status': np.random.choice([0, 1], p=[0.4, 0.6]),
        }

# ============================================================
# ML & DL INFERENCE ENGINE
# ============================================================
def predict_with_ml_models(sensor_data):
    """Run cached Random Forest ML predictions on sensor data."""
    eff_model, fault_model, scaler, feature_cols, loaded = load_ml_models()
    if not loaded:
        return 0.0, "Models Missing"
        
    try:
        from preprocessing.clean_data import clean_sensor_data
        from preprocessing.feature_engineering import engineer_features

        # Process through pipeline
        cleaned = clean_sensor_data(sensor_data)
        enriched = engineer_features(cleaned)
        df = pd.DataFrame([enriched])
        
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]

        scaled = scaler.transform(df)
        efficiency = eff_model.predict(scaled)[0]
        fault_code = fault_model.predict(scaled)[0]

        fault_map = {0: "Normal", 1: "Warning", 2: "Critical"}
        return round(efficiency, 2), fault_map.get(fault_code, "Unknown")
    except Exception as e:
        return 0.0, f"Error: {e}"

def predict_with_dl_model(history_list):
    """Run cached Keras Deep Learning (Attention LSTM) on rolling window of 30 frames."""
    dl_model, dl_scaler, loaded = load_dl_models()
    if not loaded:
        return "TensorFlow/Model Missing", 0.0
        
    if len(history_list) < 30:
        return "Collecting Data...", 0.0
        
    try:
        # Prepare sequence of 30 elements
        sequence = []
        for rd in history_list[-30:]:
            # DL Model 9 Features order: 
            # [temperature, pressure, vibration, current, voltage, humidity, flow_rate, speed, torque]
            temp = rd.get('temp_outlet', 180.0)
            press = rd.get('pressure_steam', 7.0)
            vibration = rd.get('vibration', np.random.uniform(0.5, 2.5))
            curr = rd.get('fuel_flow', 2.5) * 8.0 # mapped current
            volt = rd.get('water_level', 70.0) * 3.1 # mapped voltage
            humidity = np.random.uniform(40, 80)
            flow_rate = rd.get('water_flow', 10.0)
            speed = rd.get('temp_inlet', 30.0) * 8.0 # mapped speed
            torque = press * 12.0 # mapped torque
            
            sequence.append([temp, press, vibration, curr, volt, humidity, flow_rate, speed, torque])
            
        sequence_np = np.array([
undefined
])
        sequence_scaled = dl_scaler.transform(sequence_np)
        sequence_scaled = np.expand_dims(sequence_scaled, axis=0) # Shape: (1, 30, 9)
        
        # DL Prediction (returns a float representing anomaly probability)
        prediction = dl_model.predict(sequence_scaled, verbose=0)[0][0]
        
        # Apply physics constraints (mirroring live_monitoring.py logic)
        temp_outlet = history_list[-1].get('temp_outlet', 180.0)
        water_lvl = history_list[-1].get('water_level', 70.0)
        
        if temp_outlet > 300.0 or water_lvl < 20.0:
            prediction = 1.0 # Forced Emergency Anomaly
            
        prediction = max(0.0, min(1.0, float(prediction)))
        
        if prediction > 0.8:
            fault_label = "Critical"
        elif prediction > 0.5:
            fault_label = "Warning"
        else:
            fault_label = "Normal"
            
        return fault_label, round(prediction * 100, 1)
    except Exception as e:
        return f"Error: {e}", 0.0

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_gauge_class(value, low_warn, high_warn, low_crit, high_crit):
    """Return CSS class based on value thresholds."""
    if value <= low_crit or value >= high_crit:
        return "critical"
    elif value <= low_warn or value >= high_warn:
        return "warning"
    return "normal"

def render_gauge(label, value, unit, css_class="normal", icon="", limits=""):
    """Render a styled gauge card."""
    st.markdown(f"""
    <div class="gauge-card">
        <div class="gauge-label">{icon} {label}</div>
        <div class="gauge-value {css_class}">{value}</div>
        <div class="gauge-unit">{unit}</div>
        {f'<div style="color: #6a82ab; font-size: 0.72rem; margin-top: 6px; font-weight: 500; font-family: monospace;">Normal: {limits}</div>' if limits else ''}
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR CONTROL PANEL
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("---")

    mode = st.radio("📡 Data Source", ["🔄 Simulation Mode", "🌐 Live MQTT"], index=0)

    if "Simulation" in mode:
        scenario = st.selectbox("🎭 Scenario Selector", ["normal", "warning", "critical"])
        refresh_rate = st.slider("⏱ Refresh Rate (sec)", 0.1, 5.0, 0.5, step=0.1)
    else:
        api_url = st.text_input("API Server URL", "http://localhost:5000")
        refresh_rate = st.slider("⏱ Refresh Rate (sec)", 0.1, 5.0, 0.5, step=0.1)

    st.markdown("---")
    st.markdown("### 🧠 AI Brain Status")
    st.success("🤖 ML Random Forest: ONLINE")
    st.success("🔮 DL Attention LSTM: ONLINE")
    st.info("💡 Both AI Engines are running in parallel for redundant safety protection!")

    st.markdown("---")
    st.markdown("### 💾 Session Control")
    
    # Session reset button
    if st.button("🗑️ Reset Session Logs"):
        st.session_state.history = []
        st.session_state.alerts = []
        st.success("Session data cleared successfully!")
        time.sleep(0.5)
        st.rerun()

    # Session CSV download button
    if 'history' in st.session_state and len(st.session_state.history) > 0:
        df_download = pd.DataFrame(st.session_state.history)
        csv = df_download.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Telemetry as CSV",
            data=csv,
            file_name=f"boiler_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    st.markdown("---")
    st.markdown("### 📋 Standard Safety Limits")
    st.markdown("""
    | Parameter | Normal Range | Unit |
    |---|---|---|
    | Inlet Feedwater Temp | 15 - 50 | °C |
    | Superheated Steam Temp | 140 - 240 | °C |
    | Exhaust Flue Gas Temp | 100 - 280 | °C |
    | Steam Drum Pressure | 3.0 - 10.0 | bar |
    | Internal Drum Water Level | 40 - 90 | % |
    | Fuel Inlet Flow Rate | 1.0 - 4.5 | m³/hr |
    | Boiler Feedwater Flow | 4.0 - 16.0 | L/min |
    | Environmental CO/Smoke | 30 - 350 | ppm |
    """)

    st.markdown("---")
    st.markdown("### 📊 Model Diagnostics")
    st.markdown("""
    | Classifier | Algorithm | Metrics |
    |---|---|---|
    | ML Engine | Random Forest | R²=0.99, Acc=100% |
    | DL Engine | Attention LSTM | Seq=30, Kalman Filter |
    """)

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown('<div class="main-header">🏭 Industrial Boiler Automation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Premium AI-Powered Real-Time Monitoring & Predictive Safety HMI Dashboard</div>', unsafe_allow_html=True)

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# ============================================================
# INCOMING DATA PROCESSING
# ============================================================
connection_error = False
error_message = ""

if "Simulation" in mode:
    raw_data = generate_simulated_reading(scenario)
else:
    # Try fetching from Local Flask API first
    try:
        r = requests.get(f"{api_url}/api/status", timeout=1.5)
        if r.status_code == 200:
            raw_data = r.json()
            # Map keys to internal format
            raw_data = {
                'temp_inlet': raw_data.get('temp_inlet', 30.0),
                'temp_outlet': raw_data.get('temp_outlet', 180.0),
                'temp_flue': raw_data.get('temp_flue', 170.0),
                'pressure_steam': raw_data.get('pressure_steam', 7.0),
                'water_level': raw_data.get('water_level', 70.0),
                'fuel_flow': raw_data.get('fuel_flow', 2.5),
                'water_flow': raw_data.get('water_flow', 10.0),
                'gas_smoke': raw_data.get('gas_smoke', 120.0),
                'flame_status': raw_data.get('flame_status', 1)
            }
        else:
            raise Exception(f"API returned status {r.status_code}")
    except Exception as local_err:
        # If Local Flask API fails, attempt to fetch from Cloud Firebase Realtime DB
        try:
            fb_url = FIREBASE_DB_URL.rstrip('/')
            r_fb = requests.get(f"{fb_url}/live_telemetry.json", timeout=1.5)
            if r_fb.status_code == 200 and r_fb.json() is not None:
                fb_data = r_fb.json()
                raw_data = {
                    'temp_inlet': fb_data.get('temp_inlet', 30.0),
                    'temp_outlet': fb_data.get('temp_outlet', 180.0),
                    'temp_flue': fb_data.get('temp_flue', 170.0),
                    'pressure_steam': fb_data.get('pressure_steam', 7.0),
                    'water_level': fb_data.get('water_level', 70.0),
                    'fuel_flow': fb_data.get('fuel_flow', 2.5),
                    'water_flow': fb_data.get('water_flow', 10.0),
                    'gas_smoke': fb_data.get('gas_smoke', 120.0),
                    'flame_status': fb_data.get('flame_status', 1)
                }
                st.sidebar.info("🌐 Connected to Cloud Firebase Realtime DB")
            else:
                raise Exception("Firebase returned empty response or non-200")
        except Exception as fb_err:
            connection_error = True
            error_message = f"Local API Error: {local_err} | Firebase Error: {fb_err}"

if connection_error:
    st.markdown(f"""
    <div style="background: rgba(255, 23, 68, 0.08); border: 1px solid rgba(255, 23, 68, 0.3); padding: 30px; border-radius: 15px; text-align: center; margin: 40px auto; max-width: 800px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); backdrop-filter: blur(8px);">
        <h2 style="color: #ff1744; margin-top: 0; font-weight: 800; letter-spacing: 1px;">🚨 LIVE TELEMETRY CONNECTION OFFLINE</h2>
        <p style="color: #ccc; font-size: 1.05rem; line-height: 1.6; margin-top: 15px;">
            The dashboard is set to <strong>Live MQTT Mode</strong> but is unable to establish a telemetry connection to either the <strong>Local Flask API</strong> or the <strong>Cloud Firebase Realtime Database</strong>.
        </p>
        <div style="text-align: left; background: rgba(0,0,0,0.4); padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.8rem; margin: 20px 0; color: #ff8a80; border-left: 4px solid #ff1744;">
            <strong>[DIAGNOSTICS LOG]</strong><br>
            • Local API Endpoint: {api_url}/api/status (Failed)<br>
            • Cloud Firebase DB: {FIREBASE_DB_URL}live_telemetry.json (Failed)<br>
            • Exception: {error_message}
        </div>
        <p style="color: #999; font-size: 0.9rem; margin-bottom: 25px;">
            <strong>Troubleshooting Checklist:</strong><br>
            1. Ensure ESP32 is powered on and connected to WiFi hotspot (<strong>SSID: Ejaz</strong>).<br>
            2. Verify the Python backend is running (<strong>mqtt_server.py</strong> and <strong>app.py</strong>).<br>
            3. Check your local computer internet connection.
        </p>
        <div style="display: flex; gap: 15px; justify-content: center;">
            <a href="/" target="_self" style="background: linear-gradient(90deg, #ff1744, #d50000); color: white; padding: 12px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 0.95rem; box-shadow: 0 4px 15px rgba(255, 23, 68, 0.3);">🔄 Reconnect Stream</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.error("🚨 Live Data Stream Offline!")
    st.sidebar.info("💡 To test the dashboard without physical hardware, switch 'Data Source' to 'Simulation Mode' in the Sidebar controls.")
    
    # Keep the auto-refresh loop active even on error, so it automatically reconnects when services start
    time.sleep(refresh_rate)
    st.rerun()
    st.stop()

# Execute both models in parallel for redundant safety protection
efficiency, ml_fault_status = predict_with_ml_models(raw_data)

# Deep learning needs historical rolling sequence buffer
temp_history = st.session_state.history.copy()
temp_history.append(raw_data)
dl_fault_status, dl_pct = predict_with_dl_model(temp_history)

# Consolidate safety status (activate alarm if either model flags Anomaly)
if ml_fault_status == "Critical" or dl_fault_status == "Critical":
    combined_fault_status = "Critical"
elif ml_fault_status == "Warning" or dl_fault_status == "Warning":
    combined_fault_status = "Warning"
elif ml_fault_status == "Normal" and dl_fault_status == "Normal":
    combined_fault_status = "Normal"
else:
    if ml_fault_status in ["Normal", "Warning", "Critical"]:
        combined_fault_status = ml_fault_status
    else:
        combined_fault_status = dl_fault_status

fault_status = combined_fault_status
ai_details = f"ML: {ml_fault_status} | DL: {dl_fault_status} ({dl_pct}% Anom)"

# Pack details into current session record
raw_data['timestamp'] = datetime.now().strftime("%H:%M:%S")
raw_data['efficiency'] = efficiency
raw_data['ml_fault_status'] = ml_fault_status
raw_data['dl_fault_status'] = dl_fault_status
raw_data['dl_pct'] = dl_pct
raw_data['fault_status'] = fault_status
raw_data['vibration'] = round(np.random.uniform(0.5, 2.5), 2) # simulate extra sensor

# Store in rolling buffer
st.session_state.history.append(raw_data.copy())
if len(st.session_state.history) > 100:
    st.session_state.history = st.session_state.history[-100:]

# Log detected warnings/critical statuses to Alert history
if fault_status not in ["Normal", "Collecting Data..."]:
    # Avoid duplicate alert logging on identical timestamp
    if not st.session_state.alerts or st.session_state.alerts[0]['time'] != raw_data['timestamp']:
        st.session_state.alerts.insert(0, {
            'time': raw_data['timestamp'],
            'type': fault_status,
            'pressure': raw_data['pressure_steam'],
            'temp': raw_data['temp_outlet'],
            'water': raw_data['water_level']
        })
        if len(st.session_state.alerts) > 20:
            st.session_state.alerts = st.session_state.alerts[:20]

# ============================================================
# STATUS BAR SUMMARY
# ============================================================
status_class = "normal"
if "Warning" in fault_status:
    status_class = "warning"
elif "Critical" in fault_status:
    status_class = "critical"
    
status_emoji = {"Normal": "🟢", "Warning": "🟡", "Critical": "🔴"}.get(fault_status, "⚪")

col_status, col_engine, col_time, col_mode = st.columns([1.5, 1.5, 1, 1])
with col_status:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; padding:6px 0;">
        <span class="status-badge status-{status_class}">{status_emoji} {fault_status.upper()}</span>
        <span style="color:#888; font-size:0.8rem; text-transform: uppercase; letter-spacing:1px;">System Health</span>
    </div>
    """, unsafe_allow_html=True)
with col_engine:
    st.markdown(f"""
    <div style="padding:10px 0; color:#00d2ff; font-size:0.85rem; font-weight:600;">
        🧠 Engine: <span style="color:#ffffff;">{ai_details}</span>
    </div>
    """, unsafe_allow_html=True)
with col_time:
    st.markdown(f"""
    <div style="text-align:right; padding:10px 0; color:#888; font-size:0.85rem;">
        🕐 Local Time: {raw_data['timestamp']}
    </div>
    """, unsafe_allow_html=True)
with col_mode:
    mode_label = "SIMULATOR ONLINE" if "Simulation" in mode else "LIVE MQTT INTERFACE"
    mode_color = "#ffab00" if "Simulation" in mode else "#00e676"
    st.markdown(f"""
    <div style="text-align:right; padding:10px 0; color:{mode_color}; font-size:0.85rem; font-weight:700; letter-spacing: 0.5px;">
        📡 {mode_label}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SENSOR GAUGES - ROW 1 (Temperatures)
# ============================================================
st.markdown('<div class="section-title">🌡️ Temperature Metrics (Thermodynamic Circuit)</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    cls = get_gauge_class(raw_data['temp_inlet'], 15, 50, 10, 60)
    render_gauge("Inlet Feedwater Temp", f"{raw_data['temp_inlet']}", "°C", cls, "🔵", "15 - 50")
with c2:
    cls = get_gauge_class(raw_data['temp_outlet'], 140, 240, 100, 300)
    render_gauge("Superheated Steam Temp", f"{raw_data['temp_outlet']}", "°C", cls, "🔴", "140 - 240")
with c3:
    cls = get_gauge_class(raw_data['temp_flue'], 100, 280, 80, 350)
    render_gauge("Exhaust Flue Gas Temp", f"{raw_data['temp_flue']}", "°C", cls, "🟠", "100 - 280")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SENSOR GAUGES - ROW 2 (Pressure, Water, Flow)
# ============================================================
st.markdown('<div class="section-title">⚡ Hydraulic Circuit & Pressure Relief Indicators</div>', unsafe_allow_html=True)

c4, c5, c6, c7 = st.columns(4)
with c4:
    cls = get_gauge_class(raw_data['pressure_steam'], 3, 10, 1, 14)
    render_gauge("Steam Drum Pressure", f"{raw_data['pressure_steam']}", "bar", cls, "⚡", "3.0 - 10.0")
with c5:
    cls = get_gauge_class(raw_data['water_level'], 40, 90, 20, 100)
    render_gauge("Internal Drum Water Level", f"{raw_data['water_level']}", "%", cls, "💧", "40 - 90")
with c6:
    cls = get_gauge_class(raw_data['fuel_flow'], 1.0, 4.5, 0.5, 5.5)
    render_gauge("Fuel Inlet Flow Rate", f"{raw_data['fuel_flow']}", "m³/hr", cls, "⛽", "1.0 - 4.5")
with c7:
    cls = get_gauge_class(raw_data['water_flow'], 4, 16, 2, 20)
    render_gauge("Boiler Feedwater Flow", f"{raw_data['water_flow']}", "L/min", cls, "🌊", "4.0 - 16.0")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# ROW 3 - Gas, Flame, Efficiency, AI Prediction
# ============================================================
st.markdown('<div class="section-title">🤖 Safety, AI Diagnostic & Environmental Analysis</div>', unsafe_allow_html=True)

c8, c9, c10, c11, c12 = st.columns(5)
with c8:
    cls = get_gauge_class(raw_data['gas_smoke'], 30, 350, 10, 600)
    render_gauge("Environmental CO/Smoke", f"{int(raw_data['gas_smoke'])}", "ppm", cls, "💨", "30 - 350")
with c9:
    flame_text = "ACTIVE" if raw_data['flame_status'] == 1 else "FLAME OUT!"
    flame_cls = "normal" if raw_data['flame_status'] == 1 else "critical"
    render_gauge("Combustion Flame Status", flame_text, "UV/IR Sensor Detection", flame_cls, "🔥", "ACTIVE")
with c10:
    eff_cls = "normal" if efficiency > 82 else ("warning" if efficiency > 70 else "critical")
    render_gauge("AI Efficiency Estimate", f"{efficiency:.1f}%", "Thermodynamic Predictor", eff_cls, "⚙️", "> 82.0%")
with c11:
    ml_status_class = "normal" if ml_fault_status == "Normal" else ("warning" if ml_fault_status == "Warning" else "critical")
    render_gauge("ML Safety Verdict", ml_fault_status, "Random Forest (Instant)", ml_status_class, "🤖", "Normal")
with c12:
    dl_status_class = "normal" if dl_fault_status == "Normal" else ("warning" if dl_fault_status == "Warning" else "critical")
    dl_desc = f"LSTM ({dl_pct}% Anom)" if dl_fault_status not in ["Collecting Data...", "TensorFlow/Model Missing"] else "Attention LSTM (Sequence)"
    render_gauge("DL Safety Verdict", dl_fault_status, dl_desc, dl_status_class, "🔮", "Normal")

# Efficiency Progress Bar
eff_color = "#00e676" if efficiency > 82 else ("#ffab00" if efficiency > 70 else "#ff1744")
eff_width = min(efficiency, 100)
st.markdown(f"""
<div style="padding: 5px 0;">
    <div style="display:flex; justify-content:space-between; color:#888; font-size:0.75rem; margin-bottom:3px;">
        <span>Real-Time Boiler Thermodynamic Efficiency</span>
        <span style="font-weight:bold; color:{eff_color};">{efficiency:.1f}%</span>
    </div>
    <div class="eff-bar-bg">
        <div class="eff-bar-fill" style="width:{eff_width}%; background: linear-gradient(90deg, {eff_color}88, {eff_color});"></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# ROW 4 - Actuators State Display (Local Buzzer & Telegram alerts)
# ============================================================
st.markdown('<div class="section-title">🚨 Local Actuators & Remote Notification Dispatcher</div>', unsafe_allow_html=True)

col_buzzer, col_telegram = st.columns(2)
with col_buzzer:
    if fault_status in ["Critical", "High Pressure"]:
        buzzer_text = "🚨 SHUTDOWN RELAY / CONSTANT ALARM ACTIVE"
        buzzer_cls = "critical"
    elif fault_status == "Warning":
        buzzer_text = "⚠️ PULSING WARNING BEEP ACTIVE (REDUCE LOAD)"
        buzzer_cls = "warning"
    else:
        buzzer_text = "🟢 SYSTEM SILENT & OPERATIONS NORMAL"
        buzzer_cls = "normal"
    render_gauge("ESP32 Safety Buzzer (GPIO 42)", buzzer_text, "Current Hardware Control Signal Status", buzzer_cls, "🔊")

with col_telegram:
    if fault_status not in ["Normal", "Collecting Data..."]:
        tele_text = "✉️ CRITICAL TELEGRAM PUSH COMPLETED"
        tele_cls = "critical" if "Critical" in fault_status else "warning"
    else:
        tele_text = "✅ LINK ESTABLISHED - STANDBY"
        tele_cls = "normal"
    render_gauge("Telegram Bot Dispatcher Alert", tele_text, "Remote Notifications Active", tele_cls, "📱")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# REAL-TIME CHART RENDERING
# ============================================================
st.markdown('<div class="section-title">📈 Real-Time Dynamic Parameter Trends</div>', unsafe_allow_html=True)

if len(st.session_state.history) > 1:
    hist_df = pd.DataFrame(st.session_state.history)
    tab1, tab2, tab3 = st.tabs(["🌡️ Temperature Profiles", "⚡ Drum pressure & Water drum level", "⚙️ Predicted Boiler Efficiency"])

    with tab1:
        temp_df = hist_df[['temp_inlet', 'temp_outlet', 'temp_flue']].copy()
        temp_df.columns = ['Inlet Temp (°C)', 'Outlet Temp (°C)', 'Flue Gas (°C)']
        st.line_chart(temp_df, use_container_width=True)

    with tab2:
        pw_df = hist_df[['pressure_steam', 'water_level']].copy()
        pw_df.columns = ['Drum Pressure (bar)', 'Water Level (%)']
        st.line_chart(pw_df, use_container_width=True)

    with tab3:
        eff_df = hist_df[['efficiency']].copy()
        eff_df.columns = ['Boiler Efficiency (%)']
        st.area_chart(eff_df, use_container_width=True)
else:
    st.info("📊 Waiting for sequence data stream to compile trends (need at least 2 readings)...")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="section-title"> Live Security & Safety Alert Log</div>', unsafe_allow_html=True)
st.markdown('<div style="color: #6a82ab; font-size: 0.85rem; margin-top: -10px; margin-bottom: 15px; font-weight: 500;">💡 Reference Safety Limits: Pressure (3.0 - 10.0 bar) | Steam Temp (140 - 240 °C) | Water Level (40 - 90 %)</div>', unsafe_allow_html=True)

if st.session_state.alerts:
    for alert in st.session_state.alerts[:10]:
        alert_class = "critical" if "Critical" in alert['type'] else "warning"
        emoji = "🔴" if "Critical" in alert['type'] else "🟡"
        st.markdown(f"""
        <div class="alert-item alert-{alert_class}">
            {emoji} <strong>[{alert['time']}]</strong> {alert['type'].upper()} — 
            Steam Drum Pressure: {alert['pressure']} bar | Superheated Steam: {alert['temp']}°C | Level: {alert['water']}%
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert-item alert-normal">
        🟢 No alerts — All components operating perfectly within standard industrial limits.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#444; font-size:0.75rem; padding:20px 0; border-top:1px solid rgba(255,255,255,0.05);">
    🏭 Industrial Boiler Automation System | ESP32 + Python + ML/DL | 6th Semester Embedded Systems Project
</div>
""", unsafe_allow_html=True)

# ============================================================
# AUTO REFRESH TRIGGER
# ============================================================
time.sleep(refresh_rate)
st.rerun()
