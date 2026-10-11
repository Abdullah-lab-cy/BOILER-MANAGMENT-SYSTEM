import json
import joblib
import pandas as pd
import numpy as np
import paho.mqtt.client as mqtt
import sys
import os
import requests
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Deep Learning specific imports can go here if needed
try:
    from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_CONTROL_TOPIC, FIREBASE_DB_URL, DL_FAULT_MODEL_PATH, DL_SCALER_PATH
    from database import save_reading
except ImportError:
    from .config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_CONTROL_TOPIC, FIREBASE_DB_URL, DL_FAULT_MODEL_PATH, DL_SCALER_PATH
    from .database import save_reading

from alerts.notifier import notify_user
from preprocessing.clean_data import clean_sensor_data
from preprocessing.feature_engineering import engineer_features

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker for AI Live Monitoring (Code {rc})")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        
        # ESP32 JSON Keys Mapping
        mapped_data = {
            'temp_inlet': payload.get('inlet_temperature_C', 30.0),
            'temp_outlet': payload.get('outlet_temperature_C', 100.0),
            'pressure_steam': payload.get('steam_pressure_bar', 2.0),
            'temp_flue': payload.get('flue_gas_temperature_C', 100.0),
            'water_level': payload.get('water_level_percent', 100.0),
            'fuel_flow': payload.get('fuel_flow_m3_hr', 2.0),
            'water_flow': 10.0,
            'gas_smoke': 100.0,
            'flame_status': 1
        }
        
        # For DL monitoring, we still apply the safety threshold overrides
        fault_label = "Normal"
        p_steam = mapped_data['pressure_steam']
        t_outlet = mapped_data['temp_outlet']
        w_level = mapped_data['water_level']
        
        # Override Thresholds
        if p_steam >= 15.0 or t_outlet >= 280.0 or w_level <= 20.0:
            fault_label = "High Pressure"
            print(f"[AI CRITICAL] Pressure: {p_steam}, Temp: {t_outlet}")
        elif p_steam >= 10.0 or t_outlet >= 200.0 or w_level <= 40.0:
            fault_label = "Warning"
            print(f"[AI WARNING] Pressure: {p_steam}, Temp: {t_outlet}")
            
        # Send Control Commands based on AI / Threshold logic
        if fault_label != "Normal":
            command = "SHUTDOWN" if fault_label == "High Pressure" else "REDUCE_LOAD"
            client.publish(MQTT_CONTROL_TOPIC, command)
            print(f"[AI ALERT] Sent command to ESP32: {command}")
        else:
            client.publish(MQTT_CONTROL_TOPIC, "NORMAL")

        # Optional: Sync Deep Learning results to Firebase
        try:
            fb_url = FIREBASE_DB_URL.rstrip('/')
            firebase_payload = {
                "pressure_steam": p_steam,
                "temp_outlet": t_outlet,
                "fault_status": fault_label,
                "source": "DeepLearningMonitor",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            requests.put(f"{fb_url}/dl_live_telemetry.json", json=firebase_payload, timeout=2)
        except Exception as fe:
            print(f"[FIREBASE WARNING] DL Sync failed: {fe}")

    except Exception as e:
        print(f"[ERROR] AI Monitor processing error: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

if __name__ == "__main__":
    print("\n[SYSTEM] AI Deep Learning Monitoring Starting...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()