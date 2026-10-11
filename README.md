# 🏭 Industrial Boiler Automation System

A robust IoT and Machine Learning system for monitoring and optimizing industrial boilers using ESP32, Python, and Predictive Modeling.

## 🚀 Features
- **Real-time Monitoring:** 9 sensors (Temperature, Pressure, Flow, Gas, Flame, Water Level)
- **AI Intelligence:** Efficiency Prediction (R²=0.99) & Fault Detection (100% Accuracy)
- **Alert System:** Telegram notifications for critical faults
- **Web Dashboard:** Interactive Streamlit data visualization
- **Simulation Mode:** Full system works without real hardware

## 📂 Project Structure
```
Boiler Automation System/
├── firmware/               # ESP32 C code (sensors, MQTT, state machine)
│   ├── main.c              # Entry point - WiFi, MQTT publish loop
│   ├── sensors.c/h         # 9-sensor reading with fault simulation
│   ├── state_machine.c     # Boiler state management
│   ├── wifi_mqtt.c         # WiFi + MQTT communication
│   ├── buzzer.c            # Local alarm buzzer driver
│   ├── display.c           # LCD/OLED display driver
│   └── encoder.c           # Rotary encoder input
│
├── server/                 # Python Backend
│   ├── mqtt_server.py      # MQTT listener → Clean → Predict → Alert → Save
│   ├── app.py              # Flask REST API (/api/status, /api/history)
│   ├── database.py         # SQLite database operations
│   └── config.py           # All configuration (MQTT, DB, Telegram, Models)
│
├── data/                   # Data Pipeline
│   ├── generate_realistic_data.py  # Realistic boiler data simulator
│   ├── raw_data.csv                # 2000 simulated readings
│   └── training_dataset.csv        # Cleaned training data
│
├── preprocessing/          # Data Preprocessing Pipeline
│   ├── clean_data.py       # Outlier clamping, missing value handling
│   ├── feature_engineering.py  # 6 derived features (temp_delta, ratios, etc.)
│   └── normalize.py        # StandardScaler normalization
│
├── ml_models/              # Machine Learning
│   ├── train_models.py     # Full training pipeline with evaluation metrics
│   ├── efficiency_model.pkl    # Trained RandomForestRegressor
│   ├── fault_model.pkl         # Trained RandomForestClassifier
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── feature_columns.txt    # Feature order for inference
│
├── alerts/                 # Notification System
│   ├── telegram_bot.py     # Telegram Bot API integration
│   └── notifier.py         # Alert routing (local + Telegram)
│
├── dashboard/              # Web Frontend
│   └── app_frontend.py     # Streamlit real-time dashboard
│
├── database/               # Database
│   ├── schema.sql          # Table definitions
│   └── boiler_data.db      # SQLite database file
│
├── testing/                # Test Suite (18 tests)
│   ├── sensor_test.py      # Data generation validation
│   ├── ml_test.py          # Model loading & prediction tests
│   └── integration_test.py # Full pipeline end-to-end tests
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🧠 AI/ML Pipeline

### Data Flow
```
ESP32 Sensors → MQTT → Python Server → Clean → Features → Scale → Predict → Alert + Save → Dashboard
```

### Models
| Model | Type | Algorithm | Metric | Score |
|-------|------|-----------|--------|-------|
| Efficiency Prediction | Regression | RandomForestRegressor | R² Score | 0.9937 |
| Fault Detection | Classification | RandomForestClassifier | Accuracy | 100% |

### Training Data
- **2000 samples** generated with realistic boiler physics
- **3 scenarios:** Normal (60%), Warning (25%), Critical (15%)
- **Physics-based** efficiency calculation (heat balance method)
- **Rule-based** fault labeling (pressure, temperature, water level thresholds)

### Engineered Features
| Feature | Formula | Purpose |
|---------|---------|---------|
| temp_delta | outlet - inlet | Heat transfer indicator |
| flue_loss_ratio | flue_temp / outlet_temp | Exhaust heat loss |
| steam_fuel_ratio | water_flow / fuel_flow | Fuel efficiency proxy |
| pressure_temp_index | pressure × temp / 1000 | Abnormal combo detector |
| water_risk_score | (100-level) × pressure / 100 | Safety risk metric |
| combustion_quality | smoke / fuel_flow | Combustion efficiency |

## 🛠 Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Training Data
```bash
python data/generate_realistic_data.py
```

### 3. Train AI Models
```bash
python ml_models/train_models.py
```

### 4. Run Tests (18 tests)
```bash
python -m unittest testing.integration_test testing.ml_test testing.sensor_test -v
```

### 5. Start Server
```bash
cd server
python database.py        # Initialize DB
python mqtt_server.py     # Start MQTT listener
python app.py             # Start REST API (port 5000)
```

### 6. Launch Dashboard
```bash
streamlit run dashboard/app_frontend.py
```

### 7. (Optional) Flash ESP32
Flash the `firmware/` folder to your ESP32 using PlatformIO or Arduino IDE.

## 📡 9 Sensors

| # | Sensor | Type | Unit | Pin |
|---|--------|------|------|-----|
| 1 | Inlet Temperature | DS18B20 | °C | GPIO 4 |
| 2 | Outlet Temperature | Thermocouple K | °C | GPIO 5 |
| 3 | Flue Gas Temperature | Thermocouple K | °C | GPIO 18 |
| 4 | Steam Pressure | Pressure Transducer | bar | GPIO 34 |
| 5 | Water Level | Ultrasonic/Float | % | GPIO 35 |
| 6 | Fuel Flow | Flow Meter | m³/hr | GPIO 32 |
| 7 | Water Flow | Flow Meter | L/min | GPIO 33 |
| 8 | Gas/Smoke (CO/NOx) | MQ-2/MQ-135 | ppm | GPIO 36 |
| 9 | Flame Status | UV/IR Sensor | 0/1 | GPIO 39 |

## 🔔 Alert System
- **Normal (0):** No alerts — system operating safely
- **Warning (1):** Console log + local notification
- **Critical (2):** Telegram message + console alert + buzzer

## 📝 License
Academic Project — 6th Semester Embedded Systems
