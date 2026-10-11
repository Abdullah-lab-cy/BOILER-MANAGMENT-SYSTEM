import pandas as pd

def clean_sensor_data(raw_data):
    """
    Cleans incoming sensor data (dict or DataFrame row).
    Handles missing values, outliers, and invalid readings.
    """
    if isinstance(raw_data, dict):
        df = pd.DataFrame([raw_data])
    else:
        df = pd.DataFrame([raw_data.to_dict()])
    
    # ---- Handle Missing Values ----
    # Fill missing numeric values with column defaults (safe operating values)
    defaults = {
        'temp_inlet': 30.0,
        'temp_outlet': 180.0,
        'temp_flue': 170.0,
        'pressure_steam': 7.0,
        'water_level': 70.0,
        'fuel_flow': 2.5,
        'water_flow': 10.0,
        'gas_smoke': 100.0,
        'flame_status': 1
    }
    
    for col, default_val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)
    
    # ---- Outlier Clamping (Physical Limits) ----
    # Values cannot go below physical minimums
    clamp_rules = {
        'temp_inlet':     (0, 100),      # °C - water inlet can't be negative or boiling
        'temp_outlet':    (50, 500),      # °C - steam temperature limits
        'temp_flue':      (50, 600),      # °C - flue gas limits
        'pressure_steam': (0, 25),        # bar - physical pressure limits
        'water_level':    (0, 100),       # % - percentage bounds
        'fuel_flow':      (0, 10),        # m³/hr - flow can't be negative
        'water_flow':     (0, 30),        # L/min - flow can't be negative
        'gas_smoke':      (0, 1000),      # ppm - concentration bounds
        'flame_status':   (0, 1),         # binary
    }
    
    for col, (low, high) in clamp_rules.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
    
    # Force flame_status to integer
    if 'flame_status' in df.columns:
        df['flame_status'] = df['flame_status'].astype(int)
    
    return df.iloc[0].to_dict()


def clean_dataframe(df):
    """
    Clean an entire DataFrame of sensor readings using vectorized operations.
    """
    df = df.copy()
    
    # ---- Handle Missing Values ----
    defaults = {
        'temp_inlet': 30.0, 'temp_outlet': 180.0, 'temp_flue': 170.0,
        'pressure_steam': 7.0, 'water_level': 70.0, 'fuel_flow': 2.5,
        'water_flow': 10.0, 'gas_smoke': 100.0, 'flame_status': 1
    }
    for col, default_val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)
    
    # ---- Outlier Clamping ----
    clamp_rules = {
        'temp_inlet': (0, 100), 'temp_outlet': (50, 500), 'temp_flue': (50, 600),
        'pressure_steam': (0, 25), 'water_level': (0, 100), 'fuel_flow': (0, 10),
        'water_flow': (0, 30), 'gas_smoke': (0, 1000), 'flame_status': (0, 1)
    }
    for col, (low, high) in clamp_rules.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
    
    # Force flame_status to integer
    if 'flame_status' in df.columns:
        df['flame_status'] = df['flame_status'].astype(int)
    
    return df


if __name__ == "__main__":
    # Test with sample data including outliers and missing values
    test_data = {
        "temp_inlet": 25, 
        "temp_outlet": 180,
        "temp_flue": 170,
        "pressure_steam": -5,  # Invalid negative pressure
        "water_level": 110,    # Over 100%
        "fuel_flow": None,     # Missing value
        "water_flow": 10,
        "gas_smoke": 120,
        "flame_status": 1
    }
    
    cleaned = clean_sensor_data(test_data)
    print("Original:", test_data)
    print("Cleaned: ", cleaned)
    print("\nNotice: pressure clamped to 0, water_level clamped to 100, fuel_flow filled with default 2.5")
