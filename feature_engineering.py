import pandas as pd
import numpy as np

def engineer_features(data):
    """
    Creates meaningful engineered features from raw sensor data.
    Input can be a dict (single row) or a DataFrame.
    Returns the same type with new features added.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
        return_dict = True
    else:
        df = data.copy()
        return_dict = False

    # ---- Feature 1: Temperature Delta (Outlet - Inlet) ----
    # Higher delta = more heat transfer = better performance
    df['temp_delta'] = df['temp_outlet'] - df['temp_inlet']

    # ---- Feature 2: Flue Loss Indicator ----
    # Higher flue temp relative to outlet = more heat lost to exhaust
    df['flue_loss_ratio'] = df['temp_flue'] / (df['temp_outlet'] + 1e-6)

    # ---- Feature 3: Steam-to-Fuel Ratio ----
    # Higher ratio = more efficient use of fuel
    df['steam_fuel_ratio'] = df['water_flow'] / (df['fuel_flow'] + 1e-6)

    # ---- Feature 4: Pressure-Temperature Index ----
    # Abnormal combinations indicate potential faults
    df['pressure_temp_index'] = df['pressure_steam'] * df['temp_outlet'] / 1000.0

    # ---- Feature 5: Water Level Risk Score ----
    # Lower water + higher pressure = higher risk
    df['water_risk_score'] = (100 - df['water_level']) * df['pressure_steam'] / 100.0

    # ---- Feature 6: Combustion Quality ----
    # High smoke with flame = poor combustion; high smoke without flame = danger
    df['combustion_quality'] = df['gas_smoke'] / (df['fuel_flow'] + 1e-6)

    # Round all new features
    new_cols = ['temp_delta', 'flue_loss_ratio', 'steam_fuel_ratio', 
                'pressure_temp_index', 'water_risk_score', 'combustion_quality']
    for col in new_cols:
        df[col] = df[col].round(4)

    if return_dict:
        return df.iloc[0].to_dict()
    return df


if __name__ == "__main__":
    test_data = {
        'temp_inlet': 30, 'temp_outlet': 185, 'temp_flue': 175,
        'pressure_steam': 7.5, 'water_level': 72,
        'fuel_flow': 2.8, 'water_flow': 11,
        'gas_smoke': 130, 'flame_status': 1
    }
    
    result = engineer_features(test_data)
    print("Engineered Features:")
    for k, v in result.items():
        print(f"  {k:25s}: {v}")
