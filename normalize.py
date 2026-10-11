import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def normalize_features(df, scaler=None, fit=True, save_path=None):
    """
    Normalize (StandardScaler) numeric sensor features.
    
    Args:
        df: DataFrame with sensor features
        scaler: Pre-fitted scaler (for inference). If None, creates new one.
        fit: If True, fit the scaler on the data. If False, only transform.
        save_path: If provided, saves the fitted scaler to this path.
    
    Returns:
        scaled_df: DataFrame with scaled values
        scaler: The fitted scaler object
    """
    # Columns to scale (all numeric sensor + engineered features)
    feature_cols = [col for col in df.columns 
                    if col not in ['efficiency', 'fault_label', 'fault_text']]
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        scaled_values = scaler.fit_transform(df[feature_cols])
    else:
        scaled_values = scaler.transform(df[feature_cols])
    
    scaled_df = pd.DataFrame(scaled_values, columns=feature_cols, index=df.index)
    
    # Save scaler if path provided
    if save_path and fit:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"  Scaler saved to: {save_path}")
    
    return scaled_df, scaler


if __name__ == "__main__":
    # Quick test
    test_df = pd.DataFrame({
        'temp_inlet': [30, 35, 25],
        'temp_outlet': [180, 250, 320],
        'pressure_steam': [7, 12, 16],
    })
    
    scaled, sc = normalize_features(test_df)
    print("Original:\n", test_df)
    print("\nScaled:\n", scaled)
