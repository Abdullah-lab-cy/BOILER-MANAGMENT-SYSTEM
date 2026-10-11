"""
Proper ML Training Pipeline for Boiler Automation System
---------------------------------------------------------
Uses realistic simulated data (from data/generate_realistic_data.py)
with proper preprocessing, train/test split, and evaluation metrics.

Models:
  1. Efficiency Prediction  → RandomForestRegressor
  2. Fault Detection        → RandomForestClassifier
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.feature_engineering import engineer_features
from preprocessing.clean_data import clean_dataframe


def load_and_prepare_data():
    """Load training data and apply preprocessing pipeline."""
    data_path = os.path.join(PROJECT_ROOT, 'data', 'training_dataset.csv')
    
    if not os.path.exists(data_path):
        print(f"   Training data not found!")
        print("   Run this first:  python data/generate_realistic_data.py")
        sys.exit(1)
    
    print("   Loading training data...")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Step 1: Clean the data
    print("\nStep 1: Cleaning data...")
    df = clean_dataframe(df)
    print(f"   Cleaned {len(df)} rows")
    
    # Step 2: Feature Engineering
    print("\nStep 2: Engineering features...")
    df = engineer_features(df)
    new_features = ['temp_delta', 'flue_loss_ratio', 'steam_fuel_ratio', 
                    'pressure_temp_index', 'water_risk_score', 'combustion_quality']
    print(f"   Added {len(new_features)} new features: {new_features}")
    
    return df


def train_efficiency_model(X_train, X_test, y_train, y_test, scaler):
    """Train and evaluate the Efficiency Prediction model."""
    print("\n" + "=" * 60)
    print("  MODEL 1: EFFICIENCY PREDICTION (Regression)")
    print("=" * 60)
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n  Evaluation Metrics:")
    print(f"     MSE  (Mean Squared Error):   {mse:.4f}")
    print(f"     RMSE (Root Mean Sq Error):    {rmse:.4f}")
    print(f"     MAE  (Mean Absolute Error):   {mae:.4f}")
    print(f"     R²   (R-Squared Score):       {r2:.4f}")
    
    if r2 > 0.8:
        print(f"\n  GOOD: R² = {r2:.4f}")
    elif r2 > 0.5:
        print(f"\n  FAIR: R² = {r2:.4f}")
    else:
        print(f"\n  POOR: R² = {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    print(f"\n  Top 5 Important Features:")
    for feat, imp in feature_importance.nlargest(5).items():
        print(f"     {feat:25s}: {imp:.4f}")
    
    return model


def train_fault_model(X_train, X_test, y_train, y_test, scaler):
    """Train and evaluate the Fault Detection model."""
    print("\n" + "=" * 60)
    print("  MODEL 2: FAULT DETECTION (Classification)")
    print("=" * 60)
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  Overall Accuracy: {accuracy*100:.2f}%")
    
    print(f"\n  Classification Report:")
    fault_names = {0: 'Normal', 1: 'Warning', 2: 'Critical'}
    target_names = [fault_names[i] for i in sorted(y_test.unique())]
    report = classification_report(y_test, y_pred, target_names=target_names)
    for line in report.split('\n'):
        print(f"     {line}")
    
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"     {'':15s} {'Pred Normal':>12s} {'Pred Warning':>12s} {'Pred Critical':>13s}")
    labels = ['Act Normal', 'Act Warning', 'Act Critical']
    for i, row in enumerate(cm):
        print(f"     {labels[i]:15s} {row[0]:12d} {row[1]:12d} {row[2]:13d}")
    
    if accuracy > 0.9:
        print(f"\n  EXCELLENT: {accuracy*100:.1f}% accuracy")
    elif accuracy > 0.75:
        print(f"\n  GOOD: {accuracy*100:.1f}% accuracy")
    else:
        print(f"\n  NEEDS WORK: {accuracy*100:.1f}% accuracy")
    
    # Feature importance
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    print(f"\n  Top 5 Important Features:")
    for feat, imp in feature_importance.nlargest(5).items():
        print(f"     {feat:25s}: {imp:.4f}")
    
    return model


def main():
    print("\n" + "*" * 60)
    print("  BOILER AUTOMATION - ML TRAINING PIPELINE")
    print("*" * 60)
    
    # ---- Load & Prepare Data ----
    df = load_and_prepare_data()
    
    # ---- Define Features & Targets ----
    feature_cols = ['temp_inlet', 'temp_outlet', 'temp_flue', 'pressure_steam',
                    'water_level', 'fuel_flow', 'water_flow', 'gas_smoke', 'flame_status',
                    'temp_delta', 'flue_loss_ratio', 'steam_fuel_ratio',
                    'pressure_temp_index', 'water_risk_score', 'combustion_quality']
    
    X = df[feature_cols]
    y_efficiency = df['efficiency']
    y_fault = df['fault_label']
    
    # ---- Train/Test Split (80/20) ----
    print("\nStep 3: Train/Test Split (80/20)...")
    X_train, X_test, y_eff_train, y_eff_test, y_fault_train, y_fault_test = \
        train_test_split(X, y_efficiency, y_fault, test_size=0.2, random_state=42, stratify=y_fault)
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set:     {len(X_test)} samples")
    
    # ---- Step 4: Fit Scaler on Training Data ----
    print("\nStep 4: Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # ---- Train Models ----
    eff_model = train_efficiency_model(X_train, X_test, y_eff_train, y_eff_test, scaler)
    fault_model = train_fault_model(X_train, X_test, y_fault_train, y_fault_test, scaler)
    
    # ---- Save Everything ----
    print("\n" + "SAVED MODELS")
    print("=" * 60)
    
    models_dir = os.path.join(PROJECT_ROOT, 'ml_models')
    os.makedirs(models_dir, exist_ok=True)
    
    eff_path = os.path.join(models_dir, 'efficiency_model.pkl')
    fault_path = os.path.join(models_dir, 'fault_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    joblib.dump(eff_model, eff_path)
    print(f"  Efficiency Model  -> {eff_path}")
    
    joblib.dump(fault_model, fault_path)
    print(f"  Fault Model       -> {fault_path}")
    
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler            -> {scaler_path}")
    
    # ---- Save feature column order for inference ----
    feature_order_path = os.path.join(models_dir, 'feature_columns.txt')
    with open(feature_order_path, 'w') as f:
        for col in feature_cols:
            f.write(col + '\n')
    print(f"  Feature Order     -> {feature_order_path}")
    
    print("\n" + "*" * 60)
    print("  TRAINING COMPLETE - ALL MODELS SAVED!")
    print("*" * 60 + "\n")


if __name__ == "__main__":
    main()
