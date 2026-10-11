from .telegram_bot import send_telegram_alert

def notify_user(fault_type, sensor_data):
    """
    Decides which alerts to trigger based on fault type.
    
    Fault Types from ML Model:
      - "Normal"       → No alert
      - "Warning"      → Console log only
      - "High Pressure"→ Telegram + Console (Critical)
    
    Additional rule-based alerts:
      - Critical Temp  → Telegram
      - Gas Leak       → Telegram
      - Low Water      → Telegram
    """
    # Build detailed alert message
    message = f"*Fault Detected:* {fault_type}\n"
    message += "---\n"
    message += f"🌡 *Temperature:* {sensor_data.get('temp_outlet', 'N/A')} °C\n"
    message += f"⚡ *Pressure:* {sensor_data.get('pressure_steam', 'N/A')} bar\n"
    message += f"💧 *Water Level:* {sensor_data.get('water_level', 'N/A')}%\n"
    message += f"🔥 *Flame:* {'ON' if sensor_data.get('flame_status', 0) == 1 else 'OFF'}\n"
    message += f"💨 *Gas/Smoke:* {sensor_data.get('gas_smoke', 'N/A')} ppm\n"
    message += f"⛽ *Fuel Flow:* {sensor_data.get('fuel_flow', 'N/A')} m³/hr"

    # Always log locally
    print(f"\n{'='*40}")
    print(f"  ALERT: {fault_type}")
    print(f"{'='*40}")
    print(f"  Temp:     {sensor_data.get('temp_outlet', 'N/A')} °C")
    print(f"  Pressure: {sensor_data.get('pressure_steam', 'N/A')} bar")
    print(f"  Water:    {sensor_data.get('water_level', 'N/A')}%")
    print(f"  Gas:      {sensor_data.get('gas_smoke', 'N/A')} ppm")
    print(f"{'='*40}\n")

    # Telegram alerts for critical/warning faults
    critical_faults = ["High Pressure", "Critical Temp", "Gas Leak", "Low Water"]
    warning_faults = ["Warning"]
    
    if fault_type in critical_faults:
        # CRITICAL → Send Telegram immediately
        send_telegram_alert(f"🔴 *CRITICAL ALERT*\n\n{message}")
    elif fault_type in warning_faults:
        # WARNING → Send Telegram with lower priority
        send_telegram_alert(f"🟡 *WARNING*\n\n{message}")
    # "Normal" → No Telegram, only console log above
