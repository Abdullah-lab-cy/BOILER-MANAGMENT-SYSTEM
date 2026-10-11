#!/usr/bin/env python3
# ========================================================================
# Boiler Automation System - Pre-Flight Verification Script
# ========================================================================
# This script verifies all configurations before starting the system
# Run: python verify_system.py
# ========================================================================

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Reconfigure standard output to support UTF-8 on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # For Python < 3.7

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

# ========================================================================
# VERIFICATION CHECKS
# ========================================================================

def check_project_structure():
    """Verify all required directories and files exist"""
    print_header("📁 Project Structure Verification")
    
    required_files = {
        'firmware/src/main.cpp': 'ESP32 Firmware',
        'firmware/platformio.ini': 'PlatformIO Configuration',
        'server/config.py': 'Server Configuration',
        'server/mqtt_server.py': 'MQTT Server',
        'server/app.py': 'Flask API',
        'server/database.py': 'Database Manager',
        'dashboard/app_frontend.py': 'Streamlit Dashboard',
        'ai_monitoring/live_monitoring.py': 'Deep Learning Monitor',
        'ai_monitoring/fault_model.h5': 'ML Model (H5)',
        'requirements.txt': 'Python Dependencies'
    }
    
    project_root = Path(__file__).parent
    all_exist = True
    
    for file_path, description in required_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            print_success(f"{description}: {file_path}")
        else:
            print_error(f"{description}: {file_path} (NOT FOUND)")
            all_exist = False
    
    return all_exist

def check_python_packages():
    """Verify all required Python packages are installed"""
    print_header("📦 Python Dependencies Verification")
    
    required_packages = {
        'pandas': 'Data Processing',
        'numpy': 'Numerical Computing',
        'paho.mqtt': 'MQTT Protocol',
        'flask': 'Web Server Framework',
        'flask_cors': 'CORS Support',
        'streamlit': 'Dashboard Framework',
        'tensorflow': 'Deep Learning Framework',
        'joblib': 'Model Serialization',
        'requests': 'HTTP Client (Telegram)'
    }
    
    all_installed = True
    for package, description in required_packages.items():
        try:
            __import__(package)
            print_success(f"{description} ({package})")
        except ImportError:
            print_error(f"{description} ({package}) - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_configuration_files():
    """Verify critical configuration values"""
    print_header("⚙️ Configuration Verification")
    
    config_path = Path(__file__).parent / 'server' / 'config.py'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check MQTT Configuration
        if 'broker.hivemq.com' in content:
            print_success("MQTT Broker: broker.hivemq.com (HiveMQ Cloud)")
        else:
            print_error("MQTT Broker: Not properly configured")
        
        if 'ejaz_boiler_sys_737/sensors' in content:
            print_success("MQTT Topic: ejaz_boiler_sys_737/sensors")
        else:
            print_error("MQTT Topic: Not properly configured")
        
        if 'TELEGRAM_TOKEN' in content and '8687747811' in content:
            print_success("Telegram Bot: Configured")
        else:
            print_warning("Telegram Bot: Token may not be set")
        
        if 'DATABASE' in content or 'boiler_data.db' in content:
            print_success("Database: SQLite configured")
        else:
            print_error("Database: Not properly configured")
        
    except Exception as e:
        print_error(f"Configuration check failed: {e}")
        return False
    
    return True

def check_firmware_config():
    """Verify firmware configuration"""
    print_header("🔧 Firmware Configuration Verification")
    
    firmware_path = Path(__file__).parent / 'firmware' / 'src' / 'main.cpp'
    
    try:
        with open(firmware_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check WiFi Configuration
        if 'ssid = "ssss"' in content:
            print_success("WiFi SSID: ssss")
        else:
            print_warning("WiFi SSID: May not be set to 'ssss'")
        
        if 'password = "123456789"' in content:
            print_success("WiFi Password: ••••••••• (configured)")
        else:
            print_warning("WiFi Password: May need verification")
        
        # Check MQTT Configuration
        if 'broker.hivemq.com' in content:
            print_success("Firmware MQTT Broker: broker.hivemq.com")
        else:
            print_error("Firmware MQTT Broker: Not properly configured")
        
        # Check Pin Configuration
        pins = {
            'pinTemp1 = 1': 'Inlet Temperature (GPIO 1)',
            'pinTemp2 = 2': 'Outlet Temperature (GPIO 2)',
            'pinPress1 = 3': 'Steam Pressure (GPIO 3)',
            'pinPress2 = 4': 'Flue Gas Temperature (GPIO 4)',
            'pinVoltage = 5': 'Water Level (GPIO 5)',
            'pinCurrent = 6': 'Fuel Flow (GPIO 6)',
            'BUZZER_PIN = 42': 'Buzzer (GPIO 42)'
        }
        
        print("")
        print_info("Pin Configuration:")
        for pin_config, description in pins.items():
            if pin_config in content:
                print_success(f"  {description}")
            else:
                print_warning(f"  {description} - May need verification")
        
    except Exception as e:
        print_error(f"Firmware check failed: {e}")
        return False
    
    return True

def check_database():
    """Check database status"""
    print_header("💾 Database Status Verification")
    
    db_path = Path(__file__).parent / 'database' / 'boiler_data.db'
    
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print_success(f"Database exists: boiler_data.db ({size_mb:.2f} MB)")
        
        # Try to query it
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            conn.close()
            
            if table_count > 0:
                print_success(f"Database tables: {table_count} tables found")
            else:
                print_warning("Database: No tables found (needs initialization)")
        except Exception as e:
            print_warning(f"Database check: {e}")
    else:
        print_warning("Database: Not found (will be created on first run)")

def check_ml_models():
    """Check ML models"""
    print_header("🤖 Machine Learning Models Verification")
    
    models_dir = Path(__file__).parent / 'ai_monitoring'
    
    model_files = {
        'fault_model.h5': 'Deep Learning Fault Model (TensorFlow)',
        'scaler.save': 'Feature Scaler'
    }
    
    models_exist = 0
    for file_name, description in model_files.items():
        model_path = models_dir / file_name
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print_success(f"{description}: {file_name} ({size_mb:.2f} MB)")
            models_exist += 1
        else:
            print_warning(f"{description}: {file_name} (NOT FOUND)")
    
    # Also check traditional ML models
    ml_dir = Path(__file__).parent / 'ml_models'
    if ml_dir.exists():
        pkl_files = list(ml_dir.glob('*.pkl'))
        if pkl_files:
            print_info(f"Traditional ML models found: {len(pkl_files)} files")

def generate_summary():
    """Generate verification summary"""
    print_header("📋 Verification Summary")
    
    print(f"""
{Colors.BOLD}System Status:{Colors.ENDC}
  • Hardware: ESP32-S3 DevKit
  • Communication: MQTT (HiveMQ Cloud)
  • AI Engine: TensorFlow Deep Learning
  • Dashboard: Streamlit Web UI
  • Database: SQLite
  • Alerts: Telegram Bot

{Colors.BOLD}Required Setup:{Colors.ENDC}
  • WiFi Hotspot SSID: "Ejaz"
  • WiFi Password: "123456789"
  • Internet Connection: Required (for MQTT)
  • USB Cable: For ESP32 programming

{Colors.BOLD}Before Starting:{Colors.ENDC}
  1. ✓ Verify all project files exist
  2. ✓ Install all Python packages
  3. ✓ Enable WiFi hotspot with correct SSID/Password
  4. ✓ Connect ESP32-S3 via USB
  5. ✓ Run startup script

{Colors.BOLD}Next Steps:{Colors.ENDC}
  1. Flash firmware: platformio run -e esp32s3 -t upload
  2. Run startup script: ./START_ALL_SERVICES.ps1
  3. Open Dashboard: http://localhost:8501
  4. Monitor MQTT Server output for sensor data
""")

def main():
    """Main verification routine"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Boiler Automation System - Pre-Flight Verification{Colors.ENDC}")
    print(f"{Colors.CYAN}Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
    
    all_pass = True
    
    # Run all checks
    all_pass &= check_project_structure()
    all_pass &= check_python_packages()
    all_pass &= check_configuration_files()
    all_pass &= check_firmware_config()
    check_database()
    check_ml_models()
    
    # Generate summary
    generate_summary()
    
    # Final status
    print_header("🎯 Verification Complete")
    if all_pass:
        print_success("All critical checks passed! System is ready to start.")
        print_info("Run: ./START_ALL_SERVICES.ps1 to begin")
    else:
        print_warning("Some checks failed. Please review the errors above.")
        print_info("Fix the issues and run this script again.")
    
    print("")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Verification cancelled by user.{Colors.ENDC}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Verification error: {e}{Colors.ENDC}\n")
        sys.exit(1)
