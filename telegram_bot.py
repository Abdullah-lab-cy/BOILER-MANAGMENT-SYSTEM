import requests
import os
import sys

# Fix import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from server.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

def send_telegram_alert(message):
    """
    Sends a message to the configured Telegram Chat.
    Uses the Telegram Bot API sendMessage endpoint.
    """
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("[Telegram] Token not configured. Alert:", message)
        return None

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🚨 BOILER ALERT 🚨\n\n{message}",
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        result = response.json()
        if result.get('ok'):
            print(f"[Telegram] Alert sent successfully (msg_id: {result['result']['message_id']})")
        else:
            print(f"[Telegram] Failed: {result.get('description', 'Unknown error')}")
        return result
    except requests.exceptions.Timeout:
        print("[Telegram] Connection timed out - check internet/VPN")
        return None
    except Exception as e:
        print(f"[Telegram] Error: {e}")
        return None


if __name__ == "__main__":
    # Quick standalone test
    print("Sending test Telegram message...")
    result = send_telegram_alert("Test alert from Boiler Automation System.\nAll systems operational.")
    if result and result.get('ok'):
        print("SUCCESS - Check your Telegram!")
    else:
        print("FAILED - Check token/chat_id/network")
