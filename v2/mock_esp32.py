import requests
import time
import random

# This URL must match the one in server.py
# Since we are running locally, we use localhost
url = "http://127.0.0.1:8000/update"

print("🚀 Starting Fake ESP32...")

while True:
    # 1. Generate Fake Data
    payload = {
        "temperature": round(random.uniform(20.0, 35.0), 1),
        "humidity": round(random.uniform(40.0, 90.0), 1),
        "ph": round(random.uniform(5.5, 7.5), 1)
    }

    try:
        # 2. Send POST Request (Just like ESP32)
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print(f"✅ Sent: {payload}")
        else:
            print(f"❌ Server Error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️  Connection refused. Is server.py running?")

    # 3. Wait 3 seconds before sending again
    time.sleep(3)