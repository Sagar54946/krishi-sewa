# server.py
from fastapi import FastAPI, Request
import json
import uvicorn
import os

app = FastAPI()

# The shared file used to talk to Streamlit
DATA_FILE = "sensor_data.json"

# Initialize file if it doesn't exist
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump({"temperature": 25.0, "humidity": 50.0, "ph": 6.5}, f)

@app.post("/update")
async def receive_data(request: Request):
    try:
        data = await request.json()
        print(f"📡 Data Received from ESP32: {data}")
        
        # Save data to the file
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)
            
        return {"status": "success"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error"}

if __name__ == "__main__":
    # 0.0.0.0 makes it accessible to the ESP32 on your network
    uvicorn.run(app, host="0.0.0.0", port=8000)