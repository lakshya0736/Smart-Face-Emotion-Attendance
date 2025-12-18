from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import subprocess

app = FastAPI()

# Serve frontend files
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Homepage
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Start attendance camera
@app.post("/mark_attendance")
async def mark_attendance():
    try:
        subprocess.Popen(["python", "real_time_attendance.py"])
        return JSONResponse({
            "status": "success",
            "message": "Camera started. Please look at the camera."
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })
