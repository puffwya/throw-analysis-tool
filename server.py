from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import subprocess
import os

app = FastAPI()

UPLOAD_PATH = "uploads"
OUTPUT_PATH = "outputs"

os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- Upload & Process Video ---
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_PATH, file.filename)
    base_output_name = f"processed_{os.path.splitext(file.filename)[0]}"
    avi_output_path = os.path.join(OUTPUT_PATH, f"{base_output_name}.avi")
    mp4_output_path = os.path.join(OUTPUT_PATH, f"{base_output_name}.mp4")

    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run your processing script (outputs AVI)
    result = subprocess.run([
        "/home/ubuntu/throw_ai_env/bin/python",
        "angle_analysis.py",
        input_path,
        avi_output_path
    ], capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode != 0:
        return {"error": result.stderr}

    # Convert AVI to MP4 using H.264 for browser compatibility
    ffmpeg_result = subprocess.run([
        "ffmpeg", "-y", "-i", avi_output_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        mp4_output_path
    ], capture_output=True, text=True)

    if ffmpeg_result.returncode != 0:
        return {"error": f"FFmpeg failed: {ffmpeg_result.stderr}"}

    # Optional: remove the AVI after conversion
    if os.path.exists(avi_output_path):
        os.remove(avi_output_path)

    # Return MP4 path for streaming
    return {"output_video": f"/outputs/{os.path.basename(mp4_output_path)}"}


# --- Serve Video Properly ---
@app.get("/outputs/{filename}")
async def stream_video(filename: str):
    file_path = os.path.join(OUTPUT_PATH, filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(file_path, media_type="video/mp4")

# --- Serve Static Files for Frontend ---
app.mount("/", StaticFiles(directory="static", html=True), name="static")
