from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import subprocess
import os
import tempfile
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("clips", exist_ok=True)
app.mount("/clips", StaticFiles(directory="clips"), name="clips")


@app.post("/api/process")
async def process_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        input_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    output_fd, output_path = tempfile.mkstemp(suffix=".json")
    os.close(output_fd)

    try:
        result = subprocess.run(
            ["python", "clipify-pipeline.py", input_path, output_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline failed: {result.stderr}",
            )

        with open(output_path) as f:
            data = json.load(f)

        return data

    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
