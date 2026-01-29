"""
Frontend server for Levitate API.
Run with: uvicorn server:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import boto3

# Import the main API app
from levitate import app as api_app, s3, S3_BUCKET, logger

# ---------------- APP ----------------
app = FastAPI(title="Levitate Frontend", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the API
app.mount("/api", api_app)

# ---------------- LIST MUSIC ENDPOINT ----------------
@app.get("/music")
def list_music():
    """List all uploaded music files."""
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET)
        
        if "Contents" not in response:
            return {"files": []}
        
        files = []
        for obj in response["Contents"]:
            if obj["Key"].lower().endswith(".mp3"):
                files.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat()
                })
        
        # Sort by last modified (newest first)
        files.sort(key=lambda x: x["last_modified"], reverse=True)
        
        return {"files": files}
    except Exception as e:
        logger.exception("Failed to list music files")
        return {"files": [], "error": str(e)}

# ---------------- PLAY MUSIC ENDPOINT ----------------
@app.get("/music/play/{s3_key:path}")
def get_music_url(s3_key: str):
    """Get a presigned URL to play music."""
    try:
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=3600  # 1 hour
        )
        return {"url": presigned_url}
    except Exception as e:
        logger.exception("Failed to get music URL")
        return {"error": str(e)}

# ---------------- STATIC FILES ----------------
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")
