from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import boto3
import json
import base64
import librosa
import numpy as np
import tempfile
import os
import logging

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- APP ----------------
app = FastAPI(title="Levitate API", version="1.0")

@app.get("/")
def root():
    return {"status": "ok", "message": "Levitate API is running"}

# ---------------- AWS CLIENTS ----------------
AWS_REGION = "us-east-1"
S3_BUCKET = "music-upload-bucket2"

s3 = boto3.client("s3", region_name=AWS_REGION)

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION
)

# ---------------- AUDIO ANALYSIS ----------------
def analyze_audio(path: str) -> dict:
    y, sr = librosa.load(path)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y).mean()

    y_harm, y_perc = librosa.effects.hpss(y)
    harmonic_energy = np.mean(np.abs(y_harm))
    percussive_energy = np.mean(np.abs(y_perc))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    if rms < 0.03 and percussive_energy < harmonic_energy:
        energy = "low"
    elif rms < 0.06:
        energy = "medium"
    else:
        energy = "high"

    if harmonic_energy > percussive_energy and centroid < 2000:
        mood = "emotional / warm"
    elif contrast > 25 and zcr > 0.1:
        mood = "tense / aggressive"
    elif centroid > 3000:
        mood = "bright / uplifting"
    else:
        mood = "dark / cinematic"

    return {
        "tempo": round(float(tempo), 1),
        "energy": energy,
        "mood": mood,
        "rms": float(rms),
        "centroid": float(centroid)
    }

# ---------------- PROMPT BUILDER ----------------
def build_prompt(features: dict) -> str:
    lighting = {
        "low": "soft ambient light",
        "medium": "cinematic balanced lighting",
        "high": "dramatic volumetric lighting"
    }[features["energy"]]

    mood_style = {
        "emotional / warm": "warm sunset tones, soft glow, peaceful atmosphere",
        "dark / cinematic": "moody shadows, deep contrast",
        "bright / uplifting": "vibrant colors, hopeful sky"
    }.get(features["mood"], "cinematic lighting")

    return f"""
GAME CONCEPT ART of a vast open fantasy landscape.
Atmosphere: {mood_style}.
Motion synced with rhythm at {features["tempo"]} BPM.
Lighting: {lighting}.
Unreal Engine 5 style, AAA environment concept art.
Cinematic wide shot, ultra detailed, no text, no watermark.
""".strip()

# ---------------- BEDROCK IMAGE GENERATION ----------------
def generate_image(prompt: str) -> bytes:
    logger.info("Calling Bedrock Titan Image Generator")

    response = bedrock.invoke_model(
        # amazon.titan-image-generator-v2:0
        modelId="amazon.titan-image-generator-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 1024,
                "width": 1024,
                "cfgScale": 8.0,
                "seed": 0
            }
        })
    )

    raw = response["body"].read()
    logger.info(f"Bedrock response received ({len(raw)} bytes)")

    result = json.loads(raw)
    image_b64 = result["artifacts"][0]["base64"]

    return base64.b64decode(image_b64)

# ---------------- REQUEST MODELS ----------------
class GenerateRequest(BaseModel):
    s3_key: str

# ---------------- UPLOAD ENDPOINT ----------------
@app.post("/upload")
async def upload_mp3(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only .mp3 files are supported")

    try:
        s3.upload_fileobj(file.file, S3_BUCKET, file.filename)
        logger.info(f"Uploaded {file.filename} to S3")
        return {"status": "uploaded", "s3_key": file.filename}
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- GENERATE ENDPOINT ----------------
@app.post("/generate")
def generate_visual(req: GenerateRequest):
    logger.info(f"Generate request received: {req.s3_key}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            s3.download_fileobj(S3_BUCKET, req.s3_key, tmp)
            tmp_path = tmp.name

        logger.info("Analyzing audio")
        features = analyze_audio(tmp_path)

        logger.info("Building prompt")
        prompt = build_prompt(features)

        logger.info("Generating image")
        image_bytes = generate_image(prompt)

        os.remove(tmp_path)

        return {
            "features": features,
            "prompt": prompt,
            "image_base64": base64.b64encode(image_bytes).decode("utf-8")
        }

    except Exception as e:
        logger.exception("Generate failed")
        raise HTTPException(status_code=500, detail=str(e))

