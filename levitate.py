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
import uuid
import random
from datetime import datetime

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
AWSS3_REGION = "ap-south-2"
S3_BUCKET = "music-upload-bucket1"
S3_COVER_BUCKET = "output-covers"

s3 = boto3.client("s3", region_name=AWSS3_REGION)

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
def generate_image(prompt: str, size: int = 512) -> bytes:
    """
    Generate image using Bedrock Titan.
    Size options: 512 (fast ~10-15s) or 1024 (slow ~25-40s)
    """
    logger.info(f"Calling Bedrock Titan Image Generator ({size}x{size})")
    
    # Random seed for variation on regeneration
    seed = random.randint(0, 2147483647)

    response = bedrock.invoke_model(
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
                "height": size,
                "width": size,
                "cfgScale": 7.0,
                "seed": seed
            }
        })
    )

    raw = response["body"].read()
    logger.info(f"Bedrock response received ({len(raw)} bytes)")

    result = json.loads(raw)
    image_b64 = result["images"][0]

    return base64.b64decode(image_b64)

# ---------------- S3 IMAGE UPLOAD ----------------
def upload_image_to_s3(image_bytes: bytes, original_key: str) -> str:
    """Upload generated image to S3 and return a presigned URL."""
    # Generate unique filename based on original audio file and timestamp
    base_name = os.path.splitext(original_key)[0]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    image_key = f"{base_name}_{timestamp}_{unique_id}.png"
    
    logger.info(f"Uploading image to S3: {S3_COVER_BUCKET}/{image_key}")
    
    s3.put_object(
        Bucket=S3_COVER_BUCKET,
        Key=image_key,
        Body=image_bytes,
        ContentType="image/png"
    )
    
    # Generate a presigned URL (valid for 7 days)
    presigned_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': S3_COVER_BUCKET, 'Key': image_key},
        ExpiresIn=604800  # 7 days in seconds
    )
    
    logger.info(f"Image uploaded successfully: {image_key}")
    
    return presigned_url

# ---------------- REQUEST MODELS ----------------
class GenerateRequest(BaseModel):
    s3_key: str

# ---------------- UPLOAD ENDPOINT ----------------
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

@app.post("/upload")
async def upload_mp3(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only .mp3 files are supported")

    try:
        # Read file content to check size
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size is 5MB, got {file_size / (1024*1024):.2f}MB"
            )
        
        # Upload to S3
        from io import BytesIO
        s3.upload_fileobj(BytesIO(contents), S3_BUCKET, file.filename)
        logger.info(f"Uploaded {file.filename} to S3 ({file_size / 1024:.1f} KB)")
        return {"status": "uploaded", "s3_key": file.filename}
    except HTTPException:
        raise
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

        # Upload image to S3 and get the URL
        logger.info("Uploading image to S3")
        image_url = upload_image_to_s3(image_bytes, req.s3_key)

        return {
            "features": features,
            "prompt": prompt,
            "image_url": image_url
        }

    except Exception as e:
        logger.exception("Generate failed")
        raise HTTPException(status_code=500, detail=str(e))

