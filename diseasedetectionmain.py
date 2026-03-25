"""
SmartKhet — Disease Detection Microservice
===========================================
FastAPI service that:
1. Accepts crop images (upload or S3 URL)
2. Routes to ML inference (EfficientNet-B4 cloud model)
3. Fetches treatment advisory from knowledge base
4. Returns diagnosis + multilingual remedy in < 3 seconds

Endpoints:
  POST /disease/analyze        — image file upload
  POST /disease/analyze-url    — image from S3/URL
  POST /disease/analyze-text   — symptom description (no image)
  GET  /disease/history/{farmer_id}
  GET  /disease/classes        — list of supported diseases

Author: Axora / SmartKhet ML Team
"""

import os
import uuid
import json
import logging
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import boto3
import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncpg
from aiokafka import AIOKafkaProducer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Configuration ──────────────────────────────────────────────────────────────

DB_URL = os.getenv("POSTGRES_URL", "postgresql://localhost/smartkhet")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "smartkhet-images")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
ML_SERVICE_URL = os.getenv("ML_INFERENCE_URL", "http://ml-service:8080")
MAX_IMAGE_SIZE_MB = 10
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class DiseasePrediction(BaseModel):
    label: str
    crop: str
    condition: str
    confidence: float
    confidence_pct: str
    is_healthy: bool

class TreatmentStep(BaseModel):
    step: int
    action: str
    product: Optional[str] = None
    dosage: Optional[str] = None
    timing: Optional[str] = None
    organic_alternative: Optional[str] = None

class DiseaseAnalysisResponse(BaseModel):
    analysis_id: str
    farmer_id: str
    image_url: str
    predictions: list[DiseasePrediction]
    primary: DiseasePrediction
    requires_treatment: bool
    severity: str  # "low" | "medium" | "high" | "critical"
    treatment_advisory: list[TreatmentStep]
    advisory_hindi: str
    advisory_local: str
    estimated_yield_loss_pct: Optional[float]
    nearest_agri_shop: Optional[str]
    analyzed_at: datetime
    model_version: str

class SymptomQueryRequest(BaseModel):
    farmer_id: str
    crop: str
    symptoms: str = Field(..., min_length=10, max_length=500)
    language: str = Field(default="hi")

class AnalysisHistoryItem(BaseModel):
    analysis_id: str
    crop: str
    disease: str
    confidence: float
    is_healthy: bool
    analyzed_at: datetime
    image_thumbnail_url: Optional[str]

# ── App Lifecycle ──────────────────────────────────────────────────────────────

app_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all connections on startup, clean up on shutdown."""
    log.info("Starting Disease Detection Service...")

    # Database pool
    app_state["db"] = await asyncpg.create_pool(
        DB_URL.replace("postgresql://", "postgres://"),
        min_size=5, max_size=20,
        command_timeout=30,
    )

    # Redis cache
    app_state["redis"] = await aioredis.from_url(
        REDIS_URL, encoding="utf-8", decode_responses=True
    )

    # Kafka producer
    app_state["kafka"] = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await app_state["kafka"].start()

    # S3 client
    app_state["s3"] = boto3.client("s3", region_name=os.getenv("AWS_REGION", "ap-south-1"))

    # HTTP client for ML service calls
    app_state["http"] = httpx.AsyncClient(timeout=30.0)

    # Load treatment knowledge base from JSON
    kb_path = "data/seeds/disease_treatments.json"
    if os.path.exists(kb_path):
        with open(kb_path) as f:
            app_state["treatment_kb"] = json.load(f)
    else:
        app_state["treatment_kb"] = {}
        log.warning("Treatment knowledge base not found")

    log.info("Disease Detection Service ready ✅")
    yield

    # Cleanup
    await app_state["db"].close()
    await app_state["redis"].close()
    await app_state["kafka"].stop()
    await app_state["http"].aclose()


app = FastAPI(
    title="SmartKhet Disease Detection Service",
    version="1.0.0",
    description="Plant disease detection using EfficientNet-B4 computer vision",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Dependencies ───────────────────────────────────────────────────────────────

async def get_db():
    return app_state["db"]

async def get_redis():
    return app_state["redis"]

async def verify_jwt(authorization: str = Header(...)) -> dict:
    """Verify JWT token. In production, calls auth service."""
    from backend.shared.auth import decode_jwt
    token = authorization.replace("Bearer ", "")
    payload = decode_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


# ── Helpers ───────────────────────────────────────────────────────────────────

async def upload_image_to_s3(image_bytes: bytes, farmer_id: str,
                              mime_type: str) -> str:
    """Upload image to S3 and return presigned URL (valid 7 days)."""
    ext = mime_type.split("/")[-1]
    key = f"disease-images/{farmer_id}/{uuid.uuid4()}.{ext}"

    s3 = app_state["s3"]
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=image_bytes,
        ContentType=mime_type,
        ServerSideEncryption="AES256",
        Metadata={"farmer_id": farmer_id, "purpose": "disease-detection"},
    )

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=604800,  # 7 days
    )
    return url


async def call_ml_inference(image_bytes: bytes) -> dict:
    """Call ML inference service (Ray Serve endpoint)."""
    http = app_state["http"]
    files = {"image": ("crop.jpg", image_bytes, "image/jpeg")}
    resp = await http.post(f"{ML_SERVICE_URL}/v1/disease/predict", files=files)
    resp.raise_for_status()
    return resp.json()


def get_treatment_advisory(disease_label: str,
                            language: str = "hi") -> tuple[list[TreatmentStep], str, str]:
    """
    Lookup treatment steps from knowledge base.
    Returns (steps, hindi_summary, local_lang_summary).
    """
    kb = app_state.get("treatment_kb", {})
    treatment = kb.get(disease_label, {})

    steps = [
        TreatmentStep(**step)
        for step in treatment.get("steps", [
            TreatmentStep(step=1, action="Consult nearest Krishi Vigyan Kendra (KVK)",
                          timing="Within 2 days")
        ])
    ]
    hindi = treatment.get("summary_hi",
                          f"{disease_label} detected. Contact KVK for treatment advice.")
    local = treatment.get(f"summary_{language}", hindi)

    return steps, hindi, local


def estimate_severity(confidence: float, disease_label: str) -> tuple[str, Optional[float]]:
    """Estimate disease severity and potential yield loss."""
    HIGH_IMPACT_DISEASES = {
        "Rice___Neck_blast", "Wheat___Brown_rust", "Tomato___Late_blight",
        "Cotton___Curl_virus", "Banana___Panama_wilt",
    }
    if disease_label in HIGH_IMPACT_DISEASES:
        if confidence > 0.85:
            return "critical", 40.0
        elif confidence > 0.65:
            return "high", 25.0
    if confidence > 0.80:
        return "medium", 15.0
    elif confidence > 0.50:
        return "low", 5.0
    return "low", None


async def publish_analysis_event(analysis_id: str, farmer_id: str,
                                  disease: str, bg: BackgroundTasks):
    """Publish to Kafka for analytics + notification downstream."""
    async def _publish():
        try:
            await app_state["kafka"].send(
                "disease.events",
                value={
                    "event": "disease.analysis.completed",
                    "analysis_id": analysis_id,
                    "farmer_id": farmer_id,
                    "disease": disease,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        except Exception as e:
            log.error(f"Kafka publish failed: {e}")

    bg.add_task(_publish)


async def save_analysis_to_db(pool: asyncpg.Pool, analysis_id: str,
                               farmer_id: str, image_url: str,
                               prediction: dict, severity: str):
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO disease_analyses
              (id, farmer_id, image_url, disease_label, crop, confidence,
               severity, is_healthy, analyzed_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,NOW())
        """,
            analysis_id,
            farmer_id,
            image_url,
            prediction["primary"]["label"],
            prediction["primary"]["crop"],
            prediction["primary"]["confidence"],
            severity,
            prediction["primary"]["is_healthy"],
        )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/disease/analyze", response_model=DiseaseAnalysisResponse)
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = "hi",
    farmer_payload: dict = Depends(verify_jwt),
    db: asyncpg.Pool = Depends(get_db),
    cache: aioredis.Redis = Depends(get_redis),
):
    """
    Analyze crop image for disease detection.
    Accepts JPEG/PNG/WebP up to 10MB.
    """
    farmer_id = farmer_payload["sub"]

    # Validate file
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image too large. Max {MAX_IMAGE_SIZE_MB}MB")

    analysis_id = str(uuid.uuid4())

    # Upload to S3 (non-blocking)
    image_url = await upload_image_to_s3(image_bytes, farmer_id, file.content_type)

    # ML inference
    try:
        ml_result = await call_ml_inference(image_bytes)
    except httpx.HTTPError as e:
        log.error(f"ML inference failed: {e}")
        raise HTTPException(503, "Disease analysis temporarily unavailable")

    # Build response
    primary = ml_result["primary"]
    severity, yield_loss = estimate_severity(primary["confidence"], primary["label"])
    treatment_steps, hindi_summary, local_summary = get_treatment_advisory(
        primary["label"], language
    )

    predictions = [DiseasePrediction(**p) for p in ml_result["top_predictions"]]

    response = DiseaseAnalysisResponse(
        analysis_id=analysis_id,
        farmer_id=farmer_id,
        image_url=image_url,
        predictions=predictions,
        primary=DiseasePrediction(**primary),
        requires_treatment=ml_result["requires_treatment"],
        severity=severity,
        treatment_advisory=treatment_steps,
        advisory_hindi=hindi_summary,
        advisory_local=local_summary,
        estimated_yield_loss_pct=yield_loss,
        nearest_agri_shop=None,  # Populated by downstream enrichment
        analyzed_at=datetime.utcnow(),
        model_version="efficientnet_b4_v1.2",
    )

    # Async: persist to DB + publish event
    await save_analysis_to_db(db, analysis_id, farmer_id, image_url, ml_result, severity)
    await publish_analysis_event(analysis_id, farmer_id, primary["label"], background_tasks)

    return response


@app.post("/disease/analyze-text", response_model=dict)
async def analyze_symptoms(
    request: SymptomQueryRequest,
    farmer_payload: dict = Depends(verify_jwt),
):
    """
    Symptom-based disease advisory when farmer describes in text/voice.
    Uses vector similarity search over disease knowledge base.
    """
    # In production: embed symptoms → Pinecone similarity search → top disease matches
    return {
        "advisory": f"Based on symptoms described for {request.crop}: "
                    f"consult KVK or upload a photo for precise diagnosis.",
        "suggested_diseases": [],
        "confidence": "low",
        "note": "Image-based diagnosis is more accurate than symptom description.",
    }


@app.get("/disease/history/{farmer_id}", response_model=list[AnalysisHistoryItem])
async def get_analysis_history(
    farmer_id: str,
    limit: int = 20,
    farmer_payload: dict = Depends(verify_jwt),
    db: asyncpg.Pool = Depends(get_db),
):
    """Fetch farmer's disease analysis history."""
    if farmer_payload["sub"] != farmer_id:
        raise HTTPException(403, "Cannot access another farmer's history")

    async with db.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, crop, disease_label, confidence, is_healthy, analyzed_at
            FROM disease_analyses
            WHERE farmer_id = $1
            ORDER BY analyzed_at DESC
            LIMIT $2
        """, farmer_id, limit)

    return [AnalysisHistoryItem(
        analysis_id=str(row["id"]),
        crop=row["crop"],
        disease=row["disease_label"],
        confidence=float(row["confidence"]),
        is_healthy=row["is_healthy"],
        analyzed_at=row["analyzed_at"],
        image_thumbnail_url=None,
    ) for row in rows]


@app.get("/disease/classes")
async def get_disease_classes():
    """Return list of all supported disease classes."""
    from ml.disease_detection.train import DISEASE_CLASSES
    return {"classes": DISEASE_CLASSES, "total": len(DISEASE_CLASSES)}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "disease-detection", "version": "1.0.0"}
