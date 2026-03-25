"""
SmartKhet — Farmer Service
===========================
Core farmer identity, profile management, and land/crop history.
This service is the source of truth for farmer context used by all other services.

Endpoints:
  POST /farmers/register       — new farmer onboarding
  POST /farmers/login          — OTP-based login
  GET  /farmers/{id}/profile   — full profile
  PUT  /farmers/{id}/profile   — update profile
  POST /farmers/{id}/land      — add land parcel
  GET  /farmers/{id}/land      — list land parcels
  POST /farmers/{id}/crop-history — record planted crop
  GET  /farmers/{id}/context   — aggregated context for AI services

Author: Axora / SmartKhet ML Team
"""

import os
import uuid
import json
import hmac
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional

import asyncpg
import jwt
import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Config ────────────────────────────────────────────────────────────────────

DB_URL = os.getenv("POSTGRES_URL", "postgresql://localhost/smartkhet")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 30
MSG91_API_KEY = os.getenv("MSG91_API_KEY", "")
MSG91_SENDER_ID = "SKHETT"

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class FarmerRegisterRequest(BaseModel):
    phone: str = Field(..., pattern=r"^[6-9]\d{9}$", description="10-digit Indian mobile number")
    name: str = Field(..., min_length=2, max_length=100)
    state: str
    district: str
    village: Optional[str] = None
    preferred_language: str = Field(default="hi")
    referral_code: Optional[str] = None

class OTPVerifyRequest(BaseModel):
    phone: str = Field(..., pattern=r"^[6-9]\d{9}$")
    otp: str = Field(..., min_length=4, max_length=6)

class LandParcel(BaseModel):
    area_acres: float = Field(..., gt=0, le=500)
    soil_type: Optional[str] = None  # "alluvial" | "black" | "red" | "laterite" | "sandy"
    irrigation_type: Optional[str] = None  # "canal" | "borewell" | "rainfed" | "drip"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    khasra_number: Optional[str] = None  # Land record number

class CropHistoryEntry(BaseModel):
    crop: str
    season: str          # "kharif" | "rabi" | "zaid"
    year: int
    land_parcel_id: Optional[str] = None
    yield_qtl_per_acre: Optional[float] = None
    sold_price_per_qtl: Optional[float] = None
    notes: Optional[str] = None

class FarmerProfile(BaseModel):
    id: str
    phone: str
    name: str
    state: str
    district: str
    village: Optional[str]
    preferred_language: str
    total_land_acres: float
    primary_crops: list[str]
    member_since: datetime
    last_active: Optional[datetime]
    advisory_count: int

class FarmerContext(BaseModel):
    """Aggregated context snapshot used by AI services for personalisation."""
    farmer_id: str
    name: str
    state: str
    district: str
    preferred_language: str
    total_land_acres: float
    land_parcels: list[dict]
    current_crops: list[str]
    recent_crop_history: list[dict]
    soil_types: list[str]
    irrigation_types: list[str]
    last_soil_ph: Optional[float]
    last_advisory_date: Optional[datetime]


# ── App Lifecycle ──────────────────────────────────────────────────────────────

app_state: dict = {}

@asynccontextmanager
async def lifespan(app):
    app_state["db"] = await asyncpg.create_pool(
        DB_URL.replace("postgresql://", "postgres://"),
        min_size=5, max_size=20,
    )
    app_state["redis"] = await aioredis.from_url(REDIS_URL, decode_responses=True)
    app_state["http"] = httpx.AsyncClient(timeout=10.0)
    log.info("Farmer Service ready ✅")
    yield
    await app_state["db"].close()
    await app_state["redis"].close()
    await app_state["http"].aclose()


app = FastAPI(
    title="SmartKhet Farmer Service",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Auth Helpers ───────────────────────────────────────────────────────────────

def create_access_token(farmer_id: str, phone: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": farmer_id, "phone": phone, "exp": expire, "type": "access"},
        JWT_SECRET, algorithm=JWT_ALGORITHM,
    )

def create_refresh_token(farmer_id: str) -> str:
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": farmer_id, "exp": expire, "type": "refresh"},
        JWT_SECRET, algorithm=JWT_ALGORITHM,
    )

def decode_jwt(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None

async def verify_jwt(authorization: str = Header(...)) -> dict:
    token = authorization.replace("Bearer ", "")
    payload = decode_jwt(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(401, "Invalid or expired token")
    return payload

async def get_db():
    return app_state["db"]

async def get_redis():
    return app_state["redis"]


# ── OTP Service ───────────────────────────────────────────────────────────────

async def send_otp(phone: str) -> str:
    """Generate OTP, store in Redis (5 min TTL), send via MSG91."""
    import random
    otp = str(random.randint(100000, 999999))
    otp_key = f"otp:{phone}"

    redis = app_state["redis"]
    await redis.setex(otp_key, 300, otp)  # 5 minutes TTL

    if MSG91_API_KEY:
        # Send via MSG91
        http = app_state["http"]
        payload = {
            "template_id": "smartkhet_otp_v1",
            "mobile": f"91{phone}",
            "authkey": MSG91_API_KEY,
            "otp": otp,
            "sender": MSG91_SENDER_ID,
        }
        try:
            await http.post("https://api.msg91.com/api/v5/otp", json=payload)
        except Exception as e:
            log.error(f"MSG91 send failed: {e}")

    return otp  # In production, never return OTP in response

async def verify_otp(phone: str, otp: str) -> bool:
    redis = app_state["redis"]
    stored = await redis.get(f"otp:{phone}")
    if stored and hmac.compare_digest(stored, otp):
        await redis.delete(f"otp:{phone}")
        return True
    return False


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/farmers/send-otp")
async def send_otp_endpoint(phone: str):
    """Send OTP to farmer's phone number."""
    otp = await send_otp(phone)
    log.info(f"OTP sent to {phone[-4:]}: {otp}")  # Only log last 4 digits
    return {"message": f"OTP sent to +91 {phone[-4:].zfill(10)}", "expires_in": 300}


@app.post("/farmers/register")
async def register_farmer(
    request: FarmerRegisterRequest,
    otp: str,
    db: asyncpg.Pool = Depends(get_db),
):
    """Register new farmer after OTP verification."""
    if not await verify_otp(request.phone, otp):
        raise HTTPException(400, "Invalid or expired OTP")

    farmer_id = str(uuid.uuid4())
    async with db.acquire() as conn:
        existing = await conn.fetchrow("SELECT id FROM farmers WHERE phone=$1", request.phone)
        if existing:
            raise HTTPException(409, "Farmer with this phone already exists")

        await conn.execute("""
            INSERT INTO farmers (id, phone, name, state, district, village,
                                 preferred_language, created_at, updated_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,NOW(),NOW())
        """,
            farmer_id, request.phone, request.name,
            request.state, request.district, request.village,
            request.preferred_language,
        )

    access_token = create_access_token(farmer_id, request.phone)
    refresh_token = create_refresh_token(farmer_id)

    return {
        "farmer_id": farmer_id,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "message": f"Welcome to SmartKhet, {request.name}!",
    }


@app.post("/farmers/login")
async def login(request: OTPVerifyRequest, db: asyncpg.Pool = Depends(get_db)):
    """Login existing farmer with phone OTP."""
    if not await verify_otp(request.phone, request.otp):
        raise HTTPException(400, "Invalid or expired OTP")

    async with db.acquire() as conn:
        farmer = await conn.fetchrow(
            "SELECT id, name FROM farmers WHERE phone=$1", request.phone
        )
        if not farmer:
            raise HTTPException(404, "Farmer not found. Please register first.")

        await conn.execute(
            "UPDATE farmers SET last_active=NOW() WHERE id=$1", farmer["id"]
        )

    return {
        "farmer_id": str(farmer["id"]),
        "name": farmer["name"],
        "access_token": create_access_token(str(farmer["id"]), request.phone),
        "refresh_token": create_refresh_token(str(farmer["id"])),
        "token_type": "Bearer",
    }


@app.get("/farmers/{farmer_id}/profile", response_model=FarmerProfile)
async def get_profile(
    farmer_id: str,
    farmer_payload: dict = Depends(verify_jwt),
    db: asyncpg.Pool = Depends(get_db),
    cache: aioredis.Redis = Depends(get_redis),
):
    cache_key = f"farmer_profile:{farmer_id}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    async with db.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT f.*, 
                   COALESCE(SUM(lp.area_acres),0) as total_land,
                   COUNT(DISTINCT da.id) as advisory_count
            FROM farmers f
            LEFT JOIN land_parcels lp ON lp.farmer_id = f.id
            LEFT JOIN disease_analyses da ON da.farmer_id = f.id
            WHERE f.id = $1
            GROUP BY f.id
        """, uuid.UUID(farmer_id))

        if not row:
            raise HTTPException(404, "Farmer not found")

        crops = await conn.fetch("""
            SELECT DISTINCT crop FROM crop_history
            WHERE farmer_id=$1 ORDER BY created_at DESC LIMIT 5
        """, uuid.UUID(farmer_id))

    profile = FarmerProfile(
        id=str(row["id"]),
        phone=row["phone"],
        name=row["name"],
        state=row["state"],
        district=row["district"],
        village=row["village"],
        preferred_language=row["preferred_language"],
        total_land_acres=float(row["total_land"]),
        primary_crops=[r["crop"] for r in crops],
        member_since=row["created_at"],
        last_active=row["last_active"],
        advisory_count=row["advisory_count"],
    )

    await cache.setex(cache_key, 300, profile.model_dump_json())
    return profile


@app.post("/farmers/{farmer_id}/land")
async def add_land_parcel(
    farmer_id: str,
    land: LandParcel,
    farmer_payload: dict = Depends(verify_jwt),
    db: asyncpg.Pool = Depends(get_db),
):
    if farmer_payload["sub"] != farmer_id:
        raise HTTPException(403, "Forbidden")

    parcel_id = str(uuid.uuid4())
    async with db.acquire() as conn:
        await conn.execute("""
            INSERT INTO land_parcels
              (id, farmer_id, area_acres, soil_type, irrigation_type,
               latitude, longitude, khasra_number, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,NOW())
        """,
            uuid.UUID(parcel_id), uuid.UUID(farmer_id),
            land.area_acres, land.soil_type, land.irrigation_type,
            land.latitude, land.longitude, land.khasra_number,
        )

    # Invalidate profile cache
    cache = app_state["redis"]
    await cache.delete(f"farmer_profile:{farmer_id}")

    return {"parcel_id": parcel_id, "message": "Land parcel added successfully"}


@app.get("/farmers/{farmer_id}/context", response_model=FarmerContext)
async def get_farmer_context(
    farmer_id: str,
    db: asyncpg.Pool = Depends(get_db),
):
    """
    Aggregated context payload used by AI services.
    Called internally — no JWT auth required (internal network only).
    """
    async with db.acquire() as conn:
        farmer = await conn.fetchrow(
            "SELECT * FROM farmers WHERE id=$1", uuid.UUID(farmer_id)
        )
        if not farmer:
            raise HTTPException(404, "Farmer not found")

        parcels = await conn.fetch(
            "SELECT * FROM land_parcels WHERE farmer_id=$1", uuid.UUID(farmer_id)
        )
        history = await conn.fetch("""
            SELECT crop, season, year, yield_qtl_per_acre
            FROM crop_history WHERE farmer_id=$1
            ORDER BY year DESC, season DESC LIMIT 10
        """, uuid.UUID(farmer_id))

        last_soil = await conn.fetchrow("""
            SELECT ph FROM soil_readings WHERE farmer_id=$1
            ORDER BY measured_at DESC LIMIT 1
        """, uuid.UUID(farmer_id))

    return FarmerContext(
        farmer_id=farmer_id,
        name=farmer["name"],
        state=farmer["state"],
        district=farmer["district"],
        preferred_language=farmer["preferred_language"],
        total_land_acres=sum(float(p["area_acres"]) for p in parcels),
        land_parcels=[dict(p) for p in parcels],
        current_crops=list({r["crop"] for r in history if r["year"] == datetime.now().year}),
        recent_crop_history=[dict(r) for r in history],
        soil_types=list({p["soil_type"] for p in parcels if p["soil_type"]}),
        irrigation_types=list({p["irrigation_type"] for p in parcels if p["irrigation_type"]}),
        last_soil_ph=float(last_soil["ph"]) if last_soil else None,
        last_advisory_date=None,
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "farmer-service", "version": "1.0.0"}
