"""
SmartKhet — Crop Advisory Service
===================================
Generates personalised crop, fertilizer, and irrigation advisories
by combining soil data, weather forecasts, farmer history, and
the trained XGBoost crop recommendation model.

Endpoints:
  POST /advisory/crop         — crop recommendation from soil data
  POST /advisory/fertilizer   — NPK fertilizer schedule
  POST /advisory/irrigation   — irrigation timing & quantity
  POST /advisory/pest         — pest management advice
  GET  /advisory/daily/{id}   — today's personalised advisory for farmer
  POST /advisory/soil         — soil health report + improvement tips

Author: Axora / SmartKhet ML Team
"""

import os
import json
import logging
from datetime import datetime, date
from typing import Optional
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
import asyncpg
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Config ────────────────────────────────────────────────────────────────────

DB_URL = os.getenv("POSTGRES_URL", "postgresql://localhost/smartkhet")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ML_SERVICE_URL = os.getenv("ML_INFERENCE_URL", "http://ml-service:8080")
WEATHER_SERVICE_URL = os.getenv("WEATHER_SERVICE_URL", "http://weather-service:8007")
FARMER_SERVICE_URL = os.getenv("FARMER_SERVICE_URL", "http://farmer-service:8001")


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class SoilInput(BaseModel):
    nitrogen: float = Field(..., ge=0, le=500, description="N content kg/ha")
    phosphorus: float = Field(..., ge=0, le=300, description="P content kg/ha")
    potassium: float = Field(..., ge=0, le=400, description="K content kg/ha")
    ph: float = Field(..., ge=1.0, le=14.0)
    moisture: float = Field(default=50.0, ge=0, le=100)

class CropAdvisoryRequest(BaseModel):
    farmer_id: str
    soil: SoilInput
    district: str
    state: str
    season: str = Field(default="kharif",
                        pattern="^(kharif|rabi|zaid|perennial)$")
    preferred_crops: Optional[list[str]] = None   # Farmer's preference hint

class FertilizerRequest(BaseModel):
    farmer_id: str
    crop: str
    soil: SoilInput
    area_acres: float = Field(..., gt=0)
    growth_stage: Optional[str] = None  # "sowing"|"vegetative"|"flowering"|"maturity"

class IrrigationRequest(BaseModel):
    farmer_id: str
    crop: str
    soil_moisture: float = Field(..., ge=0, le=100)
    district: str
    state: str
    crop_stage: Optional[str] = None

class CropRecommendation(BaseModel):
    rank: int
    crop: str
    confidence: float
    confidence_pct: str
    why: str                         # Human-readable reason
    estimated_yield_qtl_acre: Optional[float]
    market_price_inr_qtl: Optional[float]
    expected_income_per_acre: Optional[float]

class FertilizerSchedule(BaseModel):
    crop: str
    area_acres: float
    urea_kg_total: float
    dap_kg_total: float
    mop_kg_total: float
    application_stages: list[dict]
    ph_correction: Optional[str]
    estimated_cost_inr: float
    organic_alternatives: list[str]

class IrrigationPlan(BaseModel):
    crop: str
    current_moisture_pct: float
    needs_irrigation: bool
    recommended_in_days: int
    water_mm_per_hectare: float
    method: str                    # "flood"|"drip"|"sprinkler"
    weather_advisory: str
    next_check_date: date


# ── App Lifecycle ──────────────────────────────────────────────────────────────

app_state: dict = {}

@asynccontextmanager
async def lifespan(app):
    app_state["db"] = await asyncpg.create_pool(
        DB_URL.replace("postgresql://", "postgres://"), min_size=3, max_size=15
    )
    app_state["redis"] = await aioredis.from_url(REDIS_URL, decode_responses=True)
    app_state["http"] = httpx.AsyncClient(timeout=15.0)

    # Load agronomic knowledge base
    kb_path = "data/seeds/crop_advisory_kb.json"
    if os.path.exists(kb_path):
        with open(kb_path) as f:
            app_state["agri_kb"] = json.load(f)
    else:
        app_state["agri_kb"] = {}

    log.info("Crop Advisory Service ready ✅")
    yield
    await app_state["db"].close()
    await app_state["redis"].close()
    await app_state["http"].aclose()


app = FastAPI(
    title="SmartKhet Crop Advisory Service",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Helpers ───────────────────────────────────────────────────────────────────

async def get_weather_context(district: str, state: str) -> dict:
    """Fetch 7-day weather forecast from weather service."""
    cache = app_state["redis"]
    key = f"weather:{district}:{state}"
    cached = await cache.get(key)
    if cached:
        return json.loads(cached)

    try:
        resp = await app_state["http"].get(
            f"{WEATHER_SERVICE_URL}/weather/forecast",
            params={"district": district, "state": state, "days": 7}
        )
        data = resp.json()
        await cache.setex(key, 3600, json.dumps(data))
        return data
    except Exception:
        return {"avg_temp": 27, "avg_humidity": 65, "rainfall_7d_mm": 0,
                "forecast_available": False}


async def get_market_context(crop: str, district: str) -> dict:
    """Fetch current market price for crop from market service."""
    try:
        resp = await app_state["http"].get(
            f"http://market-service:8004/market/price/{crop}",
            params={"district": district}
        )
        return resp.json()
    except Exception:
        return {}


async def call_ml_crop_recommender(features: dict) -> list[dict]:
    """Call ML inference service for crop recommendation."""
    cache = app_state["redis"]
    import hashlib
    cache_key = "crop_pred:" + hashlib.md5(
        json.dumps(features, sort_keys=True).encode()
    ).hexdigest()

    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    try:
        resp = await app_state["http"].post(
            f"{ML_SERVICE_URL}/v1/crop/predict",
            json=features,
            timeout=10.0,
        )
        result = resp.json()
        await cache.setex(cache_key, 1800, json.dumps(result))
        return result
    except Exception as e:
        log.error(f"ML crop recommender failed: {e}")
        return []


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/advisory/crop", response_model=list[CropRecommendation])
async def get_crop_recommendation(request: CropAdvisoryRequest):
    """
    Top-3 crop recommendations based on soil + weather + market context.
    Combines ML model output with agronomic rules and market pricing.
    """
    weather = await get_weather_context(request.district, request.state)

    features = {
        "nitrogen":        request.soil.nitrogen,
        "phosphorus":      request.soil.phosphorus,
        "potassium":       request.soil.potassium,
        "ph":              request.soil.ph,
        "moisture":        request.soil.moisture,
        "temperature":     weather.get("avg_temp", 27),
        "humidity":        weather.get("avg_humidity", 65),
        "rainfall":        weather.get("annual_rainfall_mm", 800),
        "season_encoded":  {"kharif": 0, "rabi": 1, "zaid": 2, "perennial": 3}
                           .get(request.season, 0),
        "district_encoded": 0.5,   # Placeholder; real value from district freq map
        "market_demand":   0.5,
    }

    ml_results = await call_ml_crop_recommender(features)

    recommendations = []
    for item in ml_results[:3]:
        crop = item["crop"]
        market = await get_market_context(crop, request.district)
        kb = app_state["agri_kb"].get(crop, {})

        # Build human-readable reason
        reasons = []
        if request.soil.ph < 6.5 and kb.get("prefers_acidic"):
            reasons.append("आपकी मिट्टी का pH इस फ़सल के लिए उचित है")
        if weather.get("avg_temp", 27) in kb.get("optimal_temp_range", [20, 35]):
            reasons.append("मौसम अनुकूल है")
        if not reasons:
            reasons.append(f"मिट्टी और मौसम के अनुसार उपयुक्त")

        modal_price = market.get("current_price", 0)
        est_yield = kb.get("avg_yield_qtl_acre", 10)
        expected_income = round(modal_price * est_yield, 2) if modal_price else None

        recommendations.append(CropRecommendation(
            rank=item["rank"],
            crop=crop,
            confidence=item["confidence"],
            confidence_pct=item["confidence_pct"],
            why=" | ".join(reasons),
            estimated_yield_qtl_acre=est_yield,
            market_price_inr_qtl=modal_price or None,
            expected_income_per_acre=expected_income,
        ))

    return recommendations


@app.post("/advisory/fertilizer", response_model=FertilizerSchedule)
async def get_fertilizer_advisory(request: FertilizerRequest):
    """
    Generates a complete fertilizer schedule: total quantities + application stages.
    Based on ICAR recommendations per crop, soil deficit, and growth stage.
    """
    from ml.crop_recommendation.train import CropRecommender
    recommender = CropRecommender.__new__(CropRecommender)

    advisory = recommender._generate_fertilizer_advisory(
        crop=request.crop,
        n=request.soil.nitrogen,
        p=request.soil.phosphorus,
        k=request.soil.potassium,
        ph=request.soil.ph,
    )

    area = request.area_acres
    urea_total = round(advisory["urea_kg_ha"] * area * 0.405, 1)  # ha = acres * 0.405
    dap_total  = round(advisory["dap_kg_ha"]  * area * 0.405, 1)
    mop_total  = round(advisory["mop_kg_ha"]  * area * 0.405, 1)

    # Prices (approx 2024 rates)
    urea_price, dap_price, mop_price = 266, 1350, 1700  # ₹ per 45kg bag

    cost = round(
        (urea_total / 45) * urea_price +
        (dap_total  / 50) * dap_price +
        (mop_total  / 50) * mop_price, 2
    )

    kb = app_state["agri_kb"].get(request.crop, {})
    stages = kb.get("fertilizer_stages", [
        {"stage": "बुवाई के समय", "urea_pct": 30, "dap_pct": 100, "mop_pct": 100},
        {"stage": "30 दिन बाद (CRI)", "urea_pct": 40, "dap_pct": 0, "mop_pct": 0},
        {"stage": "60 दिन बाद", "urea_pct": 30, "dap_pct": 0, "mop_pct": 0},
    ])

    return FertilizerSchedule(
        crop=request.crop,
        area_acres=area,
        urea_kg_total=urea_total,
        dap_kg_total=dap_total,
        mop_kg_total=mop_total,
        application_stages=stages,
        ph_correction=advisory.get("ph_correction"),
        estimated_cost_inr=cost,
        organic_alternatives=[
            "वर्मीकम्पोस्ट 2 टन/एकड़",
            "हरी खाद (ढैंचा) 6 सप्ताह पहले",
            "जैव खाद — राइज़ोबियम + PSB",
        ],
    )


@app.post("/advisory/irrigation", response_model=IrrigationPlan)
async def get_irrigation_advisory(request: IrrigationRequest):
    """
    Irrigation recommendation based on soil moisture + crop water requirements
    + 7-day rainfall forecast.
    """
    weather = await get_weather_context(request.district, request.state)
    kb = app_state["agri_kb"].get(request.crop, {})

    # Critical soil moisture threshold per crop
    critical_moisture = kb.get("critical_moisture_pct", 40)
    optimal_moisture  = kb.get("optimal_moisture_pct", 65)

    rainfall_7d = weather.get("rainfall_7d_mm", 0)
    needs_irrigation = (request.soil_moisture < critical_moisture
                        and rainfall_7d < 10)

    # Estimate days until irrigation needed
    daily_evapotranspiration = kb.get("etc_mm_day", 5)
    current_deficit = optimal_moisture - request.soil_moisture
    days_until_needed = max(0, int(current_deficit / daily_evapotranspiration))
    if rainfall_7d > 20:
        days_until_needed += 3

    # Water quantity
    water_mm = max(0, (optimal_moisture - request.soil_moisture) * 3)  # approx

    # Weather advisory
    if weather.get("rain_probability_7d", 0) > 60:
        weather_advisory = "अगले 7 दिनों में बारिश संभव है। सिंचाई टालें।"
    elif needs_irrigation:
        weather_advisory = "सिंचाई ज़रूरी है। तुरंत करें।"
    else:
        weather_advisory = f"{days_until_needed} दिन बाद सिंचाई करें।"

    from datetime import timedelta
    next_check = date.today() + timedelta(days=max(1, days_until_needed))

    method = kb.get("preferred_irrigation", "flood")

    return IrrigationPlan(
        crop=request.crop,
        current_moisture_pct=request.soil_moisture,
        needs_irrigation=needs_irrigation,
        recommended_in_days=days_until_needed,
        water_mm_per_hectare=round(water_mm, 1),
        method=method,
        weather_advisory=weather_advisory,
        next_check_date=next_check,
    )


@app.get("/advisory/daily/{farmer_id}")
async def get_daily_advisory(
    farmer_id: str,
    db: asyncpg.Pool = Depends(lambda: app_state["db"]),
    cache: aioredis.Redis = Depends(lambda: app_state["redis"]),
):
    """
    Personalised daily advisory for a farmer.
    Combines weather alert, crop stage advisory, and market tip.
    Cached per farmer for 6 hours.
    """
    cache_key = f"daily_advisory:{farmer_id}:{date.today()}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Fetch farmer context from farmer service
    try:
        resp = await app_state["http"].get(
            f"{FARMER_SERVICE_URL}/farmers/{farmer_id}/context"
        )
        ctx = resp.json()
    except Exception:
        raise HTTPException(503, "Could not fetch farmer context")

    district = ctx.get("district", "")
    state = ctx.get("state", "")
    language = ctx.get("preferred_language", "hi")
    current_crops = ctx.get("current_crops", [])

    weather = await get_weather_context(district, state)

    advisories = []

    # 1. Weather alert
    if weather.get("rain_probability_7d", 0) > 70:
        advisories.append({
            "type": "weather",
            "priority": "high",
            "title": "बारिश की चेतावनी ⛈️",
            "message": "अगले 48 घंटों में भारी बारिश संभव। कटाई रोकें, फ़सल सुरक्षित करें।",
            "icon": "rain",
        })

    # 2. Crop-specific advisory
    for crop in current_crops[:2]:
        kb = app_state["agri_kb"].get(crop, {})
        month = datetime.now().month
        seasonal_tip = kb.get("monthly_tips", {}).get(str(month))
        if seasonal_tip:
            advisories.append({
                "type": "crop_stage",
                "priority": "medium",
                "title": f"{crop.title()} सलाह 🌾",
                "message": seasonal_tip,
                "icon": "crop",
            })

    # 3. Market tip
    for crop in current_crops[:1]:
        try:
            market = await get_market_context(crop, district)
            if market.get("signal") in ("SELL_NOW", "SELL"):
                advisories.append({
                    "type": "market",
                    "priority": "high",
                    "title": f"{crop.title()} बेचें 💰",
                    "message": market.get("signal_reason", "भाव अच्छा है।"),
                    "icon": "market",
                })
        except Exception:
            pass

    result = {
        "farmer_id": farmer_id,
        "date": str(date.today()),
        "language": language,
        "advisories": advisories or [{
            "type": "general",
            "priority": "low",
            "title": "आज कोई विशेष सलाह नहीं",
            "message": "फ़सल की निगरानी जारी रखें। कोई समस्या हो तो फ़ोटो भेजें।",
            "icon": "info",
        }],
        "generated_at": datetime.utcnow().isoformat(),
    }

    await cache.setex(cache_key, 21600, json.dumps(result, default=str))  # 6h TTL
    return result


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "crop-advisory-service", "version": "1.0.0"}
