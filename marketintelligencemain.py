"""
SmartKhet — Market Intelligence Service
========================================
Aggregates Agmarknet mandi data, runs price prediction,
and provides sell/hold signals to farmers.

Endpoints:
  GET  /market/price/{commodity}       — current + 7-day predicted price
  GET  /market/mandis/{district}       — nearby mandis + their today's prices
  GET  /market/sell-signal/{commodity} — AI sell recommendation
  GET  /market/top-prices              — highest paying mandis today
  POST /market/alert                   — subscribe to price alert

Data Sources:
  - Agmarknet API (real-time arrival + prices)
  - eNAM (National Agriculture Market)
  - Custom scraper for regional mandis not on Agmarknet

Author: Axora / SmartKhet ML Team
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta, date
from typing import Optional

import httpx
import redis.asyncio as aioredis
import asyncpg
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Config ────────────────────────────────────────────────────────────────────

DB_URL = os.getenv("POSTGRES_URL", "postgresql://localhost/smartkhet")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
AGMARKNET_API = os.getenv("AGMARKNET_API_URL", "https://agmarknet.gov.in/api")
AGMARKNET_API_KEY = os.getenv("AGMARKNET_API_KEY", "")
ML_SERVICE_URL = os.getenv("ML_INFERENCE_URL", "http://ml-service:8080")

# Cache TTLs (seconds)
TTL_CURRENT_PRICE = 900      # 15 minutes
TTL_PREDICTION = 3600        # 1 hour
TTL_MANDI_LIST = 86400       # 24 hours


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class MandiPrice(BaseModel):
    mandi_name: str
    district: str
    state: str
    commodity: str
    variety: Optional[str]
    min_price: float
    max_price: float
    modal_price: float
    arrival_tonnes: Optional[float]
    date: date

class PriceForecast(BaseModel):
    commodity: str
    district: str
    current_price: float
    predicted_price_7d: float
    predicted_price_14d: float
    predicted_price_30d: float
    trend: str        # "rising" | "falling" | "stable"
    trend_strength: str  # "strong" | "moderate" | "weak"
    forecast_generated_at: datetime

class SellSignal(BaseModel):
    commodity: str
    district: str
    signal: str     # "SELL_NOW" | "SELL" | "HOLD" | "HOLD_FOR_MSP"
    reason: str
    current_price: float
    predicted_price: float
    msp: Optional[float]
    best_mandi: Optional[str]
    best_mandi_price: Optional[float]
    confidence: float
    valid_until: datetime

class PriceAlertRequest(BaseModel):
    farmer_id: str
    commodity: str
    district: str
    alert_above_price: Optional[float] = None
    alert_below_price: Optional[float] = None
    phone: str


# ── App Lifecycle ──────────────────────────────────────────────────────────────

app_state: dict = {}

@asynccontextmanager
async def lifespan(app):
    app_state["db"] = await asyncpg.create_pool(
        DB_URL.replace("postgresql://", "postgres://"), min_size=3, max_size=15
    )
    app_state["redis"] = await aioredis.from_url(REDIS_URL, decode_responses=True)
    app_state["http"] = httpx.AsyncClient(timeout=15.0)

    # Load MSP data
    msp_path = "data/seeds/msp_2024_25.json"
    if os.path.exists(msp_path):
        with open(msp_path) as f:
            app_state["msp"] = json.load(f)
    else:
        app_state["msp"] = {}

    log.info("Market Intelligence Service ready ✅")
    yield
    await app_state["db"].close()
    await app_state["redis"].close()
    await app_state["http"].aclose()


app = FastAPI(title="SmartKhet Market Intelligence Service", version="1.0.0",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Agmarknet API Client ──────────────────────────────────────────────────────

async def fetch_agmarknet_prices(commodity: str, state: str = None,
                                  district: str = None) -> list[dict]:
    """
    Fetch today's prices from Agmarknet API.
    Falls back to database (yesterday's prices) if API is down.
    """
    cache = app_state["redis"]
    cache_key = f"agmarknet:{commodity}:{state}:{district}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    http = app_state["http"]
    params = {
        "commodity": commodity,
        "api_key": AGMARKNET_API_KEY,
        "format": "json",
        "date": datetime.now().strftime("%d-%b-%Y"),
    }
    if state:
        params["state"] = state
    if district:
        params["district"] = district

    try:
        resp = await http.get(f"{AGMARKNET_API}/mandi-prices", params=params)
        resp.raise_for_status()
        data = resp.json().get("records", [])
        await cache.setex(cache_key, TTL_CURRENT_PRICE, json.dumps(data))
        return data
    except Exception as e:
        log.error(f"Agmarknet API failed: {e}. Using DB fallback.")
        return await _get_prices_from_db(commodity, state, district)


async def _get_prices_from_db(commodity: str, state: str = None,
                               district: str = None) -> list[dict]:
    """Fallback to most recent prices in our DB when Agmarknet is down."""
    db = app_state["db"]
    query = """
        SELECT mandi_name, district, state, commodity, variety,
               min_price, max_price, modal_price, arrival_tonnes, price_date
        FROM mandi_prices
        WHERE commodity = $1 AND price_date >= CURRENT_DATE - INTERVAL '3 days'
    """
    params = [commodity]
    if state:
        query += " AND state = $2"
        params.append(state)
    if district:
        query += f" AND district = ${len(params)+1}"
        params.append(district)
    query += " ORDER BY price_date DESC LIMIT 50"

    async with db.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [dict(r) for r in rows]


async def get_price_prediction(commodity: str, district: str,
                                horizon: int = 7) -> dict:
    """Call ML service for price forecast."""
    cache = app_state["redis"]
    cache_key = f"price_pred:{commodity}:{district}:{horizon}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    http = app_state["http"]
    try:
        resp = await http.get(
            f"{ML_SERVICE_URL}/v1/market/predict",
            params={"commodity": commodity, "district": district, "horizon": horizon}
        )
        resp.raise_for_status()
        result = resp.json()
        await cache.setex(cache_key, TTL_PREDICTION, json.dumps(result))
        return result
    except Exception as e:
        log.warning(f"ML price prediction failed: {e}")
        return {}


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/market/price/{commodity}", response_model=PriceForecast)
async def get_price_with_forecast(
    commodity: str,
    district: str = Query(...),
    state: str = Query(default=None),
):
    """
    Get current mandi price + 7/14/30-day forecast for a commodity in a district.
    """
    commodity = commodity.lower().strip()

    # Get current prices
    raw_prices = await fetch_agmarknet_prices(commodity, state, district)
    if not raw_prices:
        raise HTTPException(404, f"No price data found for {commodity} in {district}")

    # Average modal price across mandis in district
    modal_prices = [p.get("modal_price", 0) or p.get("Modal_Price", 0)
                    for p in raw_prices if p.get("modal_price") or p.get("Modal_Price")]
    current_price = sum(modal_prices) / len(modal_prices) if modal_prices else 0

    # Get predictions
    pred_7d = await get_price_prediction(commodity, district, 7)
    pred_14d = await get_price_prediction(commodity, district, 14)
    pred_30d = await get_price_prediction(commodity, district, 30)

    predicted_7d = pred_7d.get("predicted_price", current_price * 1.02)
    predicted_14d = pred_14d.get("predicted_price", current_price * 1.03)
    predicted_30d = pred_30d.get("predicted_price", current_price * 1.05)

    # Trend calculation
    price_change = ((predicted_7d - current_price) / current_price) * 100
    if price_change > 3:
        trend, strength = "rising", "strong" if price_change > 8 else "moderate"
    elif price_change < -3:
        trend, strength = "falling", "strong" if price_change < -8 else "moderate"
    else:
        trend, strength = "stable", "weak"

    return PriceForecast(
        commodity=commodity,
        district=district,
        current_price=round(current_price, 2),
        predicted_price_7d=round(predicted_7d, 2),
        predicted_price_14d=round(predicted_14d, 2),
        predicted_price_30d=round(predicted_30d, 2),
        trend=trend,
        trend_strength=strength,
        forecast_generated_at=datetime.utcnow(),
    )


@app.get("/market/mandis/{district}", response_model=list[MandiPrice])
async def get_mandis_in_district(
    district: str,
    commodity: str = Query(...),
    state: str = Query(default=None),
):
    """List all mandis in a district with today's prices for a commodity."""
    raw = await fetch_agmarknet_prices(commodity, state, district)
    result = []
    for r in raw:
        try:
            result.append(MandiPrice(
                mandi_name=r.get("Mandi_Name", r.get("mandi_name", "Unknown")),
                district=r.get("District_Name", district),
                state=r.get("State_Name", state or ""),
                commodity=r.get("Commodity", commodity),
                variety=r.get("Variety"),
                min_price=float(r.get("Min_Price", r.get("min_price", 0))),
                max_price=float(r.get("Max_Price", r.get("max_price", 0))),
                modal_price=float(r.get("Modal_Price", r.get("modal_price", 0))),
                arrival_tonnes=float(r["Arrivals_Tonnes"]) if r.get("Arrivals_Tonnes") else None,
                date=datetime.now().date(),
            ))
        except (ValueError, KeyError):
            continue

    # Sort by modal price descending (best price first)
    return sorted(result, key=lambda x: x.modal_price, reverse=True)


@app.get("/market/sell-signal/{commodity}", response_model=SellSignal)
async def get_sell_signal(
    commodity: str,
    district: str = Query(...),
    state: str = Query(default=None),
):
    """
    AI-powered sell/hold recommendation combining current price,
    forecast, MSP, and seasonal patterns.
    """
    commodity = commodity.lower()

    # Get current + forecast
    try:
        forecast = await get_price_with_forecast(commodity, district=district, state=state)
    except HTTPException:
        raise HTTPException(404, f"Cannot generate sell signal — no price data for {commodity}")

    current_price = forecast.current_price
    predicted_7d = forecast.predicted_price_7d
    msp = app_state["msp"].get(commodity)
    price_change_pct = ((predicted_7d - current_price) / current_price) * 100

    # Get best mandi in district
    mandis = await get_mandis_in_district(district, commodity=commodity, state=state)
    best_mandi = mandis[0] if mandis else None

    # Signal logic
    if price_change_pct > 5:
        signal = "HOLD"
        reason = f"{commodity.title()} ka daam agli 7 dinon mein {price_change_pct:.1f}% badhne ka anuman hai. Abhi rukein."
        confidence = min(0.85, 0.5 + abs(price_change_pct) / 100)
    elif price_change_pct < -5:
        signal = "SELL_NOW"
        reason = f"Bhav girane ki sambhavana hai ({price_change_pct:.1f}%). Abhi bechna behtar hai."
        confidence = min(0.85, 0.5 + abs(price_change_pct) / 100)
    elif msp and current_price < msp:
        signal = "HOLD_FOR_MSP"
        reason = f"Bhav ₹{current_price:.0f} MSP ₹{msp:.0f} se kam hai. FCI ya cooperative mein bechne ka prayaas karein."
        confidence = 0.90
    else:
        signal = "SELL"
        reason = f"Bhav thheek hai. {best_mandi.mandi_name if best_mandi else district} mandi mein bechna uchit hai."
        confidence = 0.70

    return SellSignal(
        commodity=commodity,
        district=district,
        signal=signal,
        reason=reason,
        current_price=current_price,
        predicted_price=predicted_7d,
        msp=msp,
        best_mandi=best_mandi.mandi_name if best_mandi else None,
        best_mandi_price=best_mandi.modal_price if best_mandi else None,
        confidence=confidence,
        valid_until=datetime.utcnow() + timedelta(hours=24),
    )


@app.get("/market/top-prices")
async def get_top_prices(
    commodity: str = Query(...),
    state: str = Query(default=None),
    limit: int = Query(default=10, le=50),
):
    """Top-paying mandis across the state for a commodity today."""
    raw = await fetch_agmarknet_prices(commodity, state=state)
    top = sorted(raw, key=lambda x: float(x.get("Modal_Price", x.get("modal_price", 0))),
                 reverse=True)[:limit]
    return {"commodity": commodity, "top_mandis": top, "fetched_at": datetime.utcnow()}


@app.post("/market/alert")
async def subscribe_price_alert(request: PriceAlertRequest,
                                 db: asyncpg.Pool = Depends(lambda: app_state["db"])):
    """Subscribe to price alert — notified via SMS/WhatsApp when price crosses threshold."""
    async with db.acquire() as conn:
        await conn.execute("""
            INSERT INTO price_alerts
              (farmer_id, commodity, district, alert_above, alert_below, phone, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,NOW())
            ON CONFLICT (farmer_id, commodity, district)
            DO UPDATE SET alert_above=$4, alert_below=$5, updated_at=NOW()
        """,
            request.farmer_id, request.commodity, request.district,
            request.alert_above_price, request.alert_below_price, request.phone,
        )
    return {"message": f"Price alert set for {request.commodity} in {request.district}"}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "market-service", "version": "1.0.0"}
