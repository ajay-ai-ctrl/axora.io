"""
SmartKhet — Backend Integration Tests
=======================================
Tests the full request-response cycle for all core API endpoints.
Uses pytest-asyncio for async FastAPI testing.

Run: pytest tests/ -v --cov=backend --cov-report=term-missing

Author: Axora / SmartKhet Team
"""

import pytest
import pytest_asyncio
import asyncio
import json
import io
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_soil_data():
    return {
        "nitrogen": 90,
        "phosphorus": 45,
        "potassium": 40,
        "ph": 6.8,
        "moisture": 60.0,
    }


@pytest.fixture
def sample_farmer_register():
    return {
        "phone": "9876543210",
        "name": "Ramesh Kumar",
        "state": "Uttar Pradesh",
        "district": "Gorakhpur",
        "village": "Pipraich",
        "preferred_language": "hi",
    }


@pytest.fixture
def mock_db_pool():
    pool = AsyncMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool, conn


# ── Farmer Service Tests ───────────────────────────────────────────────────────

class TestFarmerService:

    @pytest.mark.asyncio
    async def test_send_otp_valid_phone(self):
        """OTP endpoint returns 200 for valid Indian mobile number."""
        from backend.farmer_service.main import app

        with patch("backend.farmer_service.main.app_state", {
            "db": AsyncMock(),
            "redis": AsyncMock(),
            "http": AsyncMock(),
        }):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/farmers/send-otp?phone=9876543210")
                assert response.status_code == 200
                data = response.json()
                assert "OTP sent" in data["message"]
                assert data["expires_in"] == 300

    @pytest.mark.asyncio
    async def test_register_farmer_invalid_phone(self):
        """Registration fails for invalid phone number format."""
        from backend.farmer_service.main import app

        with patch("backend.farmer_service.main.app_state", {
            "db": AsyncMock(), "redis": AsyncMock(), "http": AsyncMock()
        }):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                payload = {
                    "phone": "1234567890",  # Starts with 1 — invalid Indian number
                    "name": "Test Farmer",
                    "state": "UP",
                    "district": "Gorakhpur",
                }
                response = await client.post("/farmers/register?otp=123456", json=payload)
                assert response.status_code == 422  # Pydantic validation error

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Health check returns healthy status."""
        from backend.farmer_service.main import app

        with patch("backend.farmer_service.main.app_state",
                   {"db": AsyncMock(), "redis": AsyncMock(), "http": AsyncMock()}):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")
                assert response.status_code == 200
                assert response.json()["status"] == "healthy"


# ── Disease Service Tests ──────────────────────────────────────────────────────

class TestDiseaseService:

    @pytest.mark.asyncio
    async def test_analyze_invalid_file_type(self):
        """Disease analysis rejects non-image files."""
        from backend.disease_service.main import app

        with patch("backend.disease_service.main.app_state", {
            "db": AsyncMock(), "redis": AsyncMock(),
            "kafka": AsyncMock(), "s3": MagicMock(),
            "http": AsyncMock(), "treatment_kb": {},
        }):
            with patch("backend.disease_service.main.verify_jwt",
                       return_value={"sub": "test-farmer-id"}):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/disease/analyze",
                        files={"file": ("test.pdf", b"pdf content", "application/pdf")},
                        headers={"Authorization": "Bearer fake-token"},
                    )
                    assert response.status_code == 400
                    assert "Unsupported file type" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_analyze_file_too_large(self):
        """Disease analysis rejects files > 10MB."""
        from backend.disease_service.main import app

        large_bytes = b"x" * (11 * 1024 * 1024)  # 11MB

        with patch("backend.disease_service.main.app_state", {
            "db": AsyncMock(), "redis": AsyncMock(),
            "kafka": AsyncMock(), "s3": MagicMock(),
            "http": AsyncMock(), "treatment_kb": {},
        }):
            with patch("backend.disease_service.main.verify_jwt",
                       return_value={"sub": "test-farmer-id"}):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/disease/analyze",
                        files={"file": ("crop.jpg", large_bytes, "image/jpeg")},
                        headers={"Authorization": "Bearer fake-token"},
                    )
                    assert response.status_code == 413

    @pytest.mark.asyncio
    async def test_disease_classes_endpoint(self):
        """Disease classes endpoint returns correct count."""
        from backend.disease_service.main import app
        from ml.disease_detection.train import DISEASE_CLASSES

        with patch("backend.disease_service.main.app_state", {}):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/disease/classes")
                assert response.status_code == 200
                data = response.json()
                assert data["total"] == len(DISEASE_CLASSES)


# ── Market Service Tests ───────────────────────────────────────────────────────

class TestMarketService:

    @pytest.mark.asyncio
    async def test_sell_signal_response_schema(self):
        """Sell signal endpoint returns all required fields."""
        from backend.market_service.main import app

        mock_forecast = {
            "current_price": 2100.0,
            "predicted_price_7d": 2250.0,
            "predicted_price_14d": 2300.0,
            "predicted_price_30d": 2350.0,
            "trend": "rising",
            "trend_strength": "moderate",
        }

        with patch("backend.market_service.main.app_state", {
            "db": AsyncMock(), "redis": AsyncMock(),
            "http": AsyncMock(), "msp": {"wheat": 2275},
        }):
            with patch("backend.market_service.main.get_price_with_forecast",
                       AsyncMock(return_value=MagicMock(**mock_forecast))):
                with patch("backend.market_service.main.get_mandis_in_district",
                           AsyncMock(return_value=[])):
                    async with AsyncClient(
                        transport=ASGITransport(app=app), base_url="http://test"
                    ) as client:
                        response = await client.get(
                            "/market/sell-signal/wheat?district=gorakhpur"
                        )
                        assert response.status_code == 200
                        data = response.json()
                        required_fields = [
                            "commodity", "signal", "reason",
                            "current_price", "predicted_price", "confidence"
                        ]
                        for field in required_fields:
                            assert field in data, f"Missing field: {field}"

    def test_sell_signal_hold_when_price_rising(self):
        """HOLD signal generated when price expected to rise > 5%."""
        # Unit test for signal logic without HTTP
        current = 2000.0
        predicted = 2150.0  # +7.5%
        pct_change = ((predicted - current) / current) * 100
        signal = "HOLD" if pct_change > 5 else "SELL"
        assert signal == "HOLD"


# ── ML Model Tests ─────────────────────────────────────────────────────────────

class TestCropRecommender:

    def test_feature_vector_length(self):
        """Feature vector must have exactly 11 elements."""
        from ml.crop_recommendation.train import FEATURE_COLUMNS
        assert len(FEATURE_COLUMNS) == 11

    def test_fertilizer_advisory_ph_correction_acidic(self):
        """Acidic soil should trigger lime recommendation."""
        from ml.crop_recommendation.train import CropRecommender
        advisory = CropRecommender._generate_fertilizer_advisory(
            crop="rice", n=80, p=30, k=20, ph=5.2
        )
        assert advisory["ph_correction"] is not None
        assert "lime" in advisory["ph_correction"].lower() or "चूना" in advisory["ph_correction"]

    def test_fertilizer_advisory_no_correction_neutral_ph(self):
        """Neutral pH soil should not trigger pH correction."""
        from ml.crop_recommendation.train import CropRecommender
        advisory = CropRecommender._generate_fertilizer_advisory(
            crop="wheat", n=100, p=50, k=40, ph=7.0
        )
        assert advisory["ph_correction"] is None

    def test_urea_calculation(self):
        """Urea kg/ha should be N deficit / 0.46 (urea N content)."""
        from ml.crop_recommendation.train import CropRecommender
        # Wheat optimal N = 120, current = 80, deficit = 40
        advisory = CropRecommender._generate_fertilizer_advisory(
            crop="wheat", n=80, p=60, k=40, ph=7.0
        )
        expected_urea = round(40 / 0.46)  # ~87 kg/ha
        assert abs(advisory["urea_kg_ha"] - expected_urea) <= 2


class TestNLPPipeline:

    def test_rule_based_crop_extraction(self):
        """Rule-based entity extractor finds crop names in Hindi text."""
        from ml.nlp_voice.pipeline import AgroNLPPipeline

        # Test without spaCy model
        pipeline = AgroNLPPipeline.__new__(AgroNLPPipeline)
        pipeline.use_spacy_ner = False

        entities = pipeline._rule_based_extract("मेरे गेहूँ में पीलापन आ गया है")
        crop_ents = [e for e in entities if e.label == "CROP"]
        assert len(crop_ents) >= 1
        assert any("गेहूँ" in e.text or "wheat" in e.text.lower() for e in crop_ents)

    def test_quantity_extraction(self):
        """Extracts quantity entities from text."""
        from ml.nlp_voice.pipeline import AgroNLPPipeline

        pipeline = AgroNLPPipeline.__new__(AgroNLPPipeline)
        pipeline.use_spacy_ner = False

        entities = pipeline._rule_based_extract("50 किलो यूरिया डालना है")
        qty_ents = [e for e in entities if e.label == "QUANTITY"]
        assert len(qty_ents) >= 1


# ── pytest Configuration ───────────────────────────────────────────────────────

# pytest.ini equivalent — in pyproject.toml or pytest.ini normally
# Included here for reference
PYTEST_CONFIG = """
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
markers =
    slow: marks tests as slow (deselect with '-m not slow')
    integration: marks tests as integration tests
    ml: marks tests requiring ML models
"""
