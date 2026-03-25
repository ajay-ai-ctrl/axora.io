-- SmartKhet — PostgreSQL Database Schema
-- =========================================
-- Version: 1.0.0
-- Database: PostgreSQL 16 + PostGIS extension
-- Author: Axora / SmartKhet Team

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fuzzy text search
CREATE EXTENSION IF NOT EXISTS btree_gin; -- compound GIN indexes

-- ══════════════════════════════════════════════════════
-- FARMERS
-- ══════════════════════════════════════════════════════

CREATE TABLE farmers (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone               VARCHAR(15) UNIQUE NOT NULL,
    name                VARCHAR(120) NOT NULL,
    state               VARCHAR(60) NOT NULL,
    district            VARCHAR(60) NOT NULL,
    village             VARCHAR(100),
    pincode             CHAR(6),
    preferred_language  CHAR(5) NOT NULL DEFAULT 'hi',
    profile_complete    BOOLEAN DEFAULT FALSE,
    is_active           BOOLEAN DEFAULT TRUE,
    last_active         TIMESTAMPTZ,
    referral_code       VARCHAR(20),
    referred_by         UUID REFERENCES farmers(id),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_farmers_phone     ON farmers(phone);
CREATE INDEX idx_farmers_district  ON farmers(state, district);
CREATE INDEX idx_farmers_pincode   ON farmers(pincode);

-- ══════════════════════════════════════════════════════
-- LAND PARCELS
-- ══════════════════════════════════════════════════════

CREATE TABLE land_parcels (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farmer_id       UUID NOT NULL REFERENCES farmers(id) ON DELETE CASCADE,
    area_acres      NUMERIC(8, 3) NOT NULL CHECK (area_acres > 0),
    soil_type       VARCHAR(30) CHECK (soil_type IN
                      ('alluvial', 'black', 'red', 'laterite',
                       'sandy', 'loamy', 'clayey', 'saline', 'peaty')),
    irrigation_type VARCHAR(30) CHECK (irrigation_type IN
                      ('canal', 'borewell', 'rainfed', 'drip', 'sprinkler', 'tube_well')),
    geom            GEOMETRY(POINT, 4326),  -- WGS84 GPS coordinates
    khasra_number   VARCHAR(50),
    survey_number   VARCHAR(50),
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_parcels_farmer ON land_parcels(farmer_id);
CREATE INDEX idx_parcels_geom   ON land_parcels USING GIST(geom);

-- ══════════════════════════════════════════════════════
-- SOIL READINGS
-- ══════════════════════════════════════════════════════

CREATE TABLE soil_readings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farmer_id       UUID NOT NULL REFERENCES farmers(id),
    parcel_id       UUID REFERENCES land_parcels(id),
    nitrogen        NUMERIC(8, 2),      -- kg/ha
    phosphorus      NUMERIC(8, 2),      -- kg/ha
    potassium       NUMERIC(8, 2),      -- kg/ha
    ph              NUMERIC(4, 2) CHECK (ph BETWEEN 1 AND 14),
    moisture_pct    NUMERIC(5, 2) CHECK (moisture_pct BETWEEN 0 AND 100),
    ec_ms_cm        NUMERIC(6, 3),      -- Electrical conductivity
    organic_carbon  NUMERIC(5, 3),      -- % organic carbon
    source          VARCHAR(30) DEFAULT 'manual',  -- 'manual' | 'iot_sensor' | 'lab'
    lab_report_url  TEXT,
    geom            GEOMETRY(POINT, 4326),
    measured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    synced_at       TIMESTAMPTZ
);

CREATE INDEX idx_soil_farmer  ON soil_readings(farmer_id);
CREATE INDEX idx_soil_time    ON soil_readings(measured_at DESC);
CREATE INDEX idx_soil_geom    ON soil_readings USING GIST(geom);

-- ══════════════════════════════════════════════════════
-- CROP HISTORY
-- ══════════════════════════════════════════════════════

CREATE TABLE crop_history (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farmer_id           UUID NOT NULL REFERENCES farmers(id),
    parcel_id           UUID REFERENCES land_parcels(id),
    crop                VARCHAR(60) NOT NULL,
    variety             VARCHAR(80),
    season              VARCHAR(20) NOT NULL CHECK (season IN ('kharif', 'rabi', 'zaid', 'perennial')),
    year                SMALLINT NOT NULL CHECK (year BETWEEN 2000 AND 2100),
    sowing_date         DATE,
    harvest_date        DATE,
    area_acres          NUMERIC(8, 3),
    yield_qtl_per_acre  NUMERIC(8, 3),
    total_yield_qtl     NUMERIC(10, 3),
    sold_price_per_qtl  NUMERIC(10, 2),
    sold_at_mandi       VARCHAR(100),
    total_income        NUMERIC(12, 2),
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_crop_history_farmer  ON crop_history(farmer_id);
CREATE INDEX idx_crop_history_crop    ON crop_history(crop, season, year);

-- ══════════════════════════════════════════════════════
-- DISEASE ANALYSES
-- ══════════════════════════════════════════════════════

CREATE TABLE disease_analyses (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farmer_id       UUID NOT NULL REFERENCES farmers(id),
    parcel_id       UUID REFERENCES land_parcels(id),
    image_s3_key    TEXT,
    image_url       TEXT,
    disease_label   VARCHAR(100) NOT NULL,
    crop            VARCHAR(60),
    confidence      NUMERIC(5, 4) CHECK (confidence BETWEEN 0 AND 1),
    severity        VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    is_healthy      BOOLEAN NOT NULL DEFAULT FALSE,
    model_version   VARCHAR(30),
    geom            GEOMETRY(POINT, 4326),
    advisory_text   TEXT,
    treatment_json  JSONB,
    analyzed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    synced_at       TIMESTAMPTZ
);

CREATE INDEX idx_disease_farmer  ON disease_analyses(farmer_id);
CREATE INDEX idx_disease_label   ON disease_analyses(disease_label);
CREATE INDEX idx_disease_time    ON disease_analyses(analyzed_at DESC);
CREATE INDEX idx_disease_geom    ON disease_analyses USING GIST(geom);

-- ══════════════════════════════════════════════════════
-- MANDI PRICES (historical store)
-- ══════════════════════════════════════════════════════

CREATE TABLE mandi_prices (
    id              BIGSERIAL PRIMARY KEY,
    mandi_name      VARCHAR(120) NOT NULL,
    district        VARCHAR(80) NOT NULL,
    state           VARCHAR(60) NOT NULL,
    commodity       VARCHAR(80) NOT NULL,
    variety         VARCHAR(80),
    min_price       NUMERIC(10, 2),
    max_price       NUMERIC(10, 2),
    modal_price     NUMERIC(10, 2) NOT NULL,
    arrivals_tonnes NUMERIC(10, 3),
    price_date      DATE NOT NULL,
    source          VARCHAR(20) DEFAULT 'agmarknet',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_mandi_unique ON mandi_prices(mandi_name, commodity, price_date);
CREATE INDEX idx_mandi_commodity_district ON mandi_prices(commodity, district, price_date DESC);
CREATE INDEX idx_mandi_date ON mandi_prices(price_date DESC);

-- ══════════════════════════════════════════════════════
-- PRICE ALERTS
-- ══════════════════════════════════════════════════════

CREATE TABLE price_alerts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farmer_id       UUID NOT NULL REFERENCES farmers(id),
    commodity       VARCHAR(80) NOT NULL,
    district        VARCHAR(80) NOT NULL,
    alert_above     NUMERIC(10, 2),
    alert_below     NUMERIC(10, 2),
    phone           VARCHAR(15) NOT NULL,
    is_active       BOOLEAN DEFAULT TRUE,
    last_triggered  TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (farmer_id, commodity, district)
);

-- ══════════════════════════════════════════════════════
-- ADVISORY LOG
-- ══════════════════════════════════════════════════════

CREATE TABLE advisory_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farmer_id       UUID NOT NULL REFERENCES farmers(id),
    channel         VARCHAR(20) NOT NULL CHECK (channel IN
                      ('app', 'whatsapp', 'sms', 'ivr', 'push')),
    advisory_type   VARCHAR(50) NOT NULL,
    content         TEXT,
    content_lang    CHAR(5) DEFAULT 'hi',
    was_read        BOOLEAN DEFAULT FALSE,
    feedback        SMALLINT CHECK (feedback BETWEEN 1 AND 5),
    delivered_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_advisory_farmer ON advisory_logs(farmer_id);
CREATE INDEX idx_advisory_type   ON advisory_logs(advisory_type);

-- ══════════════════════════════════════════════════════
-- TRIGGERS — auto-update updated_at
-- ══════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER farmers_updated_at
  BEFORE UPDATE ON farmers
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER alerts_updated_at
  BEFORE UPDATE ON price_alerts
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ══════════════════════════════════════════════════════
-- VIEWS
-- ══════════════════════════════════════════════════════

CREATE VIEW farmer_summary AS
SELECT
    f.id,
    f.name,
    f.phone,
    f.district,
    f.state,
    f.preferred_language,
    COALESCE(SUM(lp.area_acres), 0) AS total_land_acres,
    COUNT(DISTINCT da.id) AS total_disease_analyses,
    COUNT(DISTINCT ch.id) AS total_crop_records,
    MAX(al.delivered_at) AS last_advisory_at,
    f.created_at AS member_since
FROM farmers f
LEFT JOIN land_parcels lp ON lp.farmer_id = f.id AND lp.is_active
LEFT JOIN disease_analyses da ON da.farmer_id = f.id
LEFT JOIN crop_history ch ON ch.farmer_id = f.id
LEFT JOIN advisory_logs al ON al.farmer_id = f.id
GROUP BY f.id;

CREATE VIEW latest_mandi_prices AS
SELECT DISTINCT ON (commodity, district)
    commodity, district, state, mandi_name,
    modal_price, min_price, max_price,
    arrivals_tonnes, price_date
FROM mandi_prices
ORDER BY commodity, district, price_date DESC;
