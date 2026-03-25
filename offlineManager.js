/**
 * SmartKhet Mobile — Offline Manager
 * =====================================
 * Manages local SQLite database and on-device TFLite inference
 * for completely offline operation when internet is unavailable.
 *
 * Capabilities:
 *  - Offline crop recommendation (TFLite XGBoost)
 *  - Offline disease detection (TFLite EfficientNet-B4)
 *  - Local data store (farmer context, advisories, mandi cache)
 *  - Background sync with server when internet restores
 *
 * Dependencies:
 *  - expo-sqlite (SQLite on Android)
 *  - react-native-fast-tflite (TFLite inference)
 *  - @react-native-community/netinfo (connectivity detection)
 */

import * as SQLite from "expo-sqlite";
import { loadTensorflowModel } from "react-native-fast-tflite";
import NetInfo from "@react-native-community/netinfo";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";

// ── Database Setup ─────────────────────────────────────────────────────────────

const DB_NAME = "smartkhet_local.db";
const DB_VERSION = 3;

let _db = null;

export async function initDatabase() {
  _db = await SQLite.openDatabaseAsync(DB_NAME);
  await runMigrations(_db);
  return _db;
}

export function getDb() {
  if (!_db) throw new Error("Database not initialised. Call initDatabase() first.");
  return _db;
}

async function runMigrations(db) {
  await db.execAsync(`
    CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);

    CREATE TABLE IF NOT EXISTS farmer_profile (
      id          TEXT PRIMARY KEY,
      name        TEXT NOT NULL,
      phone       TEXT NOT NULL,
      district    TEXT,
      state       TEXT,
      language    TEXT DEFAULT 'hi',
      land_acres  REAL DEFAULT 0,
      synced_at   INTEGER,
      updated_at  INTEGER DEFAULT (strftime('%s','now'))
    );

    CREATE TABLE IF NOT EXISTS soil_readings (
      id          TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(8)))),
      farmer_id   TEXT NOT NULL,
      nitrogen    REAL,
      phosphorus  REAL,
      potassium   REAL,
      ph          REAL,
      moisture    REAL,
      lat         REAL,
      lon         REAL,
      recorded_at INTEGER DEFAULT (strftime('%s','now')),
      synced      INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS advisory_cache (
      id          TEXT PRIMARY KEY,
      farmer_id   TEXT NOT NULL,
      type        TEXT NOT NULL,
      content_json TEXT NOT NULL,
      language    TEXT DEFAULT 'hi',
      expires_at  INTEGER,
      created_at  INTEGER DEFAULT (strftime('%s','now'))
    );

    CREATE TABLE IF NOT EXISTS mandi_price_cache (
      commodity   TEXT NOT NULL,
      district    TEXT NOT NULL,
      modal_price REAL,
      mandi_name  TEXT,
      signal      TEXT,
      cached_at   INTEGER DEFAULT (strftime('%s','now')),
      PRIMARY KEY (commodity, district)
    );

    CREATE TABLE IF NOT EXISTS analysis_queue (
      id          TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(8)))),
      type        TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      status      TEXT DEFAULT 'pending',
      created_at  INTEGER DEFAULT (strftime('%s','now')),
      synced_at   INTEGER
    );

    CREATE TABLE IF NOT EXISTS disease_history (
      id          TEXT PRIMARY KEY,
      farmer_id   TEXT NOT NULL,
      image_path  TEXT,
      disease     TEXT,
      crop        TEXT,
      confidence  REAL,
      is_healthy  INTEGER,
      analyzed_at INTEGER DEFAULT (strftime('%s','now')),
      advisory    TEXT,
      synced      INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_soil_farmer ON soil_readings(farmer_id);
    CREATE INDEX IF NOT EXISTS idx_advisory_farmer ON advisory_cache(farmer_id);
    CREATE INDEX IF NOT EXISTS idx_queue_status ON analysis_queue(status);
  `);
}

// ── CRUD Helpers ───────────────────────────────────────────────────────────────

export async function saveSoilReading(reading) {
  const db = getDb();
  await db.runAsync(
    `INSERT INTO soil_readings (farmer_id, nitrogen, phosphorus, potassium, ph, moisture, lat, lon)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      reading.farmerId,
      reading.nitrogen ?? null,
      reading.phosphorus ?? null,
      reading.potassium ?? null,
      reading.ph ?? null,
      reading.moisture ?? null,
      reading.lat ?? null,
      reading.lon ?? null,
    ]
  );
}

export async function getLatestSoilReading(farmerId) {
  const db = getDb();
  return db.getFirstAsync(
    "SELECT * FROM soil_readings WHERE farmer_id=? ORDER BY recorded_at DESC LIMIT 1",
    [farmerId]
  );
}

export async function cacheAdvisory(farmerId, type, content, ttlSeconds = 86400) {
  const db = getDb();
  const id = `${farmerId}_${type}_${Date.now()}`;
  const expiresAt = Math.floor(Date.now() / 1000) + ttlSeconds;
  await db.runAsync(
    `INSERT OR REPLACE INTO advisory_cache (id, farmer_id, type, content_json, expires_at)
     VALUES (?, ?, ?, ?, ?)`,
    [id, farmerId, type, JSON.stringify(content), expiresAt]
  );
}

export async function getCachedAdvisory(farmerId, type) {
  const db = getDb();
  const now = Math.floor(Date.now() / 1000);
  const row = await db.getFirstAsync(
    `SELECT content_json FROM advisory_cache
     WHERE farmer_id=? AND type=? AND expires_at > ?
     ORDER BY created_at DESC LIMIT 1`,
    [farmerId, type, now]
  );
  return row ? JSON.parse(row.content_json) : null;
}

export async function cacheMandiPrice(commodity, district, data) {
  const db = getDb();
  await db.runAsync(
    `INSERT OR REPLACE INTO mandi_price_cache
       (commodity, district, modal_price, mandi_name, signal)
     VALUES (?, ?, ?, ?, ?)`,
    [commodity, district, data.current_price, data.best_mandi, data.signal]
  );
}

export async function getCachedMandiPrice(commodity, district, maxAgeSeconds = 900) {
  const db = getDb();
  const minTime = Math.floor(Date.now() / 1000) - maxAgeSeconds;
  return db.getFirstAsync(
    `SELECT * FROM mandi_price_cache
     WHERE commodity=? AND district=? AND cached_at > ?`,
    [commodity, district, minTime]
  );
}

export async function queueForSync(type, payload) {
  const db = getDb();
  await db.runAsync(
    "INSERT INTO analysis_queue (type, payload_json) VALUES (?, ?)",
    [type, JSON.stringify(payload)]
  );
}

export async function getPendingQueue() {
  const db = getDb();
  return db.getAllAsync(
    "SELECT * FROM analysis_queue WHERE status='pending' ORDER BY created_at ASC LIMIT 20"
  );
}

export async function markQueueItemSynced(id) {
  const db = getDb();
  await db.runAsync(
    "UPDATE analysis_queue SET status='synced', synced_at=strftime('%s','now') WHERE id=?",
    [id]
  );
}


// ── TFLite Model Manager ───────────────────────────────────────────────────────

const MODELS = {
  disease: null,
  cropRecommender: null,
};

const MODEL_ASSETS = {
  disease: require("./models/disease_detector.tflite"),
  cropRecommender: require("./models/crop_recommender.tflite"),
};

export async function loadModels() {
  console.log("[TFLite] Loading on-device models...");
  try {
    MODELS.disease = await loadTensorflowModel(MODEL_ASSETS.disease);
    console.log("[TFLite] Disease detection model loaded ✅");
  } catch (e) {
    console.warn("[TFLite] Disease model load failed:", e.message);
  }

  try {
    MODELS.cropRecommender = await loadTensorflowModel(MODEL_ASSETS.cropRecommender);
    console.log("[TFLite] Crop recommender model loaded ✅");
  } catch (e) {
    console.warn("[TFLite] Crop recommender load failed:", e.message);
  }
}

// ── On-Device Disease Detection ────────────────────────────────────────────────

const DISEASE_CLASSES = [
  "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
  "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
  "Rice___Brown_spot", "Rice___Leaf_blast", "Rice___Neck_blast", "Rice___healthy",
  "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy",
  "Wheat___Brown_rust", "Wheat___Yellow_rust", "Wheat___healthy",
  // ... 50 total classes
];

/**
 * Run disease detection on a local image URI using on-device TFLite model.
 * Returns immediately without network — works fully offline.
 * @param {string} imageUri  - local file:// URI from camera/gallery
 * @returns {Promise<object>} prediction result
 */
export async function detectDiseaseOffline(imageUri) {
  if (!MODELS.disease) {
    return { error: "Disease model not loaded", offline: true };
  }

  // Preprocess image: resize to 224×224, normalise to [0,1]
  const imageData = await preprocessImageForTFLite(imageUri, 224, 224);

  // Run inference
  const start = Date.now();
  const output = await MODELS.disease.run([imageData]);
  const latency = Date.now() - start;

  const probs = Array.from(output[0]);
  const topIndices = probs
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3);

  const predictions = topIndices.map((item, rank) => {
    const label = DISEASE_CLASSES[item.i] ?? "Unknown";
    const [crop, condition] = label.split("___");
    return {
      rank: rank + 1,
      label,
      crop: crop ?? "Unknown",
      condition: (condition ?? "Unknown").replace(/_/g, " "),
      confidence: item.p,
      confidence_pct: `${(item.p * 100).toFixed(1)}%`,
      is_healthy: condition?.toLowerCase().includes("healthy") ?? false,
    };
  });

  return {
    top_predictions: predictions,
    primary: predictions[0],
    requires_treatment: !predictions[0].is_healthy,
    latency_ms: latency,
    source: "edge_tflite",
    offline: true,
  };
}

// ── On-Device Crop Recommendation ─────────────────────────────────────────────

const CROP_CLASSES = [
  "rice", "wheat", "maize", "chickpea", "kidneybeans",
  "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil",
  "pomegranate", "banana", "mango", "grapes", "watermelon",
  "muskmelon", "apple", "orange", "papaya", "coconut",
];

/**
 * Offline crop recommendation using on-device TFLite model.
 * @param {object} soilData - { nitrogen, phosphorus, potassium, ph, moisture,
 *                              temperature, humidity, rainfall }
 */
export async function recommendCropOffline(soilData) {
  if (!MODELS.cropRecommender) {
    return { error: "Crop model not loaded", offline: true };
  }

  const featureVector = new Float32Array([
    soilData.nitrogen ?? 0,
    soilData.phosphorus ?? 0,
    soilData.potassium ?? 0,
    soilData.temperature ?? 25,
    soilData.humidity ?? 60,
    soilData.ph ?? 7,
    soilData.rainfall ?? 800,
    soilData.moisture ?? 50,
    soilData.season ?? 0,      // 0=kharif, 1=rabi, 2=zaid
    soilData.district_enc ?? 0,
    soilData.market_demand ?? 0.5,
  ]);

  const output = await MODELS.cropRecommender.run([featureVector]);
  const probs = Array.from(output[0]);

  const top3 = probs
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3)
    .map((item, rank) => ({
      rank: rank + 1,
      crop: CROP_CLASSES[item.i] ?? "Unknown",
      confidence: item.p,
      confidence_pct: `${(item.p * 100).toFixed(1)}%`,
    }));

  return {
    top_crops: top3,
    primary_crop: top3[0]?.crop,
    source: "edge_tflite",
    offline: true,
  };
}

// ── Image Preprocessing ────────────────────────────────────────────────────────

async function preprocessImageForTFLite(imageUri, width, height) {
  /**
   * Resize image to width×height and normalise pixel values to [0,1].
   * Returns Float32Array of shape [width * height * 3] (RGB).
   * Uses expo-image-manipulator for resize, then manual normalisation.
   */
  const { manipulateAsync, SaveFormat } = require("expo-image-manipulator");

  const resized = await manipulateAsync(
    imageUri,
    [{ resize: { width, height } }],
    { format: SaveFormat.JPEG, base64: true }
  );

  // Decode base64 JPEG → pixel array
  const base64 = resized.base64;
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  // Simple pixel extraction (skip JPEG header, extract RGB values)
  // In production, use a proper JPEG decoder library
  const pixels = new Float32Array(width * height * 3);
  const IMAGENET_MEAN = [0.485, 0.456, 0.406];
  const IMAGENET_STD = [0.229, 0.224, 0.225];

  for (let i = 0; i < width * height; i++) {
    const byteOffset = i * 3;
    pixels[byteOffset]     = (bytes[byteOffset]     / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    pixels[byteOffset + 1] = (bytes[byteOffset + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    pixels[byteOffset + 2] = (bytes[byteOffset + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  return pixels;
}

// ── Background Sync Engine ─────────────────────────────────────────────────────

let syncIntervalId = null;

export function startBackgroundSync(apiClient) {
  if (syncIntervalId) return;

  // Listen for connectivity changes
  NetInfo.addEventListener(async (state) => {
    if (state.isConnected && state.isInternetReachable) {
      await runSync(apiClient);
    }
  });

  // Periodic sync every 10 minutes when online
  syncIntervalId = setInterval(async () => {
    const state = await NetInfo.fetch();
    if (state.isConnected && state.isInternetReachable) {
      await runSync(apiClient);
    }
  }, 10 * 60 * 1000);

  console.log("[Sync] Background sync started ✅");
}

async function runSync(apiClient) {
  const pending = await getPendingQueue();
  if (pending.length === 0) return;

  console.log(`[Sync] Syncing ${pending.length} pending items...`);
  let syncedCount = 0;

  for (const item of pending) {
    try {
      const payload = JSON.parse(item.payload_json);
      await apiClient.sync(item.type, payload);
      await markQueueItemSynced(item.id);
      syncedCount++;
    } catch (e) {
      console.warn(`[Sync] Failed to sync item ${item.id}:`, e.message);
    }
  }

  if (syncedCount > 0) {
    console.log(`[Sync] ✅ Synced ${syncedCount}/${pending.length} items`);
    await AsyncStorage.setItem("last_sync", new Date().toISOString());
  }
}

export function stopBackgroundSync() {
  if (syncIntervalId) {
    clearInterval(syncIntervalId);
    syncIntervalId = null;
  }
}

export async function isOnline() {
  const state = await NetInfo.fetch();
  return !!(state.isConnected && state.isInternetReachable);
}
