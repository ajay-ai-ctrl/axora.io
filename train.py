"""
SmartKhet — Crop Recommendation Model
======================================
Algorithm: XGBoost + RandomForest Voting Ensemble
Features : Soil NPK, pH, moisture, temperature, rainfall, humidity,
           district, season, market demand index
Target   : Crop label (20 crops covering 85% of Indian agricultural land)

Author   : Axora / SmartKhet ML Team
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CROP_LABELS = [
    "rice", "wheat", "maize", "chickpea", "kidneybeans",
    "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil",
    "pomegranate", "banana", "mango", "grapes", "watermelon",
    "muskmelon", "apple", "orange", "papaya", "coconut",
]

FEATURE_COLUMNS = [
    "nitrogen",        # N content in soil (kg/ha)
    "phosphorus",      # P content in soil (kg/ha)
    "potassium",       # K content in soil (kg/ha)
    "temperature",     # Average temperature (°C)
    "humidity",        # Relative humidity (%)
    "ph",              # Soil pH
    "rainfall",        # Annual rainfall (mm)
    "moisture",        # Soil moisture (%)
    "season_encoded",  # Kharif=0, Rabi=1, Zaid=2
    "district_encoded",# District numeric code
    "market_demand",   # Market demand index 0–1 (from mandi data)
]

MODEL_SAVE_PATH = "models/crop_recommender.pkl"
SCALER_SAVE_PATH = "models/crop_scaler.pkl"
ENCODER_SAVE_PATH = "models/crop_label_encoder.pkl"


# ── Data Loading & Preprocessing ──────────────────────────────────────────────

def load_and_preprocess(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw soil+crop CSV, encode categoricals, and return X, y.
    Expected CSV columns: nitrogen, phosphorus, potassium, temperature,
                          humidity, ph, rainfall, moisture, season,
                          district, market_demand, label
    """
    log.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Dataset shape: {df.shape}")

    # Encode season
    season_map = {"kharif": 0, "rabi": 1, "zaid": 2, "perennial": 3}
    df["season_encoded"] = df["season"].str.lower().map(season_map).fillna(0).astype(int)

    # Encode district using frequency encoding (handles unseen districts at inference)
    district_freq = df["district"].value_counts(normalize=True)
    df["district_encoded"] = df["district"].map(district_freq).fillna(0)

    # Clip outliers at 1st/99th percentile
    numeric_cols = ["nitrogen", "phosphorus", "potassium", "temperature",
                    "humidity", "ph", "rainfall", "moisture"]
    for col in numeric_cols:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    X = df[FEATURE_COLUMNS]
    y = df["label"].str.lower()

    log.info(f"Class distribution:\n{y.value_counts()}")
    return X, y


# ── Model Definition ───────────────────────────────────────────────────────────

def build_ensemble() -> VotingClassifier:
    """
    Soft-voting ensemble of XGBoost and RandomForest.
    XGBoost handles feature interactions; RF provides calibration and diversity.
    """
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist",       # Fast histogram method
        random_state=42,
        n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",  # Handles class imbalance
        random_state=42,
        n_jobs=-1,
    )

    # XGBoost gets slightly higher weight — consistently better on tabular agri data
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        voting="soft",
        weights=[0.6, 0.4],
    )
    return ensemble


def build_pipeline(ensemble: VotingClassifier) -> Pipeline:
    """
    Full sklearn Pipeline: impute → scale → ensemble.
    Ensures same transformations at training and inference time.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ensemble),
    ])
    return pipeline


# ── Training ───────────────────────────────────────────────────────────────────

def train(csv_path: str, experiment_name: str = "smartkhet-crop-recommendation"):
    """
    Full training run with MLflow tracking, SMOTE oversampling,
    cross-validation, and model artifact saving.
    """
    mlflow.set_experiment(experiment_name)

    X, y_raw = load_and_preprocess(csv_path)

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    log.info(f"Classes: {le.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE to handle class imbalance in minority crops
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    log.info(f"After SMOTE: {X_train_res.shape}")

    ensemble = build_ensemble()
    pipeline = build_pipeline(ensemble)

    with mlflow.start_run(run_name="xgb_rf_voting_ensemble"):

        # Cross-validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(
            pipeline, X_train_res, y_train_res,
            cv=cv, scoring="f1_weighted", n_jobs=-1
        )
        log.info(f"CV F1 (weighted): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        # Fit final model
        log.info("Training final pipeline...")
        pipeline.fit(X_train_res, y_train_res)

        # Evaluate on held-out test set
        y_pred = pipeline.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        log.info(f"\nClassification Report:\n{report}")

        # MLflow logging
        mlflow.log_param("model_type", "xgb_rf_soft_voting")
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("n_classes", len(le.classes_))
        mlflow.log_param("smote", True)
        mlflow.log_metric("cv_f1_mean", cv_f1.mean())
        mlflow.log_metric("cv_f1_std", cv_f1.std())
        mlflow.log_metric("test_f1_weighted", test_f1)
        mlflow.sklearn.log_model(pipeline, "crop_recommender_pipeline")

        # Save artifacts locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, MODEL_SAVE_PATH)
        joblib.dump(le, ENCODER_SAVE_PATH)

        # Save class mapping for API use
        class_map = {int(i): name for i, name in enumerate(le.classes_)}
        with open("models/crop_class_map.json", "w") as f:
            json.dump(class_map, f, indent=2)

        log.info(f"✅ Model saved: {MODEL_SAVE_PATH}")
        log.info(f"✅ Test F1 Weighted: {test_f1:.4f}")

    return pipeline, le


# ── Inference ──────────────────────────────────────────────────────────────────

class CropRecommender:
    """
    Production inference class. Loads pipeline and label encoder from disk.
    Thread-safe for use in FastAPI async context (no mutable state after init).
    """

    def __init__(
        self,
        model_path: str = MODEL_SAVE_PATH,
        encoder_path: str = ENCODER_SAVE_PATH,
        class_map_path: str = "models/crop_class_map.json",
    ):
        log.info("Loading CropRecommender from disk...")
        self.pipeline = joblib.load(model_path)
        self.le = joblib.load(encoder_path)
        with open(class_map_path) as f:
            self.class_map = json.load(f)
        log.info("CropRecommender ready ✅")

    def predict(self, features: dict, top_k: int = 3) -> list[dict]:
        """
        Given a dict of feature values, return top-k crop recommendations
        with confidence scores.

        Args:
            features: dict with keys matching FEATURE_COLUMNS
            top_k: number of top crops to return

        Returns:
            List of {"crop": str, "confidence": float, "rank": int}
        """
        # Build feature vector in correct column order
        row = pd.DataFrame([[features.get(col, 0) for col in FEATURE_COLUMNS]],
                           columns=FEATURE_COLUMNS)

        # Get probability distribution over all classes
        probs = self.pipeline.predict_proba(row)[0]

        # Rank by probability
        ranked_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            results.append({
                "rank": rank,
                "crop": self.class_map[str(idx)],
                "confidence": round(float(probs[idx]), 4),
                "confidence_pct": f"{probs[idx] * 100:.1f}%",
            })
        return results

    def predict_with_advisory(self, features: dict) -> dict:
        """
        Returns top-3 crops + basic fertilizer advisory based on soil NPK.
        """
        recommendations = self.predict(features, top_k=3)
        top_crop = recommendations[0]["crop"]

        advisory = self._generate_fertilizer_advisory(
            crop=top_crop,
            n=features.get("nitrogen", 0),
            p=features.get("phosphorus", 0),
            k=features.get("potassium", 0),
            ph=features.get("ph", 7.0),
        )

        return {
            "top_crops": recommendations,
            "primary_crop": top_crop,
            "fertilizer_advisory": advisory,
        }

    @staticmethod
    def _generate_fertilizer_advisory(crop: str, n: float, p: float,
                                       k: float, ph: float) -> dict:
        """
        Rule-based fertilizer advisory. Complements ML output.
        Optimal ranges sourced from ICAR recommendations.
        """
        # ICAR optimal soil NPK targets (kg/ha)
        OPTIMAL = {
            "rice":    {"n": 120, "p": 60,  "k": 60},
            "wheat":   {"n": 120, "p": 60,  "k": 40},
            "maize":   {"n": 150, "p": 75,  "k": 50},
            "default": {"n": 100, "p": 50,  "k": 50},
        }
        target = OPTIMAL.get(crop, OPTIMAL["default"])

        deficit = {
            "nitrogen":   max(0, target["n"] - n),
            "phosphorus": max(0, target["p"] - p),
            "potassium":  max(0, target["k"] - k),
        }

        # pH-based lime/sulfur recommendation
        ph_action = None
        if ph < 6.0:
            lime_kg = round((6.5 - ph) * 1250)  # approx. kg/ha
            ph_action = f"Soil is acidic (pH {ph}). Apply {lime_kg} kg/ha agricultural lime."
        elif ph > 8.0:
            ph_action = f"Soil is alkaline (pH {ph}). Apply gypsum 250 kg/ha + sulfur 50 kg/ha."

        return {
            "crop": crop,
            "nutrient_deficit_kg_ha": deficit,
            "urea_kg_ha": round(deficit["nitrogen"] / 0.46),       # 46% N
            "dap_kg_ha":  round(deficit["phosphorus"] / 0.46),     # 46% P2O5
            "mop_kg_ha":  round(deficit["potassium"] / 0.60),      # 60% K2O
            "ph_correction": ph_action,
        }


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SmartKhet Crop Recommender")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to CSV training data")
    parser.add_argument("--experiment", type=str,
                        default="smartkhet-crop-recommendation")
    args = parser.parse_args()

    train(csv_path=args.data, experiment_name=args.experiment)
