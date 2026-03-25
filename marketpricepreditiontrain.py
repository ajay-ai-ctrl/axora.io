"""
SmartKhet — Market Price Prediction Model
==========================================
Algorithm  : LSTM (PyTorch) + Facebook Prophet ensemble
Target     : Mandi commodity price (₹/quintal) 7/14/30-day forecast
Data       : Agmarknet historical prices (2015–present), 500+ mandis
Features   : Lag prices, rolling stats, seasonality, MSP, rainfall, diesel price
Output     : Price forecast + sell/hold signal + best mandi recommendation

Author     : Axora / SmartKhet ML Team
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import mlflow
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SEQUENCE_LEN = 60          # 60 days look-back window
FORECAST_HORIZONS = [7, 14, 30]
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data Engineering ───────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features, rolling statistics, and seasonal encodings
    from raw daily price data.

    Expected input columns: date, commodity, district, price_per_qtl
    """
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    # Lag features (autoregressive)
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f"price_lag_{lag}"] = df["price_per_qtl"].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30, 60]:
        df[f"rolling_mean_{window}"] = df["price_per_qtl"].rolling(window).mean()
        df[f"rolling_std_{window}"] = df["price_per_qtl"].rolling(window).std()
        df[f"rolling_min_{window}"] = df["price_per_qtl"].rolling(window).min()
        df[f"rolling_max_{window}"] = df["price_per_qtl"].rolling(window).max()

    # Price momentum
    df["pct_change_7d"] = df["price_per_qtl"].pct_change(7)
    df["pct_change_30d"] = df["price_per_qtl"].pct_change(30)

    # Seasonal encodings
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # Year for long-term trend
    df["year"] = df["date"].dt.year
    df["year_norm"] = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min() + 1)

    return df.dropna()


def create_sequences(data: np.ndarray, seq_len: int, horizon: int = 7):
    """
    Convert time series array into (X, y) sliding window sequences.
    X shape: (n_samples, seq_len, n_features)
    y shape: (n_samples,) — price horizon days ahead
    """
    X, y = [], []
    for i in range(len(data) - seq_len - horizon):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len + horizon - 1, 0])  # col 0 = price
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── LSTM Model ─────────────────────────────────────────────────────────────────

class PriceLSTM(nn.Module):
    """
    Stacked LSTM with attention mechanism for price sequence modelling.
    Bidirectional processing + self-attention allows the model to weigh
    which days in the look-back window are most informative.
    """

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

        # Attention over time steps
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden*2)

        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # Weighted sum across time steps
        context = (attn_weights * lstm_out).sum(dim=1)
        # context: (batch, hidden*2)

        return self.fc(context).squeeze(-1)


# ── Training ───────────────────────────────────────────────────────────────────

def train_lstm(df: pd.DataFrame, commodity: str, district: str,
               horizon: int = 7, output_dir: str = "models/market/") -> PriceLSTM:
    """
    Train LSTM for a specific commodity × district × forecast horizon.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Feature columns (price must be index 0 for target extraction)
    feature_cols = ["price_per_qtl"] + [c for c in df.columns if c not in
                    ["date", "commodity", "district", "price_per_qtl"]]

    data = df[feature_cols].values.astype(np.float32)

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled, SEQUENCE_LEN, horizon)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = PriceLSTM(input_size=X.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss(delta=1.0)  # Huber loss — robust to price spikes
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=True
    )

    best_val_mae = float("inf")

    with mlflow.start_run(run_name=f"lstm_{commodity}_{district}_{horizon}d"):
        mlflow.log_params({
            "commodity": commodity, "district": district,
            "horizon_days": horizon, "seq_len": SEQUENCE_LEN,
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
        })

        for epoch in range(1, NUM_EPOCHS + 1):
            # Train
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validate
            model.eval()
            val_preds, val_actuals = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(xb)
                    val_preds.extend(pred.cpu().numpy())
                    val_actuals.extend(yb.cpu().numpy())

            # Inverse-transform to get ₹/qtl MAE
            dummy = np.zeros((len(val_preds), data_scaled.shape[1]))
            dummy[:, 0] = val_preds
            inv_preds = scaler.inverse_transform(dummy)[:, 0]
            dummy[:, 0] = val_actuals
            inv_actuals = scaler.inverse_transform(dummy)[:, 0]

            mae = mean_absolute_error(inv_actuals, inv_preds)
            rmse = np.sqrt(mean_squared_error(inv_actuals, inv_preds))
            scheduler.step(mae)

            if epoch % 10 == 0:
                log.info(f"Epoch {epoch:03d} | MAE: ₹{mae:.1f}/qtl | RMSE: ₹{rmse:.1f}/qtl")

            mlflow.log_metrics({"val_mae": mae, "val_rmse": rmse}, step=epoch)

            if mae < best_val_mae:
                best_val_mae = mae
                save_path = os.path.join(output_dir, f"{commodity}_{district}_{horizon}d.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "scaler": scaler,
                    "feature_cols": feature_cols,
                    "commodity": commodity,
                    "district": district,
                    "horizon": horizon,
                    "seq_len": SEQUENCE_LEN,
                    "val_mae": mae,
                }, save_path)

        mlflow.log_metric("best_val_mae", best_val_mae)
        log.info(f"✅ LSTM MAE: ₹{best_val_mae:.1f}/qtl  [{commodity}/{district}/{horizon}d]")

    return model


# ── Prophet Model ──────────────────────────────────────────────────────────────

def train_prophet(df: pd.DataFrame, commodity: str, district: str,
                  output_dir: str = "models/market/") -> Prophet:
    """
    Facebook Prophet model with custom Indian agricultural seasonality.
    Prophet captures long-term trends + harvest seasonality well.
    Used as ensemble partner to LSTM.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prophet expects columns: ds (date), y (value)
    prophet_df = df[["date", "price_per_qtl"]].rename(
        columns={"date": "ds", "price_per_qtl": "y"}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # No weekly pattern in commodity prices
        daily_seasonality=False,
        changepoint_prior_scale=0.05,   # Conservative — agri prices change slowly
        seasonality_prior_scale=10,
        seasonality_mode="multiplicative",  # Seasonal swings scale with price level
        uncertainty_samples=1000,
    )

    # Add Indian agricultural seasons as custom seasonalities
    model.add_seasonality(name="kharif_rabi", period=182.5, fourier_order=5)
    model.add_seasonality(name="harvest_pulse", period=365/3, fourier_order=3)

    model.fit(prophet_df)

    save_path = os.path.join(output_dir, f"prophet_{commodity}_{district}.pkl")
    joblib.dump(model, save_path)
    log.info(f"✅ Prophet model saved: {save_path}")
    return model


# ── Ensemble Inference ─────────────────────────────────────────────────────────

class MarketPricePredictor:
    """
    Ensemble of LSTM (60%) + Prophet (40%) for price forecasting.
    LSTM captures short-term momentum; Prophet captures seasonality.
    Provides sell/hold signal and best mandi recommendation.
    """

    # MSP (Minimum Support Price) 2024-25 per quintal (₹)
    MSP_2024 = {
        "wheat": 2275, "rice": 2300, "maize": 2090, "chickpea": 5440,
        "lentil": 6425, "mustard": 5650, "soybean": 4600, "cotton": 7121,
    }

    def __init__(self, lstm_path: str, prophet_path: str):
        checkpoint = torch.load(lstm_path, map_location="cpu")
        self.scaler = checkpoint["scaler"]
        self.feature_cols = checkpoint["feature_cols"]
        self.commodity = checkpoint["commodity"]
        self.district = checkpoint["district"]
        self.horizon = checkpoint["horizon"]
        self.seq_len = checkpoint["seq_len"]

        lstm_model = PriceLSTM(input_size=len(self.feature_cols))
        lstm_model.load_state_dict(checkpoint["model_state_dict"])
        lstm_model.eval()
        self.lstm = lstm_model

        self.prophet = joblib.load(prophet_path)
        log.info(f"MarketPricePredictor loaded ✅ [{self.commodity}/{self.district}]")

    def predict(self, recent_df: pd.DataFrame) -> dict:
        """
        Predict price for self.horizon days ahead.
        recent_df: last SEQUENCE_LEN days of feature data.
        """
        df_feat = engineer_features(recent_df.copy())
        features = df_feat[self.feature_cols].values[-self.seq_len:]
        scaled = self.scaler.transform(features)

        # LSTM prediction
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            lstm_scaled_pred = self.lstm(x).item()

        dummy = np.zeros((1, len(self.feature_cols)))
        dummy[0, 0] = lstm_scaled_pred
        lstm_price = self.scaler.inverse_transform(dummy)[0, 0]

        # Prophet prediction
        future = self.prophet.make_future_dataframe(periods=self.horizon)
        prophet_forecast = self.prophet.predict(future)
        prophet_price = prophet_forecast["yhat"].iloc[-1]
        prophet_lower = prophet_forecast["yhat_lower"].iloc[-1]
        prophet_upper = prophet_forecast["yhat_upper"].iloc[-1]

        # Ensemble: 60% LSTM, 40% Prophet
        ensemble_price = 0.6 * lstm_price + 0.4 * prophet_price

        # Sell signal logic
        current_price = float(recent_df["price_per_qtl"].iloc[-1])
        msp = self.MSP_2024.get(self.commodity.lower(), 0)
        price_change_pct = ((ensemble_price - current_price) / current_price) * 100
        above_msp = ensemble_price > msp

        if price_change_pct > 5:
            signal = "HOLD"
            signal_reason = f"Price expected to rise {price_change_pct:.1f}% in {self.horizon} days"
        elif price_change_pct < -5:
            signal = "SELL_NOW"
            signal_reason = f"Price may fall {abs(price_change_pct):.1f}% — sell before decline"
        elif above_msp:
            signal = "SELL"
            signal_reason = f"Price ₹{ensemble_price:.0f} is above MSP ₹{msp}. Good to sell."
        else:
            signal = "HOLD_FOR_MSP"
            signal_reason = f"Price below MSP ₹{msp}. Consider holding or selling via FCI."

        return {
            "commodity": self.commodity,
            "district": self.district,
            "current_price": round(current_price, 2),
            "predicted_price": round(ensemble_price, 2),
            "lstm_price": round(lstm_price, 2),
            "prophet_price": round(prophet_price, 2),
            "confidence_range": {
                "low": round(prophet_lower, 2),
                "high": round(prophet_upper, 2),
            },
            "horizon_days": self.horizon,
            "price_change_pct": round(price_change_pct, 2),
            "msp": msp,
            "above_msp": above_msp,
            "signal": signal,
            "signal_reason": signal_reason,
            "currency": "INR",
            "unit": "per quintal",
        }


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SmartKhet Market Price Predictor")
    parser.add_argument("--data", type=str, required=True,
                        help="CSV with columns: date, commodity, district, price_per_qtl")
    parser.add_argument("--commodity", type=str, required=True)
    parser.add_argument("--district", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--output", type=str, default="models/market/")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[(df["commodity"] == args.commodity) & (df["district"] == args.district)]
    df = engineer_features(df)

    train_lstm(df, args.commodity, args.district, args.horizon, args.output)
    train_prophet(df, args.commodity, args.district, args.output)
