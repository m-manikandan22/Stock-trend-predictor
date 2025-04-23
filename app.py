import os
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ── Disable GPU so TensorFlow runs on CPU ────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ── Mapping sectors to .h5 model filenames ───────────────────────────────
MODEL_DIR = "models/"
sector_model_map = {
    "Technology": "Technology.h5",
    "Financial Services": "finance_model.h5",
    "Healthcare": "Healthcare.h5",
    "Energy": "Energy.h5",
    "Consumer Defensive": "Consumer Goods.h5",
    "Consumer Cyclical": "Consumer Services.h5",
    "Industrials": "industrials.h5",
    "Real Estate": "Real Estate.h5",
    "Communication Services": "communication.h5",
    "Utilities": "Utilities.h5",
    "Basic Materials": "Material.h5"
}

# ── Get sector via Yahoo Finance ─────────────────────────────────────────
def get_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector", "Technology")
    except Exception:
        return "Technology"

# ── Load Keras .h5 model on CPU without compiling ────────────────────────
def load_model_keras(sector: str):
    model_file = sector_model_map.get(sector)
    if not model_file:
        return None
    path = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(path):
        return None
    return load_model(path, compile=False)

# ── Fetch historical stock data ──────────────────────────────────────────
def fetch_stock_data(ticker: str, days: int = 50) -> pd.DataFrame | None:
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=days * 2)
    df = yf.download(ticker, start=start_date, end=end_date)
    return df.tail(days).reset_index() if not df.empty else None

# ── Scale features and pad if needed ─────────────────────────────────────
def prepare_data(df: pd.DataFrame):
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    data = df[feature_cols].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Pad to 12 features
    expected_features = 12
    padded_data = np.pad(
        data_scaled,
        ((0, 0), (0, expected_features - data_scaled.shape[1])),
        mode='constant'
    )

    return padded_data, scaler

# ── Predict next N days ──────────────────────────────────────────────────
def predict_next_days(model, data_scaled, seq_length: int, scaler, days: int = 5):
    preds = []
    last_seq = np.expand_dims(data_scaled[-seq_length:], axis=0)
    last_known_row = data_scaled[-1].copy()
    expected_features = data_scaled.shape[1]

    for _ in range(days):
        scaled_pred = model.predict(last_seq, verbose=0)[0, 0]
        new_row = last_known_row.copy()
        if expected_features >= 4:
            new_row[3] = scaled_pred  # 'Close'

        inv_input = new_row[:5]
        inv_scaled = scaler.inverse_transform([inv_input])[0]
        actual_close = inv_scaled[3]
        preds.append(actual_close)

        padded = np.pad(inv_scaled, (0, expected_features-5), mode='constant')
        last_seq = np.concatenate([last_seq[:,1:,:], padded.reshape(1,1,-1)], axis=1)
        last_known_row = padded

    return preds

# ── Streamlit UI ─────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.title("📊 Sector‑Based Stock Price Predictor ")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, MSFT, TSLA):").upper()
if st.button("Predict"):
    if not ticker:
        st.warning("⚠️ Please enter a stock ticker.")
        st.stop()

    df = fetch_stock_data(ticker)
    if df is None or len(df) < 50:
        st.error("❌ Not enough data to predict. Try another ticker.")
        st.stop()

    sector = get_sector(ticker)
    model = load_model_keras(sector)
    if model is None:
        st.error(f"❌ No model found for sector: {sector}")
        st.stop()
    st.success(f"✅ Sector: {sector}")

    data_scaled, scaler = prepare_data(df)
    preds = predict_next_days(model, data_scaled, seq_length=50, scaler=scaler, days=5)

    # Prepare dates
    df["Date"] = pd.to_datetime(df["Date"])
    actual_dates = df["Date"].tail(30)
    actual_prices = df["Close"].tail(30).values

    last_date = actual_dates.iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5, freq="B")

    # Show result table
    result_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Close Price": np.round(preds, 2)
    })
    st.write(result_df)

    # Plot with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    # Past 30 days
    ax.plot(actual_dates, actual_prices,
            color="blue", marker="o", linestyle="-", label="Actual (30d)", markersize=8)
    # Next 5 days
    ax.plot(future_dates, preds,
            color="red", marker="o", linestyle="-", label="Predicted (5d)", markersize=8)
    # Formatting
    ax.set_title(f"{ticker} Close Price: Last 30 Days & Next 5 Days Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    fig.autofmt_xdate()
    ax.grid(True)

    st.pyplot(fig)

