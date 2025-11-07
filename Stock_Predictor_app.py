# This app implements a Streamlit dashboard for real-time stock price prediction 
# using a pre-trained LSTM model from Stock_Price Predictor project.

# -------------------------
# Weeks 11-12: Streamlit Dashboard
# -------------------------
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Disables threading lock

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import load_model
from keras.optimizers import Adam

st.title("📈 Real-Time Stock Price Prediction Dashboard")

# -------------------------
# Load pre-trained models
# -------------------------
@st.cache_resource
def load_models():
    try:
        stacked = load_model("stacked_lstm_model.h5", compile=False)
    except:
        stacked = None
    try:
        optimized = load_model("optimized_lstm_model.h5", compile=False)
    except:
        optimized = None
    return stacked, optimized

stacked_model, optimized_model = load_models()

model_choice = st.selectbox("Choose Prediction Model:", ("Stacked LSTM", "Optimized LSTM"))
active_model = stacked_model if model_choice == "Stacked LSTM" else optimized_model

if active_model is None:
    st.error(f"❌ The selected model file for **{model_choice}** is missing.")
    st.stop()

# -------------------------
# Functions
# -------------------------
def create_latest_sequence(data, window_size=60):
    return np.reshape(data[-window_size:], (1, window_size, 1))

# -------------------------
# User input: ticker
# -------------------------
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL").upper()
window_size = 60

# -------------------------
# Real-time next-day prediction
# -------------------------
if st.button("Predict Next Price"):

    # Fetch last 1 year daily data
    try:
        df = yf.download(ticker, period="1y", interval="1d")
    except:
        st.error("❌ Could not connect to data source. Check your internet.")
        st.stop()

    if df.empty:
        st.error("⚠️ No price data found. Try another ticker symbol.")
        st.stop()

    data = df["Close"].values.reshape(-1,1)
    if len(data) < window_size:
        st.error(f"⚠️ Not enough recent data (need at least {window_size} days).")
        st.stop()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X_latest = create_latest_sequence(scaled_data, window_size)
    prediction = active_model.predict(X_latest, verbose=0)
    price = scaler.inverse_transform(prediction)[0][0]

    st.success(f"Predicted Next Closing Price for **{ticker}**: **${price:.2f}**")


# -------------------------
# # Weeks 13–14: Model Comparison + Trend Forecasting + Analytics 
# -------------------------

# -------------------------
# Trend Forecasting & Analytics
# -------------------------
st.subheader("📊 Model Comparison & 7-Day Trend Forecasting")

# Fetch historical data for long-term trend
try:
    full_data = yf.download(ticker, start="2015-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))
except:
    st.error("❌ Could not download historical data. Check your ticker or internet.")
    st.stop()

if full_data.empty or "Close" not in full_data:
    st.error("⚠️ No historical price data available for forecasting. Try a different ticker.")
    st.stop()

close_prices = full_data["Close"].values.reshape(-1,1)
if len(close_prices) < window_size:
    st.error(f"⚠️ Not enough historical data (need at least {window_size} days).")
    st.stop()

# Scale full data
scaler2 = MinMaxScaler()
scaled_full = scaler2.fit_transform(close_prices)

# Forecast next 7 days
forecast_input = scaled_full[-window_size:].reshape(1, window_size, 1)
future_predictions = []
curr_sequence = forecast_input.copy()

for _ in range(7):
    next_val = active_model.predict(curr_sequence, verbose=0)[0][0]
    future_predictions.append(next_val)
    curr_sequence = np.append(curr_sequence[:, 1:, :], [[[next_val]]], axis=1)

future_predictions = scaler2.inverse_transform(np.array(future_predictions).reshape(-1,1))

# Display trend chart
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(close_prices[-100:], label="Recent Actual Prices")
ax.plot(range(len(close_prices[-100:]), len(close_prices[-100:])+7),
         future_predictions, marker='o', label="7-Day Prediction")
ax.set_title(f"{ticker} - Short-Term Price Projection ({model_choice})")
ax.set_xlabel("Days")
ax.set_ylabel("Price ($)")
ax.legend()
st.pyplot(fig)

# Evaluate model performance on recent segment
test_data = scaled_full[-(window_size+30):]  # last 30 predictions
X_eval, y_eval = [], []

for i in range(window_size, len(test_data)):
    X_eval.append(test_data[i-window_size:i, 0])
    y_eval.append(test_data[i,0])

X_eval = np.array(X_eval).reshape(-1, window_size,1)
y_eval = np.array(y_eval)

y_pred_eval = active_model.predict(X_eval, verbose=0)
y_pred_eval = scaler2.inverse_transform(y_pred_eval)
y_eval_actual = scaler2.inverse_transform(y_eval.reshape(-1,1))

rmse_eval = np.sqrt(mean_squared_error(y_eval_actual, y_pred_eval))
mape_eval = mean_absolute_percentage_error(y_eval_actual, y_pred_eval)*100

st.write("### 📈 Model Performance Evaluation")
st.write(f"**RMSE:** {rmse_eval:.3f}")
st.write(f"**MAPE:** {mape_eval:.2f}%")
st.info("Lower RMSE and MAPE values indicate better accuracy.")


# # Note: -------------------------
# # To run this app, save it as Stock_Predictor_app.py
# # To run this app, ensure you have Streamlit and required libraries installed.
# # Run the app with EXACTLY THIS: 
# #               streamlit run /Users/sigmus_aimofumhe/Stock_Predictor_app.py
