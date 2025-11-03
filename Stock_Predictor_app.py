# This app implements a Streamlit dashboard for real-time stock price prediction 
# using a pre-trained LSTM model from Stock_Price Predictor project.

# -------------------------
# Weeks 11-12: Streamlit Dashboard
# -------------------------
import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

model = load_model("optimized_lstm_model.h5")

def create_latest_sequence(data, window_size=60):
    X = data[-window_size:]
    return np.reshape(X, (1, window_size, 1))

st.title("📈 Real-Time Stock Price Prediction Dashboard")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")

if st.button("Predict Next Price"):
    data = yf.download(ticker, period="1y", interval="1d")["Close"].values.reshape(-1,1)
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X_latest = create_latest_sequence(scaled, 60)
    prediction = model.predict(X_latest)
    price = scaler.inverse_transform(prediction)[0][0]

    st.success(f"Predicted Next Closing Price for {ticker}: **${price:.2f}**")
