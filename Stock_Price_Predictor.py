# dataset @ https://finance.yahoo.com/quote/

# Weeks 1–2 Code: Data Collection, EDA, Preprocessing

# --- Imports ---
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# --- Step 1: Download Stock Data ---
# Example: Apple (AAPL) and Tesla (TSLA), 2015-2023
tickers = ["AAPL", "TSLA"]
start_date = "2015-01-01"
end_date = "2023-12-31"

data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Extract closing prices for convenience
close_prices = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tickers})
print("Data Shape:", close_prices.shape)
print(close_prices.head())

# --- Step 2: Exploratory Data Analysis ---
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(close_prices.index, close_prices[ticker], label=ticker)
plt.title("Stock Closing Prices (2015–2023)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

# Daily log returns
log_returns = np.log(close_prices / close_prices.shift(1))
plt.figure(figsize=(12, 6))
plt.plot(log_returns.index, log_returns['AAPL'], label="AAPL Log Return")
plt.title("Apple Daily Log Returns")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.legend()
plt.show()

# --- Step 3: Preprocessing Plan ---
# Scale values (for LSTM input later)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_aapl = scaler.fit_transform(close_prices[['AAPL']].dropna())

# --- Step 4: Sliding Window Function ---
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])  # past window_size days
        y.append(data[i, 0])                # next day
    return np.array(X), np.array(y)

window_size = 60
X_aapl, y_aapl = create_sequences(scaled_aapl, window_size=window_size)

print("Shape of X:", X_aapl.shape)  # (samples, timesteps)
print("Shape of y:", y_aapl.shape)

# --- Step 5: Train/Validation/Test Split (initial plan) ---
train_size = int(0.8 * len(X_aapl))
X_train, X_test = X_aapl[:train_size], X_aapl[train_size:]
y_train, y_test = y_aapl[:train_size], y_aapl[train_size:]

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))


# -------------------------
# Weeks 3–4: Baseline LSTM
# -------------------------

# --- Step 1: Download Data ---
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2023-12-31")

# Closing prices
prices = data['Close'].values.reshape(-1, 1)

# --- Step 2: Scale Data ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# --- Step 3: Create Sliding Window Dataset ---
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])  # last 60 days
        y.append(data[i, 0])                # next day
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)

# Reshape to (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# --- Step 4: Train/Test Split ---
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Train samples:", X_train.shape, y_train.shape)
print("Test samples:", X_test.shape, y_test.shape)

# --- Step 5: Build Baseline LSTM ---
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Dense(1)  # Predict next scaled price
])

model.compile(optimizer="adam", loss="mean_squared_error")

# --- Step 6: Train Model ---
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

# --- Step 7: Evaluate on Test Data ---
y_pred = model.predict(X_test)

# Inverse scale predictions
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# --- Step 8: Plot Predictions vs Actual ---
plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled, label="Actual Price")
plt.plot(y_pred_rescaled, label="Predicted Price")
plt.title(f"{ticker} Baseline LSTM Prediction (Test Set)")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

# --- Step 9: Compute Simple Metrics ---
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)

print("MSE:", mse)
print("MAE:", mae)


# --------------------------------------
# Weeks 5–8: Stacked and Bidirectional LSTM Models
# --------------------------------------

# --- Step 1: Load Data ---
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2023-12-31")
prices = data["Close"].values.reshape(-1, 1)

# --- Step 2: Scale Data ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# --- Step 3: Create Sequences ---
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # reshape for LSTM input

# --- Step 4: Train/Test Split ---
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Train:", X_train.shape, "Test:", X_test.shape)

# --- Step 5: Stacked LSTM Model ---
stacked_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(1)
])

stacked_model.compile(optimizer="adam", loss="mean_squared_error")
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history_stacked = stacked_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

# --- Step 6: Evaluate Stacked LSTM ---
y_pred_stacked = stacked_model.predict(X_test)
y_pred_stacked_rescaled = scaler.inverse_transform(y_pred_stacked)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

mse_stacked = mean_squared_error(y_test_rescaled, y_pred_stacked_rescaled)
mae_stacked = mean_absolute_error(y_test_rescaled, y_pred_stacked_rescaled)
print(f"Stacked LSTM - MSE: {mse_stacked:.4f}, MAE: {mae_stacked:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual Price")
plt.plot(y_pred_stacked_rescaled, label="Stacked LSTM Prediction")
plt.title("Stacked LSTM - AAPL Prediction (Test Set)")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

# --- Step 7: Bidirectional LSTM ---
bilstm_model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.3),
    Dense(1)
])

bilstm_model.compile(optimizer="adam", loss="mean_squared_error")

history_bi = bilstm_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

# --- Step 8: Evaluate Bidirectional LSTM ---
y_pred_bi = bilstm_model.predict(X_test)
y_pred_bi_rescaled = scaler.inverse_transform(y_pred_bi)

mse_bi = mean_squared_error(y_test_rescaled, y_pred_bi_rescaled)
mae_bi = mean_absolute_error(y_test_rescaled, y_pred_bi_rescaled)
print(f"Bidirectional LSTM - MSE: {mse_bi:.4f}, MAE: {mae_bi:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual Price")
plt.plot(y_pred_bi_rescaled, label="Bidirectional LSTM Prediction")
plt.title("Bidirectional LSTM - AAPL Prediction (Test Set)")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

# --- Step 9: Compare Models ---
print("\nModel Performance Summary:")
print(f"Stacked LSTM → MSE: {mse_stacked:.4f}, MAE: {mae_stacked:.4f}")
print(f"Bidirectional LSTM → MSE: {mse_bi:.4f}, MAE: {mae_bi:.4f}")

# --- Step 10: Save Models for Next Phase (Week 8–10) ---
stacked_model.save("stacked_lstm_model.h5")
bilstm_model.save("bilstm_model.h5")
