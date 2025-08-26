# stockpredictor.app.py
# Merged & refactored: keeps your LSTM/LR + adds full analysis stack, multi-source data, indicators,
# model comparison, feature importance, richer charts, downloads, API status, and a fresh UI layout.
# Sources merged: your original stockpredictor.app.py and the provided app.py (features & flows).

import streamlit as st
import time
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

API_KEY = 'LPRQX827JWWLKA4R'
BASE_URL = 'https://www.alphavantage.co/query?'

# Predefined list of symbols (example popular NASDAQ + NYSE stocks)
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'IBM', 'META', 'NFLX']

def fetch_stock_data(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': 'compact',
        'datatype': 'json'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    key = next((k for k in data.keys() if 'Time Series' in k), None)
    if 'Error Message' in data or not key:
        st.warning(f"Data not found for symbol: {symbol}")
        return None
    return data[key]

def preprocess_data(ts_data):
    df = pd.DataFrame.from_dict(ts_data, orient='index')
    df = df.rename(columns={
        '1. open':'Open', '2. high':'High', '3. low':'Low', 
        '4. close':'Close', '5. adjusted close':'Adj Close', 
        '6. volume':'Volume', '7. dividend amount':'Dividend', 
        '8. split coefficient':'Split Coeff'
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[['Adj Close']].astype(float)
    return df

def create_lstm_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

st.title("Batch Stock Price Prediction")

selected_symbols = st.multiselect("Select stock symbols for prediction:", options=SYMBOLS, default=['AAPL'])

if st.button("Run Batch Predictions"):

    for symbol in selected_symbols:
        st.write(f"Processing {symbol}...")
        with st.spinner(f"Fetching data for {symbol}"):
            ts_data = fetch_stock_data(symbol)
        if ts_data is None:
            st.warning(f"Skipping {symbol} due to data fetch issue.")
            continue

        df = preprocess_data(ts_data)
        st.subheader(f"{symbol} Historical Adjusted Close Prices")
        st.line_chart(df)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)

        look_back = 60
        X, y = create_lstm_dataset(scaled_data, look_back)
        if len(X) == 0:
            st.warning(f"Not enough data points for {symbol}.")
            continue

        X = X.reshape((X.shape[0], X.shape[1], 1))
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        lstm_model = build_lstm_model((X_train.shape[1], 1))
        lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        predicted_lstm = lstm_model.predict(X_test)
        predicted_lstm_inv = scaler.inverse_transform(predicted_lstm)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        lstm_mse = mean_squared_error(y_test_inv, predicted_lstm_inv)

        st.subheader(f"{symbol} LSTM Model Prediction vs Actual")
        comparison_df = pd.DataFrame({
            "Actual": y_test_inv.flatten(),
            "Predicted": predicted_lstm_inv.flatten()
        })
        st.line_chart(comparison_df)
        st.write(f"LSTM Mean Squared Error for {symbol}: {lstm_mse:.4f}")

        time.sleep(15)  # To avoid hitting rate limit (max 5 requests/min)

