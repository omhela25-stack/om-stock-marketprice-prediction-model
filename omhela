import streamlit as st
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

# Functions: fetch data, preprocess, create dataset, build & train models
# (Use the previously shared functions here: fetch_stock_data, preprocess_data, create_lstm_dataset)
# Define them above or import from a separate module

def fetch_stock_data(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': 'full',
        'datatype': 'json'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    key = next((k for k in data.keys() if 'Time Series' in k), None)
    if not key:
        st.error("Error fetching data or symbol not found.")
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
    df = df[['Adj Close']]
    df = df.astype(float)
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

# Streamlit UI
st.title("Stock Price Prediction with LSTM and Linear Regression")

symbol = st.text_input("Enter Stock Symbol (e.g. IBM, AAPL):", value="IBM")

if st.button("Run Prediction"):
    with st.spinner('Fetching and processing data...'):
        ts_data = fetch_stock_data(symbol)
        if ts_data is None:
            st.stop()
        
        df = preprocess_data(ts_data)
        st.subheader("Historical Adjusted Close Prices")
        st.line_chart(df)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        
        look_back = 60
        X, y = create_lstm_dataset(scaled_data, look_back)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        lstm_model = build_lstm_model((X_train.shape[1], 1))
        
        # To keep app fast, use small epochs or consider loading a pre-trained model
        lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        predicted_lstm = lstm_model.predict(X_test)
        predicted_lstm_inv = scaler.inverse_transform(predicted_lstm)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        lstm_mse = mean_squared_error(y_test_inv, predicted_lstm_inv)
        
        st.subheader("LSTM Model Prediction vs Actual")
        comparison_df = pd.DataFrame({
            "Actual": y_test_inv.flatten(),
            "Predicted": predicted_lstm_inv.flatten()
        })
        st.line_chart(comparison_df)
        st.write(f"LSTM Mean Squared Error: {lstm_mse:.4f}")
        
        # Linear regression baseline
        X_lr = np.arange(len(scaled_data)).reshape(-1, 1)
        y_lr = scaled_data
        X_train_lr, X_test_lr = X_lr[:train_size+look_back], X_lr[train_size+look_back:]
        y_train_lr, y_test_lr = y_lr[:train_size+look_back], y_lr[train_size+look_back:]
        
        from sklearn.linear_model import LinearRegression
        lr_model = LinearRegression()
        lr_model.fit(X_train_lr, y_train_lr)
        
        predicted_lr = lr_model.predict(X_test_lr)
        predicted_lr_inv = scaler.inverse_transform(predicted_lr)
        y_test_lr_inv = scaler.inverse_transform(y_test_lr)
        
        lr_mse = mean_squared_error(y_test_lr_inv, predicted_lr_inv)
        
        st.subheader("Linear Regression Model Prediction vs Actual")
        comparison_lr_df = pd.DataFrame({
            "Actual": y_test_lr_inv.flatten(),
            "Predicted": predicted_lr_inv.flatten()
        })
        st.line_chart(comparison_lr_df)
        st.write(f"Linear Regression Mean Squared Error: {lr_mse:.4f}")
