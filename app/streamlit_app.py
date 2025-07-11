import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


st.set_page_config(page_title="LSTM Stock Forecasting", layout="wide")
st.title("📈 AAPL Stock Price Forecasting using LSTM")


st.header("Data Collection")
df = pd.read_csv('data/AAPL.csv')
df.columns = df.columns.str.strip()
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(subset=['Close'], inplace=True)

if st.checkbox("Show raw stock data"):
    st.dataframe(df)


st.header("Preprocessing & Visualization")
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']])

fig2, ax2 = plt.subplots()
ax2.plot(df['Close'], label='Original Close Price')
ax2.set_title("Original Close Prices")
st.pyplot(fig2)


st.header(" Statistical Overview")
st.write("🔹 Mean Close Price:", round(df['Close'].mean(), 2))
st.write("🔹 Max Close Price:", round(df['Close'].max(), 2))
st.write("🔹 Min Close Price:", round(df['Close'].min(), 2))


st.header(" ARIMA Forecasting")
model_arima = ARIMA(df['Close'], order=(5,1,0))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=10)
st.line_chart(forecast_arima)


st.header("SARIMA Forecasting")
model_sarima = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1,1,1,12))
model_sarima_fit = model_sarima.fit(disp=False)
forecast_sarima = model_sarima_fit.forecast(steps=10)
st.line_chart(forecast_sarima)

st.header(" LSTM Model Training")
X, y = [], []
for i in range(3, len(scaled_close)):
    X.append(scaled_close[i-3:i, 0])
    y.append(scaled_close[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

if st.button("Train LSTM Model"):
    with st.spinner("Training the LSTM model..."):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=1, verbose=0)

        predicted = model.predict(X)
        predicted_prices = scaler.inverse_transform(predicted)
        actual = scaler.inverse_transform(y.reshape(-1, 1))

        st.header(" Model Evaluation & Visualization")
        fig3, ax3 = plt.subplots()
        ax3.plot(actual, label='Actual')
        ax3.plot(predicted_prices, label='Predicted')
        ax3.set_title("Actual vs Predicted Close Prices")
        ax3.legend()
        st.pyplot(fig3)

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(actual, predicted_prices)
        mae = mean_absolute_error(actual, predicted_prices)
        st.success(f"✅ Model Trained!  \n📉 MSE: {mse:.4f}  \n📊 MAE: {mae:.4f}")
