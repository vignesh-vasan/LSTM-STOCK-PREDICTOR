import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

st.set_page_config(page_title="LSTM Stock Forecasting", layout="wide")
st.title("📈 AAPL Stock Price Forecasting using LSTM")

# STEP 1: Data Collection
st.header("1️⃣ Data Collection")
df = pd.read_csv('../data/AAPL.csv')
df.columns = df.columns.str.strip()
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(subset=['Close'], inplace=True)

if st.checkbox("Show raw stock data"):
    st.dataframe(df)

# STEP 2: Preprocessing and Visualization
st.header("2️⃣ Preprocessing & Visualization")
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']])

fig2, ax2 = plt.subplots()
ax2.plot(df['Close'], label='Original Close Price')
ax2.set_title("Original Close Prices")
st.pyplot(fig2)

# STEP 3: Statistical Modeling
st.header("3️⃣ Statistical Overview")
st.write("🔹 Mean Close Price:", round(df['Close'].mean(), 2))
st.write("🔹 Max Close Price:", round(df['Close'].max(), 2))
st.write("🔹 Min Close Price:", round(df['Close'].min(), 2))

# STEP 4: LSTM Model
st.header("4️⃣ LSTM Model Training")
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

        # STEP 5: Evaluation & Visualization
        st.header("5️⃣ Model Evaluation & Visualization")
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
st.subheader("📉 Actual vs Predicted Prices")
fig, ax = plt.subplots()
ax.plot(actual, label="Actual")
ax.plot(predicted_prices, label="Predicted")
ax.set_title("AAPL LSTM Price Forecast")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)