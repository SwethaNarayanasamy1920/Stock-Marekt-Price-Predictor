import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import date, timedelta
import plotly.graph_objects as go

# --- CONFIGURATION ---
NIFTY_TICKER = '^NSEI' 
END_DATE = date.today().strftime('%Y-%m-%d')
# Use 15 years of data for training, as desired
START_DATE = (date.today() - timedelta(days=15 * 365)).strftime('%Y-%m-%d') 
FORECAST_DAYS = 30
SEQUENCE_LENGTH = 60 # Number of past days to look at for prediction

# --- STREAMLIT APP LAYOUT ---
st.title('📈 STOCK MARKET FUTURE PRICE PREDICTOR')
st.markdown(f'**Ticker:** `{NIFTY_TICKER}` | **Data Range:** {START_DATE} to {END_DATE}')

# --- DATA FETCHING (using st.cache_data for performance) ---
@st.cache_data(ttl=24*3600) 
def load_data(ticker, start, end):
  data = yf.download(ticker, start=start, end=end, progress=False)
  if data.empty:
    st.error("Failed to load data. Please check the ticker or date range.")
    return None
  return data

data = load_data(NIFTY_TICKER, START_DATE, END_DATE)

if data is not None:
  st.subheader('Raw Data')
  st.write(data.tail())

  # --- DATA PREPROCESSING & TRAINING ---
  with st.spinner('Preparing data and training model...'):
    # Use only the 'Close' price
    data_close = data['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close)

    # Create training sequences (X) and target values (Y)
    X_train, y_train = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
      X_train.append(scaled_data[i-SEQUENCE_LENGTH:i, 0])
      y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # --- LSTM MODEL CONSTRUCTION AND TRAINING ---
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
    
    # --- FORECASTING ---
    last_sequence = scaled_data[-SEQUENCE_LENGTH:].copy()
    future_predictions = []
    current_batch = last_sequence.reshape(1, SEQUENCE_LENGTH, 1)
    
    for i in range(FORECAST_DAYS):
      predicted_scale_price = model.predict(current_batch, verbose=0)[0]
      future_predictions.append(predicted_scale_price[0])
      new_val = np.array([[predicted_scale_price[0]]])
      current_batch = np.append(current_batch[:, 1:, :], new_val.reshape(1, 1, 1), axis=1)

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    forecast_prices = scaler.inverse_transform(future_predictions) # <-- This is the core 30-day price array

  st.success('✅ Model Trained and Forecast Generated!')

# -------------------------------------------------------------------------------------
# --- VISUALIZATION & ERROR FIXES ---
# -------------------------------------------------------------------------------------
# --- VISUALIZATION (Final Corrected Block) ---
  st.subheader(f'Forecasted Next {FORECAST_DAYS} Days')

  last_date_obj = data.index[-1]
  last_date_str = last_date_obj.strftime('%Y-%m-%d') 

  # 1. Create the base 30-day forecast dates
  forecast_dates = [last_date_obj + timedelta(days=i) for i in range(1, FORECAST_DAYS + 1)] 
    
    # 2. FIX: Create the 31-element lists ONLY for the continuous chart line
    # This guarantees the historical line touches the forecast line.
  forecast_x = [last_date_obj] + forecast_dates
  forecast_y = [data['Close'].iloc[-1]] + forecast_prices.flatten().tolist()

  historical_trace = go.Scatter(
        x=data.index, 
        y=data['Close'], 
        mode='lines', 
        name='Historical Price (Close)', 
        line=dict(color='blue')
    )
  forecast_trace = go.Scatter(
        x=forecast_x, 
        y=forecast_y, 
        mode='lines', 
        name='Forecasted Price', 
        line=dict(color='red', dash='dash')
    )

  fig = go.Figure(data=[historical_trace, forecast_trace])
  fig.update_layout(
    title='Nifty 50 Historical and Forecasted Prices', 
        xaxis_title='Date', 
        yaxis_title='Close Price (INR)', 
    hovermode='x unified', 
        height=600, 
        xaxis=dict(autorange=True), 
        yaxis=dict(autorange=True)
  )
  
  fig.add_shape(type="line", x0=last_date_str, x1=last_date_str, y0=0, y1=1, xref='x', yref='paper', line=dict(color="green", width=2, dash="dash"))
  fig.add_annotation(x=last_date_str, y=1, text="End of Historical Data", xref="x", yref="paper", showarrow=False, xanchor="left", yanchor="bottom", font=dict(color="green"))

  st.plotly_chart(fig, use_container_width=True)

  st.subheader('Forecasted Values')
    
  forecast_df = pd.DataFrame({
    'Date': forecast_dates[:FORECAST_DAYS],
    'Forecasted Price (INR)': forecast_prices.flatten()[:FORECAST_DAYS]
  }).set_index('Date')
    
  st.dataframe(forecast_df.head(10))

  st.info(
    "**Disclaimer:** This is a demonstration of an LSTM model for time series forecasting. Stock market prediction is highly volatile and complex, and this model should **not** be used for actual trading decisions."
  )