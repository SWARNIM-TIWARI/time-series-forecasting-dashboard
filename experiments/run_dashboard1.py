# run_dashboard.py
# ========================================
# All-in-One: Merge 33 stock CSVs + Fast Streamlit Dashboard
# ========================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer

# -----------------------------
# Metrics
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# -----------------------------
# Step 1: Merge individual stock CSVs
# -----------------------------
input_folder = "individual_stocks"
output_folder = "processed_stocks"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

stock_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
processed_files = []

for stock_file in stock_files:
    file_path = os.path.join(input_folder, stock_file)
    try:
        df = pd.read_csv(file_path)

        # Detect date column
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if not date_cols:
            print(f"Skipping {stock_file}: No date column found")
            continue
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df.rename(columns={date_cols[0]: 'date'}, inplace=True)

        # Detect value column
        value_cols = [c for c in df.columns if c.lower() in ['close','value','adj close']]
        if not value_cols:
            print(f"Skipping {stock_file}: No value column found")
            continue
        df = df[['date', value_cols[0]]].rename(columns={value_cols[0]: 'value'})

        df.sort_values('date', inplace=True)
        df.fillna(method='ffill', inplace=True)

        output_file = os.path.join(output_folder, f"your_timeseries_{stock_file}")
        df.to_csv(output_file, index=False)
        processed_files.append(output_file)
        print(f"Processed and saved: {output_file}")
    except Exception as e:
        print(f"Error processing {stock_file}: {e}")

if not processed_files:
    raise Exception("No stocks processed! Check your CSVs in individual_stocks.")

print("✅ All stocks processed successfully!")

# -----------------------------
# Step 2: Streamlit Dashboard
# -----------------------------
st.set_page_config(page_title="Time-Series Forecasting Dashboard", layout="wide")
st.title("📈 Time-Series Forecasting Dashboard")

# Dropdown: pick a stock
stock_selection = st.selectbox(
    "Select a stock:",
    [os.path.basename(f) for f in processed_files]
)
data_file = os.path.join(output_folder, stock_selection)
df = pd.read_csv(data_file, parse_dates=['date'])

st.subheader(f"Dataset Overview: {stock_selection}")
st.dataframe(df.tail())

fig = px.line(df, x='date', y='value', title='Original Time Series')
st.plotly_chart(fig)

# -----------------------------
# Step 3: Classical Models
# -----------------------------
def train_arima(df, forecast_steps=30, order=(5,1,0)):
    model = ARIMA(df['value'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    return forecast

def train_prophet(df, forecast_steps=30):
    df_prophet = df.rename(columns={'date':'ds','value':'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_steps)
    forecast = model.predict(future)
    return forecast[['ds','yhat']]

arima_forecast = train_arima(df)
prophet_forecast = train_prophet(df)

# -----------------------------
# Step 4: LSTM (optimized + cached)
# -----------------------------
@st.cache_data
def train_lstm_model(series, seq_length=30, epochs=3):
    class TimeSeriesDataset(Dataset):
        def __init__(self, series, seq_length=30):
            self.series = series
            self.seq_length = seq_length
        def __len__(self): return len(self.series)-self.seq_length
        def __getitem__(self, idx):
            x = self.series[idx:idx+self.seq_length]
            y = self.series[idx+self.seq_length]
            return torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)

    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size,1)
            self.dropout = nn.Dropout(0.2)
        def forward(self,x):
            out,_ = self.lstm(x.unsqueeze(-1))
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TimeSeriesDataset(series, seq_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for x_batch,y_batch in dataloader:
            x_batch,y_batch = x_batch.to(device),y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch).squeeze(), y_batch)
            loss.backward()
            optimizer.step()
    return model, device

@st.cache_data
def forecast_lstm(model, device, series, steps=30, seq_length=30, n_samples=20):
    model.eval()
    preds_samples=[]
    for _ in range(n_samples):
        for m in model.modules():
            if isinstance(m, nn.Dropout): m.train()
        seq = series[-seq_length:].tolist()
        preds=[]
        for _ in range(steps):
            x = torch.tensor(seq[-seq_length:],dtype=torch.float32).unsqueeze(0).to(device)
            y_pred = model(x).item()
            preds.append(y_pred)
            seq.append(y_pred)
        preds_samples.append(preds)
    preds_samples=np.array(preds_samples)
    mean = preds_samples.mean(axis=0)
    lower = np.percentile(preds_samples,5,axis=0)
    upper = np.percentile(preds_samples,95,axis=0)
    return mean, lower, upper

series = df['value'].values
lstm_model, lstm_device = train_lstm_model(series)
lstm_mean, lstm_lower, lstm_upper = forecast_lstm(lstm_model,lstm_device,series)

# -----------------------------
# Step 5: TFT (optimized + cached)
# -----------------------------
@st.cache_data
def train_tft(df, max_encoder_length=30, max_prediction_length=30):
    df_copy = df.copy()
    df_copy['time_idx']=np.arange(len(df_copy))
    df_copy['series_id']=0
    training = TimeSeriesDataSet(
        df_copy[:-max_prediction_length],
        time_idx='time_idx', target='value', group_ids=['series_id'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length
    )
    validation = TimeSeriesDataSet.from_dataset(training, df_copy[-max_prediction_length:], predict=True, stop_randomization=True)
    train_dl = training.to_dataloader(train=True, batch_size=16)
    val_dl = validation.to_dataloader(train=False, batch_size=16)
    tft_model = TemporalFusionTransformer.from_dataset(
        training, learning_rate=0.03, hidden_size=8, attention_head_size=1,
        dropout=0.1, hidden_continuous_size=4, output_size=1, loss=QuantileLoss()
    )
    trainer = Trainer(max_epochs=2, enable_progress_bar=False)
    trainer.fit(tft_model, train_dl, val_dl)
    return tft_model, val_dl

tft_model, tft_val_dl = train_tft(df)
raw_pred, x = tft_model.predict(tft_val_dl, mode="raw", return_x=True)
tft_forecast = raw_pred['prediction'].numpy().flatten()
tft_lower = raw_pred['prediction'][...,0].numpy().flatten()
tft_upper = raw_pred['prediction'][...,2].numpy().flatten()

# -----------------------------
# Step 6: Evaluation & Plot
# -----------------------------
true_future = series[-30:]
metrics_df = pd.DataFrame({
    'Model': ['ARIMA','Prophet','LSTM','TFT'],
    'RMSE': [
        rmse(true_future, arima_forecast.values[:30]),
        rmse(true_future, prophet_forecast['yhat'].values[-30:]),
        rmse(true_future, lstm_mean),
        rmse(true_future, tft_forecast)
    ],
    'MAPE': [
        mape(true_future, arima_forecast.values[:30]),
        mape(true_future, prophet_forecast['yhat'].values[-30:]),
        mape(true_future, lstm_mean),
        mape(true_future, tft_forecast)
    ]
})
st.subheader("Forecast Evaluation Metrics")
st.dataframe(metrics_df)

forecast_dates = pd.date_range(start=df['date'].iloc[-1]+pd.Timedelta(days=1), periods=30)
st.subheader("Forecast Comparison (Next 30 Days)")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines', name='Actual', line=dict(color='black', dash='dash')))
fig3.add_trace(go.Scatter(x=forecast_dates, y=arima_forecast.values[:30], mode='lines', name='ARIMA'))
fig3.add_trace(go.Scatter(x=forecast_dates, y=prophet_forecast['yhat'].values[-30:], mode='lines', name='Prophet'))
fig3.add_trace(go.Scatter(x=forecast_dates, y=lstm_mean, mode='lines', name='LSTM Mean', line=dict(color='blue')))
fig3.add_trace(go.Scatter(x=forecast_dates, y=lstm_upper, fill=None, mode='lines', line=dict(color='blue', dash='dot'), showlegend=False))
fig3.add_trace(go.Scatter(x=forecast_dates, y=lstm_lower, fill='tonexty', mode='lines', line=dict(color='blue', dash='dot'), name='LSTM 90% CI'))
fig3.add_trace(go.Scatter(x=forecast_dates, y=tft_forecast, mode='lines', name='TFT Mean', line=dict(color='red')))
fig3.add_trace(go.Scatter(x=forecast_dates, y=tft_upper, fill=None, mode='lines', line=dict(color='red', dash='dot'), showlegend=False))
fig3.add_trace(go.Scatter(x=forecast_dates, y=tft_lower, fill='tonexty', mode='lines', line=dict(color='red', dash='dot'), name='TFT 80% CI'))

st.plotly_chart(fig3)
