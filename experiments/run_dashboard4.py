# ========================================
# run_dashboard.py - All-in-One TS Forecasting Dashboard
# Supports 33 stocks, ARIMA + Prophet + LSTM + TFT
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
# Merge all CSVs into processed_stocks
# -----------------------------
def merge_stocks(input_folder="individual_stocks", output_folder="processed_stocks"):
    os.makedirs(output_folder, exist_ok=True)
    stock_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    st.info(f"Merging {len(stock_files)} stock files...")
    progress = st.progress(0)

    for i, file in enumerate(stock_files):
        try:
            df = pd.read_csv(os.path.join(input_folder, file))
            # Handle column name variations
            if 'date' not in df.columns:
                for col in df.columns:
                    if 'date' in col.lower():
                        df.rename(columns={col: 'date'}, inplace=True)
            if 'value' not in df.columns:
                for col in df.columns:
                    if 'close' in col.lower() or 'adj close' in col.lower():
                        df.rename(columns={col: 'value'}, inplace=True)
            df = df[['date','value']]
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df.fillna(method='ffill', inplace=True)

            out_file = os.path.join(output_folder, f"{file.split('.')[0]}.csv")
            df.to_csv(out_file, index=False)
        except Exception as e:
            st.warning(f"Error processing {file}: {e}")
        progress.progress((i+1)/len(stock_files))
    st.success("✅ All stocks processed successfully!")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

# -----------------------------
# LSTM Dataset & Model
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_length=30):
        self.series = series
        self.seq_length = seq_length

    def __len__(self):
        return len(self.series) - self.seq_length

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_length]
        y = self.series[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# -----------------------------
# Train LSTM (with caching)
# -----------------------------
@st.cache_resource(show_spinner=True)
def train_lstm_model(series, seq_length=30, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TimeSeriesDataset(series, seq_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
    return model, device

# LSTM Forecast
def forecast_lstm(model, device, series, steps=30, seq_length=30, n_samples=50):
    model.eval()
    preds_samples = []
    for _ in range(n_samples):
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        input_seq = series[-seq_length:].tolist()
        preds = []
        for _ in range(steps):
            x = torch.tensor(input_seq[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
            y_pred = model(x).item()
            preds.append(y_pred)
            input_seq.append(y_pred)
        preds_samples.append(preds)
    preds_samples = np.array(preds_samples)
    mean_preds = preds_samples.mean(axis=0)
    lower = np.percentile(preds_samples, 5, axis=0)
    upper = np.percentile(preds_samples, 95, axis=0)
    return mean_preds, lower, upper

# -----------------------------
# Train TFT (robust)
# -----------------------------
@st.cache_resource(show_spinner=True)
def train_tft_model(df, min_length=10):
    n = len(df)
    if n < min_length:
        st.warning("TFT skipped: series too short")
        return None, None

    max_encoder_length = min(30, n // 2)
    max_prediction_length = min(30, n - max_encoder_length)
    if max_prediction_length < 1:
        st.warning("TFT skipped: not enough data")
        return None, None

    tft_df = df.copy()
    tft_df['time_idx'] = np.arange(len(tft_df))
    tft_df['series_id'] = 0

    try:
        training = TimeSeriesDataSet(
            tft_df[:-max_prediction_length],
            time_idx='time_idx',
            target='value',
            group_ids=['series_id'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
        )
        validation = TimeSeriesDataSet.from_dataset(
            training,
            tft_df[-max_prediction_length:],
            predict=True,
            stop_randomization=True
        )

        train_loader = training.to_dataloader(train=True, batch_size=16)
        val_loader = validation.to_dataloader(train=False, batch_size=16)

        tft_model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=1,
            loss=QuantileLoss(),
        )

        trainer = Trainer(max_epochs=5, enable_progress_bar=False)
        trainer.fit(tft_model, train_loader, val_loader)

        return tft_model, val_loader
    except AssertionError as e:
        st.warning(f"TFT skipped due to dataset too short or invalid: {e}")
        return None, None

# -----------------------------
# Main Dashboard
# -----------------------------
def main():
    st.title("📈 Multi-Model Time Series Forecasting Dashboard")
    merge_stocks()  # Merge all stocks first

    data_folder = "processed_stocks"
    stock_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    stock_files.sort()

    stock_selection = st.selectbox("Select a stock:", stock_files)
    data_file = os.path.join(data_folder, stock_selection)
    df = load_data(data_file)

    st.subheader(f"Dataset Overview: {stock_selection}")
    st.dataframe(df.tail())

    fig = px.line(df, x='date', y='value', title='Original Time Series')
    st.plotly_chart(fig)

    # -----------------------------
    # Classical Models
    # -----------------------------
    # ARIMA
    arima_forecast = ARIMA(df['value'], order=(5,1,0)).fit().forecast(steps=30)

    # Prophet
    df_prophet = df.rename(columns={'date':'ds','value':'y'})
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=30)
    prophet_forecast = prophet_model.predict(future)['yhat'][-30:]

    # -----------------------------
    # LSTM
    # -----------------------------
    lstm_model, lstm_device = train_lstm_model(df['value'].values)
    lstm_mean, lstm_lower, lstm_upper = forecast_lstm(
        lstm_model, lstm_device, df['value'].values
    )

    # -----------------------------
    # TFT
    # -----------------------------
    tft_model, val_loader = train_tft_model(df)
    if tft_model is not None:
        raw_pred, x = tft_model.predict(val_loader, mode="raw", return_x=True)
        tft_forecast = raw_pred['prediction'].numpy().flatten()
        tft_lower = raw_pred['prediction'][...,0].numpy().flatten()
        tft_upper = raw_pred['prediction'][...,2].numpy().flatten()
    else:
        tft_forecast = tft_lower = tft_upper = np.full(30, np.nan)

    # -----------------------------
    # Evaluation Table
    # -----------------------------
    true_future = df['value'].values[-30:]
    metrics_df = pd.DataFrame({
        'Model': ['ARIMA','Prophet','LSTM','TFT'],
        'RMSE': [
            rmse(true_future, arima_forecast[:30]),
            rmse(true_future, prophet_forecast.values),
            rmse(true_future, lstm_mean),
            rmse(true_future, tft_forecast)
        ],
        'MAPE': [
            mape(true_future, arima_forecast[:30]),
            mape(true_future, prophet_forecast.values),
            mape(true_future, lstm_mean),
            mape(true_future, tft_forecast)
        ]
    })
    st.subheader("Forecast Evaluation Metrics")
    st.dataframe(metrics_df)

    # -----------------------------
    # Forecast Comparison Plot
    # -----------------------------
    forecast_dates = pd.date_range(df['date'].iloc[-1]+pd.Timedelta(days=1), periods=30)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines', name='Actual', line=dict(color='black', dash='dash')))
    fig3.add_trace(go.Scatter(x=forecast_dates, y=arima_forecast[:30], mode='lines', name='ARIMA'))
    fig3.add_trace(go.Scatter(x=forecast_dates, y=prophet_forecast.values, mode='lines', name='Prophet'))
    fig3.add_trace(go.Scatter(x=forecast_dates, y=lstm_mean, mode='lines', name='LSTM Mean', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=forecast_dates, y=lstm_upper, fill=None, mode='lines', line=dict(color='blue', dash='dot'), showlegend=False))
    fig3.add_trace(go.Scatter(x=forecast_dates, y=lstm_lower, fill='tonexty', mode='lines', line=dict(color='blue', dash='dot'), name='LSTM 90% CI'))
    fig3.add_trace(go.Scatter(x=forecast_dates, y=tft_forecast, mode='lines', name='TFT Mean', line=dict(color='red')))
    fig3.add_trace(go.Scatter(x=forecast_dates, y=tft_upper, fill=None, mode='lines', line=dict(color='red', dash='dot'), showlegend=False))
    fig3.add_trace(go.Scatter(x=forecast_dates, y=tft_lower, fill='tonexty', mode='lines', line=dict(color='red', dash='dot'), name='TFT 80% CI'))
    st.subheader("Forecast Comparison (Next 30 Days)")
    st.plotly_chart(fig3)

if __name__ == "__main__":
    main()
