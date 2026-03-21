# ========================================
# run_dashboard.py
# Fully integrated forecasting dashboard for 33 stocks
# With live progress for LSTM + TFT
# ========================================

# -----------------------------
# Imports
# -----------------------------
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

# Classical models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Deep learning
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# TFT
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer

# -----------------------------
# 1. Metrics
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# -----------------------------
# 2. Merge all individual stocks
# -----------------------------
def merge_stocks(individual_folder="individual_stocks", processed_folder="processed_stocks"):
    os.makedirs(processed_folder, exist_ok=True)
    stock_files = [f for f in os.listdir(individual_folder) if f.endswith(".csv")]
    st.info(f"Processing {len(stock_files)} stocks...")
    progress_bar = st.progress(0)

    for i, f in enumerate(stock_files):
        try:
            df = pd.read_csv(os.path.join(individual_folder, f))
            # Detect columns
            if 'date' not in df.columns:
                date_col = [c for c in df.columns if 'date' in c.lower()]
                value_col = [c for c in df.columns if 'close' in c.lower() or 'value' in c.lower() or 'adj close' in c.lower()]
                if date_col and value_col:
                    df = df[[date_col[0], value_col[0]]]
                    df.columns = ['date', 'value']
                else:
                    st.warning(f"Skipping {f}: cannot find date/value columns")
                    continue
            else:
                if 'value' not in df.columns:
                    val_col = [c for c in df.columns if 'close' in c.lower() or 'adj close' in c.lower()]
                    df['value'] = df[val_col[0]]
                df = df[['date', 'value']]

            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df.fillna(method='ffill', inplace=True)
            df.to_csv(os.path.join(processed_folder, f"{os.path.splitext(f)[0]}.csv"), index=False)
        except Exception as e:
            st.warning(f"Error processing {f}: {e}")
        progress_bar.progress((i+1)/len(stock_files))
    progress_bar.empty()
    st.success("✅ All stocks processed successfully!")

# -----------------------------
# 3. Load data
# -----------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

# -----------------------------
# 4. Classical Models
# -----------------------------
def train_arima(df, steps=30, order=(5,1,0)):
    model = ARIMA(df['value'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def train_prophet(df, steps=30):
    df_prophet = df.rename(columns={'date':'ds','value':'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast[['ds','yhat']]

# -----------------------------
# 5. LSTM Model
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
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,1)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        out,_ = self.lstm(x.unsqueeze(-1))
        out = self.dropout(out[:,-1,:])
        out = self.fc(out)
        return out

def train_lstm_model(series, progress_text, progress_bar, epochs=10, seq_length=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TimeSeriesDataset(series, seq_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        progress_text.text(f"LSTM Training: Epoch {epoch+1}/{epochs}")
        for i, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        progress_bar.progress((epoch+1)/epochs)
    progress_text.text("LSTM Training Done ✅")
    progress_bar.empty()
    return model, device

def forecast_lstm(model, series, device, steps=30, seq_length=30, n_samples=50):
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
    mean = preds_samples.mean(axis=0)
    lower = np.percentile(preds_samples,5,axis=0)
    upper = np.percentile(preds_samples,95,axis=0)
    return mean, lower, upper

# -----------------------------
# 6. TFT Model
# -----------------------------
def train_tft_model(df, progress_text, progress_bar, max_encoder=30, max_pred=30, epochs=5):
    if len(df) < max_encoder + max_pred:
        progress_text.text("TFT skipped: series too short ⚠️")
        return None, None
    tft_df = df.copy()
    tft_df['time_idx'] = np.arange(len(tft_df))
    tft_df['series_id'] = 0
    try:
        training = TimeSeriesDataSet(
            tft_df[:-max_pred],
            time_idx='time_idx',
            target='value',
            group_ids=['series_id'],
            max_encoder_length=max_encoder,
            max_prediction_length=max_pred
        )
        validation = TimeSeriesDataSet.from_dataset(training, tft_df[-max_pred:], predict=True, stop_randomization=True)
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
            loss=QuantileLoss()
        )
        trainer = Trainer(max_epochs=epochs, enable_progress_bar=False)
        for epoch in range(epochs):
            progress_text.text(f"TFT Training: Epoch {epoch+1}/{epochs}")
            trainer.fit(tft_model, train_loader, val_loader)
            progress_bar.progress((epoch+1)/epochs)
        progress_text.text("TFT Training Done ✅")
        progress_bar.empty()
        return tft_model, val_loader
    except Exception as e:
        progress_text.text(f"TFT skipped: {e}")
        return None, None

# -----------------------------
# 7. Main App
# -----------------------------
def main():
    st.set_page_config(page_title="Time-Series Forecasting Dashboard", layout="wide")
    st.title("📈 Time-Series Forecasting Dashboard ")
    
    # Merge stocks
    merge_stocks()
    
    # Select stock
    data_folder = "processed_stocks"
    stock_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".csv")])
    stock_selection = st.selectbox("Select a stock:", stock_files)
    df = load_data(os.path.join(data_folder, stock_selection))
    
    st.subheader(f"Dataset Overview: {stock_selection}")
    st.dataframe(df.tail())
    
    # Original series chart
    fig_orig = px.line(df, x='date', y='value', title="Original Time Series", template="plotly_dark")
    st.plotly_chart(fig_orig)
    
    # Classical models
    arima_forecast = train_arima(df).values
    prophet_forecast = train_prophet(df)['yhat'].values
    
    # LSTM
    lstm_text = st.empty()
    lstm_bar = st.progress(0)
    lstm_model, lstm_device = train_lstm_model(df['value'].values, lstm_text, lstm_bar)
    lstm_mean, lstm_lower, lstm_upper = forecast_lstm(lstm_model, df['value'].values, lstm_device)
    
    # TFT
    tft_text = st.empty()
    tft_bar = st.progress(0)
    tft_model, val_loader = train_tft_model(df, tft_text, tft_bar)
    tft_available = tft_model is not None
    if tft_available:
        raw_pred, _ = tft_model.predict(val_loader, mode="raw", return_x=True)
        tft_forecast = raw_pred['prediction'].numpy().flatten()
        tft_lower = raw_pred['prediction'][...,0].numpy().flatten()
        tft_upper = raw_pred['prediction'][...,2].numpy().flatten()
    else:
        tft_forecast = tft_lower = tft_upper = None
    
    # Forecast comparison chart
    forecast_dates = pd.date_range(df['date'].iloc[-1]+pd.Timedelta(days=1), periods=30)
    fig = go.Figure()
    
    # Dark theme colors
    colors = {
        'Actual':'lightblue','ARIMA':'orange','Prophet':'green',
        'LSTM Mean':'cyan','LSTM CI':'lightcyan','TFT Mean':'magenta','TFT CI':'pink'
    }
    
    # Actual
    fig.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines', name='Actual', line=dict(color=colors['Actual'], dash='solid')))
    # ARIMA
    fig.add_trace(go.Scatter(x=forecast_dates, y=arima_forecast[:30], mode='lines', name='ARIMA', line=dict(color=colors['ARIMA'])))
    # Prophet
    fig.add_trace(go.Scatter(x=forecast_dates, y=prophet_forecast[:30], mode='lines', name='Prophet', line=dict(color=colors['Prophet'])))
    # LSTM
    fig.add_trace(go.Scatter(x=forecast_dates, y=lstm_mean, mode='lines', name='LSTM Mean', line=dict(color=colors['LSTM Mean'])))
    fig.add_trace(go.Scatter(x=forecast_dates, y=lstm_upper, fill=None, mode='lines', line=dict(color=colors['LSTM CI'], dash='dot'), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_dates, y=lstm_lower, fill='tonexty', mode='lines', line=dict(color=colors['LSTM CI'], dash='dot'), name='LSTM 90% CI'))
    # TFT
    if tft_available:
        fig.add_trace(go.Scatter(x=forecast_dates, y=tft_forecast, mode='lines', name='TFT Mean', line=dict(color=colors['TFT Mean'])))
        fig.add_trace(go.Scatter(x=forecast_dates, y=tft_upper, fill=None, mode='lines', line=dict(color=colors['TFT CI'], dash='dot'), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_dates, y=tft_lower, fill='tonexty', mode='lines', line=dict(color=colors['TFT CI'], dash='dot'), name='TFT 80% CI'))
    
    fig.update_layout(title="Forecast Comparison (Next 30 Days)", xaxis_title="Date", yaxis_title="Value", template="plotly_dark")
    st.subheader("Forecast Comparison (Next 30 Days)")
    st.plotly_chart(fig)

# -----------------------------
# 8. Run
# -----------------------------
if __name__ == "__main__":
    main()
