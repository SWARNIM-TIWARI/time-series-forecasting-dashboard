# ========================================
# run_dashboard.py - Merge + Streamlit Dashboard
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
from torch.utils.data import Dataset, DataLoader

# TFT
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer

# -----------------------------
# Helper functions
# -----------------------------
def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred)**2))
def mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred)/y_true))*100

# -----------------------------
# 1. Merge all stock CSVs into processed_stocks
# -----------------------------
def merge_all_stocks(input_folder="individual_stocks", output_folder="processed_stocks"):
    os.makedirs(output_folder, exist_ok=True)
    stock_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    
    st.info("🔄 Merging all stock CSVs...")
    progress = st.progress(0)
    
    for i, f in enumerate(tqdm(stock_files)):
        try:
            df = pd.read_csv(os.path.join(input_folder,f))
            # Detect the date column automatically
            date_col = None
            for col in df.columns:
                if 'date' in col.lower():
                    date_col = col
                    break
            if date_col is None:
                st.warning(f"Skipping {f}, no date column detected.")
                continue
            
            # Detect value column (just pick the second column if no 'close/price/value')
            value_col = None
            for col in df.columns:
                if col.lower() in ['close','price','value']:
                    value_col = col
                    break
            if value_col is None:
                value_col = df.columns[1]
            
            df = df[[date_col, value_col]].rename(columns={date_col:'date', value_col:'value'})
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df.to_csv(os.path.join(output_folder,f"{f.replace('.csv','')}_processed.csv"), index=False)
        except Exception as e:
            st.error(f"Error processing {f}: {e}")
        progress.progress((i+1)/len(stock_files))
    st.success("✅ All stocks merged successfully!")

# -----------------------------
# 2. Load dataset
# -----------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

# -----------------------------
# 3. Classical Models
# -----------------------------
def train_arima(df, steps=30, order=(5,1,0)):
    model = ARIMA(df['value'], order=order)
    return model.fit().forecast(steps=steps)

def train_prophet(df, steps=30):
    dfp = df.rename(columns={'date':'ds','value':'y'})
    model = Prophet()
    model.fit(dfp)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast[['ds','yhat']]

# -----------------------------
# 4. LSTM
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_length=30):
        self.series, self.seq_length = series, seq_length
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

@st.cache_resource(show_spinner=True)
def train_lstm_model(series, seq_length=30, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TimeSeriesDataset(series, seq_length)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch).squeeze(), y_batch)
            loss.backward()
            optimizer.step()
    return model

def forecast_lstm(model, series, seq_length=30, steps=30, n_samples=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
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
    preds_samples = np.array(preds_samples)
    mean = preds_samples.mean(axis=0)
    lower = np.percentile(preds_samples,5,axis=0)
    upper = np.percentile(preds_samples,95,axis=0)
    return mean, lower, upper

# -----------------------------
# 5. Streamlit App
# -----------------------------
def main():
    st.title("📈 Time-Series Forecasting Dashboard")
    
    # Merge stocks
    merge_all_stocks()

    # List processed stocks
    data_folder = "processed_stocks"
    stock_files = sorted([f for f in os.listdir(data_folder) if f.endswith("_processed.csv")])
    stock_selection = st.selectbox("Select a stock:", stock_files)
    
    df = load_data(os.path.join(data_folder, stock_selection))
    st.subheader(f"Dataset Preview: {stock_selection}")
    st.dataframe(df.tail())

    fig = px.line(df, x='date', y='value', title="Original Time Series")
    st.plotly_chart(fig)

    # Forecast
    st.subheader("Forecasting Models")
    arima_forecast = train_arima(df)
    prophet_forecast = train_prophet(df)

    series = df['value'].values
    lstm_model = train_lstm_model(series)
    lstm_mean, lstm_lower, lstm_upper = forecast_lstm(lstm_model, series)

    # Show metrics
    true_future = series[-30:]
    metrics_df = pd.DataFrame({
        'Model': ['ARIMA','Prophet','LSTM'],
        'RMSE': [rmse(true_future, arima_forecast[:30]),
                 rmse(true_future, prophet_forecast['yhat'].values[-30:]),
                 rmse(true_future, lstm_mean)],
        'MAPE': [mape(true_future, arima_forecast[:30]),
                 mape(true_future, prophet_forecast['yhat'].values[-30:]),
                 mape(true_future, lstm_mean)]
    })
    st.subheader("Forecast Evaluation")
    st.dataframe(metrics_df)

    # Plot forecasts
    forecast_dates = pd.date_range(df['date'].iloc[-1]+pd.Timedelta(days=1), periods=30)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines', name='Actual'))
    fig2.add_trace(go.Scatter(x=forecast_dates, y=arima_forecast[:30], mode='lines', name='ARIMA'))
    fig2.add_trace(go.Scatter(x=forecast_dates, y=prophet_forecast['yhat'].values[-30:], mode='lines', name='Prophet'))
    fig2.add_trace(go.Scatter(x=forecast_dates, y=lstm_mean, mode='lines', name='LSTM Mean'))
    fig2.add_trace(go.Scatter(x=forecast_dates, y=lstm_upper, fill=None, mode='lines', line=dict(color='blue', dash='dot'), showlegend=False))
    fig2.add_trace(go.Scatter(x=forecast_dates, y=lstm_lower, fill='tonexty', mode='lines', line=dict(color='blue', dash='dot'), name='LSTM 90% CI'))
    st.plotly_chart(fig2)

# -----------------------------
if __name__ == "__main__":
    main()
