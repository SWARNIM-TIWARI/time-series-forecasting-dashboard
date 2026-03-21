# ========================================
# Time-Series Forecasting Dashboard - Deployable
# With LSTM & TFT Uncertainty Intervals
# ========================================

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer

# -----------------------------
# Metrics
# -----------------------------
def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred)**2))
def mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred)/y_true))*100

# -----------------------------
# Load & preprocess
# -----------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

st.title("Time-Series Forecasting Dashboard")
data_file = "your_timeseries.csv"  # 500-row stock CSV
df = load_data(data_file)
st.subheader("Dataset Overview")
st.dataframe(df.tail())

# -----------------------------
# ARIMA
# -----------------------------
def train_arima(df, steps=30, order=(5,1,0)):
    model = ARIMA(df['value'], order=order)
    return model.fit().forecast(steps=steps)

arima_forecast = train_arima(df)

# -----------------------------
# Prophet
# -----------------------------
def train_prophet(df, steps=30):
    df_prophet = df.rename(columns={'date':'ds','value':'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast[['ds','yhat']]

prophet_forecast = train_prophet(df)

# -----------------------------
# LSTM
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_length=30):
        self.series = series
        self.seq_length = seq_length
    def __len__(self): return len(self.series)-self.seq_length
    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_length]
        y = self.series[idx+self.seq_length]
        return torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)

series = df['value'].values
seq_length = 30
dataset = TimeSeriesDataset(series, seq_length)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=64,num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size,1)
    def forward(self,x):
        out,_ = self.lstm(x.unsqueeze(-1))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

def train_lstm(model,dataloader,epochs=10):
    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch).squeeze(), y_batch)
            loss.backward()
            optimizer.step()
    st.write("✅ LSTM Training Complete")

train_lstm(lstm_model, dataloader)

def forecast_lstm_with_uncertainty(model, series, steps=30, seq_length=30, n_samples=50):
    model.eval()
    preds_samples=[]
    for _ in range(n_samples):
        for m in model.modules():
            if isinstance(m, nn.Dropout): m.train()  # MC Dropout
        input_seq = series[-seq_length:].tolist()
        preds=[]
        for _ in range(steps):
            x = torch.tensor(input_seq[-seq_length:],dtype=torch.float32).unsqueeze(0).to(device)
            y_pred = model(x).item()
            preds.append(y_pred)
            input_seq.append(y_pred)
        preds_samples.append(preds)
    preds_samples = np.array(preds_samples)
    mean_preds = preds_samples.mean(axis=0)
    lower = np.percentile(preds_samples,5,axis=0)
    upper = np.percentile(preds_samples,95,axis=0)
    return mean_preds, lower, upper

lstm_mean, lstm_lower, lstm_upper = forecast_lstm_with_uncertainty(lstm_model, series)

# -----------------------------
# TFT
# -----------------------------
tft_df = df.copy()
tft_df['time_idx'] = np.arange(len(tft_df))
tft_df['series_id'] = 0

max_encoder_length = 30
max_prediction_length = 30

training = TimeSeriesDataSet(
    tft_df[:-max_prediction_length],
    time_idx='time_idx', target='value', group_ids=['series_id'],
    max_encoder_length=max_encoder_length, max_prediction_length=max_prediction_length
)

validation = TimeSeriesDataSet.from_dataset(training, tft_df[-max_prediction_length:], predict=True, stop_randomization=True)
train_loader = training.to_dataloader(train=True,batch_size=16)
val_loader = validation.to_dataloader(train=False,batch_size=16)

tft_model = TemporalFusionTransformer.from_dataset(training, learning_rate=0.03,
    hidden_size=16, attention_head_size=1, dropout=0.1,
    hidden_continuous_size=8, output_size=1, loss=QuantileLoss()
)

trainer = Trainer(max_epochs=5, enable_progress_bar=False)
trainer.fit(tft_model, train_loader, val_loader)

raw_pred, x = tft_model.predict(val_loader, mode="raw", return_x=True)
tft_forecast = raw_pred['prediction'].numpy().flatten()
tft_lower = raw_pred['prediction'][...,0].numpy().flatten()
tft_upper = raw_pred['prediction'][...,2].numpy().flatten()

# -----------------------------
# Evaluation
# -----------------------------
true_future = series[-30:]
metrics_df = pd.DataFrame({
    'Model':['ARIMA','Prophet','LSTM','TFT'],
    'RMSE':[rmse(true_future, arima_forecast.values[:30]),
           rmse(true_future, prophet_forecast['yhat'].values[-30:]),
           rmse(true_future, lstm_mean),
           rmse(true_future, tft_forecast)],
    'MAPE':[mape(true_future, arima_forecast.values[:30]),
            mape(true_future, prophet_forecast['yhat'].values[-30:]),
            mape(true_future, lstm_mean),
            mape(true_future, tft_forecast)]
})
st.subheader("Forecast Metrics")
st.dataframe(metrics_df)

# -----------------------------
# Forecast Comparison with Confidence Intervals
# -----------------------------
forecast_dates = pd.date_range(df['date'].iloc[-1]+pd.Timedelta(days=1), periods=30)
st.subheader("Forecast Comparison (Next 30 Days)")
fig3 = go.Figure()
# Actual
fig3.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines', name='Actual', line=dict(color='black', dash='dash')))
# ARIMA & Prophet
fig3.add_trace(go.Scatter(x=forecast_dates, y=arima_forecast.values[:30], mode='lines', name='ARIMA'))
fig3.add_trace(go.Scatter(x=forecast_dates, y=prophet_forecast['yhat'].values[-30:], mode='lines', name='Prophet'))
# LSTM with CI
fig3.add_trace(go.Scatter(x=forecast_dates, y=lstm_mean, mode='lines', name='LSTM Mean', line=dict(color='blue')))
fig3.add_trace(go.Scatter(x=forecast_dates, y=lstm_upper, fill=None, mode='lines', line=dict(color='blue', dash='dot'), showlegend=False))
fig3.add_trace(go.Scatter(x=forecast_dates, y=lstm_lower, fill='tonexty', mode='lines', line=dict(color='blue', dash='dot'), name='LSTM 90% CI'))
# TFT with CI
fig3.add_trace(go.Scatter(x=forecast_dates, y=tft_forecast, mode='lines', name='TFT Mean', line=dict(color='red')))
fig3.add_trace(go.Scatter(x=forecast_dates, y=tft_upper, fill=None, mode='lines', line=dict(color='red', dash='dot'), showlegend=False))
fig3.add_trace(go.Scatter(x=forecast_dates, y=tft_lower, fill='tonexty', mode='lines', line=dict(color='red', dash='dot'), name='TFT 80% CI'))
st.plotly_chart(fig3)
