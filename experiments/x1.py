# ===============================
# Time-Series Forecasting Project
# ===============================

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Classical Models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Deep Learning
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# TFT (PyTorch Forecasting)
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import RMSE, QuantileLoss

# Dashboard
import streamlit as st
import plotly.express as px

# --------------------------
# 1. Data Loading & Preprocessing
# --------------------------
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

data_file = "your_timeseries.csv"  # CSV with 'date' and 'value' columns
df = load_data(data_file)

st.title("Time-Series Forecasting Dashboard")
st.write("Dataset Overview")
st.dataframe(df.head())

# Plot original time series
fig = px.line(df, x='date', y='value', title='Original Time Series')
st.plotly_chart(fig)

# --------------------------
# 2. Classical Models
# --------------------------

# ---- ARIMA ----
def train_arima(df, order=(5,1,0)):
    model = ARIMA(df['value'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)  # predict next 30 points
    return forecast

arima_forecast = train_arima(df)
st.write("ARIMA Forecast (Next 30 Steps)")
st.line_chart(arima_forecast)

# ---- Prophet ----
def train_prophet(df):
    df_prophet = df.rename(columns={'date': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds','yhat']]

prophet_forecast = train_prophet(df)
st.write("Prophet Forecast (Next 30 Steps)")
st.line_chart(prophet_forecast.set_index('ds')['yhat'])

# --------------------------
# 3. Deep Learning Models
# --------------------------

# ---- LSTM ----
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

# Prepare data
series = df['value'].values
seq_length = 30
dataset = TimeSeriesDataset(series, seq_length)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        out = self.fc(out[:, -1, :])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

# Train LSTM (skeleton)
def train_lstm(model, dataloader, epochs=5):
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
    st.write("LSTM Training Complete")

train_lstm(lstm_model, dataloader)

# --------------------------
# 4. Temporal Fusion Transformer (TFT)
# --------------------------
# For TFT, the dataset must be in long format: id, time_idx, value, optional covariates
tft_df = df.copy()
tft_df['time_idx'] = np.arange(len(tft_df))
tft_df['series_id'] = 0  # single series

max_encoder_length = 30
max_prediction_length = 30

training = TimeSeriesDataSet(
    tft_df[:-30],
    time_idx='time_idx',
    target='value',
    group_ids=['series_id'],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
)

validation = TimeSeriesDataSet.from_dataset(training, tft_df[-30:], predict=True, stop_randomization=True)
train_dataloader = training.to_dataloader(train=True, batch_size=16)
val_dataloader = validation.to_dataloader(train=False, batch_size=16)

tft_model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=1,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# Skeleton training
# trainer = pl.Trainer(max_epochs=5)
# trainer.fit(tft_model, train_dataloader, val_dataloader)

st.write("TFT model setup complete (training skeleton ready)")

# --------------------------
# 5. Streamlit Dashboard Placeholders
# --------------------------
st.subheader("Forecast Comparison")
st.line_chart(pd.DataFrame({
    'ARIMA': arima_forecast,
    'Prophet': prophet_forecast['yhat'].values[:30],
    # LSTM & TFT predictions will go here after training
}))

st.write("Add LSTM and TFT predictions after training completes.")
