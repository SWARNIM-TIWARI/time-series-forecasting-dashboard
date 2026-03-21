# ========================================
# Stock Forecasting Dashboard
# Models: ARIMA, Prophet, LSTM, TFT
# ========================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Stats models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Deep learning stuff
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# TFT imports
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Metrics - basic stuff
# -----------------------------
def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred)**2))

def mape(actual, pred):
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

# -----------------------------
# Merge CSVs from individual_stocks folder
# -----------------------------
def process_stock_files(raw_folder="individual_stocks", clean_folder="processed_stocks"):
    os.makedirs(clean_folder, exist_ok=True)
    
    if not os.path.exists(raw_folder):
        st.warning(f"Create '{raw_folder}' folder and add your CSV files!")
        os.makedirs(raw_folder, exist_ok=True)
        return
    
    files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]
    if not files:
        st.warning(f"No CSV files found in '{raw_folder}'")
        return
    
    st.info(f"Processing {len(files)} stock files...")
    pbar = st.progress(0)
    
    for i, fname in enumerate(files):
        try:
            df = pd.read_csv(os.path.join(raw_folder, fname))
            
            # Find date and value columns (case insensitive)
            date_col = next((c for c in df.columns if 'date' in c.lower()), None)
            value_col = next((c for c in df.columns if any(x in c.lower() for x in ['close', 'value', 'price', 'adj close'])), None)
            
            if not date_col or not value_col:
                st.warning(f"Skipping {fname} - can't find date/value columns")
                continue
            
            df = df[[date_col, value_col]].copy()
            df.columns = ['date', 'value']
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df = df.ffill()
            
            output = os.path.join(clean_folder, fname)
            df.to_csv(output, index=False)
            
        except Exception as e:
            st.warning(f"Error with {fname}: {e}")
        
        pbar.progress((i+1)/len(files))
    
    pbar.empty()
    st.success(f"✅ Processed {len(files)} files")

# -----------------------------
# Load data with cache
# -----------------------------
@st.cache_data
def load_stock(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

# -----------------------------
# ARIMA Forecasting
# -----------------------------
def forecast_arima(df, horizon=30):
    try:
        model = ARIMA(df['value'], order=(5,1,0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        return forecast.values
    except:
        # Fallback: just repeat last value
        return np.full(horizon, df['value'].iloc[-1])

# -----------------------------
# Prophet Forecasting
# -----------------------------
def forecast_prophet(df, horizon=30):
    try:
        df_p = df[['date', 'value']].rename(columns={'date':'ds', 'value':'y'})
        m = Prophet(daily_seasonality=False, weekly_seasonality=False)
        m.fit(df_p)
        future = m.make_future_dataframe(periods=horizon)
        pred = m.predict(future)
        return pred['yhat'].values[-horizon:]
    except:
        return np.full(horizon, df['value'].iloc[-1])

# -----------------------------
# LSTM Model
# -----------------------------
class StockDataset(Dataset):
    def __init__(self, data, window=30):
        self.data = data
        self.window = window
    
    def __len__(self):
        return max(0, len(self.data) - self.window)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window]
        y = self.data[idx+self.window]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMNet(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)  # add feature dim
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm(data, pbar_txt, pbar, epochs=10, window=30):
    if len(data) < window + 20:
        pbar_txt.text(f"⚠️ LSTM needs {window+20}+ points, got {len(data)}")
        pbar.empty()
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StockDataset(data, window)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = LSTMNet().to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for e in range(epochs):
        pbar_txt.text(f"🔄 LSTM: Epoch {e+1}/{epochs}")
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        pbar.progress((e+1)/epochs)
    
    pbar_txt.text("✅ LSTM Done")
    pbar.empty()
    return model

def predict_lstm(model, data, horizon=30, window=30, samples=50):
    if model is None:
        return np.full(horizon, np.nan), np.full(horizon, np.nan), np.full(horizon, np.nan)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Monte Carlo dropout for uncertainty
    all_preds = []
    for _ in range(samples):
        model.train()  # enable dropout
        seq = data[-window:].tolist()
        preds = []
        for _ in range(horizon):
            x = torch.FloatTensor(seq[-window:]).unsqueeze(0).to(device)
            with torch.no_grad():
                y = model(x).item()
            preds.append(y)
            seq.append(y)
        all_preds.append(preds)
    
    all_preds = np.array(all_preds)
    mean = all_preds.mean(axis=0)
    lower = np.percentile(all_preds, 5, axis=0)
    upper = np.percentile(all_preds, 95, axis=0)
    return mean, lower, upper

# -----------------------------
# TFT Model
# -----------------------------
class TFTProgressCallback(Callback):
    def __init__(self, txt, bar, total):
        super().__init__()
        self.txt = txt
        self.bar = bar
        self.total = total
    
    def on_epoch_end(self, trainer, model):
        epoch = trainer.current_epoch + 1
        self.txt.text(f"🔄 TFT: Epoch {epoch}/{self.total}")
        self.bar.progress(epoch/self.total)

def train_tft(df, pbar_txt, pbar, epochs=5):
    n = len(df)
    min_needed = 100
    
    if n < min_needed:
        pbar_txt.text(f"⚠️ TFT needs {min_needed}+ points, got {n}")
        pbar.empty()
        return None
    
    # Adaptive window sizing
    enc_len = min(30, n//4)
    pred_len = min(30, n//4)
    
    if enc_len < 10 or pred_len < 10:
        pbar_txt.text("⚠️ TFT: Not enough data for stable training")
        pbar.empty()
        return None
    
    df_tft = df.copy()
    df_tft['time_idx'] = range(len(df_tft))
    df_tft['group'] = 'stock'
    
    try:
        cutoff = n - pred_len
        
        train_data = TimeSeriesDataSet(
            df_tft[:cutoff],
            time_idx='time_idx',
            target='value',
            group_ids=['group'],
            max_encoder_length=enc_len,
            max_prediction_length=pred_len,
        )
        
        val_data = TimeSeriesDataSet.from_dataset(train_data, df_tft, predict=True)
        
        train_loader = train_data.to_dataloader(train=True, batch_size=16, num_workers=0)
        val_loader = val_data.to_dataloader(train=False, batch_size=16, num_workers=0)
        
        tft = TemporalFusionTransformer.from_dataset(
            train_data,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,
            loss=QuantileLoss(),
        )
        
        callback = TFTProgressCallback(pbar_txt, pbar, epochs)
        trainer = Trainer(
            max_epochs=epochs,
            enable_progress_bar=False,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            accelerator='cpu'
        )
        
        trainer.fit(tft, train_loader, val_loader)
        
        pbar_txt.text("✅ TFT Done")
        pbar.empty()
        return tft, val_loader
        
    except Exception as e:
        pbar_txt.text(f"⚠️ TFT failed: {str(e)[:60]}")
        pbar.empty()
        return None

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
    st.title("📈 Multi-Model Stock Forecasting")
    
    # Process files
    process_stock_files()
    
    # Load available stocks
    folder = "processed_stocks"
    if not os.path.exists(folder):
        st.error("No processed stocks found!")
        return
    
    stocks = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
    if not stocks:
        st.error("No stock files available!")
        return
    
    selected = st.selectbox("Select Stock:", stocks)
    df = load_stock(os.path.join(folder, selected))
    
    # Show data overview
    st.subheader(f"📊 {selected}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Data Points", len(df))
    c2.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
    c3.metric("Latest Price", f"${df['value'].iloc[-1]:.2f}")
    
    st.dataframe(df.tail(10), use_container_width=True)
    
    # Historical chart
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=df['date'], y=df['value'], name='Historical', line=dict(color='lightblue')))
    fig_hist.update_layout(title="Historical Price", template="plotly_dark", height=400)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # FORECAST SECTION
    st.subheader("🔮 Generate 30-Day Forecast")
    horizon = 30
    
    if st.button("Run Forecasting Models", type="primary"):
        
        # Classical models (fast)
        with st.spinner("Running ARIMA..."):
            arima_fc = forecast_arima(df, horizon)
        st.success("✅ ARIMA")
        
        with st.spinner("Running Prophet..."):
            prophet_fc = forecast_prophet(df, horizon)
        st.success("✅ Prophet")
        
        # LSTM
        lstm_txt = st.empty()
        lstm_bar = st.progress(0)
        lstm_model = train_lstm(df['value'].values, lstm_txt, lstm_bar)
        lstm_mean, lstm_low, lstm_high = predict_lstm(lstm_model, df['value'].values, horizon)
        
        # TFT
        tft_txt = st.empty()
        tft_bar = st.progress(0)
        tft_result = train_tft(df, tft_txt, tft_bar)
        
        if tft_result is not None:
            tft_model, tft_loader = tft_result
            raw_pred, _ = tft_model.predict(tft_loader, mode="raw", return_x=True)
            tft_fc = raw_pred['prediction'][0, :horizon, 3].numpy()
            tft_low = raw_pred['prediction'][0, :horizon, 1].numpy()
            tft_high = raw_pred['prediction'][0, :horizon, 5].numpy()
        else:
            tft_fc = tft_low = tft_high = np.full(horizon, np.nan)
        
        # Generate future dates
        last_date = df['date'].iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)
        
        # BACKTESTING METRICS (on last 30 historical days)
        st.subheader("📊 Backtest Performance (Last 30 Days)")
        
        train_df = df[:-horizon]
        test_df = df[-horizon:]
        actual = test_df['value'].values
        
        # Retrain on training data for fair comparison
        arima_bt = forecast_arima(train_df, horizon)
        prophet_bt = forecast_prophet(train_df, horizon)
        
        metrics = pd.DataFrame({
            'Model': ['ARIMA', 'Prophet', 'LSTM', 'TFT'],
            'RMSE': [
                rmse(actual, arima_bt),
                rmse(actual, prophet_bt),
                rmse(actual, lstm_mean) if lstm_model else np.nan,
                rmse(actual, tft_fc) if tft_result else np.nan
            ],
            'MAPE (%)': [
                mape(actual, arima_bt),
                mape(actual, prophet_bt),
                mape(actual, lstm_mean) if lstm_model else np.nan,
                mape(actual, tft_fc) if tft_result else np.nan
            ]
        })
        
        st.dataframe(metrics.style.format({'RMSE': '{:.2f}', 'MAPE (%)': '{:.2f}'}).highlight_min(subset=['RMSE', 'MAPE (%)'], color='lightgreen'))
        
        # FORECAST VISUALIZATION
        st.subheader("🔮 Future Forecast (Next 30 Days)")
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['value'], 
            name='Historical',
            line=dict(color='white', width=2)
        ))
        
        # ARIMA forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=arima_fc,
            name='ARIMA',
            line=dict(color='orange', dash='dash')
        ))
        
        # Prophet forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=prophet_fc,
            name='Prophet',
            line=dict(color='green', dash='dash')
        ))
        
        # LSTM with confidence interval
        if lstm_model:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=lstm_mean,
                name='LSTM',
                line=dict(color='cyan', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=future_dates.tolist() + future_dates.tolist()[::-1],
                y=lstm_high.tolist() + lstm_low.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,255,255,0.2)',
                line=dict(width=0),
                name='LSTM 90% CI',
                showlegend=True
            ))
        
        # TFT with confidence interval
        if tft_result:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=tft_fc,
                name='TFT',
                line=dict(color='magenta', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=future_dates.tolist() + future_dates.tolist()[::-1],
                y=tft_high.tolist() + tft_low.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,0,255,0.2)',
                line=dict(width=0),
                name='TFT 80% CI',
                showlegend=True
            ))
        
        # Add vertical line at forecast start
        fig.add_vline(x=last_date, line_dash="dot", line_color="gray", annotation_text="Forecast Start")
        
        fig.update_layout(
            title="Forecast Comparison",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            hovermode='x unified',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 The forecast shows what each model predicts for the next 30 days. Shaded regions = uncertainty intervals.")

if __name__ == "__main__":
    main()