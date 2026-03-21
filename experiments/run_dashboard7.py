# ========================================
# run_dashboard.py - OPTIMIZED VERSION (FIXED TFT)
# Robust handling of small datasets
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
from pytorch_lightning.callbacks import Callback

# -----------------------------
# 1. Metrics
# -----------------------------
def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# -----------------------------
# 2. Merge all individual stocks
# -----------------------------
def merge_stocks(individual_folder="individual_stocks", processed_folder="processed_stocks"):
    """Process and standardize all stock CSV files"""
    os.makedirs(processed_folder, exist_ok=True)
    
    if not os.path.exists(individual_folder):
        st.warning(f"⚠️ Folder '{individual_folder}' not found. Creating it...")
        os.makedirs(individual_folder, exist_ok=True)
        return
    
    stock_files = [f for f in os.listdir(individual_folder) if f.endswith(".csv")]
    
    if not stock_files:
        st.warning(f"⚠️ No CSV files found in '{individual_folder}'. Please add stock data files.")
        return
    
    st.info(f"📂 Processing {len(stock_files)} stock files...")
    progress_bar = st.progress(0)

    for i, f in enumerate(stock_files):
        try:
            df = pd.read_csv(os.path.join(individual_folder, f))
            
            # Detect and standardize columns
            if 'date' not in df.columns:
                date_col = [c for c in df.columns if 'date' in c.lower()]
                value_col = [c for c in df.columns if 'close' in c.lower() or 'value' in c.lower() or 'adj close' in c.lower()]
                
                if date_col and value_col:
                    df = df[[date_col[0], value_col[0]]]
                    df.columns = ['date', 'value']
                else:
                    st.warning(f"⚠️ Skipping {f}: cannot find date/value columns")
                    continue
            else:
                if 'value' not in df.columns:
                    val_col = [c for c in df.columns if 'close' in c.lower() or 'adj close' in c.lower()]
                    if val_col:
                        df['value'] = df[val_col[0]]
                df = df[['date', 'value']]

            # Clean and sort
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df = df.ffill()  # Fixed deprecation warning
            
            # Save processed file
            output_path = os.path.join(processed_folder, f"{os.path.splitext(f)[0]}.csv")
            df.to_csv(output_path, index=False)
            
        except Exception as e:
            st.warning(f"⚠️ Error processing {f}: {e}")
        
        progress_bar.progress((i+1)/len(stock_files))
    
    progress_bar.empty()
    st.success("✅ All stocks processed successfully!")

# -----------------------------
# 3. Load data with caching
# -----------------------------
@st.cache_data
def load_data(file_path):
    """Load and preprocess stock data (cached for efficiency)"""
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df = df.ffill()  # Fixed deprecation warning
    return df

# -----------------------------
# 4. Classical Models
# -----------------------------
def train_arima(df, steps=30, order=(5,1,0)):
    """Train ARIMA model and forecast"""
    try:
        model = ARIMA(df['value'], order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        st.warning(f"⚠️ ARIMA failed: {e}")
        return np.full(steps, df['value'].iloc[-1])

def train_prophet(df, steps=30):
    """Train Prophet model and forecast"""
    try:
        df_prophet = df.rename(columns={'date':'ds','value':'y'})
        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        return forecast[['ds','yhat']]
    except Exception as e:
        st.warning(f"⚠️ Prophet failed: {e}")
        result = pd.DataFrame({
            'ds': pd.date_range(df['date'].iloc[-1] + pd.Timedelta(days=1), periods=steps),
            'yhat': df['value'].iloc[-1]
        })
        return result

# -----------------------------
# 5. LSTM Model
# -----------------------------
class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series"""
    def __init__(self, series, seq_length=30):
        self.series = series
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.series) - self.seq_length)

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_length]
        y = self.series[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    """LSTM neural network for time series forecasting"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def train_lstm_model(series, progress_text, progress_bar, epochs=10, seq_length=30):
    """Train LSTM with live progress feedback"""
    # Check if we have enough data
    if len(series) < seq_length + 10:
        progress_text.text(f"⚠️ LSTM skipped: need at least {seq_length + 10} data points, have {len(series)}")
        progress_bar.empty()
        return None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TimeSeriesDataset(series, seq_length)
    
    if len(dataset) == 0:
        progress_text.text("⚠️ LSTM skipped: insufficient data")
        progress_bar.empty()
        return None, None
    
    dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
    
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        progress_text.text(f"🔄 LSTM Training: Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        progress_bar.progress((epoch+1)/epochs)
    
    progress_text.text("✅ LSTM Training Complete!")
    progress_bar.empty()
    return model, device

def forecast_lstm(model, series, device, steps=30, seq_length=30, n_samples=50):
    """Generate LSTM forecast with uncertainty intervals"""
    if model is None:
        return np.full(steps, np.nan), np.full(steps, np.nan), np.full(steps, np.nan)
    
    model.eval()
    preds_samples = []
    
    for _ in range(n_samples):
        # Enable dropout for uncertainty estimation
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        
        input_seq = series[-seq_length:].tolist()
        preds = []
        
        for _ in range(steps):
            x = torch.tensor(input_seq[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                y_pred = model(x).item()
            preds.append(y_pred)
            input_seq.append(y_pred)
        
        preds_samples.append(preds)
    
    preds_samples = np.array(preds_samples)
    mean = preds_samples.mean(axis=0)
    lower = np.percentile(preds_samples, 5, axis=0)
    upper = np.percentile(preds_samples, 95, axis=0)
    
    return mean, lower, upper

# -----------------------------
# 6. TFT Model (FIXED - ROBUST VERSION)
# -----------------------------
class ProgressCallback(Callback):
    """Custom callback to update Streamlit progress during TFT training"""
    def __init__(self, progress_text, progress_bar, total_epochs):
        super().__init__()
        self.progress_text = progress_text
        self.progress_bar = progress_bar
        self.total_epochs = total_epochs
        
    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        self.progress_text.text(f"🔄 TFT Training: Epoch {epoch}/{self.total_epochs}")
        self.progress_bar.progress(epoch / self.total_epochs)

def train_tft_model(df, progress_text, progress_bar, epochs=5):
    """Train TFT model with adaptive sizing (FIXED)"""
    n = len(df)
    
    # Calculate safe encoder/prediction lengths based on data size
    # Rule: need at least 3x the total sequence length
    min_required = 90  # Minimum for stable TFT training
    
    if n < min_required:
        progress_text.text(f"⚠️ TFT skipped: need {min_required}+ points, have {n}")
        progress_bar.empty()
        return None, None
    
    # Adaptive sizing based on available data
    max_encoder_length = min(30, n // 4)
    max_prediction_length = min(30, n // 4)
    
    # Ensure we have enough data for train + validation split
    total_seq_length = max_encoder_length + max_prediction_length
    if n < total_seq_length * 2:
        max_encoder_length = n // 6
        max_prediction_length = n // 6
    
    if max_encoder_length < 5 or max_prediction_length < 5:
        progress_text.text("⚠️ TFT skipped: insufficient data for minimum sequence lengths")
        progress_bar.empty()
        return None, None
    
    tft_df = df.copy()
    tft_df['time_idx'] = np.arange(len(tft_df))
    tft_df['series_id'] = 0
    
    try:
        # Calculate split point
        split_point = n - max_prediction_length
        
        # Create training dataset
        training = TimeSeriesDataSet(
            tft_df[:split_point],
            time_idx='time_idx',
            target='value',
            group_ids=['series_id'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            min_encoder_length=max_encoder_length // 2,
            min_prediction_length=1,
        )
        
        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training, 
            tft_df,  # Use full dataset
            predict=True, 
            stop_randomization=True
        )
        
        train_loader = training.to_dataloader(train=True, batch_size=16, num_workers=0)
        val_loader = validation.to_dataloader(train=False, batch_size=16, num_workers=0)
        
        # Initialize model
        tft_model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # 7 quantiles for better uncertainty
            loss=QuantileLoss(),
        )
        
        # Train with progress callback
        progress_callback = ProgressCallback(progress_text, progress_bar, epochs)
        trainer = Trainer(
            max_epochs=epochs, 
            enable_progress_bar=False,
            callbacks=[progress_callback],
            logger=False,
            enable_checkpointing=False
        )
        
        trainer.fit(tft_model, train_loader, val_loader)
        
        progress_text.text("✅ TFT Training Complete!")
        progress_bar.empty()
        
        return tft_model, validation
        
    except Exception as e:
        progress_text.text(f"⚠️ TFT skipped: {str(e)[:100]}")
        progress_bar.empty()
        return None, None

# -----------------------------
# 7. Main App
# -----------------------------
def main():
    st.set_page_config(
        page_title="Time-Series Forecasting Dashboard", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Advanced Time-Series Forecasting Dashboard")
    st.markdown("*Multi-model forecasting with ARIMA, Prophet, LSTM, and Temporal Fusion Transformer*")
    
    # Merge stocks
    merge_stocks()
    
    # Stock selection
    data_folder = "processed_stocks"
    
    if not os.path.exists(data_folder):
        st.error("❌ No processed stock files found! Please add CSV files to 'individual_stocks/' folder.")
        st.info("📁 Expected CSV format: columns named 'Date' and 'Close' (or similar)")
        return
    
    stock_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".csv")])
    
    if not stock_files:
        st.error("❌ No processed stock files found!")
        return
    
    stock_selection = st.selectbox("📊 Select a stock:", stock_files)
    df = load_data(os.path.join(data_folder, stock_selection))
    
    # Dataset overview
    st.subheader(f"📋 Dataset Overview: {stock_selection}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
    with col3:
        st.metric("Latest Value", f"${df['value'].iloc[-1]:.2f}")
    
    st.dataframe(df.tail(10), use_container_width=True)
    
    # Original series chart
    fig_orig = px.line(
        df, 
        x='date', 
        y='value', 
        title="📉 Historical Time Series", 
        template="plotly_dark"
    )
    fig_orig.update_traces(line_color='lightblue')
    st.plotly_chart(fig_orig, use_container_width=True)
    
    # Model training section
    st.subheader(" Training Models")
    
    with st.spinner("Training ARIMA..."):
        arima_forecast = train_arima(df).values
    st.success("✅ ARIMA trained")
    
    with st.spinner("Training Prophet..."):
        prophet_forecast = train_prophet(df)['yhat'].values
    st.success("✅ Prophet trained")
    
    # LSTM with progress
    lstm_text = st.empty()
    lstm_bar = st.progress(0)
    lstm_model, lstm_device = train_lstm_model(df['value'].values, lstm_text, lstm_bar)
    lstm_mean, lstm_lower, lstm_upper = forecast_lstm(lstm_model, df['value'].values, lstm_device)
    
    # TFT with progress
    tft_text = st.empty()
    tft_bar = st.progress(0)
    tft_model, tft_dataset = train_tft_model(df, tft_text, tft_bar)
    
    tft_available = tft_model is not None
    if tft_available:
        # Get predictions from TFT
        tft_loader = tft_dataset.to_dataloader(train=False, batch_size=16, num_workers=0)
        raw_pred, _ = tft_model.predict(tft_loader, mode="raw", return_x=True)
        
        # Extract forecast (take last prediction)
        predictions = raw_pred['prediction']
        tft_forecast = predictions[0, :, 3].numpy()  # median (quantile index 3)
        tft_lower = predictions[0, :, 1].numpy()      # 10th percentile
        tft_upper = predictions[0, :, 5].numpy()      # 90th percentile
        
        # Ensure we have 30 points
        if len(tft_forecast) < 30:
            tft_forecast = np.pad(tft_forecast, (0, 30 - len(tft_forecast)), mode='edge')
            tft_lower = np.pad(tft_lower, (0, 30 - len(tft_lower)), mode='edge')
            tft_upper = np.pad(tft_upper, (0, 30 - len(tft_upper)), mode='edge')
        tft_forecast = tft_forecast[:30]
        tft_lower = tft_lower[:30]
        tft_upper = tft_upper[:30]
    else:
        tft_forecast = tft_lower = tft_upper = np.full(30, np.nan)
    
    # ============================================
    # EVALUATION METRICS
    # ============================================
    st.subheader("📊 Model Evaluation Metrics")
    
    # Get actual future values for comparison (last 30 points)
    true_future = df['value'].values[-30:]
    
    metrics_data = {
        'Model': ['ARIMA', 'Prophet', 'LSTM', 'TFT'],
        'RMSE': [
            rmse(true_future, arima_forecast[:30]),
            rmse(true_future, prophet_forecast[-30:]),
            rmse(true_future, lstm_mean) if lstm_model else np.nan,
            rmse(true_future, tft_forecast) if tft_available else np.nan
        ],
        'MAPE (%)': [
            mape(true_future, arima_forecast[:30]),
            mape(true_future, prophet_forecast[-30:]),
            mape(true_future, lstm_mean) if lstm_model else np.nan,
            mape(true_future, tft_forecast) if tft_available else np.nan
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Style the metrics table
    st.dataframe(
        metrics_df.style.format({
            'RMSE': '{:.4f}',
            'MAPE (%)': '{:.2f}'
        }).highlight_min(subset=['RMSE', 'MAPE (%)'], color='lightgreen'),
        use_container_width=True
    )
    
    # Best model highlight (excluding NaN values)
    valid_rmse = metrics_df[metrics_df['RMSE'].notna()]
    valid_mape = metrics_df[metrics_df['MAPE (%)'].notna()]
    
    if len(valid_rmse) > 0 and len(valid_mape) > 0:
        best_model_rmse = valid_rmse.loc[valid_rmse['RMSE'].idxmin(), 'Model']
        best_model_mape = valid_mape.loc[valid_mape['MAPE (%)'].idxmin(), 'Model']
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🏆 Best RMSE: **{best_model_rmse}**")
        with col2:
            st.info(f"🏆 Best MAPE: **{best_model_mape}**")
    
    # ============================================
    # FORECAST COMPARISON CHART
    # ============================================
    forecast_dates = pd.date_range(df['date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
    
    fig = go.Figure()
    
    # Color scheme
    colors = {
        'Actual': 'lightblue',
        'ARIMA': 'orange',
        'Prophet': 'green',
        'LSTM Mean': 'cyan',
        'LSTM CI': 'rgba(0,255,255,0.2)',
        'TFT Mean': 'magenta',
        'TFT CI': 'rgba(255,0,255,0.2)'
    }
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['value'], 
        mode='lines', 
        name='Historical', 
        line=dict(color=colors['Actual'], width=2)
    ))
    
    # ARIMA
    fig.add_trace(go.Scatter(
        x=forecast_dates, 
        y=arima_forecast[:30], 
        mode='lines', 
        name='ARIMA', 
        line=dict(color=colors['ARIMA'], dash='dash')
    ))
    
    # Prophet
    fig.add_trace(go.Scatter(
        x=forecast_dates, 
        y=prophet_forecast[-30:], 
        mode='lines', 
        name='Prophet', 
        line=dict(color=colors['Prophet'], dash='dash')
    ))
    
    # LSTM with confidence intervals
    if lstm_model is not None:
        fig.add_trace(go.Scatter(
            x=forecast_dates, 
            y=lstm_mean, 
            mode='lines', 
            name='LSTM Mean', 
            line=dict(color=colors['LSTM Mean'], width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, 
            y=lstm_upper, 
            fill=None, 
            mode='lines', 
            line=dict(color='cyan', width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, 
            y=lstm_lower, 
            fill='tonexty', 
            mode='lines', 
            line=dict(color='cyan', width=0),
            fillcolor=colors['LSTM CI'],
            name='LSTM 90% CI'
        ))
    
    # TFT with confidence intervals (if available)
    if tft_available:
        fig.add_trace(go.Scatter(
            x=forecast_dates, 
            y=tft_forecast, 
            mode='lines', 
            name='TFT Mean', 
            line=dict(color=colors['TFT Mean'], width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, 
            y=tft_upper, 
            fill=None, 
            mode='lines', 
            line=dict(color='magenta', width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, 
            y=tft_lower, 
            fill='tonexty', 
            mode='lines', 
            line=dict(color='magenta', width=0),
            fillcolor=colors['TFT CI'],
            name='TFT 80% CI'
        ))
    
    fig.update_layout(
        title=" 30-Day Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_dark",
        hovermode='x unified',
        height=600
    )
    
    st.subheader("🔮 Forecast Comparison")
    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit • Models: ARIMA, Prophet, LSTM, TFT")

# -----------------------------
# 8. Run
# -----------------------------
if __name__ == "__main__":
    main()