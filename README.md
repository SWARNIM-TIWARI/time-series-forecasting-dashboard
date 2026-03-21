# 📈 Time-Series Forecasting Dashboard

An interactive dashboard to compare classical and deep learning models for time-series forecasting on stock data.

## 🚀 Features

- Compare multiple models:
  - ARIMA
  - Prophet
  - LSTM (with uncertainty intervals)
  - Temporal Fusion Transformer (TFT)

- 📊 Backtesting on last 30 days
- 📉 RMSE & MAPE evaluation metrics
- 📈 Interactive visualizations
- ⚡ Real-time training progress in UI

## 🧠 Key Insight

In many cases, simpler statistical models (ARIMA, Prophet) produce more stable forecasts than deep learning models, especially on smaller datasets.

## 📂 Project Structure

time-series-forecasting-dashboard/
- app.py
- merge_stocks.py
- requirements.txt
- README.md
- individual_stocks/
  - WMT_2006-01-01_to_2018-01-01.csv
  - XOM_2006-01-01_to_2018-01-01.csv
  - all_stocks_2006-01-01_to_2018-01-01.csv
  - all_stocks_2017-01-01_to_2018-01-01.csv
- processed_stocks/

## ▶️ How to Run

### 1. Install dependencies

pip install -r requirements.txt


### 2. Run the app

streamlit run app.py


## 📁 Data Format

Input CSV files should have:
- A date column (e.g., Date)
- A value column (e.g., Close price)

Place them inside:

individual_stocks/


## 🛠 Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- PyTorch
- PyTorch Forecasting
- ARIMA (statsmodels)
- Prophet

## 📌 Notes

- TFT is automatically disabled for small datasets to prevent overfitting.
- LSTM includes uncertainty estimation using dropout.

---
Built for learning and comparing forecasting techniques.
