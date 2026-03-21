# 📊 Time-Series Forecasting Dashboard

An interactive dashboard to compare classical and deep learning models for time-series forecasting on stock data.

> **Key Insight:** In many cases, simpler statistical models (ARIMA, Prophet) produce more stable forecasts than deep learning models, especially on smaller datasets.

---

## 🚀 Features

- Compare forecasts from multiple models side-by-side:
  - ARIMA
  - Prophet
  - LSTM (with uncertainty intervals)
  - Temporal Fusion Transformer (TFT)
- 📊 Backtesting on the last 30 days
- 📉 RMSE & MAPE evaluation metrics
- 📈 Interactive visualizations with uncertainty bands
- ⚡ Real-time training progress in the UI
- 🔄 Automated processing of raw stock CSVs

---

## 🛠 Tech Stack

- **Python** · **Streamlit** · **Pandas** · **NumPy**
- **PyTorch** · **PyTorch Forecasting**
- **Prophet** · **ARIMA (statsmodels)**

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/SWARNIM-TIWARI/time-series-forecasting-dashboard.git
cd time-series-forecasting-dashboard
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
```

3. **Activate the virtual environment**

- Windows (PowerShell):
```bash
  .\venv\Scripts\Activate.ps1
```
- Windows (CMD):
```bash
  .\venv\Scripts\activate.bat
```
- macOS/Linux:
```bash
  source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your default browser. Select a stock from the sidebar to visualize historical data, forecasts, and model metrics.

---

## 📂 Project Structure
```
time-series-forecasting-dashboard/
├── app.py
├── merge_stocks.py
├── requirements.txt
├── README.md
├── individual_stocks/
│   ├── WMT_2006-01-01_to_2018-01-01.csv
│   ├── XOM_2006-01-01_to_2018-01-01.csv
│   ├── all_stocks_2006-01-01_to_2018-01-01.csv
│   └── all_stocks_2017-01-01_to_2018-01-01.csv
└── processed_stocks/
```

---

## 📁 Data Format

Input CSV files should have:
- A **date** column (e.g., `Date`)
- A **value** column (e.g., `Close`)

Place raw CSV files inside `individual_stocks/`. Run `merge_stocks.py` to process and standardize them into `processed_stocks/`.

---

## 📌 Notes

- TFT is automatically disabled for small datasets to prevent overfitting.
- LSTM includes uncertainty estimation using dropout.
- The dashboard automatically detects all available stocks.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*Built for learning and comparing forecasting techniques.*
