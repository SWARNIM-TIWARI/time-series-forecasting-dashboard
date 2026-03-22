# 📈 Time-Series Forecasting Dashboard

An interactive dashboard that lets you compare how classical statistical models and modern deep learning models perform on the same stock data — side by side, with uncertainty intervals and real evaluation metrics.

The honest finding from building this: simpler models often win. ARIMA and Prophet consistently produce stable, interpretable forecasts on smaller datasets where LSTM and TFT struggle to generalize. This dashboard was built to make that tradeoff visible, not to declare a winner.

---

## What It Does

You upload a stock CSV, the dashboard processes it, trains four different forecasting models, and shows you 30-day forecasts with confidence intervals and backtesting metrics — all in one place.

The four models it runs:

**ARIMA** — a classical statistical model that works well on stationary data with clear trends. Fast, interpretable, and surprisingly competitive on shorter series.

**Prophet** — Meta's forecasting library, designed for business time series with seasonality. Handles missing data and outliers gracefully.

**LSTM** — a recurrent neural network that learns sequential patterns. Includes dropout-based uncertainty estimation, so you get a confidence band around the forecast, not just a single line.

**Temporal Fusion Transformer (TFT)** — a transformer-based architecture built specifically for time series. The most complex model here. Automatically disabled on small datasets to prevent it from overfitting and producing garbage forecasts.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **UI** | Streamlit, Plotly |
| **Classical Models** | statsmodels (ARIMA), Prophet |
| **Deep Learning** | PyTorch, PyTorch Forecasting, PyTorch Lightning |
| **Data** | Pandas, NumPy |

---

## 📂 Project Structure

```
time-series-forecasting-dashboard/
├── app.py                  # Main dashboard — all models, metrics, and visualizations
├── merge_stocks.py         # Preprocesses raw CSVs into standardized format
├── requirements.txt
├── README.md
├── individual_stocks/     
│   ├── WMT_2006-01-01_to_2018-01-01.csv
│   ├── XOM_2006-01-01_to_2018-01-01.csv
│   ├── all_stocks_2006-01-01_to_2018-01-01.csv
│   └── all_stocks_2017-01-01_to_2018-01-01.csv
└── processed_stocks/       # Auto-generated after running merge_stocks.py
```

---

## ⚙️ Installation & Running

1. **Clone the repository**

```bash
git clone https://github.com/SWARNIM-TIWARI/time-series-forecasting-dashboard.git
cd time-series-forecasting-dashboard
```

2. **Create a virtual environment**

```bash
python -m venv venv
```

3. **Activate it**

- Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
- Windows (CMD): `.\venv\Scripts\activate.bat`
- macOS/Linux: `source venv/bin/activate`

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Add your stock data**

Place CSV files in the `individual_stocks/` folder. The dashboard expects a date column and a close/value column — it auto-detects common column names.

6. **Run**

```bash
streamlit run app.py
```

The dashboard opens in your browser. Select a stock from the dropdown and the models start training immediately with live progress indicators.

---

## 📁 Data Format

Your CSV files should have at minimum:

- A date column — named `Date`, `date`, or anything containing "date"
- A value column — named `Close`, `Adj Close`, `close`, or `value`

The dashboard detects these automatically. If columns don't match, it skips that file with a warning rather than crashing.

---

## 📊 How Evaluation Works

The dashboard holds out the last 30 days of each stock as a test set and evaluates all four models against those real values. Two metrics are reported:

**RMSE** (Root Mean Squared Error) — penalizes large errors more heavily. Lower is better.

**MAPE** (Mean Absolute Percentage Error) — percentage-based, so it's comparable across different price scales. Lower is better.

The best performing model on each metric is highlighted in the results table. The winner changes depending on the stock and dataset size — which is exactly the point.

---

## 📌 Notes Worth Knowing

TFT requires at least 90 data points to train. On smaller datasets it silently disables itself and shows NaN in the metrics table rather than producing unreliable forecasts.

LSTM uncertainty bands are generated using Monte Carlo dropout — running inference 50 times with dropout enabled and taking the 5th and 95th percentiles. This gives a rough but useful sense of forecast confidence.

The merge step inside the dashboard runs automatically on startup, so you don't need to run `merge_stocks.py` manually — though you can run it standalone if you want to preprocess files separately.

All models are retrained fresh each session. There's no caching of trained weights between runs.

---

## 📄 License

MIT License

---

*Built for learning and comparing forecasting techniques.*
