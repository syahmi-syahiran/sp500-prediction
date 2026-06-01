# S&P 500 MLOps Project — Instruction Guide

## Overview

This project builds a production-grade MLOps pipeline that predicts the next day's S&P 500 (SPY ETF) closing price, automatically collects ground truth, and monitors model performance on a daily basis via a Streamlit dashboard.

---

## Prerequisites

- Python 3.10+
- Git & GitHub account
- Accounts on: [Render](https://render.com), [Neon](https://neon.tech) (free PostgreSQL)
- Basic familiarity with environment variables and virtual environments

---

## Project Structure

```
stock-mlops/
├── data/
│   └── fetch.py                  # Fetch OHLCV data via yfinance
├── features/
│   └── engineering.py            # Compute technical indicators
├── training/
│   ├── train.py                  # XGBoost training + MLflow logging
│   └── evaluate.py               # Metrics computation
├── serving/
│   └── api.py                    # FastAPI prediction endpoint
├── monitoring/
│   ├── compare.py                # Compare predictions vs actuals
│   ├── drift.py                  # Evidently AI drift reports
│   └── dashboard.py              # Streamlit monitoring UI
├── pipelines/
│   └── daily_pipeline.py         # End-to-end orchestration script
├── .github/
│   └── workflows/
│       └── daily.yml             # GitHub Actions cron job (6 PM ET)
├── requirements.txt
└── config.py                     # Centralized config (DB URL, thresholds)
```

---

## Step-by-Step Build Order

### Step 1 — Data Pipeline

**File:** `data/fetch.py`

- Use `yfinance` to fetch daily OHLCV data for `SPY`
- Fetch at least 2 years of history for initial training
- Store raw data in PostgreSQL table `raw_prices`

```bash
pip install yfinance psycopg2-binary
python data/fetch.py
```

---

### Step 2 — Feature Engineering

**File:** `features/engineering.py`

Compute the following technical indicators from daily OHLCV:

| Feature | Description |
|---------|-------------|
| `SMA_5`, `SMA_20`, `SMA_50` | Simple moving averages |
| `RSI_14` | Relative strength index |
| `MACD`, `MACD_signal` | Moving average convergence/divergence |
| `BB_upper`, `BB_lower` | Bollinger bands |
| `Volume_change_pct` | Day-over-day volume change |
| `Day_of_week`, `Month` | Calendar features |
| `VIX` | CBOE Volatility Index (fetch separately via yfinance `^VIX`) |

Store computed features in PostgreSQL table `features`.

---

### Step 3 — Model Training

**File:** `training/train.py`

- Model: `XGBoostRegressor` predicting next day's close price
- Split: chronological train/validation (no random shuffle — this is time series data)
- Log experiments with **MLflow**: parameters, metrics, and model artifact
- Register best model to MLflow Model Registry under name `spy_price_predictor`

```bash
pip install xgboost mlflow scikit-learn
python training/train.py
```

---

### Step 4 — FastAPI Serving

**File:** `serving/api.py`

Expose two endpoints:

- `POST /predict` — accepts latest features, returns predicted close price
- `GET /metrics` — returns last 30-day MAE and RMSE from the DB

Load model from MLflow registry at startup. Log every prediction to PostgreSQL table `predictions`:

```sql
predictions (id, timestamp, input_features JSONB, predicted_price FLOAT)
```

```bash
pip install fastapi uvicorn
uvicorn serving.api:app --reload
```

---

### Step 5 — Daily Pipeline

**File:** `pipelines/daily_pipeline.py`

Orchestrates the full daily flow in sequence:

1. Fetch today's OHLCV from yfinance
2. Compute features
3. Call `/predict` endpoint
4. Store prediction in DB
5. Fetch yesterday's actual closing price
6. Compare yesterday's prediction vs actual, compute error
7. Log error to `actuals` table
8. Run Evidently drift report
9. Send email alert if 7-day rolling MAE exceeds threshold (default: 15% degradation)

```bash
python pipelines/daily_pipeline.py
```

---

### Step 6 — GitHub Actions Cron

**File:** `.github/workflows/daily.yml`

Trigger the daily pipeline automatically at 6:05 PM ET (after market close):

```yaml
on:
  schedule:
    - cron: '5 23 * * 1-5'   # 23:05 UTC = 6:05 PM ET, weekdays only
```

Store secrets in GitHub → Settings → Secrets:
- `DATABASE_URL`
- `API_URL`
- `SMTP_PASSWORD` (for email alerts)

---

### Step 7 — Monitoring Dashboard

**File:** `monitoring/dashboard.py`

Build a Streamlit app with the following views:

- **Prediction vs Actual** — rolling 30-day line chart
- **Daily MAE / RMSE** — trend over time
- **Directional Accuracy** — % of days the model correctly predicted up/down
- **Evidently Drift Report** — embedded HTML report showing feature distribution shift
- **Retrain Button** — triggers a GitHub Actions workflow dispatch via API call

```bash
pip install streamlit evidently
streamlit run monitoring/dashboard.py
```

---

## Database Schema

```sql
-- Raw price data
CREATE TABLE raw_prices (
    date DATE PRIMARY KEY,
    open FLOAT, high FLOAT, low FLOAT,
    close FLOAT, volume BIGINT
);

-- Engineered features
CREATE TABLE features (
    date DATE PRIMARY KEY,
    sma_5 FLOAT, sma_20 FLOAT, sma_50 FLOAT,
    rsi_14 FLOAT, macd FLOAT, macd_signal FLOAT,
    bb_upper FLOAT, bb_lower FLOAT,
    volume_change_pct FLOAT, vix FLOAT,
    day_of_week INT, month INT
);

-- Model predictions
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    prediction_date DATE,
    target_date DATE,
    predicted_price FLOAT,
    input_features JSONB,
    model_version TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Actuals and errors
CREATE TABLE actuals (
    date DATE PRIMARY KEY,
    actual_price FLOAT,
    predicted_price FLOAT,
    absolute_error FLOAT,
    direction_correct BOOLEAN
);
```

---

## Deployment on Render (Free Tier)

### Services to deploy:

| Service | Type | Render Plan |
|---------|------|-------------|
| FastAPI | Web Service | Free |
| Streamlit Dashboard | Web Service | Free |
| MLflow Tracking Server | Web Service | Free |

### Steps:
1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service → Connect GitHub repo
3. Set environment variables (`DATABASE_URL`, `MLFLOW_TRACKING_URI`, etc.)
4. Deploy each service from its respective start command:
   - FastAPI: `uvicorn serving.api:app --host 0.0.0.0 --port $PORT`
   - Streamlit: `streamlit run monitoring/dashboard.py --server.port $PORT --server.address 0.0.0.0`

### Database:
1. Go to [neon.tech](https://neon.tech) → Create project → Copy connection string
2. Add as `DATABASE_URL` environment variable in all Render services

---

## Retraining Strategy

| Trigger | Action |
|---------|--------|
| 7-day rolling MAE degrades > 15% | Email alert + auto-retrain via GitHub Actions |
| Evidently detects feature drift (p-value < 0.05) | Flag in dashboard, notify |
| Manual button in Streamlit dashboard | GitHub Actions `workflow_dispatch` |
| Every Sunday at midnight | Scheduled full retrain on full history |

---

## Environment Variables

```env
DATABASE_URL=postgresql://user:password@host/dbname
MLFLOW_TRACKING_URI=https://your-mlflow-service.onrender.com
API_URL=https://your-fastapi-service.onrender.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL=your@gmail.com
MAE_ALERT_THRESHOLD=0.15
```

---

## Key Dependencies

```
yfinance>=0.2.0
xgboost>=2.0.0
scikit-learn>=1.3.0
mlflow>=2.9.0
fastapi>=0.104.0
uvicorn>=0.24.0
streamlit>=1.28.0
evidently>=0.4.0
psycopg2-binary>=2.9.0
pandas>=2.0.0
numpy>=1.24.0
ta>=0.10.0            # Technical analysis library
httpx>=0.25.0
python-dotenv>=1.0.0
```

---

## Daily Monitoring Checklist

Once deployed, your daily routine:

| Time | What Happens |
|------|-------------|
| 6:05 PM ET | GitHub Actions runs `daily_pipeline.py` automatically |
| Anytime | Open Streamlit dashboard to review yesterday's error |
| If alert email arrives | Check dashboard → decide whether to retrain |
| Sunday | Scheduled retrain runs automatically |

**Healthy system signals:**
- 7-day rolling MAE is stable or decreasing
- Directional accuracy stays above ~52% (better than random)
- No drift alerts from Evidently for more than 2 consecutive weeks

---

## Notes

- **This is not a trading system.** Predictions are for MLOps learning purposes only.
- SPY data via `yfinance` is free but has a 15-minute delay on live data; for post-market close fetches this is not an issue.
- Render free tier services spin down after inactivity. Use a cron ping (e.g., UptimeRobot) to keep them warm if needed.
- MLflow artifacts (model files) should be stored in a persistent volume or an S3-compatible bucket for production reliability.
