import datetime
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx
import numpy as np
import pandas as pd
import sys
import os

# Adjust path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data import fetch
from features import engineering

def run_evidently_drift(conn):
    """
    Runs Evidently AI feature drift detection over the entire panel of stocks and saves it.
    """
    try:
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
        except ImportError:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
            
        # Load features data across all symbols
        df = pd.read_sql_query("SELECT * FROM features ORDER BY date DESC LIMIT 500", conn)
        if len(df) < 100:
            print("Not enough feature records to compute drift (need at least 100).")
            return
            
        # Set reference (older features) and current (latest features)
        reference = df.tail(400).drop(columns=['date', 'symbol'])
        current = df.head(100).drop(columns=['date', 'symbol'])
        
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        
        os.makedirs("monitoring", exist_ok=True)
        snapshot.save_html("monitoring/drift_report.html")
        print("Evidently Data Drift report successfully created at monitoring/drift_report.html")
    except Exception as e:
        print(f"Warning: Could not generate Evidently drift report ({e})")

def check_yesterday_prediction(conn, symbol, today_str):
    """
    Fetches yesterday's prediction for the symbol, matches it with today's actual close,
    computes error, and inserts it into the actuals table.
    """
    cursor = conn.cursor()
    placeholder = "?" if config.DB_TYPE == "sqlite" else "%s"
    
    # Yesterday prediction: targeting today_str
    cursor.execute(f"""
    SELECT prediction_date, predicted_price 
    FROM predictions 
    WHERE symbol = {placeholder} AND target_date = {placeholder} 
    ORDER BY created_at DESC LIMIT 1
    """, (symbol, today_str))
    pred_row = cursor.fetchone()
    
    if not pred_row:
        return
        
    pred_date = pred_row[0]
    pred_price = pred_row[1]
    
    if isinstance(pred_date, datetime.date):
        pred_date = pred_date.strftime('%Y-%m-%d')
        
    # Get actual close price
    cursor.execute(f"SELECT close FROM raw_prices WHERE symbol = {placeholder} AND date = {placeholder}", (symbol, today_str))
    today_row = cursor.fetchone()
    if not today_row:
        return
    actual_price = today_row[0]
    
    # Get previous day close for directional calculation
    cursor.execute(f"SELECT close FROM raw_prices WHERE symbol = {placeholder} AND date = {placeholder}", (symbol, pred_date))
    pred_date_row = cursor.fetchone()
    if not pred_date_row:
        return
    prev_close = pred_date_row[0]
    
    # Compute error metrics
    abs_error = abs(actual_price - pred_price)
    actual_up = actual_price > prev_close
    pred_up = pred_price > prev_close
    direction_correct = 1 if (actual_up == pred_up) else 0
    
    # Log to actuals table
    cursor.execute(f"SELECT 1 FROM actuals WHERE symbol = {placeholder} AND date = {placeholder}", (symbol, today_str))
    if cursor.fetchone():
        cursor.execute(f"""
        UPDATE actuals SET
            actual_price = {placeholder},
            predicted_price = {placeholder},
            absolute_error = {placeholder},
            direction_correct = {placeholder}
        WHERE symbol = {placeholder} AND date = {placeholder}
        """, (actual_price, pred_price, abs_error, direction_correct, symbol, today_str))
    else:
        cursor.execute(f"""
        INSERT INTO actuals (symbol, date, actual_price, predicted_price, absolute_error, direction_correct)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        """, (symbol, today_str, actual_price, pred_price, abs_error, direction_correct))
        
    conn.commit()
    print(f"[{symbol}] Prediction evaluated: Actual={actual_price:.2f}, Predicted={pred_price:.2f}, Abs Error={abs_error:.2f}, Direction Correct={bool(direction_correct)}")

def send_alert_email(current_mae, benchmark_mae):
    """
    Sends SMTP alert email.
    """
    if not config.SMTP_USER or not config.SMTP_PASSWORD or not config.ALERT_EMAIL:
        print("SMTP email configuration is missing. Console alert logged.")
        print(f"ALERT: Global MLOps 7-day MAE ({current_mae:.4f}) is degraded by > 15% compared to 30-day average ({benchmark_mae:.4f}).")
        return
        
    try:
        msg = MIMEMultipart()
        msg['From'] = config.SMTP_USER
        msg['To'] = config.ALERT_EMAIL
        msg['Subject'] = f"🚨 S&P 30 MLOps Model degradation alert!"
        
        body = f"""\
        <h3>MLOps Panel Prediction Alert</h3>
        <p>Your multi-stock prediction model has experienced a global degradation in performance.</p>
        <ul>
            <li><strong>7-day rolling global MAE:</strong> {current_mae:.4f}</li>
            <li><strong>30-day baseline global MAE:</strong> {benchmark_mae:.4f}</li>
            <li><strong>Degradation Percentage:</strong> {((current_mae - benchmark_mae) / benchmark_mae) * 100:.2f}%</li>
        </ul>
        <p>Please open your <a href="{config.API_URL}">Streamlit Dashboard</a> and consider retraining the model.</p>
        """
        msg.attach(MIMEText(body, 'html'))
        
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.send_message(msg)
            
        print("Degradation alert email sent successfully.")
    except Exception as e:
        print(f"Error sending alert email: {e}")

def monitor_performance(conn):
    """
    Checks if global 7-day rolling MAE exceeds 15% of the 30-day historical average.
    """
    query = "SELECT date, absolute_error FROM actuals ORDER BY date DESC LIMIT 300"
    df = pd.read_sql_query(query, conn)
    
    if len(df) < 50:
        print("Not enough history in actuals table to monitor MAE degradation.")
        return
        
    df = df.iloc[::-1].copy()
    
    # Global metrics
    current_mae = df['absolute_error'].tail(50).mean()
    benchmark_mae = df['absolute_error'].mean()
    
    degradation = (current_mae - benchmark_mae) / benchmark_mae if benchmark_mae > 0 else 0
    print(f"Global Performance: 7-day MAE={current_mae:.4f}, 30-day baseline={benchmark_mae:.4f}, Degradation={degradation * 100:.2f}%")
    
    if degradation > config.MAE_ALERT_THRESHOLD:
        send_alert_email(current_mae, benchmark_mae)

def run_daily_prediction(symbol, today_str, tomorrow_str):
    """
    Loads newly computed features for a symbol, hits the FastAPI endpoint, and predicts next close.
    """
    conn = config.get_db_connection()
    feat = None
    col_names = []
    try:
        cursor = conn.cursor()
        placeholder = "?" if config.DB_TYPE == "sqlite" else "%s"
        
        # Fetch today's feature vector
        cursor.execute(f"SELECT * FROM features WHERE symbol = {placeholder} AND date = {placeholder}", (symbol, today_str))
        feat = cursor.fetchone()
        
        if not feat:
            print(f"No engineered features available for {symbol} on {today_str}.")
            return
            
        # Get column names
        cursor.execute("SELECT * FROM features LIMIT 1")
        col_names = [col[0] for col in cursor.description]
    finally:
        conn.close()
        
    # Get today's close price directly from raw_prices
    conn = config.get_db_connection()
    try:
        cursor = conn.cursor()
        placeholder = "?" if config.DB_TYPE == "sqlite" else "%s"
        cursor.execute(f"SELECT close FROM raw_prices WHERE symbol = {placeholder} AND date = {placeholder}", (symbol, today_str))
        close_row = cursor.fetchone()
        if not close_row:
            return
        close_price = float(close_row[0])
    finally:
        conn.close()
        
    feat_dict = dict(zip(col_names, feat))
    
    # Prepare API payload (including close and symbol)
    payload = {
        "symbol": symbol,
        "date": today_str,
        "target_date": tomorrow_str,
        "close": close_price,
        "sma_5": float(feat_dict['sma_5']),
        "sma_20": float(feat_dict['sma_20']),
        "sma_50": float(feat_dict['sma_50']),
        "rsi_14": float(feat_dict['rsi_14']),
        "macd": float(feat_dict['macd']),
        "macd_signal": float(feat_dict['macd_signal']),
        "bb_upper": float(feat_dict['bb_upper']),
        "bb_lower": float(feat_dict['bb_lower']),
        "volume_change_pct": float(feat_dict['volume_change_pct']),
        "vix": float(feat_dict['vix']),
        "day_of_week": int(feat_dict['day_of_week']),
        "month": int(feat_dict['month'])
    }
    
    try:
        url = f"{config.API_URL}/predict"
        resp = httpx.post(url, json=payload, timeout=10.0)
        if resp.status_code == 200:
            result = resp.json()
            print(f"[{symbol}] Prediction: Tomorrow's predicted close = ${result['predicted_price']:.2f}")
            return
    except Exception as e:
        print(f"API call failed for {symbol}: {e}")

def main():
    # Today's date and next trading day date
    today = datetime.datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    
    tomorrow = today + datetime.timedelta(days=1)
    if tomorrow.weekday() == 5:
        tomorrow = tomorrow + datetime.timedelta(days=2)
    elif tomorrow.weekday() == 6:
        tomorrow = tomorrow + datetime.timedelta(days=1)
    tomorrow_str = tomorrow.strftime('%Y-%m-%d')
    
    print(f"--- Running Daily MLOps Panel Pipeline ({today_str}) ---")
    
    # 1. Fetch latest raw data for all 30 stocks
    conn = config.get_db_connection()
    try:
        fetch.init_db(conn)
        fetch.fetch_and_save_data(conn)
    finally:
        conn.close()
    
    # 2. Compute technical features for all 30 stocks
    conn = config.get_db_connection()
    try:
        engineering.engineer_features(conn)
    finally:
        conn.close()
    
    # Determine prediction base date (fallback to latest feature date if today is not computed yet)
    conn = config.get_db_connection()
    try:
        cursor = conn.cursor()
        placeholder = "?" if config.DB_TYPE == "sqlite" else "%s"
        cursor.execute(f"SELECT 1 FROM features WHERE date = {placeholder}", (today_str,))
        if not cursor.fetchone():
            cursor.execute("SELECT date FROM features ORDER BY date DESC LIMIT 1")
            latest_row = cursor.fetchone()
            if latest_row:
                today_str = latest_row[0]
                if isinstance(today_str, datetime.date):
                    today_dt = today_str
                    today_str = today_str.strftime('%Y-%m-%d')
                else:
                    today_dt = datetime.datetime.strptime(today_str, '%Y-%m-%d')
                tomorrow = today_dt + datetime.timedelta(days=1)
                if tomorrow.weekday() == 5:
                    tomorrow = tomorrow + datetime.timedelta(days=2)
                elif tomorrow.weekday() == 6:
                    tomorrow = tomorrow + datetime.timedelta(days=1)
                tomorrow_str = tomorrow.strftime('%Y-%m-%d')
                print(f"Timezone/Market-Close Fallback: Using latest feature date {today_str} to predict for {tomorrow_str}")
    finally:
        conn.close()
    
    # 3. Evaluate yesterday's predictions vs today's actual close prices for all stocks
    conn = config.get_db_connection()
    try:
        for symbol in config.MONITORED_SYMBOLS:
            check_yesterday_prediction(conn, symbol, today_str)
    finally:
        conn.close()
    
    # 4. Generate predictions for tomorrow for all stocks
    for symbol in config.MONITORED_SYMBOLS:
        run_daily_prediction(symbol, today_str, tomorrow_str)
    
    # 5. Monitor MAE degradation globally across all stocks
    conn = config.get_db_connection()
    try:
        monitor_performance(conn)
    finally:
        conn.close()
    
    # 6. Generate Evidently Data Drift report over the entire panel
    conn = config.get_db_connection()
    try:
        run_evidently_drift(conn)
    finally:
        conn.close()
    
    print("Daily MLOps panel pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
