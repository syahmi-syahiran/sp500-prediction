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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data import fetch
from features import engineering

def run_evidently_drift(conn):
    """
    Runs Evidently AI feature drift detection and saves the report as HTML.
    """
    try:
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
        except ImportError:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
        
        # Load features data
        df = pd.read_sql_query("SELECT * FROM features ORDER BY date DESC LIMIT 100", conn)
        if len(df) < 30:
            print("Not enough feature records to compute drift (need at least 30).")
            return
            
        # Set reference (past 20-100 days) and current (latest 10 days)
        reference = df.tail(80).drop(columns=['date'])
        current = df.head(20).drop(columns=['date'])
        
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        
        os.makedirs("monitoring", exist_ok=True)
        snapshot.save_html("monitoring/drift_report.html")
        print("Evidently Data Drift report successfully created at monitoring/drift_report.html")
    except Exception as e:
        print(f"Warning: Could not generate Evidently drift report ({e})")

def check_yesterday_prediction(conn, today_str):
    """
    Fetches yesterday's prediction and today's actual close, computes error,
    and inserts it into the actuals table.
    """
    cursor = conn.cursor()
    placeholder = "?" if config.DB_TYPE == "sqlite" else "%s"
    
    # Yesterday prediction: a prediction targetted for today_str
    cursor.execute(f"""
    SELECT prediction_date, predicted_price 
    FROM predictions 
    WHERE target_date = {placeholder} 
    ORDER BY created_at DESC LIMIT 1
    """, (today_str,))
    pred_row = cursor.fetchone()
    
    if not pred_row:
        print(f"No prediction found targeting today's date ({today_str}). Cannot evaluate prediction yet.")
        return
        
    pred_date = pred_row[0] if config.DB_TYPE == "sqlite" else pred_row['prediction_date']
    pred_price = pred_row[1] if config.DB_TYPE == "sqlite" else pred_row['predicted_price']
    
    # Convert dates to string just in case
    if isinstance(pred_date, datetime.date):
        pred_date = pred_date.strftime('%Y-%m-%d')
        
    # Get actual today's close price from raw_prices table
    cursor.execute(f"SELECT close FROM raw_prices WHERE date = {placeholder}", (today_str,))
    today_row = cursor.fetchone()
    if not today_row:
        print(f"Actual close price for today ({today_str}) not yet fetched. Cannot calculate error.")
        return
    actual_price = today_row[0]
    
    # Get close price from prediction date (T-1) to calculate directional accuracy
    cursor.execute(f"SELECT close FROM raw_prices WHERE date = {placeholder}", (pred_date,))
    pred_date_row = cursor.fetchone()
    if not pred_date_row:
        print(f"Could not retrieve close price for prediction date ({pred_date}). Directional accuracy skipped.")
        return
    prev_close = pred_date_row[0]
    
    # Compute error metrics
    abs_error = abs(actual_price - pred_price)
    
    # Direction calculations
    actual_up = actual_price > prev_close
    pred_up = pred_price > prev_close
    direction_correct = 1 if (actual_up == pred_up) else 0
    
    # Log to actuals table
    cursor.execute(f"SELECT 1 FROM actuals WHERE date = {placeholder}", (today_str,))
    if cursor.fetchone():
        cursor.execute(f"""
        UPDATE actuals SET
            actual_price = {placeholder},
            predicted_price = {placeholder},
            absolute_error = {placeholder},
            direction_correct = {placeholder}
        WHERE date = {placeholder}
        """, (actual_price, pred_price, abs_error, direction_correct, today_str))
    else:
        cursor.execute(f"""
        INSERT INTO actuals (date, actual_price, predicted_price, absolute_error, direction_correct)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        """, (today_str, actual_price, pred_price, abs_error, direction_correct))
        
    conn.commit()
    print(f"Prediction for {today_str} evaluated: Actual={actual_price:.2f}, Predicted={pred_price:.2f}, Abs Error={abs_error:.2f}, Direction Correct={bool(direction_correct)}")

def send_alert_email(current_mae, benchmark_mae):
    """
    Sends SMTP alert email.
    """
    if not config.SMTP_USER or not config.SMTP_PASSWORD or not config.ALERT_EMAIL:
        print("SMTP email configuration is missing in environment variables. Console alert logged.")
        print(f"ALERT: Model 7-day MAE ({current_mae:.4f}) is degraded by > 15% compared to 30-day average ({benchmark_mae:.4f}).")
        return
        
    try:
        msg = MIMEMultipart()
        msg['From'] = config.SMTP_USER
        msg['To'] = config.ALERT_EMAIL
        msg['Subject'] = f"🚨 S&P 500 prediction model degradation alert!"
        
        body = f"""\
        <h3>MLOps prediction Alert</h3>
        <p>Your S&P 500 prediction model has experienced a degradation in performance.</p>
        <ul>
            <li><strong>7-day rolling MAE:</strong> {current_mae:.4f}</li>
            <li><strong>30-day baseline MAE:</strong> {benchmark_mae:.4f}</li>
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
    Checks if 7-day rolling MAE exceeds 15% of the 30-day historical average baseline MAE.
    """
    query = "SELECT date, absolute_error FROM actuals ORDER BY date DESC LIMIT 30"
    df = pd.read_sql_query(query, conn)
    
    if len(df) < 10:
        print("Not enough history in actuals table to monitor MAE degradation.")
        return
        
    # Chronological sort
    df = df.iloc[::-1].copy()
    
    # 7-day rolling MAE
    current_mae = df['absolute_error'].tail(7).mean()
    # 30-day historical MAE
    benchmark_mae = df['absolute_error'].mean()
    
    degradation = (current_mae - benchmark_mae) / benchmark_mae if benchmark_mae > 0 else 0
    print(f"Performance Check: 7-day MAE={current_mae:.4f}, 30-day baseline={benchmark_mae:.4f}, Degradation={degradation * 100:.2f}%")
    
    if degradation > config.MAE_ALERT_THRESHOLD:
        send_alert_email(current_mae, benchmark_mae)

def run_daily_prediction(today_str, tomorrow_str):
    """
    Loads today's newly computed features, sends a request to FastAPI serving app (or local fallback),
    and obtains next day's prediction.
    """
    conn = config.get_db_connection()
    feat = None
    col_names = []
    try:
        cursor = conn.cursor()
        placeholder = "?" if config.DB_TYPE == "sqlite" else "%s"
        
        # Fetch today's feature vector
        cursor.execute(f"SELECT * FROM features WHERE date = {placeholder}", (today_str,))
        feat = cursor.fetchone()
        
        if not feat:
            print(f"No engineered features available for today ({today_str}). Cannot predict next price.")
            return
            
        # Get column names of features table
        cursor.execute("SELECT * FROM features LIMIT 1")
        col_names = [col[0] for col in cursor.description]
    finally:
        conn.close()
        
    # Create feature dictionary
    feat_dict = dict(zip(col_names, feat))
    
    # Prepare API payload
    payload = {
        "date": today_str,
        "target_date": tomorrow_str,
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
    
    # Try calling FastAPI serving instance
    try:
        url = f"{config.API_URL}/predict"
        print(f"Calling FastAPI prediction endpoint: {url}...")
        resp = httpx.post(url, json=payload, timeout=10.0)
        if resp.status_code == 200:
            result = resp.json()
            print(f"Prediction received from API: Tomorrow's predicted price={result['predicted_price']:.2f}")
            return
    except Exception as e:
        print(f"API call failed ({e}). Running local model fallback...")
        
    # Local fallback prediction in case API is sleeping / offline
    conn = config.get_db_connection()
    try:
        cursor = conn.cursor()
        import mlflow.xgboost
        run_id_file = "models/active_run_id.txt"
        if os.path.exists(run_id_file):
            with open(run_id_file, "r") as f:
                run_id = f.read().strip()
                
            model_path = f"./mlruns/0/{run_id}/artifacts/model"
            if not os.path.exists(model_path):
                model_path = f"runs:/{run_id}/model"
                
            model = mlflow.xgboost.load_model(model_path)
            
            # Predict
            df_feat = pd.DataFrame([payload]).drop(columns=['date', 'target_date'])
            pred_price = float(model.predict(df_feat)[0])
            
            # Log directly to DB
            placeholder = "?" if config.DB_TYPE == "sqlite" else "%s"
            input_features_serialized = json.dumps(payload)
            if config.DB_TYPE == "postgresql":
                input_features_serialized = payload
                
            cursor.execute(f"""
            INSERT INTO predictions (
                prediction_date, target_date, predicted_price, input_features, model_version
            ) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """, (
                today_str,
                tomorrow_str,
                pred_price,
                input_features_serialized,
                f"Local_Fallback_{run_id[:8]}"
            ))
            conn.commit()
            print(f"Local Fallback Prediction logged: Tomorrow's predicted price={pred_price:.2f}")
    except Exception as ex:
        print(f"Local fallback prediction also failed: {ex}")
    finally:
        conn.close()

def main():
    conn = config.get_db_connection()
    try:
        # Today's date and next trading day date (approximated for demo)
        today = datetime.datetime.now()
        today_str = today.strftime('%Y-%m-%d')
        
        # Calculate next weekday (tomorrow or Monday if today is Friday)
        tomorrow = today + datetime.timedelta(days=1)
        if tomorrow.weekday() == 5: # Saturday
            tomorrow = tomorrow + datetime.timedelta(days=2)
        elif tomorrow.weekday() == 6: # Sunday
            tomorrow = tomorrow + datetime.timedelta(days=1)
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')
        
        print(f"--- Running Daily MLOps Pipeline ({today_str}) ---")
        
        # 1. Fetch latest raw data
        conn = config.get_db_connection()
        try:
            fetch.fetch_and_save_data(conn)
        finally:
            conn.close()
        
        # 2. Compute technical features
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
                    # Re-calculate tomorrow_str
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
        
        # 3. Evaluate yesterday's prediction vs today's actual close price
        conn = config.get_db_connection()
        try:
            check_yesterday_prediction(conn, today_str)
        finally:
            conn.close()
        
        # 4. Generate prediction for tomorrow
        run_daily_prediction(today_str, tomorrow_str)
        
        # 5. Monitor MAE degradation and send alerts
        conn = config.get_db_connection()
        try:
            monitor_performance(conn)
        finally:
            conn.close()
        
        # 6. Generate Evidently Data Drift report
        conn = config.get_db_connection()
        try:
            run_evidently_drift(conn)
        finally:
            conn.close()
        
        print("Daily pipeline execution completed successfully.")
    except Exception as e:
        print(f"Error executing daily pipeline: {e}")

if __name__ == "__main__":
    main()
