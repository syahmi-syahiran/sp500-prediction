import datetime
import pandas as pd
import yfinance as yf
import sys
import os

# Adjust path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def init_db(conn):
    """
    Creates tables in SQLite or PostgreSQL database based on DB_TYPE.
    """
    cursor = conn.cursor()
    
    if config.DB_TYPE == "sqlite":
        # Create SQLite tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_prices (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            date TEXT PRIMARY KEY,
            sma_5 REAL,
            sma_20 REAL,
            sma_50 REAL,
            rsi_14 REAL,
            macd REAL,
            macd_signal REAL,
            bb_upper REAL,
            bb_lower REAL,
            volume_change_pct REAL,
            vix REAL,
            day_of_week INTEGER,
            month INTEGER
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT,
            target_date TEXT,
            predicted_price REAL,
            input_features TEXT,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS actuals (
            date TEXT PRIMARY KEY,
            actual_price REAL,
            predicted_price REAL,
            absolute_error REAL,
            direction_correct INTEGER
        );
        """)
    else:
        # Create PostgreSQL tables (Neon / Render compatible)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_prices (
            date DATE PRIMARY KEY,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            date DATE PRIMARY KEY,
            sma_5 DOUBLE PRECISION,
            sma_20 DOUBLE PRECISION,
            sma_50 DOUBLE PRECISION,
            rsi_14 DOUBLE PRECISION,
            macd DOUBLE PRECISION,
            macd_signal DOUBLE PRECISION,
            bb_upper DOUBLE PRECISION,
            bb_lower DOUBLE PRECISION,
            volume_change_pct DOUBLE PRECISION,
            vix DOUBLE PRECISION,
            day_of_week INT,
            month INT
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            prediction_date DATE,
            target_date DATE,
            predicted_price DOUBLE PRECISION,
            input_features JSONB,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS actuals (
            date DATE PRIMARY KEY,
            actual_price DOUBLE PRECISION,
            predicted_price DOUBLE PRECISION,
            absolute_error DOUBLE PRECISION,
            direction_correct BOOLEAN
        );
        """)
        
    conn.commit()
    print("Database initialized successfully.")

def fetch_and_save_data(conn, start_date=None, end_date=None):
    """
    Fetches daily OHLCV historical data for SPY from yfinance and stores it.
    """
    if start_date is None:
        # Fetch at least 2.5 years of history to have 2 full years after engineering lag features
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365*3)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
    print(f"Fetching SPY data from {start_date} to {end_date}...")
    ticker = yf.Ticker("SPY")
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    
    if df.empty:
        print("Warning: yfinance returned empty DataFrame.")
        return
        
    # Reset index to get the date column and format it as string YYYY-MM-DD
    df = df.reset_index()
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    cursor = conn.cursor()
    placeholder = "?" if config.DB_TYPE == "sqlite" else "%s"
    
    rows_inserted = 0
    for _, row in df.iterrows():
        # Check if record already exists
        cursor.execute(f"SELECT 1 FROM raw_prices WHERE date = {placeholder}", (row['Date'],))
        if cursor.fetchone():
            continue
            
        cursor.execute(f"""
        INSERT INTO raw_prices (date, open, high, low, close, volume)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        """, (
            row['Date'],
            float(row['Open']),
            float(row['High']),
            float(row['Low']),
            float(row['Close']),
            int(row['Volume'])
        ))
        rows_inserted += 1
        
    conn.commit()
    print(f"Successfully fetched and saved {rows_inserted} new daily records to 'raw_prices'.")

if __name__ == "__main__":
    conn = config.get_db_connection()
    try:
        init_db(conn)
        fetch_and_save_data(conn)
    finally:
        conn.close()
