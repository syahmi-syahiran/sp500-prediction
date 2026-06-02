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
    Drops old tables if they are single-stock and recreates them
    with multi-stock panel columns and composite primary keys.
    """
    cursor = conn.cursor()
    
    # Clean drop old tables to ensure seamless schema upgrade
    # (Checking if existing raw_prices is missing 'symbol' column)
    upgrade_needed = False
    try:
        cursor.execute("SELECT symbol FROM raw_prices LIMIT 1")
    except Exception:
        upgrade_needed = True
        # In PostgreSQL, an exception aborts the active transaction block.
        # We must rollback to reset the transaction state before running any other queries.
        try:
            conn.rollback()
            cursor = conn.cursor()
        except Exception:
            pass
        
    if upgrade_needed:
        print("Migrating database to multi-stock panel schema...")
        try:
            cursor.execute("DROP TABLE IF EXISTS raw_prices CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS features CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS predictions CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS actuals CASCADE;")
            conn.commit()
        except Exception as e:
            print(f"Migration drop warning: {e}")
            try:
                conn.rollback()
                cursor = conn.cursor()
            except Exception:
                pass
    
    if config.DB_TYPE == "sqlite":
        # Create SQLite tables with symbol column and composite PKs
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_prices (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            symbol TEXT,
            date TEXT,
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
            month INTEGER,
            PRIMARY KEY (symbol, date)
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
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
            symbol TEXT,
            date TEXT,
            actual_price REAL,
            predicted_price REAL,
            absolute_error REAL,
            direction_correct INTEGER,
            PRIMARY KEY (symbol, date)
        );
        """)
    else:
        # Create PostgreSQL tables (Neon / Render compatible)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_prices (
            symbol VARCHAR(12),
            date DATE,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT,
            PRIMARY KEY (symbol, date)
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            symbol VARCHAR(12),
            date DATE,
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
            month INT,
            PRIMARY KEY (symbol, date)
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(12),
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
            symbol VARCHAR(12),
            date DATE,
            actual_price DOUBLE PRECISION,
            predicted_price DOUBLE PRECISION,
            absolute_error DOUBLE PRECISION,
            direction_correct BOOLEAN,
            PRIMARY KEY (symbol, date)
        );
        """)
        
    conn.commit()
    print("Database schema successfully upgraded and initialized.")

def fetch_and_save_data(conn, start_date=None, end_date=None):
    """
    Fetches daily OHLCV historical data for all 30 monitored stocks from yfinance and stores it.
    """
    if start_date is None:
        # Fetch 3 years of history to allow for lagged indicator calculation (SMA_50, etc.)
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365*3)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
    cursor = conn.cursor()
    placeholder = "?" if config.DB_TYPE == "sqlite" else "%s"
    
    # Load all existing keys in memory to avoid N network roundtrips
    existing_keys = set()
    try:
        cursor.execute("SELECT symbol, date FROM raw_prices")
        for row in cursor.fetchall():
            sym = row[0]
            dt = row[1]
            if not isinstance(dt, str) and dt is not None:
                dt = dt.strftime('%Y-%m-%d')
            existing_keys.add((sym, dt))
    except Exception:
        pass

    for symbol in config.MONITORED_SYMBOLS:
        print(f"Fetching {symbol} daily data from {start_date} to {end_date}...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                print(f"Warning: yfinance returned empty DataFrame for symbol {symbol}")
                continue
                
            # Reset index to get the date column and format it as string YYYY-MM-DD
            df = df.reset_index()
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            insert_data = []
            for _, row in df.iterrows():
                row_date = row['Date']
                if (symbol, row_date) in existing_keys:
                    continue
                insert_data.append((
                    symbol,
                    row_date,
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume'])
                ))
                
            if insert_data:
                cursor.executemany(f"""
                INSERT INTO raw_prices (symbol, date, open, high, low, close, volume)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                """, insert_data)
                conn.commit()
                rows_inserted = len(insert_data)
            else:
                rows_inserted = 0
                
            print(f"Successfully processed {symbol}: {rows_inserted} new records stored.")
            total_inserted += rows_inserted
        except Exception as e:
            print(f"Error fetching data for symbol {symbol}: {e}")
            
    print(f"Fetch completed. Total raw price rows added: {total_inserted}")

if __name__ == "__main__":
    conn = config.get_db_connection()
    try:
        init_db(conn)
        fetch_and_save_data(conn)
    finally:
        conn.close()
