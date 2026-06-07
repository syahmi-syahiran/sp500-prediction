import pandas as pd
import numpy as np
import requests
import yfinance as yf
import sys
import os

# Adjust path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def fetch_vix_data(start_date, end_date):
    """
    Fetches VIX historical data from yfinance.
    """
    print(f"Fetching VIX data from {start_date} to {end_date}...")
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    ticker = yf.Ticker("^VIX", session=session)
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    if df.empty:
        print("Warning: VIX data returned empty.")
        return pd.DataFrame(columns=['date', 'vix'])
    
    df = df.reset_index()
    df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = df.rename(columns={'Close': 'vix'})
    return df[['date', 'vix']]

def compute_rsi(series, period=14):
    """
    Computes Relative Strength Index (RSI_14) using pandas.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()
    
    # Calculate exponentially weighted moving average
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def engineer_features(conn):
    """
    Loads raw prices, computes features group-by-symbol, joins VIX, and saves to features table.
    """
    cursor = conn.cursor()
    
    # Read raw prices from DB
    query = "SELECT symbol, date, open, high, low, close, volume FROM raw_prices ORDER BY date ASC"
    df_prices = pd.read_sql_query(query, conn)
    
    if df_prices.empty or len(df_prices) < 50:
        raise ValueError(f"No raw price data in DB (found {len(df_prices)} records, need at least 50) to engineer features.")
        
    # Format date as string to prevent type mismatch (e.g. datetime.date vs str)
    df_prices['date'] = pd.to_datetime(df_prices['date']).dt.strftime('%Y-%m-%d')
    
    start_date = df_prices['date'].min()
    end_date = (pd.to_datetime(df_prices['date'].max()) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Fetch VIX data once globally
    df_vix = fetch_vix_data(start_date, end_date)
    vix_dict = dict(zip(df_vix['date'], df_vix['vix']))
    
    # Load all existing keys in memory to avoid N network roundtrips
    existing_features = set()
    try:
        cursor.execute("SELECT symbol, date FROM features")
        for row in cursor.fetchall():
            sym = row[0]
            dt = row[1]
            if not isinstance(dt, str) and dt is not None:
                dt = dt.strftime('%Y-%m-%d')
            existing_features.add((sym, dt))
    except Exception:
        pass

    symbols = df_prices['symbol'].unique()
    total_added = 0
    placeholder = "?" if config.DB_TYPE == "sqlite" else "%s"
    
    for symbol in symbols:
        print(f"Engineering features for {symbol}...")
        df = df_prices[df_prices['symbol'] == symbol].copy()
        
        if len(df) < 50:
            print(f"Warning: Not enough history for symbol {symbol} (need at least 50 days). Skipping.")
            continue
            
        # Map VIX globally
        df['vix'] = df['date'].map(vix_dict)
        df['vix'] = df['vix'].ffill().bfill()
        
        # Compute Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Compute Volume Change Percent
        df['volume_change_pct'] = df['volume'].pct_change()
        
        # Compute Bollinger Bands
        bb_middle = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = bb_middle + (2 * bb_std)
        df['bb_lower'] = bb_middle - (2 * bb_std)
        
        # Compute RSI 14
        df['rsi_14'] = compute_rsi(df['close'], period=14)
        
        # Compute MACD & MACD signal
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calendar features
        df_dates = pd.to_datetime(df['date'])
        df['day_of_week'] = df_dates.dt.dayofweek
        df['month'] = df_dates.dt.month
        
        # Drop rows that have NaN values due to rolling windows
        df_features = df.dropna().copy()
        
        insert_data = []
        for _, row in df_features.iterrows():
            row_date = row['date']
            if (symbol, row_date) in existing_features:
                continue
            insert_data.append((
                symbol,
                row_date,
                float(row['sma_5']), float(row['sma_20']), float(row['sma_50']),
                float(row['rsi_14']), float(row['macd']), float(row['macd_signal']),
                float(row['bb_upper']), float(row['bb_lower']), float(row['volume_change_pct']),
                float(row['vix']), int(row['day_of_week']), int(row['month'])
            ))
            
        if insert_data:
            if config.DB_TYPE == "sqlite":
                cursor.executemany(f"""
                INSERT INTO features (
                    symbol, date, sma_5, sma_20, sma_50, rsi_14, macd, macd_signal, bb_upper, bb_lower, volume_change_pct, vix, day_of_week, month
                ) VALUES (
                    {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}
                )
                """, insert_data)
            else:
                from psycopg2.extras import execute_values
                execute_values(cursor, """
                INSERT INTO features (
                    symbol, date, sma_5, sma_20, sma_50, rsi_14, macd, macd_signal, bb_upper, bb_lower, volume_change_pct, vix, day_of_week, month
                ) VALUES %s
                """, insert_data)
            conn.commit()
            rows_inserted = len(insert_data)
        else:
            rows_inserted = 0
            
        print(f"Successfully processed {symbol}: {rows_inserted} new feature records stored.")
        total_added += rows_inserted
        
    print(f"Feature engineering completed. Total feature vectors added/updated: {total_added}")
    
    # Verify that we have enough records in the features table
    cursor.execute("SELECT COUNT(*) FROM features")
    total_features = cursor.fetchone()[0]
    if total_features < 50:
        raise ValueError(f"Not enough features in DB after engineering (found {total_features} records, need at least 50).")

if __name__ == "__main__":
    conn = config.get_db_connection()
    try:
        engineer_features(conn)
    finally:
        conn.close()
