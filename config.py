import os
import sqlite3
from dotenv import load_dotenv

# Set MLflow filesystem backend permission
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Load local .env file if it exists
load_dotenv()

# Central environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Monitored stock symbols
MONITORED_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "LLY",
    "JPM", "V", "XOM", "TSM", "WMT", "PG", "JNJ", "AVGO", "COST", "MA",
    "CVX", "HD", "MRK", "ASML", "PEP", "ABBV", "KO", "ORCL", "BAC", "AMD"
]

# SMTP Credentials & Alert settings
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
MAE_ALERT_THRESHOLD = float(os.getenv("MAE_ALERT_THRESHOLD", "0.15"))

# Determine DB Type
if DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"):
    DB_TYPE = "postgresql"
else:
    DB_TYPE = "sqlite"

def get_db_connection():
    """
    Returns a unified database connection object based on DB_TYPE.
    Auto-creates database file parent directory if it's SQLite.
    """
    if DB_TYPE == "sqlite":
        # Handle SQLite path from database URL if provided, or default to data/stock_mlops.db
        db_path = "data/stock_mlops.db"
        if DATABASE_URL.startswith("sqlite:///"):
            db_path = DATABASE_URL.replace("sqlite:///", "")
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Return SQLite connection with support for dict-like rows and WAL mode
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception as e:
            pass
        return conn
    else:
        # Import psycopg2 dynamically so psycopg2 is only required for production Postgres
        import psycopg2
        import psycopg2.extras
        import time
        
        # Use a retry loop (5 attempts, 5s delay) to handle Neon cold starts / scale-to-zero wakeups
        max_attempts = 5
        retry_delay = 5
        last_error = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Add connect_timeout to allow time for server to wake up
                conn = psycopg2.connect(DATABASE_URL, connect_timeout=15)
                return conn
            except psycopg2.OperationalError as e:
                last_error = e
                print(f"Database connection attempt {attempt}/{max_attempts} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                
        raise last_error
