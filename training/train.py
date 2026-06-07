import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Adjust path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
try:
    import evaluate
except ModuleNotFoundError:
    from training import evaluate

def load_training_data(conn):
    """
    Loads features and raw prices, shifts close prices to align target (t+1),
    transforms price features to scale-invariant ratios, and returns panel data.
    """
    # Join features with raw_prices to get the actual target (close price at t+1)
    query = """
    SELECT 
        f.symbol, f.date,
        f.sma_5, f.sma_20, f.sma_50,
        f.rsi_14, f.macd, f.macd_signal,
        f.bb_upper, f.bb_lower,
        f.volume_change_pct, f.vix,
        f.day_of_week, f.month,
        p.close as current_close
    FROM features f
    JOIN raw_prices p ON f.symbol = p.symbol AND f.date = p.date
    ORDER BY f.symbol ASC, f.date ASC
    """
    df = pd.read_sql_query(query, conn)
    
    # Normalize dates to string YYYY-MM-DD format
    if not df.empty:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    if df.empty or len(df) < 50:
        raise ValueError("Not enough features data in DB to train model.")
        
    # Calculate next day's close within each symbol group
    df['next_close'] = df.groupby('symbol')['current_close'].shift(-1)
    
    # Target is next day's percentage return
    df['target_return'] = (df['next_close'] - df['current_close']) / df['current_close']
    
    # Save the latest features row per symbol (where next_close is NaN) for prediction
    latest_rows = df[df['next_close'].isna()].copy()
    
    # Drop rows with NaN (the last trading day per symbol which has no future close price yet)
    df_train = df.dropna().copy()
    
    # Transform price-based indicators into scale-invariant ratios relative to the current_close
    for col in ['sma_5', 'sma_20', 'sma_50', 'bb_upper', 'bb_lower', 'macd', 'macd_signal']:
        df_train[col] = df_train[col] / df_train['current_close']
        
    return df_train, latest_rows

def train_model():
    """
    Trains global XGBoost model on next-day return percentages, logs metrics to MLflow, and registers.
    """
    conn = config.get_db_connection()
    try:
        df, latest_rows = load_training_data(conn)
    finally:
        conn.close()
        
    # Define scale-invariant features
    feature_cols = [
        'sma_5', 'sma_20', 'sma_50', 'rsi_14', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'volume_change_pct', 'vix', 'day_of_week', 'month'
    ]
    X = df[feature_cols]
    y = df['target_return']
    
    # Chronological Split across dates to prevent future-looking leakage
    unique_dates = sorted(df['date'].unique())
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    
    train_mask = df['date'] < split_date
    val_mask = df['date'] >= split_date
    
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    
    # Model parameters
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Set MLflow Tracking URI
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("SPY_ETF_Prediction") # Maintain experiment naming for seamless sync
    
    with mlflow.start_run() as run:
        print("Training Multi-Stock XGBoost Regressor model on returns...")
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predicted returns on validation set
        y_pred_return = model.predict(X_val)
        
        # Convert returns back to absolute predicted prices to compute validation metrics
        val_close = df['current_close'][val_mask].values
        val_actual_close = df['next_close'][val_mask].values
        val_pred_close = val_close * (1 + y_pred_return)
        
        # Calculate validation metrics in absolute dollar terms
        mae = mean_absolute_error(val_actual_close, val_pred_close)
        rmse = np.sqrt(mean_squared_error(val_actual_close, val_pred_close))
        
        # Directional Accuracy calculation
        dir_accuracy = evaluate.compute_directional_accuracy(val_actual_close, val_pred_close, val_close)
        
        print(f"Validation MAE (absolute $): {mae:.4f}")
        print(f"Validation RMSE (absolute $): {rmse:.4f}")
        print(f"Validation Directional Accuracy: {dir_accuracy * 100:.2f}%")
        
        # Log params & metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("val_mae", float(mae))
        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_metric("val_directional_accuracy", float(dir_accuracy))
        
        # Log model signature
        signature = mlflow.models.infer_signature(X_val, y_pred_return)
        
        # Register Model to MLflow Model Registry under name spy_price_predictor
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="spy_price_predictor",
            signature=signature
        )
        
        print(f"Model logged to MLflow Run: {run.info.run_id} and registered as 'spy_price_predictor'.")
        
        # Save run_id locally
        os.makedirs("models", exist_ok=True)
        with open("models/active_run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
    return run.info.run_id

if __name__ == "__main__":
    train_model()
