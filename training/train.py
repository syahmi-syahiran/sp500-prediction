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
    and returns features and target.
    """
    # Join features with raw_prices to get the actual target (close price at t+1)
    query = """
    SELECT 
        f.date,
        f.sma_5, f.sma_20, f.sma_50,
        f.rsi_14, f.macd, f.macd_signal,
        f.bb_upper, f.bb_lower,
        f.volume_change_pct, f.vix,
        f.day_of_week, f.month,
        p.close as current_close
    FROM features f
    JOIN raw_prices p ON f.date = p.date
    ORDER BY f.date ASC
    """
    df = pd.read_sql_query(query, conn)
    
    if df.empty or len(df) < 50:
        raise ValueError("Not enough features data in DB to train model.")
        
    # Shift 'current_close' by -1 to get next day's close (t+1 target)
    df['target_close'] = df['current_close'].shift(-1)
    
    # Save the latest features row (which has target_close = NaN) for tomorrow's prediction
    latest_row = df.iloc[-1:].copy()
    
    # Drop rows with NaN (which is the last row because the target close is in the future)
    df_train = df.dropna().copy()
    
    return df_train, latest_row

def train_model():
    """
    Trains XGBoost model, logs metrics to MLflow, and registers best model.
    """
    conn = config.get_db_connection()
    try:
        df, latest_row = load_training_data(conn)
    finally:
        conn.close()
        
    # Define features and target
    feature_cols = [
        'sma_5', 'sma_20', 'sma_50', 'rsi_14', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'volume_change_pct', 'vix', 'day_of_week', 'month'
    ]
    X = df[feature_cols]
    y = df['target_close']
    
    # Chronological Split (No random shuffle for time series!)
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
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
    mlflow.set_experiment("SPY_ETF_Prediction")
    
    with mlflow.start_run() as run:
        print("Training XGBoost Regressor model...")
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_val)
        
        # Calculate validation metrics
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Directional Accuracy calculation
        val_close = df['current_close'].iloc[split_idx:]
        dir_accuracy = evaluate.compute_directional_accuracy(y_val, y_pred, val_close)
        
        print(f"Validation MAE: {mae:.4f}")
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation Directional Accuracy: {dir_accuracy * 100:.2f}%")
        
        # Log params & metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("val_mae", float(mae))
        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_metric("val_directional_accuracy", float(dir_accuracy))
        
        # Log model signature & input example
        signature = mlflow.models.infer_signature(X_val, y_pred)
        
        # Register Model to MLflow Model Registry
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="spy_price_predictor",
            signature=signature
        )
        
        print(f"Model logged to MLflow Run: {run.info.run_id} and registered as 'spy_price_predictor'.")
        
        # Save run_id and latest features locally for convenient FastAPI and Streamlit usage
        os.makedirs("models", exist_ok=True)
        with open("models/active_run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
    return run.info.run_id

if __name__ == "__main__":
    train_model()
