import datetime
import json
import os
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Adjust path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training import train

app = FastAPI(title="Multi-Stock Prediction API", version="2.0.0")

# Global model reference
model = None
model_version = "v2.0"

class PredictionRequest(BaseModel):
    symbol: str
    date: str
    target_date: str
    close: float # Today's absolute close price, essential for return calculations and feature ratios
    sma_5: float
    sma_20: float
    sma_50: float
    rsi_14: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    volume_change_pct: float
    vix: float
    day_of_week: int
    month: int

def load_active_model():
    """
    Tries loading the registered model from MLflow Model Registry,
    and falls back to local file system runs if the registry is unreachable.
    """
    global model, model_version
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    
    try:
        print("Attempting to load model from MLflow Model Registry...")
        model = mlflow.xgboost.load_model("models:/spy_price_predictor/latest")
        model_version = "Registry_Latest"
        print("Model loaded successfully from MLflow registry.")
        return
    except Exception as e:
        print(f"Registry load failed/unreachable ({e}). Trying filesystem fallback...")
        
    # Local fallback: read active_run_id.txt and load from local mlruns
    try:
        run_id_file = "models/active_run_id.txt"
        if os.path.exists(run_id_file):
            with open(run_id_file, "r") as f:
                run_id = f.read().strip()
            
            local_paths = [
                f"./mlruns/0/{run_id}/artifacts/model",
                f"./mlruns/1/{run_id}/artifacts/model",
            ]
            for path in local_paths:
                if os.path.exists(path):
                    model = mlflow.xgboost.load_model(path)
                    model_version = f"Local_Run_{run_id[:8]}"
                    print(f"Model loaded successfully from local runs folder: {path}")
                    return
            
            # Direct mlflow runs loader
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
            model_version = f"Runs_URI_{run_id[:8]}"
            print("Model loaded successfully via runs URI.")
            return
    except Exception as e:
        print(f"Local runs fallback failed: {e}")
        
    print("Warning: No pre-trained model could be loaded at startup.")

@app.on_event("startup")
def startup_event():
    load_active_model()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "model_version": model_version,
        "db_type": config.DB_TYPE
    }

@app.post("/predict")
def predict_price(request: PredictionRequest):
    """
    Generates next day's price prediction for a specific symbol and registers it to the DB.
    """
    global model
    if model is None:
        load_active_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Prediction model is not loaded/available.")
            
    # Perform scale-invariant feature transformation relative to the current close price
    scaled_sma_5 = request.sma_5 / request.close
    scaled_sma_20 = request.sma_20 / request.close
    scaled_sma_50 = request.sma_50 / request.close
    scaled_bb_upper = request.bb_upper / request.close
    scaled_bb_lower = request.bb_lower / request.close
    scaled_macd = request.macd / request.close
    scaled_macd_signal = request.macd_signal / request.close
    
    # Prepare scaled features vector
    features_df = pd.DataFrame([{
        'sma_5': scaled_sma_5,
        'sma_20': scaled_sma_20,
        'sma_50': scaled_sma_50,
        'rsi_14': request.rsi_14,
        'macd': scaled_macd,
        'macd_signal': scaled_macd_signal,
        'bb_upper': scaled_bb_upper,
        'bb_lower': scaled_bb_lower,
        'volume_change_pct': request.volume_change_pct,
        'vix': request.vix,
        'day_of_week': request.day_of_week,
        'month': request.month
    }])
    
    # Predict percentage return
    predicted_return = float(model.predict(features_df)[0])
    
    # Reconstruct absolute predicted close price
    predicted_price = float(request.close * (1 + predicted_return))
    
    # Save prediction to DB predictions table (including symbol column)
    conn = config.get_db_connection()
    try:
        cursor = conn.cursor()
        placeholder = "?" if config.DB_TYPE == "sqlite" else "%s"
        
        # Serialize raw input features for log tracking
        input_features_dict = request.dict()
        input_features_serialized = json.dumps(input_features_dict)
        if config.DB_TYPE == "postgresql":
            input_features_serialized = input_features_dict
            
        cursor.execute(f"""
        INSERT INTO predictions (
            symbol, prediction_date, target_date, predicted_price, input_features, model_version
        ) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        """, (
            request.symbol,
            request.date,
            request.target_date,
            predicted_price,
            input_features_serialized,
            model_version
        ))
        conn.commit()
    except Exception as e:
        print(f"Error logging prediction to database for {request.symbol}: {e}")
    finally:
        conn.close()
        
    return {
        "symbol": request.symbol,
        "prediction_date": request.date,
        "target_date": request.target_date,
        "predicted_return_pct": round(predicted_return * 100, 4),
        "predicted_price": predicted_price,
        "model_version": model_version
    }

@app.get("/metrics")
def get_metrics(symbol: str = None):
    """
    Calculates and returns rolling 30-day metrics (MAE, RMSE, Directional Accuracy)
    from the actuals table, optionally filtered by symbol.
    """
    conn = config.get_db_connection()
    try:
        if symbol:
            query = """
            SELECT date, actual_price, predicted_price, absolute_error, direction_correct 
            FROM actuals 
            WHERE symbol = ? 
            ORDER BY date DESC LIMIT 30
            """
            df = pd.read_sql_query(query, conn, params=[symbol])
        else:
            query = """
            SELECT date, actual_price, predicted_price, absolute_error, direction_correct 
            FROM actuals 
            ORDER BY date DESC LIMIT 30
            """
            df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
        
    if df.empty:
        return {
            "symbol_filtered": symbol,
            "days_recorded": 0,
            "mae": None,
            "rmse": None,
            "directional_accuracy": None
        }
        
    mae = float(df['absolute_error'].mean())
    rmse = float(np.sqrt(( (df['actual_price'] - df['predicted_price']) ** 2 ).mean()))
    directional_accuracy = float(df['direction_correct'].mean())
    
    return {
        "symbol_filtered": symbol,
        "days_recorded": len(df),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "directional_accuracy": round(directional_accuracy * 100, 2)
    }

@app.post("/retrain")
def retrain_model_endpoint(background_tasks: BackgroundTasks):
    """
    Asynchronously triggers XGBoost training.
    """
    background_tasks.add_task(train.train_model)
    return {"status": "retraining started in background"}
