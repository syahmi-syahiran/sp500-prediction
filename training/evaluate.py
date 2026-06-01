import pandas as pd
import numpy as np

def compute_directional_accuracy(actual_prices, predicted_prices, current_closes):
    """
    Computes directional accuracy: % of times predicted direction (relative to current close)
    matches the actual direction of next day's close.
    """
    actual_series = pd.Series(actual_prices).reset_index(drop=True)
    pred_series = pd.Series(predicted_prices).reset_index(drop=True)
    close_series = pd.Series(current_closes).reset_index(drop=True)
    
    # Calculate actual movement direction (True if up, False if down/same)
    actual_dir = actual_series > close_series
    
    # Calculate predicted movement direction (True if predicted up, False if down/same)
    pred_dir = pred_series > close_series
    
    # Check if predicted matches actual
    matches = actual_dir == pred_dir
    return float(matches.mean())
