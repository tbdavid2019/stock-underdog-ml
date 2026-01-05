"""
Chronos-Bolt model for stock price prediction using AutoGluon
"""
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from typing import Tuple


def prepare_chronos_data(data: pd.DataFrame) -> TimeSeriesDataFrame:
    """
    Prepare stock data for Chronos model
    
    Args:
        data: Stock price DataFrame with Date index and Close column
    
    Returns:
        TimeSeriesDataFrame formatted for AutoGluon
    
    Raises:
        Exception: If TimeSeriesDataFrame creation fails
    """
    df = data.reset_index()
    formatted_df = pd.DataFrame({
        'item_id': ['stock'] * len(df),
        'timestamp': pd.to_datetime(df['Date']),
        'target': df['Close'].astype('float32').values.ravel()
    })
    formatted_df = formatted_df.sort_values('timestamp')
    
    try:
        ts_df = TimeSeriesDataFrame.from_data_frame(
            formatted_df,
            id_column='item_id',
            timestamp_column='timestamp'
        )
        return ts_df
    except Exception as e:
        print(f"Error creating TimeSeriesDataFrame: {str(e)}")
        raise


def train_and_predict_chronos(data: pd.DataFrame, prediction_length: int = 10) -> pd.DataFrame:
    """
    Train Chronos model and make predictions
    
    Args:
        data: Stock price DataFrame
        prediction_length: Number of periods to predict (default: 10)
    
    Returns:
        DataFrame with predictions
    """
    # Prepare data
    ts_df = prepare_chronos_data(data)
    
    # Initialize predictor
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length, 
        freq="D", 
        target="target"
    )
    
    # Train with Chronos-Bolt model
    predictor.fit(
        ts_df, 
        hyperparameters={
            "Chronos": {"model_path": "autogluon/chronos-bolt-base"}
        }
    )
    
    # Make predictions
    predictions = predictor.predict(ts_df)
    
    return predictions
