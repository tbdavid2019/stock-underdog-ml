"""
Prophet model for stock price prediction
"""
import pandas as pd
from prophet import Prophet
from typing import Tuple


def train_prophet_model(data: pd.DataFrame) -> Prophet:
    """
    Train Prophet model for stock price forecasting
    
    Args:
        data: Stock price DataFrame with Date and Close columns
    
    Returns:
        Trained Prophet model
    
    Raises:
        ValueError: If data contains negative values or insufficient data points
    """
    # Reset index and prepare data
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']  # Prophet required format
    df = df.dropna()  # Remove missing values
    
    # Ensure no negative values
    if (df['y'] < 0).any():
        raise ValueError("發現負值，無法訓練 Prophet 模型")
    
    # Check sufficient data
    if len(df) < 30:  # Need at least 30 data points
        raise ValueError("數據不足，無法訓練 Prophet 模型")
    
    # Initialize and train Prophet model
    model = Prophet(
        yearly_seasonality=True, 
        daily_seasonality=True, 
        changepoint_prior_scale=0.1
    )
    model.fit(df)
    return model


def predict_with_prophet(model: Prophet, data: pd.DataFrame, prediction_days: int = 3) -> pd.DataFrame:
    """
    Use Prophet to predict future stock prices
    
    Args:
        model: Trained Prophet model
        data: Original stock data (including Close prices)
        prediction_days: Number of days to predict (default: 3)
    
    Returns:
        DataFrame with prediction results (ds, yhat, yhat_lower, yhat_upper)
    """
    # Get latest Close value
    last_close = data['Close'].values[-1]
    
    # Create future dates
    future = model.make_future_dataframe(periods=prediction_days)
    
    # Predict future data
    forecast = model.predict(future)
    
    # Set reasonable bounds to avoid extreme predictions
    lower_bound = last_close * 0.8
    upper_bound = last_close * 1.2
    forecast['yhat'] = forecast['yhat'].apply(lambda x: min(max(x, lower_bound), upper_bound))
    
    # Return recent predictions
    return forecast.tail(prediction_days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
