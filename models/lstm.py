"""
LSTM (Long Short-Term Memory) model for stock price prediction
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from typing import Tuple


def prepare_data(data: pd.DataFrame, time_step: int = 60, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare stock data for LSTM training WITHOUT data leakage
    
    Args:
        data: Stock price DataFrame with OHLCV columns
        time_step: Number of time steps for sequence (default: 60)
        train_ratio: Ratio of data to use for training (default: 0.8)
    
    Returns:
        Tuple of (X, y, scaler) where:
            X: Input sequences (samples, time_step, features)
            y: Target values (samples, 1)
            scaler: Fitted MinMaxScaler (fitted ONLY on training data)
    
    Note: Scaler is fit ONLY on the training portion to prevent data leakage
    """
    # Split data into train/test BEFORE scaling
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    
    # Fit scaler ONLY on training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    # Transform entire dataset using scaler fitted on train data only
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i])  # Shape: (time_step, features)
        y.append(scaled_data[i, 3])  # Predict 'Close' price
    
    X, y = np.array(X), np.array(y).reshape(-1, 1)  # y shape: (samples, 1)
    return X, y, scaler


def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
    """
    Train LSTM model for stock price prediction
    
    Args:
        X_train: Training input sequences (samples, time_step, features)
        y_train: Training target values (samples, 1)
    
    Returns:
        Trained LSTM model
    """
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)  # Predict Close price
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model


def predict_stock(model: Sequential, data: pd.DataFrame, scaler: MinMaxScaler, time_step: int = 60) -> np.ndarray:
    """
    Make predictions using trained LSTM model
    
    Args:
        model: Trained LSTM model
        data: Stock price DataFrame with OHLCV columns
        scaler: Fitted scaler from training
        time_step: Number of time steps for sequence (default: 60)
    
    Returns:
        Array of predicted Close prices
    """
    # Scale data using multiple features
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    # Prepare test sequences
    X_test = [scaled_data[i-time_step:i] for i in range(time_step, len(scaled_data))]
    X_test = np.array(X_test)
    
    # LSTM prediction
    predicted_prices = model.predict(X_test, verbose=0)
    
    # Inverse transform - only denormalize 'Close' feature
    close_index = 3  # 'Close' is at index 3
    predicted_close_prices = scaler.inverse_transform(
        np.concatenate([
            np.zeros((predicted_prices.shape[0], close_index)),  # Padding for other features
            predicted_prices,  # Insert predicted Close
            np.zeros((predicted_prices.shape[0], scaled_data.shape[1] - close_index - 1))  # Padding
        ], axis=1)
    )[:, close_index]  # Extract only Close predictions
    
    return predicted_close_prices
