"""
LSTM (Long Short-Term Memory) model for NEXT-DAY stock price prediction
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from typing import Tuple


def prepare_data(data: pd.DataFrame, time_step: int = 90, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare stock data for LSTM training WITHOUT data leakage
    
    Args:
        data: Stock price DataFrame with OHLCV columns
        time_step: Number of time steps for sequence (default: 90)
        train_ratio: Ratio of data to use for training (default: 0.8)
    
    Returns:
        Tuple of (X, y, scaler) where:
            X: Input sequences (samples, time_step, features)
            y: Target values (samples, 1) - NEXT DAY's close price
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
        X.append(scaled_data[i - time_step:i])  # Past time_step days
        y.append(scaled_data[i, 3])  # NEXT day's Close price (index 3)
    
    X, y = np.array(X), np.array(y).reshape(-1, 1)
    return X, y, scaler


def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
    """
    Train LSTM model for NEXT-DAY stock price prediction
    
    Args:
        X_train: Training input sequences (samples, time_step, features)
        y_train: Training target values (samples, 1) - next day's price
    
    Returns:
        Trained LSTM model
    """
    # Improved architecture with more capacity
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=128, return_sequences=True),
        Dropout(0.3),
        LSTM(units=64, return_sequences=True),
        Dropout(0.3),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1)
    ])
    # Use better optimizer with learning rate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    # Add callbacks for better training
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
    
    model.fit(
        X_train, y_train, 
        epochs=20,
        batch_size=16,
        verbose=0,
        callbacks=[early_stop, reduce_lr],
        validation_split=0.1
    )
    return model


def predict_next_day(model: Sequential, data: pd.DataFrame, scaler: MinMaxScaler, time_step: int = 90) -> float:
    """
    Predict NEXT trading day's closing price
    
    Args:
        model: Trained LSTM model
        data: Stock price DataFrame with OHLCV columns
        scaler: Fitted scaler from training
        time_step: Number of time steps for sequence (default: 90)
    
    Returns:
        Predicted next day's closing price (single float value)
    """
    # Scale data
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    # Use the last time_step days to predict next day
    if len(scaled_data) < time_step:
        raise ValueError(f"Not enough data: need {time_step} days, got {len(scaled_data)}")
    
    last_sequence = scaled_data[-time_step:]  # Shape: (time_step, features)
    X_pred = np.array([last_sequence])  # Shape: (1, time_step, features)
    
    # Predict next day's scaled close price
    predicted_scaled = model.predict(X_pred, verbose=0)  # Shape: (1, 1)
    
    # Inverse transform to get actual price
    # Create dummy array with zeros for other features
    close_index = 3
    dummy = np.zeros((1, 5))
    dummy[0, close_index] = predicted_scaled[0, 0]
    predicted_price = scaler.inverse_transform(dummy)[0, close_index]
    
    return float(predicted_price)
