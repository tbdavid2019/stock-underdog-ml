"""
Transformer model for stock price prediction
"""
import numpy as np
import pandas as pd
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def build_transformer_model(input_shape: Tuple[int, int]) -> Model:
    """
    Build Transformer model architecture
    
    Args:
        input_shape: Tuple of (time_step, features)
    
    Returns:
        Compiled Transformer model
    """
    inputs = Input(shape=input_shape)
    
    # Transformer Encoder Layer
    attention = MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(inputs, inputs)
    attention = Dropout(0.1)(attention)
    attention = Add()([inputs, attention])  # Residual connection
    attention = LayerNormalization(epsilon=1e-6)(attention)
    
    # Feed Forward Layer
    feed_forward = Dense(64, activation="relu")(attention)
    feed_forward = Dropout(0.1)(feed_forward)
    outputs = Dense(1)(feed_forward)  # Predict Close price
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


def train_transformer_model(X_train: np.ndarray, y_train: np.ndarray, input_shape: Tuple[int, int]) -> Model:
    """
    Train Transformer model
    
    Args:
        X_train: Training input sequences
        y_train: Training target values
        input_shape: Tuple of (time_step, features)
    
    Returns:
        Trained Transformer model
    """
    model = build_transformer_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model


def predict_transformer(model: Model, data: pd.DataFrame, scaler: MinMaxScaler, time_step: int = 60) -> np.ndarray:
    """
    Make predictions using trained Transformer model
    
    Args:
        model: Trained Transformer model
        data: Stock price DataFrame
        scaler: Fitted scaler
        time_step: Number of time steps
    
    Returns:
        Array of predicted Close prices
    """
    # Scale data using multiple features
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    # Prepare test sequences
    X_test = [scaled_data[i-time_step:i] for i in range(time_step, len(scaled_data))]
    X_test = np.array(X_test)
    
    print(f"X_test shape: {X_test.shape}")
    
    # Transformer prediction
    predicted_prices = model.predict(X_test, verbose=0)
    
    # Fix predicted_prices shape if needed
    if len(predicted_prices.shape) > 2:
        predicted_prices = predicted_prices[:, -1, 0]  # Take last time step prediction
    
    print(f"predicted_prices shape after reshape: {predicted_prices.shape}")
    
    # Inverse transform - only denormalize Close feature
    close_index = 3  # Close feature index in scaled_data
    
    # Build full data structure for inverse transform
    full_predictions = np.zeros((predicted_prices.shape[0], scaled_data.shape[1]))
    full_predictions[:, close_index] = predicted_prices  # Insert Close predictions
    
    # Use scaler to inverse transform
    predicted_close_prices = scaler.inverse_transform(full_predictions)[:, close_index]
    return predicted_close_prices
