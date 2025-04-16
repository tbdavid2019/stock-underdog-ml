import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def fetch_data(ticker, period="3mo"):
    df = yf.download(ticker, period=period)[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def prepare_training_data(df, window_size=10, forecast_days=5):
    X, Y = [], []
    for i in range(len(df) - window_size - forecast_days):
        X.append(df.iloc[i:i+window_size].values.flatten())
        Y.append(df['Close'].iloc[i+window_size : i+window_size+forecast_days].values)
    return np.array(X), np.array(Y).reshape(len(Y), -1)

def main(ticker, forecast_days):
    print(f"下載 {ticker} 股票資料...")
    df = fetch_data(ticker)

    print("建立訓練資料...")
    X, Y = prepare_training_data(df, window_size=10, forecast_days=forecast_days)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("訓練 LightGBM 模型...")
    model = MultiOutputRegressor(LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        verbose=-1
    ))
    model.fit(X_scaled, Y)

    print(f"使用最新資料預測未來 {forecast_days} 天...")
    latest_input = df.iloc[-10:].values.flatten().reshape(1, -1)
    latest_input_scaled = scaler.transform(latest_input)
    forecast = model.predict(latest_input_scaled)[0]

    # 繪圖
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, forecast_days + 1), forecast, label='LightGBM', color='blue')
    plt.title(f"{ticker} Forecast for Next {forecast_days} Days (LightGBM)")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Close Price")
    plt.legend()
    plt.grid(True)

    if "DISPLAY" in os.environ:
        plt.show()
    else:
        filename = f"lightgbm_{ticker.lower()}_forecast.png"
        plt.savefig(filename)
        print(f"圖已儲存為 {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast future stock price using LightGBM")
    parser.add_argument('--ticker', type=str, default='TSLA', help='股票代碼，例如 TSLA, AAPL')
    parser.add_argument('--days', type=int, default=5, help='預測未來幾天的收盤價')
    args = parser.parse_args()

    main(args.ticker, args.days)