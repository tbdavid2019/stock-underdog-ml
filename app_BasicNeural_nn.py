import argparse
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ==== MLP 模型定義 ====
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

# ==== 資料處理 ====
def fetch_data(ticker, period="3mo"):
    df = yf.download(ticker, period=period)[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def prepare_data(df, window_size=10, forecast_days=5):
    X, Y = [], []
    for i in range(len(df) - window_size - forecast_days):
        X.append(df.iloc[i:i+window_size].values.flatten())
        Y.append(df['Close'].iloc[i+window_size:i+window_size+forecast_days].values)
    return np.array(X), np.array(Y)

# ==== 主流程 ====
def main(ticker, forecast_days):
    df = fetch_data(ticker)
    print(f"📈 訓練 MLP 模型預測 {ticker} 未來 {forecast_days} 天股價")

    X, Y = prepare_data(df, window_size=10, forecast_days=forecast_days)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    # 修正錯誤：確保 Y_tensor 是 2D，而非 [n, x, 1]
    Y_tensor = Y_tensor.squeeze(-1) if Y_tensor.ndim == 3 else Y_tensor

    input_size = X_tensor.shape[1]
    output_size = Y_tensor.shape[1]

    model = MLP(input_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("🧠 開始訓練...")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = loss_fn(pred, Y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    # 預測未來
    latest_input = df.iloc[-10:].values.flatten().reshape(1, -1)
    latest_scaled = scaler.transform(latest_input)
    latest_tensor = torch.tensor(latest_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        forecast = model(latest_tensor).numpy()[0]

    # 繪圖
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, forecast_days + 1), forecast, label='MLP (Neural Net)', color='red')
    plt.title(f"{ticker} Forecast for Next {forecast_days} Days (MLP)")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Close Price")
    plt.legend()
    plt.grid(True)

    if "DISPLAY" in os.environ:
        plt.show()
    else:
        filename = f"mlp_{ticker.lower()}_forecast.png"
        plt.savefig(filename)
        print(f"圖已儲存為 {filename}")

# ==== CLI 入口 ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP-based forecast of stock prices")
    parser.add_argument('--ticker', type=str, default='TSLA', help='股票代碼，例如 TSLA, AAPL')
    parser.add_argument('--days', type=int, default=5, help='預測未來幾天的收盤價')
    args = parser.parse_args()

    main(args.ticker, args.days)