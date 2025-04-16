import argparse
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ==== LSTM 模型定義 ====
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最後一個時間步的輸出
        out = self.fc(out)
        return out

# ==== 資料處理 ====
def fetch_data(ticker, period="3mo"):
    df = yf.download(ticker, period=period)[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def prepare_data(df, window_size=10, forecast_days=5):
    X, Y = [], []
    for i in range(len(df) - window_size - forecast_days):
        X.append(df.iloc[i:i+window_size].values)
        Y.append(df['Close'].iloc[i+window_size:i+window_size+forecast_days].values)
    return np.array(X), np.array(Y)

# ==== 主流程 ====
def main(ticker, forecast_days):
    df = fetch_data(ticker)
    print(f"📈 訓練 LSTM 模型預測 {ticker} 未來 {forecast_days} 天股價")

    X, Y = prepare_data(df, window_size=10, forecast_days=forecast_days)

    # 資料標準化
    scaler = StandardScaler()
    X_shape = X.shape
    X_reshaped = X.reshape(-1, X_shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped).reshape(X_shape)

    # 轉換為 tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    if Y_tensor.ndim == 3:
        Y_tensor = Y_tensor.squeeze(-1)  # 🔧 修復多餘維度錯誤

    # 建立模型
    model = LSTMModel(input_dim=X.shape[2], output_dim=forecast_days)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # 訓練模型
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
    latest_input = df.iloc[-10:].values.reshape(1, 10, -1)
    latest_scaled = scaler.transform(latest_input.reshape(-1, latest_input.shape[-1])).reshape(1, 10, -1)
    latest_tensor = torch.tensor(latest_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        forecast = model(latest_tensor).numpy()[0]

    # 繪圖
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, forecast_days + 1), forecast, label='LSTM (Neural Net)', color='green')
    plt.title(f"{ticker} Forecast for Next {forecast_days} Days (LSTM)")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Close Price")
    plt.legend()
    plt.grid(True)

    if "DISPLAY" in os.environ:
        plt.show()
    else:
        filename = f"lstm_{ticker.lower()}_forecast.png"
        plt.savefig(filename)
        print(f"圖已儲存為 {filename}")

# ==== CLI ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM-based forecast of stock prices")
    parser.add_argument('--ticker', type=str, default='TSLA', help='股票代碼，例如 TSLA, AAPL')
    parser.add_argument('--days', type=int, default=5, help='預測未來幾天的收盤價')
    args = parser.parse_args()

    main(args.ticker, args.days)