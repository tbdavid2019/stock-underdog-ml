import argparse
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# === ALSTM 模型 ===
class ALSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.0, output_dim=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh()
        )
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.att_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_size * 2, output_dim)

    def forward(self, x):
        B, T, D = x.shape
        x = self.encoder(x)                         # [B, T, H]
        rnn_out, _ = self.rnn(x)                    # [B, T, H]
        att_score = self.att_net(rnn_out)           # [B, T, 1]
        att_out = torch.sum(rnn_out * att_score, dim=1)  # [B, H]
        out = torch.cat([rnn_out[:, -1, :], att_out], dim=1)  # [B, H*2]
        return self.fc(out)                         # [B, output_dim]

# === 資料準備 ===
def fetch_data(ticker, period="3mo"):
    df = yf.download(ticker, period=period)[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def prepare_data(df, window_size=10, forecast_days=5):
    X, Y = [], []
    for i in range(len(df) - window_size - forecast_days):
        X.append(df.iloc[i:i+window_size].values)
        Y.append(df['Close'].iloc[i+window_size:i+window_size+forecast_days].values)
    return np.array(X), np.array(Y)

# === 主流程 ===
def main(ticker, forecast_days):
    print(f"📈 ALSTM 預測 {ticker} 未來 {forecast_days} 天收盤價")
    df = fetch_data(ticker)
    X, Y = prepare_data(df, window_size=10, forecast_days=forecast_days)

    scaler = StandardScaler()
    X_shape = X.shape
    X_scaled = scaler.fit_transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    if Y_tensor.ndim == 3:
        Y_tensor = Y_tensor.squeeze(-1)

    model = ALSTMModel(input_dim=X.shape[2], output_dim=forecast_days)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("🧠 開始訓練...")
    for epoch in range(200):
        model.train()
        pred = model(X_tensor)
        loss = loss_fn(pred, Y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # 預測
    latest = df.iloc[-10:].values.reshape(1, 10, -1)
    latest_scaled = scaler.transform(latest.reshape(-1, latest.shape[-1])).reshape(1, 10, -1)
    latest_tensor = torch.tensor(latest_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        forecast = model(latest_tensor).numpy()[0]

    # 畫圖
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, forecast_days + 1), forecast, label='ALSTM', color='purple')
    plt.title(f"{ticker} Forecast for Next {forecast_days} Days (ALSTM)")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Close Price")
    plt.legend()
    plt.grid(True)

    if "DISPLAY" in os.environ:
        plt.show()
    else:
        filename = f"alstm_{ticker.lower()}_forecast.png"
        plt.savefig(filename)
        print(f"圖已儲存為 {filename}")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALSTM-based stock price forecast")
    parser.add_argument('--ticker', type=str, default='TSLA', help='股票代碼，例如 TSLA, AAPL')
    parser.add_argument('--days', type=int, default=5, help='預測未來幾天的收盤價')
    args = parser.parse_args()

    main(args.ticker, args.days)