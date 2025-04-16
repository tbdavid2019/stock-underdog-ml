import argparse
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta

# ===== 模型定義（MLP / LSTM / GRU / ALSTM）=====
# 回溯模擬：2025/3/16 當作今天，預測接下來 10 天
# python3 app_series_BasicNeural.py --ticker AAPL --days 10 --period 6mo --cutoff 2025-03-16 --compare real
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    def forward(self, x): return self.net(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x): return self.fc(self.lstm(x)[0][:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=5):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x): return self.fc(self.gru(x)[0][:, -1, :])

class ALSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.0, output_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.Tanh())
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.att_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_size * 2, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        rnn_out, _ = self.rnn(x)
        att_score = self.att_net(rnn_out)
        att_out = torch.sum(rnn_out * att_score, dim=1)
        out = torch.cat([rnn_out[:, -1, :], att_out], dim=1)
        return self.fc(out)

# ===== 資料處理 =====

def fetch_data(ticker, period="3mo"):
    df = yf.download(ticker, period=period)[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = df.index.tz_localize(None)
    return df

def prepare_data(df, window_size=10, forecast_days=5):
    X, Y = [], []
    for i in range(len(df) - window_size - forecast_days):
        X.append(df.iloc[i:i+window_size].values)
        Y.append(df['Close'].iloc[i+window_size:i+window_size+forecast_days].values)
    return np.array(X), np.array(Y)

# ===== 模型訓練與預測 =====

def train_and_predict(model, X_tensor, Y_tensor, latest_tensor):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for _ in range(100):
        model.train()
        pred = model(X_tensor)
        loss = loss_fn(pred, Y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(latest_tensor).numpy()[0]

# ===== 主流程 =====

def main(ticker, forecast_days, selected_models, period, compare_real, cutoff_str):
    df_all = fetch_data(ticker, period)

    if cutoff_str:
        cutoff = datetime.strptime(cutoff_str, "%Y-%m-%d")
        df_train = df_all[df_all.index < cutoff]
        df_test = df_all[df_all.index >= cutoff]
    else:
        cutoff = df_all.index[-1]
        df_train = df_all
        df_test = pd.DataFrame()

    X, Y = prepare_data(df_train, window_size=10, forecast_days=forecast_days)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    X_tensor_seq = torch.tensor(X_scaled, dtype=torch.float32)
    X_tensor_flat = torch.tensor(X_scaled.reshape(X_scaled.shape[0], -1), dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).squeeze()

    latest_input = df_train.iloc[-10:].values.reshape(1, 10, -1)
    latest_scaled = scaler.transform(latest_input.reshape(-1, latest_input.shape[-1])).reshape(1, 10, -1)
    latest_tensor_seq = torch.tensor(latest_scaled, dtype=torch.float32)
    latest_tensor_flat = torch.tensor(latest_scaled.reshape(1, -1), dtype=torch.float32)

    results = {}

    if "mlp" in selected_models:
        model = MLP(input_size=X_tensor_flat.shape[1], output_size=forecast_days)
        results["MLP"] = train_and_predict(model, X_tensor_flat, Y_tensor, latest_tensor_flat)

    if "lstm" in selected_models:
        model = LSTMModel(input_dim=X_tensor_seq.shape[2], output_dim=forecast_days)
        results["LSTM"] = train_and_predict(model, X_tensor_seq, Y_tensor, latest_tensor_seq)

    if "gru" in selected_models:
        model = GRUModel(input_dim=X_tensor_seq.shape[2], output_dim=forecast_days)
        results["GRU"] = train_and_predict(model, X_tensor_seq, Y_tensor, latest_tensor_seq)

    if "alstm" in selected_models:
        model = ALSTMModel(input_dim=X_tensor_seq.shape[2], output_dim=forecast_days)
        results["ALSTM"] = train_and_predict(model, X_tensor_seq, Y_tensor, latest_tensor_seq)

    # === 畫圖 ===
    forecast_dates = [cutoff + timedelta(days=i+1) for i in range(forecast_days)]
    plt.figure(figsize=(12, 6))

    for name, forecast in results.items():
        plt.plot(forecast_dates, forecast, label=name)

    if compare_real and not df_test.empty:
        real_segment = df_test['Close'].iloc[:forecast_days]
        if len(real_segment) == forecast_days:
            plt.plot(real_segment.index, real_segment.values, label="Real", color='black', linestyle='--')

    plt.title(f"{ticker} Forecast for Next {forecast_days} Days (Neural Nets)")
    plt.xlabel("Date")
    plt.ylabel("Predicted Close Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    filename = f"series_BasicNeural_{ticker.lower()}_forecast.png"
    if "DISPLAY" in os.environ:
        plt.show()
    else:
        plt.savefig(filename)
        print(f"圖已儲存為 {filename}")

# ===== CLI =====

if __name__ == "__main__":
    import pandas as pd
    parser = argparse.ArgumentParser(description="Compare MLP, LSTM, GRU, ALSTM with optional backtest cutoff")
    parser.add_argument('--ticker', type=str, default='TSLA')
    parser.add_argument('--days', type=int, default=5)
    parser.add_argument('--model', type=str, default='mlp,lstm,gru,alstm')
    parser.add_argument('--period', type=str, default='3mo', help='資料期間，如 3mo, 6mo, 1y')
    parser.add_argument('--compare', type=str, default='', help='加上 "real" 可顯示真實價格線')
    parser.add_argument('--cutoff', type=str, default='', help='模擬預測的基準日期，如 2025-03-16')
    args = parser.parse_args()

    model_list = args.model.lower().split(',')
    compare_real = args.compare.lower() == 'real'

    main(args.ticker, args.days, model_list, args.period, compare_real, args.cutoff)