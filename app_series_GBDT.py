import argparse
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from datetime import datetime, timedelta

# 用法
# python3 app_series_GBDT.py --ticker AAPL --days 10 --period 6mo --cutoff 2025-03-16 --compare real
# ==== Double Ensemble 實作 ====
class DoubleEnsembleLightGBM:
    def __init__(self, num_models=6, **lgb_params):
        self.num_models = num_models
        self.models = []
        self.lgb_params = lgb_params

    def fit(self, X, Y):
        for i in range(self.num_models):
            print(f"[DoubleEnsemble] 訓練子模型 {i+1}/{self.num_models} ...")
            sample_idx = np.random.choice(len(X), size=int(len(X) * 0.8), replace=False)
            model = MultiOutputRegressor(LGBMRegressor(**self.lgb_params))
            model.fit(X[sample_idx], Y[sample_idx])
            self.models.append(model)

    def predict(self, X):
        preds = np.stack([model.predict(X) for model in self.models])
        return np.mean(preds, axis=0)

# ==== 資料處理 ====
def fetch_data(ticker, period="3mo"):
    df = yf.download(ticker, period=period)[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = df.index.tz_localize(None)
    return df

def prepare_training_data(df, window_size=10, forecast_days=5):
    X, Y = [], []
    for i in range(len(df) - window_size - forecast_days):
        X.append(df.iloc[i:i+window_size].values.flatten())
        Y.append(df['Close'].iloc[i+window_size : i+window_size+forecast_days].values)
    return np.array(X), np.array(Y).reshape(len(Y), -1)

# ==== 主邏輯 ====
def main(ticker, forecast_days, model_list, period, compare_real, cutoff_str):
    df_all = fetch_data(ticker, period)

    if cutoff_str:
        cutoff = datetime.strptime(cutoff_str, "%Y-%m-%d")
        df_train = df_all[df_all.index < cutoff]
        df_test = df_all[df_all.index >= cutoff]
    else:
        cutoff = df_all.index[-1]
        df_train = df_all
        df_test = pd.DataFrame()

    print(f"📈 使用 {ticker} 訓練資料（共 {len(df_train)} 筆），預測 {forecast_days} 天")

    X, Y = prepare_training_data(df_train, window_size=10, forecast_days=forecast_days)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    latest_input = df_train.iloc[-10:].values.flatten().reshape(1, -1)
    latest_input_scaled = scaler.transform(latest_input)

    results = {}

    if 'xgb' in model_list:
        print("🔧 訓練 XGBoost...")
        model = MultiOutputRegressor(XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, objective='reg:squarederror'))
        model.fit(X_scaled, Y)
        results['XGBoost'] = model.predict(latest_input_scaled)[0]

    if 'lgb' in model_list:
        print("🔧 訓練 LightGBM...")
        model = MultiOutputRegressor(LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, verbose=-1))
        model.fit(X_scaled, Y)
        results['LightGBM'] = model.predict(latest_input_scaled)[0]

    if 'cat' in model_list:
        print("🔧 訓練 CatBoost...")
        model = MultiOutputRegressor(CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6, verbose=0))
        model.fit(X_scaled, Y)
        results['CatBoost'] = model.predict(latest_input_scaled)[0]

    if 'double' in model_list:
        print("🔧 訓練 DoubleEnsemble (LightGBM)...")
        model = DoubleEnsembleLightGBM(
            num_models=6,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            verbose=-1
        )
        model.fit(X_scaled, Y)
        results['DoubleEnsemble'] = model.predict(latest_input_scaled)[0]

    # === 畫圖 ===
    forecast_dates = [cutoff + timedelta(days=i+1) for i in range(forecast_days)]

    plt.figure(figsize=(12, 6))
    for name, forecast in results.items():
        plt.plot(forecast_dates, forecast, label=name)

    # 加入真實線
    if compare_real and not df_test.empty:
        real_segment = df_test['Close'].iloc[:forecast_days]
        if len(real_segment) == forecast_days:
            plt.plot(real_segment.index, real_segment.values, label="Real", color='black', linestyle='--')

    plt.title(f"{ticker} Forecast for Next {forecast_days} Days (GBDT Models)")
    plt.xlabel("Date")
    plt.ylabel("Predicted Close Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    filename = f"series_gbdt_{ticker.lower()}_forecast.png"
    if "DISPLAY" in os.environ:
        plt.show()
    else:
        plt.savefig(filename)
        print(f"圖已儲存為 {filename}")

# ==== CLI 入口 ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple GBDT models for future stock price forecasting")
    parser.add_argument('--ticker', type=str, default='TSLA', help='股票代碼，例如 TSLA, AAPL')
    parser.add_argument('--days', type=int, default=5, help='預測未來幾天的收盤價')
    parser.add_argument('--model', type=str, default='xgb,lgb,cat,double', help='選擇模型，用逗號分隔，例如 xgb,lgb')
    parser.add_argument('--period', type=str, default='3mo', help='抓資料期間，例如 3mo, 6mo, 1y')
    parser.add_argument('--compare', type=str, default='', help='加上 "real" 顯示真實價格線')
    parser.add_argument('--cutoff', type=str, default='', help='模擬今天的日期，例如 2025-03-16')
    args = parser.parse_args()

    model_list = args.model.lower().split(',')
    compare_real = args.compare.lower() == 'real'

    main(args.ticker, args.days, model_list, args.period, compare_real, args.cutoff)