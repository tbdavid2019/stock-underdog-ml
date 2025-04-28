import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from prophet import Prophet
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
import requests
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import mysql.connector
from mysql.connector import Error
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
# --- Cross‑section 模型用 ---
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import inspect, importlib, warnings
import pandas_ta as ta         # 算技術指標

# 加載 .env 文件
load_dotenv()


# MySQL 配置
use_mysql = os.getenv("USE_MYSQL", "false").lower() == "true"

***REMOVED*** 設置
smtp_server = os.getenv("SMTP_SERVER")
port = int(os.getenv("SMTP_PORT"))
sender_email = os.getenv("SENDER_EMAIL")
password = os.getenv("EMAIL_PASSWORD")
to_emails = os.getenv("TO_EMAILS").split(",")

# Telegram 設置
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_channel_id = os.getenv("TELEGRAM_CHANNEL_ID")

# MongoDB 連接設置
mongo_uri = os.getenv("MONGO_URI")
db_name = "stock_predictions"

discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

# 是否使用 Transformer
use_transformer = os.getenv("USE_TRANSFORMER", "false").lower() == "true"
transformer_period = os.getenv("TRANSFORMER_PERIOD", "1y")  # 默認 1 年數據

# 是否使用 Prophet
use_prophet = os.getenv("USE_PROPHET", "false").lower() == "true"

# 是否使用 chronos
use_chronos = os.getenv("USE_CHRONOS", "true").lower() == "true"
chronos_period = os.getenv("CHRONOS_PERIOD", "6mo")


class MySQLManager:
    def __init__(self):
        self.enabled = use_mysql
        if not self.enabled:
            print("MySQL 功能未啟用")
            return
        
        try:
            print("嘗試連接 MySQL...")  # 添加此行
            print(f"Host: {os.getenv('MYSQL_HOST')}")  # 添加此行
            print(f"Database: {os.getenv('MYSQL_DATABASE')}")  # 添加此行
            self.connection = mysql.connector.connect(
                host=os.getenv("MYSQL_HOST"),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DATABASE"),
                port=int(os.getenv("MYSQL_PORT", "3306"))
            )
            print("MySQL 連接成功")
            self.create_prediction_table()
        except Error as e:
            print(f"MySQL 連接錯誤: {e}")
            self.connection = None
            self.enabled = False

    def create_prediction_table(self):
        if not self.enabled or not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            
            # 首先檢查表是否存在
            check_table_query = """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_name = 'stock_predictions'
            """
            cursor.execute(check_table_query, (os.getenv('MYSQL_DATABASE'),))
            table_exists = cursor.fetchone()[0] > 0

            if table_exists:
                print("stock_predictions 表已存在，跳過創建")
            else:
                # 創建表
                create_table_query = """
                CREATE TABLE IF NOT EXISTS stock_predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    calculation_date DATE,
                    calculation_time TIME,
                    index_name VARCHAR(50),
                    stock_symbol VARCHAR(20),
                    current_price DECIMAL(10,2),
                    predicted_price DECIMAL(10,2),
                    potential DECIMAL(10,4),
                    prediction_method VARCHAR(20),
                    period_param VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                cursor.execute(create_table_query)
                self.connection.commit()
                print("成功創建 stock_predictions 表")

        except Error as e:
            print(f"檢查/創建表格時發生錯誤: {e}")
        finally:
            cursor.close()




    def save_predictions(self, index_name, predictions, method, period):
        if not self.enabled or not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            current_date = datetime.datetime.now().date()
            current_time = datetime.datetime.now().time()
            
            insert_query = """
            INSERT INTO stock_predictions 
            (calculation_date, calculation_time, index_name, stock_symbol, 
             current_price, predicted_price, potential, prediction_method, period_param)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            for ticker, potential, current_price, predicted_price in predictions:
                data = (
                    current_date,
                    current_time,
                    index_name,
                    ticker,
                    float(current_price.iloc[0]) if isinstance(current_price, pd.Series) else float(current_price),
                    float(predicted_price.iloc[0]) if isinstance(predicted_price, pd.Series) else float(predicted_price),
                    float(potential.iloc[0]) if isinstance(potential, pd.Series) else float(potential),
                    method,
                    period
                )
                cursor.execute(insert_query, data)
            
            self.connection.commit()
            print(f"成功保存 {len(predictions)} 條 {method} 預測結果到 MySQL")
        
        except Error as e:
            print(f"保存到 MySQL 時發生錯誤: {e}")
        finally:
            cursor.close()

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL 連接已關閉")

def save_to_mongodb(index_name, stock_predictions):
    """
    將股票預測結果存入 MongoDB
    :param index_name: 指數名稱
    :param stock_predictions: 預測結果 (dict)
    """
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db["predictions"]

        # 設置要寫入的文件格式
        record = {
            "index": index_name,
            "timestamp": datetime.datetime.now(),
            "predictions": stock_predictions
        }

        # 寫入 MongoDB
        collection.insert_one(record)
        print(f"成功將 {index_name} 結果寫入 MongoDB")
    except Exception as e:
        print(f"⚠️ 寫入 MongoDB 失敗: {str(e)}")
    finally:
        client.close()

# Stock index mappings

def get_tw0050_stocks():
    response = requests.get('https://answerbook.david888.com/TW0050')
    data = response.json()
    
    # 取得股票代碼並加上 .TW
    stocks = [f"{code}.TW" for code in data['TW0050'].keys()]
    
    # 如果需要排序的話可以加上 sort()
    #stocks.sort()
    
    return stocks


def get_tw0051_stocks():
    response = requests.get('https://answerbook.david888.com/TW0051')
    data = response.json()
    
    # 取得股票代碼並加上 .TW
    stocks = [f"{code}.TW" for code in data['TW0051'].keys()]
    
    # 如果需要排序的話可以加上 sort()
    # stocks.sort()
    
    return stocks


def get_sp500_stocks(limit=100):
    response = requests.get('https://answerbook.david888.com/SP500')
    data = response.json()
    
    # 取得股票代碼列表並限制數量
    stocks = list(data['SP500'].keys())[:limit]
    
    # 修正特殊股票代碼格式，例如將 BRK.B 轉換為 BRK-B 以適應 yfinance
    for i, stock in enumerate(stocks):
        if stock == "BRK.B":
            stocks[i] = "BRK-B"
    
    return stocks
    

# Function to fetch NASDAQ component stocks
def get_nasdaq_stocks():
# Function to fetch Philadelphia Semiconductor Index component stocks

    response = requests.get('https://answerbook.david888.com/nasdaq100')
    data = response.json()
    
    # 取得股票代碼列表並限制數量
    stocks = list(data['nasdaq100'].keys())
    
    return stocks


def get_sox_stocks():
    return [
        "NVDA", "AVGO", "GFS", "CRUS", "ON", "ASML", "QCOM", "SWKS", "MPWR", "ADI",
        "TSM", "AMD", "TXN", "QRVO", "AMKR", "MU", "ARM", "NXPI", "TER", "ENTG",
        "LSCC", "COHR", "ONTO", "MTSI", "KLAC", "LRCX", "MRVL", "AMAT", "INTC", "MCHP"
    ]

# Function to fetch Dow Jones Industrial Average component stocks
def get_dji_stocks():

    response = requests.get('https://answerbook.david888.com/dowjones')
    data = response.json()
    
    # 取得股票代碼列表並限制數量
    stocks = list(data['dowjones'].keys())
    
    return stocks


# 獲取股票數據
def get_stock_data(ticker, period):
    try:
        print(f"正在獲取 {ticker} 的數據...")
        data = yf.download(ticker, period=period)
        print(f"獲取到 {len(data)} 條交易日數據")
        return data
    except Exception as e:
        print(f"獲取 {ticker} 數據時發生錯誤: {str(e)}")
        return pd.DataFrame()


def prepare_chronos_data(data):
    df = data.reset_index()
    formatted_df = pd.DataFrame({
        'item_id': ['stock'] * len(df),
        'timestamp': pd.to_datetime(df['Date']),
        'target': df['Close'].astype('float32').values.ravel()
    })
    formatted_df = formatted_df.sort_values('timestamp')
    try:
        ts_df = TimeSeriesDataFrame.from_data_frame(
            formatted_df,
            id_column='item_id',
            timestamp_column='timestamp'
        )
        return ts_df
    except Exception as e:
        print(f"Error creating TimeSeriesDataFrame: {str(e)}")
        raise

def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close',   'Volume']])
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i])  # 確保 X 是 (time_step, features)
        y.append(scaled_data[i, 3])  # 預測 Close 價        
    X, y = np.array(X), np.array(y).reshape(-1, 1)  # y 的形狀應為 (samples, 1)
    return X, y, scaler


def train_lstm_model(X_train, y_train):
    # 使用多個特徵作為輸入
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),  # 修改輸入形狀
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)  # 預測 Close
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model




def predict_stock(model, data, scaler, time_step=60):
    # 使用多個特徵進行標準化
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close',   'Volume']])

    # 準備測試集
    X_test = [scaled_data[i-time_step:i] for i in range(time_step, len(scaled_data))]
    X_test = np.array(X_test)

    # LSTM 預測
    predicted_prices = model.predict(X_test)

    # 反標準化，僅對 `Close` 特徵進行反標準化
    close_index = 3  # 'Close' 的索引
    predicted_close_prices = scaler.inverse_transform(
        np.concatenate([
            np.zeros((predicted_prices.shape[0], close_index)),  # 填充多餘的維度
            predicted_prices,  # 插入預測的 Close
            np.zeros((predicted_prices.shape[0], scaled_data.shape[1] - close_index - 1))  # 填充多餘的維度
        ], axis=1)
    )[:, close_index]  # 只取反標準化後的 Close
    return predicted_close_prices


# Prophet 預測股票
def train_prophet_model(data):
    # 重置索引并准备数据
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']  # Prophet 要求的格式
    df = df.dropna()  # 移除缺失值

    # 确保没有负值
    if (df['y'] < 0).any():
        raise ValueError("发现负值，无法训练 Prophet 模型")

    # 检查数据是否足够
    if len(df) < 30:  # 至少需要 30 条数据
        raise ValueError("数据不足，无法训练 Prophet 模型")

    # 初始化 Prophet 模型 
    model = Prophet(yearly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.1)
    model.fit(df)
    return model

# Prophet 预测股票
def predict_with_prophet(model, data, prediction_days=3):
    """
    使用 Prophet 預測近期股票價格
    :param model: 訓練好的 Prophet 模型
    :param data: 原始股票數據（包含 Close）
    :param prediction_days: 預測天數，默認為 1（隔日）
    :return: 預測結果的 DataFrame
    """
    # 获取最新的 Close 值
    last_close = data['Close'].values[-1]

    # 创建未来日期
    future = model.make_future_dataframe(periods=prediction_days)

    # 预测未来数据
    forecast = model.predict(future)

    # 設置合理的上下限，避免預測值過於誇張
    lower_bound = last_close * 0.8
    upper_bound = last_close * 1.2
    forecast['yhat'] = forecast['yhat'].apply(lambda x: min(max(x, lower_bound), upper_bound))

    # 返回最近的預測值
    return forecast.tail(prediction_days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# 構建 Transformer 模型
def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Transformer Encoder Layer
    attention = MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(inputs, inputs)
    attention = Dropout(0.1)(attention)
    attention = Add()([inputs, attention])  # 殘差連接
    attention = LayerNormalization(epsilon=1e-6)(attention)

    # Feed Forward Layer
    feed_forward = Dense(64, activation="relu")(attention)
    feed_forward = Dropout(0.1)(feed_forward)
    outputs = Dense(1)(feed_forward)  # 預測 Close 價
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model

# 訓練 Transformer 模型
def train_transformer_model(X_train, y_train, input_shape):
    model = build_transformer_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

# Transformer 預測
def predict_transformer(model, data, scaler, time_step=60):
    # 使用多個特徵進行標準化
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close',   'Volume']])

    # 準備測試集
    X_test = [scaled_data[i-time_step:i] for i in range(time_step, len(scaled_data))]
    X_test = np.array(X_test)

    # 打印 X_test 的形狀
    print(f"X_test shape: {X_test.shape}")

    # Transformer 預測
    predicted_prices = model.predict(X_test)

    # 修正 predicted_prices 的形狀
    if len(predicted_prices.shape) > 2:
        predicted_prices = predicted_prices[:, -1, 0]  # 取最後一個時間步的預測值

    # 打印 predicted_prices 的形狀
    print(f"predicted_prices shape after reshape: {predicted_prices.shape}")

    # 反標準化，只對 Close 特徵進行
    close_index = 3  # Close 特徵在 scaled_data 中的索引

    # 構建完整的數據結構，填充到與 scaled_data 一致的形狀
    full_predictions = np.zeros((predicted_prices.shape[0], scaled_data.shape[1]))
    full_predictions[:, close_index] = predicted_prices  # 插入 Close 預測值

    # 使用 scaler 進行反標準化
    predicted_close_prices = scaler.inverse_transform(full_predictions)[:, close_index]
    return predicted_close_prices


# =========  Cross‑section utilities  =========
CROSS_MODELS = [
    ('qlib.contrib.model.pytorch_tabnet', ['TabNet', 'TabnetModel'])
    # ('qlib.contrib.model.pytorch_sfm',    ['SFM', 'SFMModel']),
    # ('qlib.contrib.model.pytorch_add',    ['ADDModel'])
]


def train_cross_loop(model_cls, X, y, epochs, device="cpu"):
    """
    通用橫斷面訓練迴圈，支援 TabNet / SFM / ADDModel
    會根據模型的 __init__ 參數自動決定要給哪些 kwargs
    """
    import inspect
    sig = inspect.signature(model_cls.__init__)
    param_names = sig.parameters

    kw = {}
    # --- 特徵輸入維度 ---
    if 'd_feat'      in param_names: kw['d_feat']      = X.shape[1]
    if 'feature_dim' in param_names: kw['feature_dim'] = X.shape[1]
    if 'input_dim'   in param_names: kw['input_dim']   = X.shape[1]
    if 'field_dim'   in param_names: kw['field_dim']   = X.shape[1]

    # --- 輸出維度 ---
    if 'output_dim'  in param_names: kw['output_dim']  = 1
    if 'target_dim'  in param_names: kw['target_dim']  = 1
    if 'embed_dim'   in param_names: kw['embed_dim']   = 16   # 例如 SFM 用

    model = model_cls(**kw)
    net   = model.model if hasattr(model, 'model') else model

    # 有些模型本身不支援 .to()
    if hasattr(net, 'to'):
        net.to(device)

    ds = DataLoader(TensorDataset(X.to(device), y.to(device)),
                    batch_size=512, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    net.train()
    for _ in range(epochs):
        for xb, yb in ds:
            opt.zero_grad()
            out = net(xb)
            out = out[0] if isinstance(out, tuple) else out
            loss_fn(out.squeeze(), yb).backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        preds = net(X.to(device))
        preds = preds[0] if isinstance(preds, tuple) else preds
        preds = preds.squeeze().cpu().numpy()

    return preds
    
def add_indicators(df):
    """給單支股票 DataFrame 加 MA5 / MA10 / RSI14"""
    df.ta.strategy(ta.Strategy(
        name="ma_rsi",
        ta=[ {"kind":"sma","length":5},
             {"kind":"sma","length":10},
             {"kind":"rsi","length":14} ]))
    df.rename(columns={'SMA_5':'ma5','SMA_10':'ma10','RSI_14':'rsi14'}, inplace=True)
    return df

def download_many(tickers, period):
    """一次抓多股票日線，回傳扁平 DF"""
    data = yf.download(" ".join(tickers), period=period,
                       group_by='ticker', auto_adjust=True, threads=True)
    frames=[]
    if isinstance(data.columns, pd.MultiIndex):
        for tic in tickers:
            sub = data[tic].copy()
            sub.columns = ['Open','High','Low','Close','Adj Close','Volume'][:len(sub.columns)]
            sub = add_indicators(sub); sub['Ticker']=tic; frames.append(sub)
    else:  # 只抓到 1 檔
        data = add_indicators(data); data['Ticker']=tickers[0]; frames.append(data)

    df = pd.concat(frames).reset_index().rename(columns={'index':'Date'})

    # ➜ 保證六大欄都在，若缺就補 0
    base_cols = ['Open','High','Low','Close','Adj Close','Volume']
    for c in base_cols:
        if c not in df.columns:
            df[c] = 0.0    # 填 0（或用 np.nan）

    cols = ['Date','Ticker','Open','High','Low','Close','Volume','ma5','ma10','rsi14']
    df['Date'] = pd.to_datetime(df['Date'])  # 確保 Date 欄位為 datetime 型態
    df = df.reset_index(drop=True)           # 確保沒有 MultiIndex
    return df[cols].dropna().sort_values(['Date','Ticker'])



def build_cross_xy(df):
    """
    產生 tabular 特徵與標籤
    標籤 = 下一日價格差百分比 = (Close(t+1) − Close(t)) / Close(t)
    """
    df = df.copy()
    # 建立百分比標籤
    df['pct_ret1'] = (
        df.groupby('Ticker')['Close'].transform(lambda x: x.shift(-1)) - df['Close']
    ) / df['Close']

    df = df.dropna()                 # 移除最後一天無標籤資料
    feats = ['Open','High','Low','Close','Volume','ma5','ma10','rsi14']

    X = torch.tensor(df[feats].values, dtype=torch.float32)
    y = torch.tensor(df['pct_ret1'].values, dtype=torch.float32)
    meta = df[['Date','Ticker']].reset_index(drop=True)
    return X, y, meta

def import_model(mod_path, cls_list):
    try:
        m = importlib.import_module(mod_path)
        for c in cls_list:
            if hasattr(m, c): return getattr(m, c)
    except ImportError: pass
    return None

# -------------------------------------------
# 正確的 TabNet hand‑loop  (with priors tensor)
# -------------------------------------------
from qlib.contrib.model.pytorch_tabnet import TabNet

def train_tabnet(X, y, epochs=150, device="cpu"):
    inp = X.shape[1]
    net = TabNet(inp_dim=inp, out_dim=1).to(device)

    loader = DataLoader(TensorDataset(X.to(device), y.to(device)),
                        batch_size=512, shuffle=True)
    opt, loss_fn = torch.optim.Adam(net.parameters(), lr=1e-3), nn.MSELoss()

    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            pri = torch.ones(xb.size(0), inp, device=device)  # 全 1 遮罩
            opt.zero_grad()
            raw = net(xb, priors=pri)            # ← 可能 tuple
            out = raw[0] if isinstance(raw, tuple) else raw
            loss_fn(out.squeeze(), yb).backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pri_all = torch.ones(X.size(0), inp, device=device)
        raw_all = net(X.to(device), priors=pri_all)
        preds = (raw_all[0] if isinstance(raw_all, tuple) else raw_all
                ).squeeze().cpu().numpy()
    return preds
# =========  Cross‑section utilities  =========

# 發送電子郵件
def send_email(subject, body, to_emails):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(to_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    server = smtplib.SMTP_SSL(smtp_server, port)
    server.login(sender_email, password)
    server.sendmail(sender_email, to_emails, msg.as_string())
    server.quit()

# 發送 Telegram 消息
def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {"chat_id": telegram_channel_id, "text": message, "parse_mode": "HTML"}
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f"Telegram 發送失敗: {response.text}")


# 發送 Discord 消息
def send_to_discord(message):
    try:
        payload = {
            "content": message
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(discord_webhook_url, json=payload, headers=headers)  # 使用全域變數
        if response.status_code == 204:
            print("訊息已成功傳送到 Discord 頻道。")
        else:
            print(f"傳送訊息到 Discord 時發生錯誤: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"傳送訊息到 Discord 時發生錯誤: {str(e)}")

def send_results(index_name, stock_predictions):
    calculation_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"發送結果: {index_name}")

    # 將結果存入 MongoDB（可選功能）
    save_to_mongodb(index_name, stock_predictions)

    # 組裝 Email 內容
    email_subject = f"每日潛力股分析DAVID888 - {index_name} - 運算時間: {calculation_time}"
    email_body = f"運算日期和時間: {calculation_time}\n\n指數: {index_name}\n"
    for key, predictions in stock_predictions.items():
        email_body += f"\n{key}:\n"
        for stock in predictions:
            email_body += f"股票: {stock[0]}, 潛力: {stock[1]:.2%}, 現價: {stock[2]:.2f}, 預測價: {stock[3]:.2f}\n"
    send_email(email_subject, email_body, to_emails)

    # 組裝 Telegram 內容
    telegram_message = f"<b>每日潛力股分析</b>\n運算日期和時間: <b>{calculation_time}</b>\n\n指數: <b>{index_name}</b>\n"
    for key, predictions in stock_predictions.items():
        telegram_message += f"<b>{key}:</b>\n"
        for stock in predictions:
            telegram_message += f"股票: {stock[0]}, 潛力: {stock[1]:.2%}, 現價: {stock[2]:.2f}, 預測價: {stock[3]:.2f}\n"
    send_to_telegram(telegram_message)

    # 組裝 Discord 內容
    discord_message = f"**每日潛力股分析**\n運算日期和時間: **{calculation_time}**\n\n指數: **{index_name}**\n"
    for key, predictions in stock_predictions.items():
        discord_message += f"**{key}:**\n"
        for stock in predictions:
            discord_message += f"股票: {stock[0]}, 潛力: {stock[1]:.2%}, 現價: {stock[2]:.2f}, 預測價: {stock[3]:.2f}\n"
    print("[DEBUG] discord_message 組裝內容：")
    print(discord_message)
    send_to_discord(discord_message)  # 不再傳入 webhook_url


# -------------------------------------------------------
# 股票分析函數：先執行橫斷面模型，再執行時間序列模型
# -------------------------------------------------------
def get_top_and_bottom_10_potential_stocks(period, selected_indices, mysql_manager=None):
    """
    依所選指數，回傳各模型潛力排行榜（前 / 後 10）
    結構範例：
    {
        "台灣50": {
            "🥇 前十名 LSTM 🧠":    [ (ticker, pot, curr, pred), ... ],
            "📉 後十名 LSTM 🧠":    [ ... ],
            ...
            "🚀 前十名 TabNet":     [ ... ],
            "⛔ 後十名 TabNet":     [ ... ]
        }, ...
    }
    """
    results = {}
    # stock_predictions = {}  # 移除這行，改到每個指數處理時初始化

    # --- 指數 → 股票清單 ---------------------------------
    index_stock_map = {
        "台灣50":      get_tw0050_stocks(),
        "台灣中型100": get_tw0051_stocks(),
        "SP500":       get_sp500_stocks(),
        "NASDAQ":      get_nasdaq_stocks(),
        "費城半導體":   get_sox_stocks(),
        "道瓊":        get_dji_stocks(),
    }

    # --- 全域設定 ---------------------------------------
    use_cross    = os.getenv("USE_CROSS", "false").lower() == "true"
    cross_period = os.getenv("CROSS_PERIOD", "6mo")
    cross_epochs = int(os.getenv("CROSS_EPOCHS", "150"))
    device       = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 時間序列模型設定 ---------------------------------
    use_transformer = os.getenv("USE_TRANSFORMER", "false").lower() == "true"
    use_prophet = os.getenv("USE_PROPHET", "false").lower() == "true"
    use_chronos = os.getenv("USE_CHRONOS", "true").lower() == "true"

    for index_name, stock_list in index_stock_map.items():
        stock_predictions = {}  # 每個指數都重新初始化
        if index_name not in selected_indices:
            continue
        print(f"\n=== 處理指數: {index_name} ===")

        # -------- 序列模型容器 --------
        lstm_preds, prophet_preds = [], []
        transformer_preds, chronos_preds = [], []

        # ======== 先跑橫斷面模型（Cross Sectional Models） ========
        if use_cross:
            try:
                raw_df = download_many(stock_list, cross_period)
                Xc, yc, meta_c = build_cross_xy(raw_df)

                # 修正 mask_last = Series vs Series 錯誤
                max_date = pd.Timestamp(meta_c['Date'].max())
                print(f"[LOG] max_date: {max_date}, type: {type(max_date)}")
                print(f"[LOG] meta_c['Date'] head: {meta_c['Date'].head()}, dtype: {meta_c['Date'].dtype}")
                mask_last = meta_c['Date'].values == max_date
                print(f"[LOG] mask_last: {mask_last}, shape: {mask_last.shape}, type: {type(mask_last)}")
                meta_last = meta_c[mask_last].reset_index(drop=True)
                print(f"[LOG] meta_last shape: {meta_last.shape}, columns: {meta_last.columns}")

                latest_close = (
                    raw_df[raw_df['Date'] == max_date]
                    .groupby('Ticker')['Close']
                    .first()
                )
                print(f"[LOG] latest_close index: {latest_close.index}, type: {type(latest_close)}")

                # 執行 Cross 模型（TabNet，SFM，ADDModel）
                for m_path, cls_list in CROSS_MODELS:
                    ModelClass = import_model(m_path, cls_list)
                    if ModelClass is None:
                        continue
                    print(f"🔍 Cross 訓練 {ModelClass.__name__} …")
                    try:
                        if ModelClass.__name__ == "TabNet":
                            preds_all = train_tabnet(Xc, yc, epochs=cross_epochs, device=device)
                        else:
                            preds_all = train_cross_loop(ModelClass, Xc, yc, cross_epochs, device)
                        print(f"[LOG] preds_all shape: {getattr(preds_all, 'shape', None)}, type: {type(preds_all)}")
                        preds_last = preds_all[mask_last]
                        print(f"[LOG] preds_last shape: {getattr(preds_last, 'shape', None)}, type: {type(preds_last)}")

                        # 組 TabNet / SFM / ADDModel 結果
                        records = [
                            (
                                tic,
                                p,                               # 預測潛力
                                float(latest_close[tic]),        # 現價
                                float(latest_close[tic] * (1+p)) # 預測價
                            )
                            for tic, p in zip(meta_last['Ticker'], preds_last)
                        ]
                        print(f"[LOG] records sample: {records[:3]}")

                        # 寫 MySQL
                        if mysql_manager and mysql_manager.enabled:
                            mysql_manager.save_predictions(index_name, records, ModelClass.__name__, cross_period)

                        # 排行榜
                        stock_predictions.update({
                            f"🚀 前十名 {ModelClass.__name__}": sorted(records, key=lambda x:x[1], reverse=True)[:10],
                            f"⛔ 後十名 {ModelClass.__name__}": sorted(records, key=lambda x:x[1])[:10],
                        })
                        print(f"[DEBUG] stock_predictions keys after update: {list(stock_predictions.keys())}")
                        print(f"[DEBUG] stock_predictions lens after update: {[len(v) for v in stock_predictions.values()]}")

                    except Exception as e:
                        print(f"{ModelClass.__name__} 失敗: {e}")
                        continue

            except Exception as e:
                print(f"Cross‑section 流程錯誤: {e}")

        # ======== 跑時間序列模型 ========
        for tic in stock_list:
            data = get_stock_data(tic, period)
            if len(data) < 60:
                continue

            # ----- LSTM -----
            try:
                X, y, scaler = prepare_data(data)
                lstm_model = train_lstm_model(X, y)
                lstm_series = predict_stock(lstm_model, data, scaler)
                print(f"[DEBUG][LSTM] lstm_series type: {type(lstm_series)}, shape: {getattr(lstm_series, 'shape', None)}")
                print(f"[DEBUG][LSTM] data['Close'] type: {type(data['Close'])}, shape: {getattr(data['Close'], 'shape', None)}")
                cur  = float(data['Close'].iloc[-1, 0]) if isinstance(data['Close'], pd.DataFrame) else float(data['Close'].iloc[-1])
                pred = float(lstm_series.max())
                pot  = (pred - cur) / cur
                print(f"[DEBUG][LSTM] cur: {cur}, pred: {pred}, pot: {pot}")
                lstm_preds.append((tic, pot, cur, pred))
            except Exception as e:
                print(f"LSTM 失敗 {tic}: {e}")

            # ----- Transformer -----
            if use_transformer:
                try:
                    tf_data = get_stock_data(tic, transformer_period)
                    X_tf, y_tf, tf_scaler = prepare_data(tf_data)
                    tf_shape = (X_tf.shape[1], X_tf.shape[2])
                    tf_model = train_transformer_model(X_tf, y_tf, tf_shape)
                    tf_series = predict_transformer(tf_model, tf_data, tf_scaler)
                    cur  = tf_data['Close'].iloc[-1]
                    pred = float(tf_series.max())
                    pot  = (pred - cur) / cur
                    transformer_preds.append((tic, pot, cur, pred))
                except Exception as e:
                    print(f"Transformer 失敗 {tic}: {e}")

            # ----- Prophet -----
            if use_prophet:
                try:
                    p_model = train_prophet_model(data)
                    p_fore  = predict_with_prophet(p_model, data)
                    cur  = data['Close'].iloc[-1]
                    pred = float(p_fore['yhat'].max())
                    pot  = (pred - cur) / cur
                    prophet_preds.append((tic, pot, cur, pred))
                except Exception as e:
                    print(f"Prophet 失敗 {tic}: {e}")

            # ----- Chronos‑Bolt -----
            if use_chronos:
                try:
                    ch_data = get_stock_data(tic, chronos_period)
                    if len(ch_data) < 60:
                        continue
                    ts_df = prepare_chronos_data(ch_data)
                    predictor = TimeSeriesPredictor(
                        prediction_length=10, freq="D", target="target")
                    predictor.fit(ts_df, hyperparameters={
                        "Chronos": {"model_path": "autogluon/chronos-bolt-base"}
                    })
                    preds = predictor.predict(ts_df)
                    cur  = float(ch_data['Close'].iloc[-1])
                    pred = float(preds.values.max())
                    pot  = (pred - cur) / cur
                    chronos_preds.append((tic, pot, cur, pred))
                except Exception as e:
                    print(f"Chronos 失敗 {tic}: {e}")

        # --- MySQL：時間序列模型 ---------------------------
        if mysql_manager and mysql_manager.enabled:
            if lstm_preds:
                mysql_manager.save_predictions(index_name, lstm_preds, "LSTM", period)
            if use_prophet and prophet_preds:
                mysql_manager.save_predictions(index_name, prophet_preds, "Prophet", period)
            if use_transformer and transformer_preds:
                mysql_manager.save_predictions(index_name, transformer_preds, "Transformer", period)
            if use_chronos and chronos_preds:
                mysql_manager.save_predictions(index_name, chronos_preds, "Chronos-Bolt", chronos_period)

        # --- 組排行榜（時間序列） -------------------------
        stock_predictions = stock_predictions if 'stock_predictions' in locals() else {}

        stock_predictions.update({
            "🥇 前十名 LSTM 🧠": sorted(lstm_preds, key=lambda x: x[1], reverse=True)[:10],
            "📉 後十名 LSTM 🧠": sorted(lstm_preds, key=lambda x: x[1])[:10],
        })
        if use_prophet and prophet_preds:
            stock_predictions.update({
                "🚀 前十名 Prophet 🔮": sorted(prophet_preds, key=lambda x: x[1], reverse=True)[:10],
                "⛔ 後十名 Prophet 🔮": sorted(prophet_preds, key=lambda x: x[1])[:10],
            })
        if use_transformer and transformer_preds:
            stock_predictions.update({
                "🚀 前十名 Transformer 🔄": sorted(transformer_preds, key=lambda x: x[1], reverse=True)[:10],
                "⛔ 後十名 Transformer 🔄": sorted(transformer_preds, key=lambda x: x[1])[:10],
            })
        if use_chronos and chronos_preds:
            stock_predictions.update({
                "🚀 前十名 Chronos-Bolt ⚡": sorted(chronos_preds, key=lambda x: x[1], reverse=True)[:10],
                "⛔ 後十名 Chronos-Bolt ⚡": sorted(chronos_preds, key=lambda x: x[1])[:10],
            })

        # -------- 收尾 --------
        if stock_predictions:
            results[index_name] = stock_predictions

    return results


# 主函數
def main():
    try:
        # 初始化 MySQL 管理器
        mysql_manager = MySQLManager() if use_mysql else None

        calculation_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        period = "6mo"
        #selected_indices = ["台灣50", "台灣中型100", "SP500"]
        selected_indices = ["SP500"]

        print("計算潛力股...")
        analysis_results = get_top_and_bottom_10_potential_stocks(period, selected_indices, mysql_manager)

        # 分開處理每個指數的結果
        for index_name, stock_predictions in analysis_results.items():
            print(f"處理並發送結果: {index_name}")
            send_results(index_name, stock_predictions)

    except Exception as e:
        print(f"錯誤: {str(e)}")
        send_to_telegram(f"⚠️ 錯誤: {str(e)}")

    finally:
        # 關閉 MySQL 連接
        if mysql_manager:
            mysql_manager.close()        
        
if __name__ == "__main__":
    main()