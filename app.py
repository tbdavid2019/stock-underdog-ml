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

# 加載 .env 文件
load_dotenv()

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
transformer_period = os.getenv("TRANSFORMER_PERIOD", "5y")  # 默認 3 年數據



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
    return [
        "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW", "2382.TW", "2303.TW", "2882.TW", "2891.TW", "3711.TW",
        "2412.TW", "2886.TW", "2884.TW", "1216.TW", "2357.TW", "2885.TW", "2892.TW", "3034.TW", "2890.TW", "2327.TW",
        "5880.TW", "2345.TW", "3231.TW", "2002.TW", "2880.TW", "3008.TW", "2883.TW", "1303.TW", "4938.TW", "2207.TW",
        "2887.TW", "2379.TW", "1101.TW", "2603.TW", "2301.TW", "1301.TW", "5871.TW", "3037.TW", "3045.TW", "2912.TW",
        "3017.TW", "6446.TW", "4904.TW", "3661.TW", "6669.TW", "1326.TW", "5876.TW", "2395.TW", "1590.TW", "6505.TW"
    ]

def get_tw0051_stocks():
    return [
        "2371.TW", "3533.TW", "2618.TW", "3443.TW", "2347.TW", "3044.TW", "2834.TW", "2385.TW", "1605.TW", "2105.TW",
        "6239.TW", "6176.TW", "9904.TW", "1519.TW", "9910.TW", "1513.TW", "1229.TW", "9945.TW", "2313.TW", "1477.TW",
        "3665.TW", "2354.TW", "4958.TW", "8464.TW", "9921.TW", "2812.TW", "2059.TW", "1504.TW", "2542.TW", "6770.TW",
        "5269.TW", "2344.TW", "3023.TW", "1503.TW", "2049.TW", "2610.TW", "2633.TW", "3036.TW", "2368.TW", "3035.TW",
        "2027.TW", "9914.TW", "2408.TW", "2809.TW", "1319.TW", "2352.TW", "2337.TW", "2006.TW", "2206.TW", "4763.TW",
        "3005.TW", "1907.TW", "2915.TW", "1722.TW", "6285.TW", "6472.TW", "6531.TW", "3406.TW", "9958.TW", "9941.TW",
        "1795.TW", "2201.TW", "9917.TW", "2492.TW", "6890.TW", "2845.TW", "8454.TW", "8046.TW", "6789.TW", "2388.TW",
        "6526.TW", "1802.TW", "5522.TW", "6592.TW", "2204.TW", "2540.TW", "2539.TW", "3532.TW"
    ]

# Function to fetch S&P 500 component stocks
def get_sp500_stocks():
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "BRK.B", "AVGO", "GOOG",
        "UNH", "JNJ", "V", "WMT", "PG", "JPM", "MA", "LLY", "XOM", "BAC",
        "MRK", "PFE", "ABBV", "KO", "PEP", "TMO", "COST", "CSCO", "MCD", "DHR",
        "NKE", "DIS", "VZ", "ADBE", "CMCSA", "NFLX", "INTC", "WFC", "TXN", "LIN",
        "HON", "UNP", "ACN", "QCOM", "NEE", "ABT", "PM", "MDT", "BMY", "SPGI",
        "LOW", "MS", "RTX", "IBM", "CVX", "ORCL", "INTU", "AMD", "GS",
        "BLK", "ISRG", "GE", "AMT", "CAT", "DE", "LMT", "PLD",
        "SYK","MDLZ","AXP","T","EL","GILD","NOW","ADI","ZTS","PYPL",
        "MO","BKNG","SCHW","MMC","ADP","C","TJX","DUK","SO","BDX",
        "APD","PNC","USB","CI","EQIX","TGT","CB","ICE","HUM","ITW",
        "ETN","WM","ECL","FIS","NSC","REGN","FDX","D","NOC","GM",
        "SHW","PSA","GD","HCA","EMR","MCO","KLAC","EW","AON",
        "TRV","SPG","MU","FISV","BSX","AEP",
        "MRNA","LRCX","KMB","SLB"
    ]

# Function to fetch NASDAQ component stocks
def get_nasdaq_stocks():
    return [
        "AAPL", "NVDA", "MSFT", "AMZN", "GOOG", "META", "TSLA", "AVGO", "COST", 
        "NFLX", "TMUS", "ASML", "CSCO", "ADBE", "AMD", "PEP", "LIN", "AZN", "ISRG", 
        "INTU", "QCOM", "TXN", "BKNG", "CMCSA", "AMGN", "HON", "ARM", "AMAT", "PDD", 
        "PANW", "ADP", "VRTX", "GILD", "SBUX", "MU", "ADI", "MELI", "MRVL", "LRCX", 
        "CTAS", "CRWD", "INTC", "PYPL", "KLAC", "ABNB", "MDLZ", "CDNS", "REGN", "MAR", 
        "CEG", "SNPS", "FTNT", "DASH", "TEAM", "ORLY", "WDAY", "TTD", "CSX", "ADSK", 
        "CHTR", "PCAR", "ROP", "CPRT", "DDOG", "NXPI", "ROST", "AEP", "MNST", "PAYX", 
        "FANG", "FAST", "KDP", "EA", "ODFL", "LULU", "BKR", "VRSK", "XEL", "CTSH", 
        "EXC", "KHC", "GEHC", "CCEP", "IDXX", "TTWO", "CSGP", "ZS", "MCHP", "DXCM", 
        "ANSS", "ON", "WBD", "MDB", "GFS", "CDW", "BIIB", "ILMN", "MRNA", "DLTR", 
        "WBA"
    ]
# Function to fetch Philadelphia Semiconductor Index component stocks

def get_sox_stocks():
    return [
        "NVDA", "AVGO", "GFS", "CRUS", "ON", "ASML", "QCOM", "SWKS", "MPWR", "ADI",
        "TSM", "AMD", "TXN", "QRVO", "AMKR", "MU", "ARM", "NXPI", "TER", "ENTG",
        "LSCC", "COHR", "ONTO", "MTSI", "KLAC", "LRCX", "MRVL", "AMAT", "INTC", "MCHP"
    ]

# Function to fetch Dow Jones Industrial Average component stocks
def get_dji_stocks():
    return [
        "AAPL", "MSFT", "JPM", "V", "UNH", "PG", "JNJ", "WMT", "DIS", "VZ",
        "INTC", "KO", "MRK", "GS", "TRV", "IBM", "MMM", "CAT", "RTX", "CVX",
        "MCD", "HON", "AXP", "WBA", "NKE", "DOW", "BA", "HD", "CRM", "AMGN"
    ]

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

# 準備數據
# def prepare_data(data, time_step=60):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
#     X, y = [], []
#     for i in range(time_step, len(scaled_data)):
#         X.append(scaled_data[i-time_step:i, 0])
#         y.append(scaled_data[i, 0])
#     X, y = np.array(X), np.array(y)
#     X = np.reshape(X, (X.shape[0], X.shape[1], 1))
#     return X, y, scaler

def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i])  # 確保 X 是 (time_step, features)
        y.append(scaled_data[i, 3])  # 預測 Close 價        
    X, y = np.array(X), np.array(y).reshape(-1, 1)  # y 的形狀應為 (samples, 1)
    return X, y, scaler

# 訓練 LSTM 模型
# def train_lstm_model(X_train, y_train):
#     model = Sequential([
#         Input(shape=(X_train.shape[1], 1)),
#         LSTM(units=50, return_sequences=True),
#         Dropout(0.2),
#         LSTM(units=50, return_sequences=False),
#         Dropout(0.2),
#         Dense(units=1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X_train, y_train, epochs=10, batch_size=32)
#     return model
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


# LSTM 預測股票
# def predict_stock(model, data, scaler, time_step=60):
#     inputs = scaler.transform(data['Close'].values.reshape(-1, 1))
#     X_test = [inputs[i-time_step:i, 0] for i in range(time_step, len(inputs))]
#     X_test = np.array(X_test)
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#     predicted_prices = model.predict(X_test)
#     return scaler.inverse_transform(predicted_prices)

def predict_stock(model, data, scaler, time_step=60):
    # 使用多個特徵進行標準化
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

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
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

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


# 送到 discord上 
def send_to_discord(webhook_url, message):
    try:
        payload = {
            "content": message
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(webhook_url, json=payload, headers=headers)
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
    webhook_url = "https://discord.com/api/webhooks/1317462344866992159/f6_dgykIsWRxl4ttibgT62fVWPkly1cx0DseVLFYdNy4Cy2CxVNFdZSZmIpSLu5tXF4G"
    send_to_discord(webhook_url, discord_message)



# # 股票分析函數 (新增 Prophet 與排序功能)

# for index_name, stock_list in index_stock_map.items():
#     if index_name not in selected_indices:
#         continue

#     print(f"處理指數: {index_name}")
#     lstm_predictions = []
#     prophet_predictions = []
#     transformer_predictions = []  # 新增 Transformer 預測結果列表

#     for ticker in stock_list:
#         data = get_stock_data(ticker, period if not use_transformer else transformer_period)  # 根據開關選擇不同的數據期間
#         if len(data) < 60:
#             continue

#         # LSTM 預測
#         if not use_transformer:
#             X_train, y_train, lstm_scaler = prepare_data(data)
#             lstm_model = train_lstm_model(X_train, y_train)
#             lstm_predicted_prices = predict_stock(lstm_model, data, lstm_scaler)
#             lstm_current_price = data['Close'].values[-1].item()
#             lstm_predicted_price = float(lstm_predicted_prices[-1])
#             lstm_potential = (lstm_predicted_price - lstm_current_price) / lstm_current_price
#             lstm_predictions.append((ticker, lstm_potential, lstm_current_price, lstm_predicted_price))

#         # Prophet 預測
#         try:
#             prophet_model = train_prophet_model(data)
#             forecast = predict_with_prophet(prophet_model, data)
#             prophet_current_price = data['Close'].values[-1].item()
#             prophet_predicted_price = float(forecast['yhat'].iloc[-1])
#             prophet_potential = (prophet_predicted_price - prophet_current_price) / prophet_current_price
#             prophet_predictions.append((ticker, prophet_potential, prophet_current_price, prophet_predicted_price))
#         except Exception as e:
#             print(f"Prophet 預測失敗: {ticker}, 錯誤: {str(e)}")

#         # Transformer 預測
#         if use_transformer:
#             X_train, y_train, transformer_scaler = prepare_data(data)
#             input_shape = (X_train.shape[1], X_train.shape[2])
#             transformer_model = train_transformer_model(X_train, y_train, input_shape)
#             transformer_predicted_prices = predict_transformer(transformer_model, data, transformer_scaler)
#             transformer_current_price = data['Close'].values[-1].item()
#             transformer_predicted_price = float(transformer_predicted_prices[-1])
#             transformer_potential = (transformer_predicted_price - transformer_current_price) / transformer_current_price
#             transformer_predictions.append((ticker, transformer_potential, transformer_current_price, transformer_predicted_price))

#     # 排序結果
#     stock_predictions = {
#         "🥇 前十名 LSTM 🧠": sorted(lstm_predictions, key=lambda x: x[1], reverse=True)[:10] if not use_transformer else [],
#         "📉 後十名 LSTM 🧠": sorted(lstm_predictions, key=lambda x: x[1])[:10] if not use_transformer else [],
#         "🚀 前十名 Prophet 🔮": sorted(prophet_predictions, key=lambda x: x[1], reverse=True)[:10],
#         "⛔ 後十名 Prophet 🔮": sorted(prophet_predictions, key=lambda x: x[1])[:10],
#         "🚀 前十名 Transformer 🔄": sorted(transformer_predictions, key=lambda x: x[1], reverse=True)[:10] if use_transformer else [],
#         "⛔ 後十名 Transformer 🔄": sorted(transformer_predictions, key=lambda x: x[1])[:10] if use_transformer else [],
#     }

#     # 組裝並發送結果
#     send_results(index_name, stock_predictions)
    

def get_top_and_bottom_10_potential_stocks(period, selected_indices):
    index_stock_map = {
        "台灣50": get_tw0050_stocks(),
        "台灣中型100": get_tw0051_stocks(),
        "SP500": get_sp500_stocks(),
        "NASDAQ": get_nasdaq_stocks(),
        "費城半導體": get_sox_stocks(),
        "道瓊": get_dji_stocks()
    }

    for index_name, stock_list in index_stock_map.items():
        if index_name not in selected_indices:
            continue

        print(f"處理指數: {index_name}")
        lstm_predictions = []
        prophet_predictions = []
        transformer_predictions = []  # 確保變量初始化

        for ticker in stock_list:
            # LSTM 使用 `period` 參數 (3 個月)
            lstm_data = get_stock_data(ticker, period)
            if len(lstm_data) >= 60:  # 確保數據足夠
                try:
                    # LSTM 預測邏輯
                    X_train, y_train, lstm_scaler = prepare_data(lstm_data)
                    print(f"LSTM 訓練數據形狀: X_train: {X_train.shape}, y_train: {y_train.shape}")
                    lstm_model = train_lstm_model(X_train, y_train)
                    lstm_predicted_prices = predict_stock(lstm_model, lstm_data, lstm_scaler)
                    lstm_current_price = lstm_data['Close'].values[-1].item()
                    lstm_predicted_price = float(lstm_predicted_prices[-1])
                    lstm_potential = (lstm_predicted_price - lstm_current_price) / lstm_current_price
                    lstm_predictions.append((ticker, lstm_potential, lstm_current_price, lstm_predicted_price))
                    print(f"LSTM 預測完成: {ticker}")
                except Exception as e:
                    print(f"LSTM 預測失敗: {ticker}, 錯誤: {str(e)}")

            # Transformer 使用 `transformer_period` (默認 5 年)
            transformer_data = get_stock_data(ticker, period=transformer_period)
            if len(transformer_data) >= 60:  # 確保數據足夠
                try:
                    # Transformer 預測邏輯
                    X_train, y_train, transformer_scaler = prepare_data(transformer_data)
                    input_shape = (X_train.shape[1], X_train.shape[2])
                    print(f"Transformer 訓練數據形狀: X_train: {X_train.shape}, y_train: {y_train.shape}")
                    transformer_model = train_transformer_model(X_train, y_train, input_shape)
                    transformer_predicted_prices = predict_transformer(transformer_model, transformer_data, transformer_scaler)
                    transformer_current_price = transformer_data['Close'].values[-1].item()
                    transformer_predicted_price = float(transformer_predicted_prices[-1])
                    transformer_potential = (transformer_predicted_price - transformer_current_price) / transformer_current_price
                    transformer_predictions.append((ticker, transformer_potential, transformer_current_price, transformer_predicted_price))
                    print(f"Transformer 預測完成: {ticker}")
                except Exception as e:
                    print(f"Transformer 預測失敗: {ticker}, 錯誤: {str(e)}")

            # Prophet 預測
            try:
                prophet_model = train_prophet_model(lstm_data)  # Prophet 使用 LSTM 的數據（3 個月）
                forecast = predict_with_prophet(prophet_model, lstm_data)
                prophet_current_price = lstm_data['Close'].values[-1].item()

                # 安全提取 Prophet 的預測價格
                prophet_predicted_price = forecast['yhat'].iloc[-1]
                if isinstance(prophet_predicted_price, (pd.Series, np.ndarray)):
                    prophet_predicted_price = float(prophet_predicted_price.item())
                else:
                    prophet_predicted_price = float(prophet_predicted_price)

#                prophet_predicted_price = forecast['yhat'].iloc[-1].item()
                prophet_potential = (prophet_predicted_price - prophet_current_price) / prophet_current_price
                prophet_predictions.append((ticker, prophet_potential, prophet_current_price, prophet_predicted_price))
            except Exception as e:
                print(f"Prophet 預測失敗: {ticker}, 錯誤: {str(e)}")

        # 排序結果
        stock_predictions = {
            "🥇 前十名 LSTM 🧠": sorted(lstm_predictions, key=lambda x: x[1], reverse=True)[:10],
            "📉 後十名 LSTM 🧠": sorted(lstm_predictions, key=lambda x: x[1])[:10],
            "🚀 前十名 Prophet 🔮": sorted(prophet_predictions, key=lambda x: x[1], reverse=True)[:10],
            "⛔ 後十名 Prophet 🔮": sorted(prophet_predictions, key=lambda x: x[1])[:10],
            "🚀 前十名 Transformer 🔄": sorted(transformer_predictions, key=lambda x: x[1], reverse=True)[:10],
            "⛔ 後十名 Transformer 🔄": sorted(transformer_predictions, key=lambda x: x[1])[:10],
        }

        # 組裝並發送結果
        send_results(index_name, stock_predictions)

    # for index_name, stock_list in index_stock_map.items():
    #     if index_name not in selected_indices:
    #         continue

    #     print(f"處理指數: {index_name}")
    #     lstm_predictions = []
    #     prophet_predictions = []

    #     for ticker in stock_list:
    #         data = get_stock_data(ticker, period)
    #         if len(data) < 60:
    #             continue

    #         # LSTM 預測
    #         # X_train, y_train, lstm_scaler = prepare_data(data)
    #         # lstm_model = train_lstm_model(X_train, y_train)
    #         # lstm_predicted_prices = predict_stock(lstm_model, data, lstm_scaler)
    #         # lstm_current_price = data['Close'].values[-1].item()
    #         # lstm_predicted_price = float(lstm_predicted_prices[-1][0])
    #         # lstm_potential = (lstm_predicted_price - lstm_current_price) / lstm_current_price
    #         # lstm_predictions.append((ticker, lstm_potential, lstm_current_price, lstm_predicted_price))
    #         # LSTM 預測
    #         X_train, y_train, lstm_scaler = prepare_data(data)
    #         lstm_model = train_lstm_model(X_train, y_train)
    #         lstm_predicted_prices = predict_stock(lstm_model, data, lstm_scaler)
    #         lstm_current_price = data['Close'].values[-1].item()
    #         lstm_predicted_price = float(lstm_predicted_prices[-1])  # 不需要再取 [0]
    #         lstm_potential = (lstm_predicted_price - lstm_current_price) / lstm_current_price
    #         lstm_predictions.append((ticker, lstm_potential, lstm_current_price, lstm_predicted_price))

    #         # Prophet 預測
    #         try:
    #             prophet_model = train_prophet_model(data)
    #             forecast = predict_with_prophet(prophet_model, data)
    #             prophet_current_price = data['Close'].values[-1].item()
    #             prophet_predicted_price = float(forecast['yhat'].iloc[-1])
    #             prophet_potential = (prophet_predicted_price - prophet_current_price) / prophet_current_price
    #             prophet_predictions.append((ticker, prophet_potential, prophet_current_price, prophet_predicted_price))
    #         except Exception as e:
    #             print(f"Prophet 預測失敗: {ticker}, 錯誤: {str(e)}")

    #     # 排序結果
    #     stock_predictions = {
    #         "🥇 前十名 LSTM 🧠": sorted(lstm_predictions, key=lambda x: x[1], reverse=True)[:10],
    #         "📉 後十名 LSTM 🧠": sorted(lstm_predictions, key=lambda x: x[1])[:10],
    #         "🚀 前十名 Prophet 🔮": sorted(prophet_predictions, key=lambda x: x[1], reverse=True)[:10],
    #         "⛔ 後十名 Prophet 🔮": sorted(prophet_predictions, key=lambda x: x[1])[:10],
    #     }

    #     # 組裝並發送結果
    #     send_results(index_name, stock_predictions)

# 主函數
def main():
    try:
        calculation_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        period = "3mo"
        selected_indices = ["台灣50", "台灣中型100", "SP500", "NASDAQ", "費城半導體", "道瓊"]
#        selected_indices = ["SP500", "NASDAQ", "費城半導體", "道瓊"]
        print("計算潛力股...")
        analysis_results = get_top_and_bottom_10_potential_stocks(period, selected_indices)

        # 準備 Email
        print("準備 Email...")
        email_body = f"運算日期和時間: {calculation_time}\n\n"
        for index_name, stocks in analysis_results.items():
            email_body += f"\n指數: {index_name}\n"
            for key, predictions in stocks.items():
                email_body += f"\n{key}:\n"
                for stock in predictions:
                    email_body += f"股票: {stock[0]}, 潛力: {stock[1]:.2%}, 現價: {stock[2]:.2f}, 預測價: {stock[3]:.2f}\n"

        email_subject = f"每日潛力股分析DAVID888 - 運算時間: {calculation_time}"
        send_email(email_subject, email_body, to_emails)

        # 準備 Telegram
        print("準備 Telegram...")
        telegram_message = f"<b>每日潛力股分析</b>\n運算日期和時間: <b>{calculation_time}</b>\n\n"
        for index_name, stocks in analysis_results.items():
            telegram_message += f"<b>指數: {index_name}</b>\n"
            for key, predictions in stocks.items():
                telegram_message += f"<b>{key}:</b>\n"
                for stock in predictions:
                    telegram_message += f"股票: {stock[0]}, 潛力: {stock[1]:.2%}, 現價: {stock[2]:.2f}, 預測價: {stock[3]:.2f}\n"
        send_to_telegram(telegram_message)

        # 準備 Discord
        print("準備 Discord...")
        discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        discord_message = f"**每日潛力股分析**\n運算日期和時間: **{calculation_time}**\n\n"
        for index_name, stocks in analysis_results.items():
            discord_message += f"**指數: {index_name}**\n"
            for key, predictions in stocks.items():
                discord_message += f"**{key}:**\n"
                for stock in predictions:
                    discord_message += f"股票: {stock[0]}, 潛力: {stock[1]:.2%}, 現價: {stock[2]:.2f}, 預測價: {stock[3]:.2f}\n"
        send_to_discord(discord_webhook_url, discord_message)

    except Exception as e:
        print(f"錯誤: {str(e)}")
        send_to_telegram(f"⚠️ 錯誤: {str(e)}")

if __name__ == "__main__":
    main()