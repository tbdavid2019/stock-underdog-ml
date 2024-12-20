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

***REMOVED*** 設置
smtp_server = "smtp.gmail.com"
port = 465
sender_email = "asw.kong@webglsoft.com"
password = "goxcbqchwhtfvlni"
to_emails = ["david@david888.com", "jeinggoway.cats@blogger.com"]

# Telegram 設置
telegram_bot_token = "5333558888:AAEqxkrcgCE-mR8wN6DrJDz-432CrTMO_fo"
telegram_channel_id = "-1001839177148"


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
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "BRK.B", "NVDA", "UNH", 
        "JNJ", "V", "WMT", "JPM", "MA", "PG", "DIS", "HD", "PYPL", "ADBE",
        "NFLX", "CMCSA", "PEP", "KO", "MRK", "INTC", "T", "CRM", "CSCO", "PFE", 
        "XOM", "COST", "NKE", "CVX", "WFC", "MCD", "AMGN", "MDT", "IBM", "DHR",
        "LLY", "HON", "BA", "MMM", "NEE", "ACN", "UPS", "TMO", "AVGO", "PM",
        "CSX", "BKNG", "LIN", "ORCL", "QCOM", "TXN", "RTX", "LOW", "MDLZ", "TMUS",
        "ISRG", "GE", "AXP", "CAT", "INTU", "ADP", "SPGI", "AMT", "CL", "REGN",
        "GS", "VRTX", "LMT", "NOW", "PLD", "CCI", "AON", "KDP", "CHTR", "MSCI",
        "SNPS", "ICE", "APD", "FISV", "ELV", "HCA", "CTAS", "EQIX", "WELL", "TGT",
        "BK", "STZ", "GILD", "SCHW", "COP", "SBUX", "ADSK", "ATVI", "ZTS", "MRNA",
        "BLK", "PGR", "ORLY", "LRCX", "ECL", "ADI", "IDXX", "ITW", "NOC", "ETN",
        "F", "D", "HLT", "PXD", "CARR", "ALB", "WMB", "PSA", "MPC", "TRV",
        "KEYS", "ODFL", "FTV", "CLX", "LYB", "HIG", "KMX", "PRU", "AVB", "ARE",
        "KIM", "EXR", "LHX", "AMP", "ROK", "VRTX", "CHRW", "SBAC", "WST", "TT",
        "HSIC", "FE", "ULTA", "DOV", "CDNS", "FAST", "STLD", "EFX", "CE", "GL",
        "TSCO", "MTD", "CBOE", "JBHT", "EIX", "XYL", "VLO", "POOL", "TDY", "BR",
        "RSG", "PH", "AEE", "CMS", "XYL", "VTR", "EPAM", "ALGN", "CPT", "HSY",
        "STE", "KMI", "AAP", "ES", "TTWO", "LVS", "WRB", "WY", "WDC", "ROL",
        "CINF", "MKC", "EMR", "ED", "DPZ", "MSI", "HBAN", "WBA", "MOS", "NEM",
        "IEX", "SEE", "GPN", "DLR", "NTAP", "TRMB", "ETR", "NDAQ", "RCL", "AEP",
        "HOLX", "LW", "GWW", "NVR", "RF", "PFG", "HPE", "BBY", "MHK", "HPQ",
        "OKE", "APA", "ALLE", "L", "BF.B", "TSN", "HST", "TPR", "TPG", "KHC",
        "DHI", "IRM", "FMC", "CXO", "AAP", "MKTX", "HII", "MTCH", "CRL", "CPB",
        "MRO", "RJF", "JNPR", "NTRS", "LNT", "TXT", "FFIV", "PBCT", "WU", "HAS",
        "NWSA", "NWS", "FOXA", "FOX", "K", "NUE", "DISCK", "DISCA", "DISCB", "SYY",
        "STT", "AKAM", "FRT", "PKI", "DTE", "PPL", "EVRG", "DVA", "BAX", "PNC",
        "BEN", "OMC", "VAR", "AOS", "ZBRA", "ATO", "SJM", "PBCT", "JKHY", "RMD",
        "CEG", "GPC", "BLL", "CMG", "CF", "MAS", "GME", "HUM", "HRL", "DG",
        "CPRI", "HRB", "EL", "KSS", "SYK", "FL", "CNC", "M", "MGA", "APA"
    ]

# Function to fetch NASDAQ component stocks
def get_nasdaq_stocks():
    return [
        "AAPL", "NVDA", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "TSLA", "AVGO", "COST", 
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
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# 訓練 LSTM 模型
def train_lstm_model(X_train, y_train):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

# LSTM 預測股票
def predict_stock(model, data, scaler, time_step=60):
    inputs = scaler.transform(data['Close'].values.reshape(-1, 1))
    X_test = [inputs[i-time_step:i, 0] for i in range(time_step, len(inputs))]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_prices = model.predict(X_test)
    return scaler.inverse_transform(predicted_prices)

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


# 股票分析函數 (新增 Prophet 與排序功能)

def get_top_and_bottom_10_potential_stocks(period, selected_indices):
    index_stock_map = {
        "台灣50": get_tw0050_stocks(),
        "台灣中型100": get_tw0051_stocks(),
        "S&P": get_sp500_stocks(),
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

        for ticker in stock_list:
            data = get_stock_data(ticker, period)
            if len(data) < 60:
                continue

            # LSTM 預測
            X_train, y_train, lstm_scaler = prepare_data(data)
            lstm_model = train_lstm_model(X_train, y_train)
            lstm_predicted_prices = predict_stock(lstm_model, data, lstm_scaler)
            lstm_current_price = data['Close'].values[-1].item()
            lstm_predicted_price = float(lstm_predicted_prices[-1][0])
            lstm_potential = (lstm_predicted_price - lstm_current_price) / lstm_current_price
            lstm_predictions.append((ticker, lstm_potential, lstm_current_price, lstm_predicted_price))

            # Prophet 預測
            try:
                prophet_model = train_prophet_model(data)
                forecast = predict_with_prophet(prophet_model, data)
                prophet_current_price = data['Close'].values[-1].item()
                prophet_predicted_price = float(forecast['yhat'].iloc[-1])
                prophet_potential = (prophet_predicted_price - prophet_current_price) / prophet_current_price
                prophet_predictions.append((ticker, prophet_potential, prophet_current_price, prophet_predicted_price))
            except Exception as e:
                print(f"Prophet 預測失敗: {ticker}, 錯誤: {str(e)}")

        # 排序結果
        stock_predictions = {
            "🥇 前十名 LSTM 🧠": sorted(lstm_predictions, key=lambda x: x[1], reverse=True)[:10],
            "🚀 前十名 Prophet 🔮": sorted(prophet_predictions, key=lambda x: x[1], reverse=True)[:10],
            "📉 後十名 LSTM 🧠": sorted(lstm_predictions, key=lambda x: x[1])[:10],
            "⛔ 後十名 Prophet 🔮": sorted(prophet_predictions, key=lambda x: x[1])[:10],
        }

        # 組裝並發送結果
        send_results(index_name, stock_predictions)

# 主函數
def main():
    try:
        period = "3mo"
        selected_indices = ["台灣50", "台灣中型100", "S&P", "NASDAQ", "費城半導體", "道瓊"]
        print("計算潛力股...")
        get_top_and_bottom_10_potential_stocks(period, selected_indices)
    except Exception as e:
        print(f"錯誤: {str(e)}")
        send_to_telegram(f"⚠️ 錯誤: {str(e)}")

if __name__ == "__main__":
    main()