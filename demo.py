import yfinance as yf

def test_yfinance_download():
    # 設置股票代碼和下載期間
    ticker = "2330.TW"  # 蘋果公司的股票代碼
    period = "1mo"   # 過去一個月的數據

    try:
        print(f"正在下載 {ticker} 的數據...")
        data = yf.download(ticker, period=period)
        print(f"下載成功！數據摘要：")
        print(data.head())  # 打印前幾行數據
    except Exception as e:
        print(f"下載失敗，錯誤訊息：{str(e)}")

# 執行測試
if __name__ == "__main__":
    test_yfinance_download()

