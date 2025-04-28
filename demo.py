import yfinance as yf

def test_yfinance_download():
    # 設置股票代碼和下載期間
    ticker = "BRK-B"  # 伯克希爾·哈撒韋 B 類股票代碼
    period = "1mo"   # 過去一個月的數據
    
    # 也可以嘗試其他可能的格式
    alternative_tickers = ["BRK-B", "BRK_B", "BRKB"]

    # 嘗試主要的股票代碼
    try:
        print(f"正在下載 {ticker} 的數據...")
        data = yf.download(ticker, period=period)
        if not data.empty:
            print(f"下載成功！數據摘要：")
            print(data.head())  # 打印前幾行數據
        else:
            print(f"下載的數據為空，嘗試替代股票代碼...")
            
            # 嘗試替代股票代碼
            for alt_ticker in alternative_tickers:
                if alt_ticker == ticker:
                    continue  # 跳過已嘗試過的代碼
                    
                print(f"嘗試使用替代股票代碼: {alt_ticker}")
                alt_data = yf.download(alt_ticker, period=period)
                if not alt_data.empty:
                    print(f"使用 {alt_ticker} 下載成功！數據摘要：")
                    print(alt_data.head())
                    break
                else:
                    print(f"使用 {alt_ticker} 下載的數據為空")
    except Exception as e:
        print(f"下載失敗，錯誤訊息：{str(e)}")

# 執行測試
if __name__ == "__main__":
    test_yfinance_download()

