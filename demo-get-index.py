import requests
import json

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

# 主程序：測試各個 API 是否能正常工作
if __name__ == "__main__":
    print("測試各個指數 API 是否能正常工作...\n")
    
    # 測試台灣 0050 ETF 成分股
    print("=== 台灣 0050 ETF 成分股 ===")
    try:
        tw0050_stocks = get_tw0050_stocks()
        print(f"成功獲取 {len(tw0050_stocks)} 支 0050 成分股")
        print(f"前 5 支股票: {tw0050_stocks[:5]}")
        print()
    except Exception as e:
        print(f"獲取 0050 成分股時發生錯誤: {str(e)}")
        print()
    
    # 測試台灣 0051 ETF 成分股
    print("=== 台灣 0051 ETF 成分股 ===")
    try:
        tw0051_stocks = get_tw0051_stocks()
        print(f"成功獲取 {len(tw0051_stocks)} 支 0051 成分股")
        print(f"前 5 支股票: {tw0051_stocks[:5]}")
        print()
    except Exception as e:
        print(f"獲取 0051 成分股時發生錯誤: {str(e)}")
        print()
    
    # 測試 S&P 500 成分股
    print("=== S&P 500 成分股 ===")
    try:
        sp500_stocks = get_sp500_stocks(limit=100)
        print(f"成功獲取 {len(sp500_stocks)} 支 S&P 500 成分股")
        print(f"前 5 支股票: {sp500_stocks[:5]}")
        print()
    except Exception as e:
        print(f"獲取 S&P 500 成分股時發生錯誤: {str(e)}")
        print()
    
    # 測試 NASDAQ 100 成分股
    print("=== NASDAQ 100 成分股 ===")
    try:
        nasdaq_stocks = get_nasdaq_stocks()
        print(f"成功獲取 {len(nasdaq_stocks)} 支 NASDAQ 100 成分股")
        print(f"前 5 支股票: {nasdaq_stocks[:5]}")
        print()
    except Exception as e:
        print(f"獲取 NASDAQ 100 成分股時發生錯誤: {str(e)}")
        print()
    
    # 測試道瓊斯工業平均指數成分股
    print("=== 道瓊斯工業平均指數成分股 ===")
    try:
        dji_stocks = get_dji_stocks()
        print(f"成功獲取 {len(dji_stocks)} 支道瓊斯工業平均指數成分股")
        print(f"前 5 支股票: {dji_stocks[:5]}")
        print()
    except Exception as e:
        print(f"獲取道瓊斯工業平均指數成分股時發生錯誤: {str(e)}")
        print()