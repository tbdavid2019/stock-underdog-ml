import pandas as pd
import mysql.connector
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np

def get_historical_predictions(connection, start_date, end_date, method="LSTM"):
    """
    從MySQL獲取特定時期的預測記錄
    """
    query = """
    SELECT calculation_date, stock_symbol, current_price, predicted_price, potential
    FROM stock_predictions
    WHERE prediction_method = %s
    AND calculation_date BETWEEN %s AND %s
    ORDER BY calculation_date, stock_symbol
    """
    
    df = pd.read_sql_query(query, connection, params=(method, start_date, end_date))
    return df

def get_actual_future_price(symbol, prediction_date, days_forward=30):
    """
    獲取股票在預測日期之後的實際價格
    """
    end_date = prediction_date + timedelta(days=days_forward)
    data = yf.download(symbol, start=prediction_date, end=end_date)
    if len(data) > 0:
        return data['Close'].iloc[-1]
    return None

def analyze_prediction_accuracy(connection, start_date, end_date, days_forward=30):
    # 獲取歷史預測數據
    predictions_df = get_historical_predictions(connection, start_date, end_date)
    
    results = []
    for _, row in predictions_df.iterrows():
        prediction_date = row['calculation_date']
        symbol = row['stock_symbol']
        predicted_price = row['predicted_price']
        predicted_potential = row['potential']
        
        # 獲取實際未來價格
        actual_price = get_actual_future_price(symbol, prediction_date, days_forward)
        
        if actual_price is not None:
            actual_potential = (actual_price - row['current_price']) / row['current_price']
            
            result = {
                'date': prediction_date,
                'symbol': symbol,
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'predicted_potential': predicted_potential,
                'actual_potential': actual_potential,
                'prediction_error': abs(predicted_price - actual_price) / actual_price,
                'potential_error': abs(predicted_potential - actual_potential)
            }
            results.append(result)
    
    return pd.DataFrame(results)

def calculate_metrics(results_df):
    """
    計算各種評估指標
    """
    metrics = {
        'mean_price_error': results_df['prediction_error'].mean(),
        'mean_potential_error': results_df['potential_error'].mean(),
        'correct_direction': (
            (results_df['predicted_potential'] > 0) == 
            (results_df['actual_potential'] > 0)
        ).mean(),
        'total_predictions': len(results_df)
    }
    
    # 計算高潛力股票的準確率（預測潛力前25%的股票）
    high_potential = results_df[
        results_df['predicted_potential'] > 
        results_df['predicted_potential'].quantile(0.75)
    ]
    
    metrics['high_potential_accuracy'] = (
        (high_potential['predicted_potential'] > 0) == 
        (high_potential['actual_potential'] > 0)
    ).mean()
    
    return metrics

def main_backtest():
    # 連接數據庫
    connection = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE")
    )
    
    # 設置回測期間
    start_date = '2025-01-15'  # 根據你的數據開始日期調整
    end_date = '2025-02-15'    # 結束日期
    
    # 進行回測分析
    results_df = analyze_prediction_accuracy(connection, start_date, end_date)
    
    # 計算評估指標
    metrics = calculate_metrics(results_df)
    
    # 輸出結果
    print("\n=== LSTM預測回測結果 ===")
    print(f"預測總數: {metrics['total_predictions']}")
    print(f"平均價格預測誤差: {metrics['mean_price_error']:.2%}")
    print(f"平均潛力預測誤差: {metrics['mean_potential_error']:.2%}")
    print(f"預測方向準確率: {metrics['correct_direction']:.2%}")
    print(f"高潛力股票準確率: {metrics['high_potential_accuracy']:.2%}")
    
    # 保存詳細結果
    results_df.to_csv('backtest_results.csv', index=False)
    
    # 繪製視覺化圖表
    plot_results(results_df)
    
    connection.close()

def plot_results(results_df):
    import matplotlib.pyplot as plt
    
    # 預測vs實際潛力散點圖
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['predicted_potential'], 
               results_df['actual_potential'],
               alpha=0.5)
    plt.xlabel('預測潛力')
    plt.ylabel('實際潛力')
    plt.title('預測潛力 vs 實際潛力')
    plt.plot([-1, 1], [-1, 1], 'r--')  # 完美預測線
    plt.grid(True)
    plt.savefig('potential_comparison.png')
    plt.close()

if __name__ == "__main__":
    main_backtest()