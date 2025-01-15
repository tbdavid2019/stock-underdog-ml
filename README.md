# tbdavid2019/stock-underdog-ml
## 股市潛力股計算 (LSTM, Prophet, Transformer) 機器學習版本
這是一個用於計算股市潛力股的工具，結合了多種機器學習模型，包括 LSTM、Prophet 和 Transformer。該工具能根據不同時間範圍的歷史數據，預測股票價格，並計算股票的潛力。透過此應用，投資者可以快速篩選出具有潛力的股票，做出更明智的投資決策。

### 功能特點
- **多模型支持**：LSTM（長短期記憶網絡）、Prophet（時間序列分析）、Transformer（高效預測）。
- **多市場支持**：涵蓋台灣 50、SP500、NASDAQ、費城半導體等主要股市指數。
- **靈活參數配置**：可以針對不同市場和股票自定義數據抓取期間與預測方法。
- **結果導出**：支持透過電子郵件、Telegram、Discord 分享預測結果。
- **數據存儲**：預測結果同時存儲於 MySQL 和 MongoDB，方便後續查詢與分析。
- **回測分析**：提供預測準確性的回測功能，幫助評估模型效能。

### 使用方法
1. 安裝必要的套件：
    ```bash
    pip install -r requirements.txt
    ```
2. 配置 `.env` 文件以設置以下參數：
    - `SMTP_SERVER`, `SMTP_PORT`, `SENDER_EMAIL`, `EMAIL_PASSWORD`
    - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHANNEL_ID`
    - `MONGO_URI`
    - `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`

3. 執行主程序進行預測：
    ```bash
    python app.py
    ```

4. 執行回測分析：
    ```bash
    python backtest_analysis.py
    ```

### 資料結構與參數
- **輸入數據**：
    - 股票歷史數據，包括 `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`。
- **模型訓練**：
    - LSTM 使用 3 個月的數據 (`3mo`)。
    - Transformer 使用 5 年的數據 (`5y`)。
    - Prophet 自適應所有數據範圍。
- **預測結果存儲**：
    - MySQL：詳細的預測記錄，包含預測時間、股票代碼、當前價格、預測價格等。
    - MongoDB：完整的預測結果集合，包含更多元數據。

### 回測功能
- **功能說明**：
    - 分析歷史預測的準確性
    - 計算預測誤差率
    - 評估預測方向的準確度
    - 生成視覺化分析報告
- **回測指標**：
    - 價格預測誤差
    - 潛力預測誤差
    - 預測方向準確率
    - 高潛力股票準確率
- **結果輸出**：
    - CSV 格式的詳細分析報告
    - 預測vs實際結果的視覺化圖表

### 注意事項
- 使用 Yahoo Finance 獲取數據，請確保網絡連線正常。
- 預測模型需要一定的數據量，股票數據少於 60 條時將被跳過。
- 訓練和預測過程可能需要較長的時間，建議使用具備 GPU 的環境提升效能。
- 回測分析需要足夠的歷史預測數據，建議累積一定時間的預測記錄後再進行。

---

# tbdavid2019/stock-underdog-ml
## Stock Potential Predictor (LSTM, Prophet, Transformer) Machine Learning Version
This is a stock potential predictor tool combining multiple machine learning models, including LSTM, Prophet, and Transformer. The tool analyzes historical stock data to forecast prices and calculate potential. With this application, investors can identify promising stocks and make smarter investment decisions.

### Features
- **Multi-Model Support**: LSTM (Long Short-Term Memory), Prophet (Time Series Analysis), Transformer (Efficient Forecasting).
- **Market Coverage**: Supports major indices like Taiwan 50, SP500, NASDAQ, and Philadelphia Semiconductor.
- **Flexible Configuration**: Customize data periods and prediction methods for different markets and stocks.
- **Result Export**: Share predictions via Email, Telegram, and Discord.
- **Data Storage**: Store predictions in both MySQL and MongoDB for future reference and analysis.
- **Backtesting**: Provides backtesting functionality to evaluate prediction accuracy and model performance.

### How to Use
1. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Configure the `.env` file with the following parameters:
    - `SMTP_SERVER`, `SMTP_PORT`, `SENDER_EMAIL`, `EMAIL_PASSWORD`
    - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHANNEL_ID`
    - `MONGO_URI`
    - `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`

3. Run the main prediction script:
    ```bash
    python app.py
    ```

4. Run backtesting analysis:
    ```bash
    python backtest_analysis.py
    ```

### Data Structure and Parameters
- **Input Data**:
    - Historical stock data, including `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
- **Model Training**:
    - LSTM uses 3 months of data (`3mo`).
    - Transformer uses 5 years of data (`5y`).
    - Prophet adapts to all available data ranges.
- **Prediction Storage**:
    - MySQL: Detailed prediction records including prediction time, stock symbol, current price, predicted price, etc.
    - MongoDB: Complete prediction result sets with additional metadata.

### Backtesting Functionality
- **Features**:
    - Analysis of historical prediction accuracy
    - Calculation of prediction error rates
    - Evaluation of directional prediction accuracy
    - Generation of visual analysis reports
- **Metrics**:
    - Price prediction error
    - Potential prediction error
    - Directional accuracy rate
    - High-potential stock accuracy
- **Output**:
    - Detailed analysis report in CSV format
    - Visualization charts comparing predicted vs actual results

### Notes
- Data is fetched using Yahoo Finance. Ensure a stable internet connection.
- Stocks with fewer than 60 data points will be skipped.
- Training and prediction may take time; using a GPU-enabled environment is recommended for better performance.
- Backtesting analysis requires sufficient historical prediction data; it's recommended to accumulate prediction records over time before performing analysis.

### Project Structure
```
project/
├── app.py                     # Main prediction script
├── backtest_analysis.py       # Backtesting analysis script
├── requirements.txt           # Package dependencies
├── .env                      # Environment variables
└── results/                  # Results directory
    ├── backtest_results.csv  # Detailed backtesting results
    └── analysis_charts/      # Generated visualization charts
```

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
```
