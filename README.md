# tbdavid2019/stock-underdog-ml
## 股市潛力股計算 (LSTM, Prophet, Transformer) 機器學習版本

這是一個用於計算股市潛力股的工具，結合了多種機器學習模型，包括 LSTM、Prophet 和 Transformer。該工具能根據不同時間範圍的歷史數據，預測股票價格，並計算股票的潛力。透過此應用，投資者可以快速篩選出具有潛力的股票，做出更明智的投資決策。

### 功能特點
- **多模型支持**：LSTM（長短期記憶網絡）、Prophet（時間序列分析）、Transformer（高效預測）。
- **多市場支持**：涵蓋台灣 50、SP500、NASDAQ、費城半導體等主要股市指數。
- **靈活參數配置**：可以針對不同市場和股票自定義數據抓取期間與預測方法。
- **結果導出**：支持透過電子郵件、Telegram、Discord 分享預測結果。
- **數據存儲**：可將結果存儲於 MongoDB，方便後續查詢與分析。

### 使用方法
1. 安裝必要的套件：
    ```bash
    pip install -r requirements.txt
    ```
2. 配置 `.env` 文件以設置以下參數：
    - `SMTP_SERVER`, `SMTP_PORT`, `SENDER_EMAIL`, `EMAIL_PASSWORD`
    - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHANNEL_ID`
    - `MONGO_URI`
3. 執行主程序：
    ```bash
    python app.py
    ```
4. 查看預測結果並檢查控制台輸出或配置的通知方式。

### 資料結構與參數
- **輸入數據**：
    - 股票歷史數據，包括 `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`。
- **模型訓練**：
    - LSTM 使用 3 個月的數據 (`3mo`)。
    - Transformer 使用 5 年的數據 (`5y`)。
    - Prophet 自適應所有數據範圍。

### 注意事項
- 使用 Yahoo Finance 獲取數據，請確保網絡連線正常。
- 預測模型需要一定的數據量，股票數據少於 60 條時將被跳過。
- 訓練和預測過程可能需要較長的時間，建議使用具備 GPU 的環境提升效能。

---

# tbdavid2019/stock-underdog-ml
## Stock Potential Predictor (LSTM, Prophet, Transformer) Machine Learning Version

This is a stock potential predictor tool combining multiple machine learning models, including LSTM, Prophet, and Transformer. The tool analyzes historical stock data to forecast prices and calculate potential. With this application, investors can identify promising stocks and make smarter investment decisions.

### Features
- **Multi-Model Support**: LSTM (Long Short-Term Memory), Prophet (Time Series Analysis), Transformer (Efficient Forecasting).
- **Market Coverage**: Supports major indices like Taiwan 50, SP500, NASDAQ, and Philadelphia Semiconductor.
- **Flexible Configuration**: Customize data periods and prediction methods for different markets and stocks.
- **Result Export**: Share predictions via Email, Telegram, and Discord.
- **Data Storage**: Save results in MongoDB for future reference and analysis.

### How to Use
1. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Configure the `.env` file with the following parameters:
    - `SMTP_SERVER`, `SMTP_PORT`, `SENDER_EMAIL`, `EMAIL_PASSWORD`
    - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHANNEL_ID`
    - `MONGO_URI`
3. Run the main script:
    ```bash
    python app.py
    ```
4. View prediction results in the console or via configured notifications.

### Data Structure and Parameters
- **Input Data**:
    - Historical stock data, including `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
- **Model Training**:
    - LSTM uses 3 months of data (`3mo`).
    - Transformer uses 5 years of data (`5y`).
    - Prophet adapts to all available data ranges.

### Notes
- Data is fetched using Yahoo Finance. Ensure a stable internet connection.
- Stocks with fewer than 60 data points will be skipped.
- Training and prediction may take time; using a GPU-enabled environment is recommended for better performance.