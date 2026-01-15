# LLM 開發指南

給未來接手此專案的 AI 助手（Claude, GPT, Gemini 等）的開發筆記。

---

## 🎯 專案核心理念

### 從錯誤中學習
這個專案經歷了一次**重大的預測邏輯重構**：

**錯誤的起點：**
- 原本預測「歷史資料的最大值」
- 用 `predictions.max()` 作為預測價格
- 這根本不是「預測未來」，而是「擬合過去」

**正確的方向：**
- 改為預測「下一個交易日的收盤價」
- 使用 LSTM 的最後一個時間步預測下一步
- 這才有實際的交易參考價值

**教訓：** 永遠先確認「預測的是什麼」，不要假設現有邏輯是正確的。

---

## 🔍 關鍵發現

### 1. 模型選擇的現實

**測試結果（543 筆回測資料）：**

| 模型 | 方向準確度 | 結論 |
|------|-----------|------|
| LSTM | 54.6% | 略高於隨機，可用 |
| Prophet | 46.9% | 低於隨機，不適合短期預測 |
| Chronos | N/A | 太慢，執行失敗 |

**重要洞察：**
- Prophet 設計用於「長期趨勢+季節性」，不適合「明天漲跌」
- Chronos 是大模型，需要下載 Hugging Face 權重，太慢
- 簡單的 LSTM 反而最實用

**建議：** 不要被「新模型」吸引，先測試再決定。

### 2. 回測的重要性

**發現的問題：**
- 舊回測腳本試圖驗證「6個月後」的價格
- 但預測資料只有 4 天（2026-01-05 ~ 2026-01-08）
- 結果是「997 筆成功驗證」，但其實是錯的

**正確做法：**
- 預測「下一交易日」→ 回測也用「下一交易日」
- 用 yfinance 的 `hist.index` 自動處理交易日（週末/假日）
- 注意 `datetime.date` vs `datetime.datetime` 的比對問題

**教訓：** 回測邏輯必須與預測邏輯一致，否則數據毫無意義。

### 3. 資料洩漏陷阱

**常見錯誤：**
```python
# 錯誤：在整個資料集上 fit scaler
scaler.fit(data)
scaled_data = scaler.transform(data)
X_train, X_test = split(scaled_data)
```

**正確做法：**
```python
# 正確：只在訓練集上 fit
X_train, X_test = split(data)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**教訓：** 任何「從資料學習的操作」都只能在訓練集上做。

---

## 🛠️ 技術要點

### Supabase 使用

**關鍵點：**
1. **優先使用 `SUPABASE_SERVICE_KEY`**
   - 不是 `SUPABASE_ANON_KEY`
   - Service key 繞過 RLS，適合後端操作

2. **批次更新要小心**
   - Supabase Python client 不支援 `UPDATE ... WHERE` 批次操作
   - 需要逐筆更新或用 RPC

3. **日期格式**
   - Supabase 回傳的 timestamp 是 ISO 格式字串
   - 需要 `.replace('Z', '+00:00')` 才能正確解析

### yfinance 陷阱

**問題：**
```python
hist = yf.Ticker("2330.TW").history(period='1mo')
# hist.index 是 pandas.DatetimeIndex
# 但元素是 datetime.date，不是 datetime.datetime！
```

**解決：**
```python
# 錯誤
if check_date.date() in hist.index:  # 永遠 False

# 正確
if check_date.date() in [d.date() if hasattr(d, 'date') else d for d in hist.index]:
```

### LSTM 優化技巧

**有效的改進：**
- ✅ 增加網路深度（2層 → 3層）
- ✅ 增加訓練時間（10 epochs → 100 epochs）
- ✅ Early Stopping（防止過擬合）
- ✅ Learning Rate Scheduling
- ✅ 增加 time_step（60 → 90 天）

**無效的改進：**
- ❌ 過度複雜的架構（容易過擬合）
- ❌ 過小的 batch size（<8，訓練不穩定）

---

## 📋 接手開發檢查清單

### 第一步：理解現狀

- [ ] 閱讀 `README.md` 和 `CHANGELOG.md`
- [ ] 檢查 `.env` 設定（哪些模型開啟？）
- [ ] 查看最近的 `logs/app.log`
- [ ] 執行 `python scripts/check_db_contents.py` 查看資料庫狀態

### 第二步：驗證系統

- [ ] 執行 `python scripts/test_next_day_prediction.py`
- [ ] 確認預測邏輯正確（是預測「明天」，不是「歷史最大值」）
- [ ] 檢查回測結果：`python scripts/backfill_historical.py`
- [ ] 驗證方向準確度 >50%（至少要比隨機好）

### 第三步：了解限制

- [ ] LSTM 方向準確度只有 54.6%（還有改進空間）
- [ ] 台灣50 表現最差（41.5%）
- [ ] 回測需要 2.6 分鐘處理 1000 筆
- [ ] 部分股票會下市（2809.TW, 2888.TW 等）

---

## 🚨 常見陷阱

### 1. 不要輕易改預測邏輯

**問題：** 使用者可能會說「預測不準，改成預測 3 天後吧」

**回應：**
- 預測期間越長，準確度通常越低
- 改邏輯前，先用現有資料回測驗證
- 記得同步更新回測腳本

### 2. 不要假設舊程式碼是對的

**問題：** 專案中有很多舊檔案（`elder/`, `app.py` 等）

**回應：**
- 這些是歷史遺留，可能有錯誤
- 以 `main.py` 和 `models/lstm.py` 為準
- 舊檔案已移至 `scripts/`，標記為「已棄用」

### 3. 不要過度優化

**問題：** 想要加入更多模型、更多特徵

**回應：**
- 先把 LSTM 優化到 60%+ 再說
- 每次只改一個變數，用回測驗證
- 記錄在 CHANGELOG.md

---

## 💡 優化建議

### 短期（可立即嘗試）

1. **特徵工程**
   - 加入技術指標（RSI, MACD, Bollinger Bands）
   - 加入成交量變化率
   - 加入市場情緒指標

2. **超參數調整**
   - Grid search: units (64/128/256), layers (2/3/4)
   - 調整 dropout rate (0.2/0.3/0.4)
   - 調整 learning rate (0.0001/0.001/0.01)

3. **資料增強**
   - 增加訓練資料期間（6mo → 1y）
   - 使用更多股票訓練（transfer learning）

### 中期（需要較多工作）

1. **Ensemble 模型**
   - LSTM + XGBoost
   - 多個 LSTM 投票

2. **注意力機制**
   - 在 LSTM 上加 Attention layer
   - Transformer encoder

3. **強化學習**
   - 用 RL 優化買賣時機
   - 結合風險管理

### 長期（需要重構）

1. **即時預測**
   - WebSocket 接收即時股價
   - 盤中更新預測

2. **基本面整合**
   - 財報數據
   - 新聞情緒分析

3. **自動交易**
   - 紙上交易驗證
   - 風險控制系統

---

## 📝 開發流程建議

### 1. 改動前

```bash
# 1. 備份資料庫
python scripts/check_db_contents.py > backup_$(date +%Y%m%d).txt

# 2. 記錄當前表現
python scripts/backfill_historical.py > baseline.txt
```

### 2. 開發中

```bash
# 1. 小步迭代
# 2. 每次只改一個東西
# 3. 立即測試
python scripts/test_next_day_prediction.py
```

### 3. 改動後

```bash
# 1. 回測驗證
python scripts/backfill_historical.py

# 2. 比較結果
# 3. 更新 CHANGELOG.md
```

---

## 🔗 重要檔案索引

### 核心邏輯
- `models/lstm.py` - LSTM 預測邏輯（最重要！）
- `parallel_processor.py` - 並行處理和模型調用
- `main.py` - 主程式入口

### 設定
- `.env` - 環境變數（模型開關、API keys）
- `config.py` - 設定管理類別

### 回測
- `backtest/backtest.py` - 回測主腳本
- `backtest/analyze_backtest.py` - 結果分析

### 資料庫
- `supabase_schema.sql` - 完整 schema（含回測欄位）
- `database.py` - Supabase 操作封裝

### 文檔
- `README.md` - 專案總覽
- `CHANGELOG.md` - 版本記錄
- `SETUP.md` - 安裝指南
- `CLAUDE.md` - 本檔案

---

## 🎓 學到的教訓

1. **永遠先驗證假設** - 不要假設現有程式碼是對的
2. **回測是必須的** - 沒有回測就是盲目飛行
3. **簡單優於複雜** - LSTM 比 Chronos 更實用
4. **文檔很重要** - 未來的你（或其他 AI）會感謝現在的你
5. **小步快跑** - 每次只改一個東西，立即驗證

---

## 🤝 給下一位 AI 的話

這個專案已經走過很多彎路，但現在的架構是經過實戰驗證的：

- ✅ 預測邏輯正確（下一交易日）
- ✅ 回測系統完整
- ✅ 文檔齊全
- ✅ 專案結構清晰

**你的任務是：**
1. 把 LSTM 方向準確度從 54.6% 提升到 60%+
2. 保持程式碼品質
3. 更新 CHANGELOG.md

**記住：**
- 使用者可能不懂技術細節，要耐心解釋
- 每個決策都要有數據支持（回測結果）
- 保持謙虛，承認模型的限制

祝你好運！🚀

---

**最後更新：** 2026-01-08  
**作者：** Claude (Anthropic)  
**專案版本：** v2.0.0
