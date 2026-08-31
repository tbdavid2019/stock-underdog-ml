## Context

專案現有架構將資料下載、特徵計算、模型訓練、評分邏輯、通知與儲存高度耦合於 `main.py`，且保留了早期 MySQL / MongoDB 的未清理代碼（詳見 `proposal.md`）。

為了因應未來持續擴充各類型選股策略（如技術面動量、價量突破、籌碼面多因子、新 ML 模型如 Transformer/LightGBM 等），本次重構將「策略引擎」設計為 **插件式註冊架構（Pluggable Strategy Registry Pattern）**，讓新增任何新策略只需建立單一策略類別並註冊，完全不需要修改核心管線或資料庫主程式。

## Goals / Non-Goals

**Goals:**
- **模組化分層架構**：建立 `core/`、`data/`、`strategies/`、`evaluators/`、`pipeline/` 模組，各模組職責單一且可獨立單元測試。
- **高擴展性策略插件系統 (Strategy Registry)**：
  - 新增策略只需繼承 `BaseStrategy` 並透過 `@register_strategy("name")` 註冊。
  - 支援設定檔啟用/停用特定策略列表（`ENABLED_STRATEGIES`），管線動態多型執行。
- **統一硬體設備管理**：實作 `DeviceManager`，支援 CUDA / Apple Silicon MPS / CPU 自動探測、配置切換與安全降級。
- **I/O 與運算分流**：建立兩階段管線（Stage 1 批次下載與快取 ➔ Stage 2 向量化策略計算與 ML 預測），避免 GIL 下的執行緒競爭。
- **多策略動態綜合評價**：`CompositeEvaluator` 支援任意 N 個策略的動態交集分析、權重配置、標籤聚合與 Top-N 排序。
- **完整向下相容**：維持既有 Supabase `predictions` 欄位格式與 Telegram / Discord 通知輸出內容。

**Non-Goals:**
- 不變更玄鐵重劍技術指標（MA60/MA120/MA250 回調）的原始數學邏輯。
- 不更改 Supabase 資料表綱要（Schema），非傳統欄位採用 `signals`/`metrics` JSON 封裝或向下相容對應。
- 不引入重型分散式調度框架（如 Celery/Airflow），保持單機高效與輕量部署。

## Decisions

### 1. 模組架構與職責分配
- `core/`:
  - `device.py`: `DeviceManager`（自動偵測與封裝 `cuda` / `mps` / `cpu`）。
  - `config.py`: 精簡後的設定類別，移除 MySQL/MongoDB 廢棄變數。
- `data/`:
  - `fetcher.py`: 統一股票行情下載（批次下載 + 智慧重試）。
  - `fundamentals.py`: 基本面指標（PE/PB/EV/EBITDA）批次擷取與快取。
  - `cache.py`: 統一快取管理器（支援成分股 JSON 快取與股價 Pickle 快取）。
- `strategies/`:
  - `base.py`: 定義 `BaseStrategy` 抽象基礎類別、`StockContext` 與 `StrategyResult` 資料結構。
  - `registry.py`: `StrategyRegistry`（策略註冊中心與工廠函式）。
  - `xuantie.py`: 玄鐵重劍波段策略封裝。
  - `lstm_strategy.py`: LSTM 短線預測模型與訓練/推論生命週期管理。
- `evaluators/`:
  - `composite_evaluator.py`: 多策略交叉比對、動態權重評分、標籤聚合與排序。
  - `formatter.py`: 終端機表格美化格式化工具。
- `pipeline/`:
  - `orchestrator.py`: 兩階段管線執行引擎（Index ➔ Batch Fetch ➔ Strategy Evaluate ➔ Composite Scoring ➔ Persist/Notify）。
- `storage/`:
  - `supabase_adapter.py`: 收攏 Supabase 寫入與 JSON 安全校驗（取代肥大且含廢碼的 `database.py`）。

### 2. 策略擴充介面與註冊機制 (Pluggable Strategy Pattern)

未來新增策略（如 `BreakoutStrategy` 或 `MomentumStrategy`）時，只需撰寫一個獨立檔案：

```python
# strategies/base.py
@dataclass
class StockContext:
    ticker: str
    df: pd.DataFrame                # 歷史行情 OHLCV + 基礎技術指標
    fundamentals: Dict[str, Any]    # 基本面數據 (PE, PB, EV/EBITDA)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyResult:
    ticker: str
    strategy_name: str
    is_hit: bool                   # 是否符合買入/選股標準
    score: float                   # 策略獨立評分 (0~100 或漲幅潛力)
    current_price: float
    predicted_price: Optional[float] = None
    potential: Optional[float] = None
    signals: Dict[str, Any] = field(default_factory=dict)   # 自訂信號 (如 pullback_type, rsi)
    metrics: Dict[str, Any] = field(default_factory=dict)   # 自訂指標 (如 ma60, volume_surge)
    tags: List[str] = field(default_factory=list)           # 策略生成之標籤

class BaseStrategy(ABC):
    name: str = "base"
    category: str = "technical"    # technical | ml | fundamental | chip
    required_lookback: int = 60    # 所需歷史數據天數

    @abstractmethod
    def evaluate(self, context: StockContext) -> StrategyResult:
        """單檔選股邏輯"""
        pass
        
    def evaluate_batch(self, contexts: Dict[str, StockContext]) -> List[StrategyResult]:
        """批次選股邏輯 (預設遍歷 evaluate，支援子類別覆寫為矩陣化/GPU批次推論)"""
        return [self.evaluate(ctx) for ctx in contexts.values()]

# 註冊中心
class StrategyRegistry:
    _registry: Dict[str, Type[BaseStrategy]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(subclass: Type[BaseStrategy]):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_strategy(cls, name: str, **kwargs) -> BaseStrategy:
        if name not in cls._registry:
            raise ValueError(f"Unknown strategy: {name}")
        return cls._registry[name](**kwargs)
```

### 3. 動態多策略綜合評價 (Dynamic Composite Evaluator)
`CompositeEvaluator` 不硬編碼特定策略名稱，而是接受動態權重與交集規則：
- **動態權重矩陣**：例如 `weights = {"xuantie": 0.4, "lstm": 0.4, "momentum": 0.2}`。
- **動態交集規則 (N-of-M Match)**：可設定「只要命中任 2 個以上策略」或「指定必中特定核心策略」。
- **標籤自動聚合**：收集各命中策略回傳的 `tags` 與基本面標籤，組合成最終綜合評價標籤。

### 4. 兩階段管線並發設計 (Two-Stage Pipeline)
- **Stage 1 (I/O Bound)**：使用 `ThreadPoolExecutor` 批次下載各股票的 OHLCV 與基本面數據，建立 `StockContext` 記憶體字典與本機快取。
- **Stage 2 (Compute Bound)**：管線遍歷 `ENABLED_STRATEGIES`，多型呼叫各策略的 `evaluate_batch(contexts)`；ML 模型推論透過 `DeviceManager` 綁定至指定硬體（MPS/CUDA/CPU）。

### 5. 歷史廢碼退役
徹底移除 `database.py` 中的 `MySQLManager` 與 `save_to_mongodb`，並刪除根目錄過渡期腳本 `parallel_processor.py`、`main_lstm_only.py`。

## Risks / Trade-offs

- **[Risk] 新策略可能需要額外特徵數據（如大盤指數、籌碼面、新聞情緒）**
  - *Mitigation*: `StockContext` 提供 `metadata` 字典擴充槽，資料層預留擴充 provider 介面。
- **[Risk] yfinance 基本面大量查詢觸發 Rate Limit**
  - *Mitigation*: 在 `fundamentals.py` 實作 local TTL 快取（24 小時），並加入請求間隔與失敗重試保護。
- **[Risk] Apple Silicon MPS 與 CUDA 相容性問題**
  - *Mitigation*: `DeviceManager` 在初始化時執行輕量 tensor 測試運算，若失敗則無縫降級至 CPU 並發出警告日誌。

## Migration Plan

1. **Phase 1: 核心與基礎層**（`core/device.py`, `core/config.py`, 清理 `database.py`）
2. **Phase 2: 資料層與快取重構**（`data/fetcher.py`, `data/fundamentals.py`, `data/cache.py`）
3. **Phase 3: 策略標準化與評價引擎**（`strategies/base.py`, `strategies/registry.py`, `strategies/xuantie.py`, `strategies/lstm.py`, `evaluators/`）
4. **Phase 4: 管線排程與主入口**（`pipeline/orchestrator.py`, 重構 `main.py`）
5. **Phase 5: 回歸測試與文件同步**（更新 `README.md` 與 `docs/CHANGELOG.md`）
