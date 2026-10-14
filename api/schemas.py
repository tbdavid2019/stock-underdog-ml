"""
api/schemas.py - Pydantic Data Models for Quantitative API & MCP Tools
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from core.version import APP_VERSION


class StockPredictionItem(BaseModel):
    index_name: Optional[str] = Field(None, description="指數名稱 (台灣50, 台灣中型100, S&P500 等)")
    model_name: Optional[str] = Field(None, description="模型/策略名稱 (玄鐵重劍, LSTM, 多維共振)")
    strategy_type: Optional[str] = Field(None, description="策略類別")
    ticker: str = Field(..., description="股票代號 (如 2330.TW, AAPL)")
    current_price: Optional[float] = Field(None, description="當前/最新收盤價")
    predicted_price: Optional[float] = Field(None, description="預測明日目標價")
    potential: Optional[float] = Field(None, description="預期漲跌幅百分比 (%)")
    ma5: Optional[float] = Field(None, description="5日均線")
    ma10: Optional[float] = Field(None, description="10日均線")
    ma60: Optional[float] = Field(None, description="60日均線 (季線)")
    ma120: Optional[float] = Field(None, description="120日均線 (半年線)")
    ma250: Optional[float] = Field(None, description="250日均線 (年線)")
    pullback_type: Optional[str] = Field(None, description="均線回調類型 (如 MA60 (+2.1%))")
    pe: Optional[float] = Field(None, description="本益比 (Price to Earnings)")
    pb: Optional[float] = Field(None, description="股價淨值比 (Price to Book)")
    forward_pe: Optional[float] = Field(None, description="預估本益比")
    ev_ebitda: Optional[float] = Field(None, description="企業價值倍數")
    period: Optional[str] = Field(None, description="訓練週期")
    timestamp: Optional[str] = Field(None, description="預測產出時間戳記 (ISO 8601)")
    analysis_date: Optional[str] = Field(None, description="分析所屬日期 (YYYY-MM-DD)")
    data_as_of: Optional[str] = Field(None, description="資料產生時間戳記 (ISO 8601)")
    age_hours: Optional[float] = Field(None, description="資料產生迄今經過小時數")
    is_stale: Optional[bool] = Field(None, description="資料是否已超過 48 小時過期")
    macro_regime: Optional[str] = Field(None, description="當時美股宏觀市場環境")
    trust_net_5d: Optional[int] = Field(None, description="投信 5 日累計買賣超 (張)")
    foreign_net_5d: Optional[int] = Field(None, description="外資 5 日累計買賣超 (張)")
    risk_reward_ratio: Optional[float] = Field(None, description="盈虧比 (TimesFM 90%分位數上行 / 10%下行)")
    tags: Optional[str] = Field(None, description="綜合標籤 (如 🏆三重共振 | 土洋合買)")


class PredictionResponse(BaseModel):
    success: bool = True
    batch_date: Optional[str] = Field(None, description="最新批次分析日期 (YYYY-MM-DD)")
    latest_batch_timestamp: Optional[str] = Field(None, description="最新批次完整時間戳記")
    is_stale: bool = Field(False, description="整個批次資料是否已逾期 (> 48h)")
    count: int = Field(..., description="符合條件的記錄總數")
    data: List[StockPredictionItem] = Field(default_factory=list, description="預測結果清單")


class TickerResolutionResponse(BaseModel):
    success: bool = True
    query: str
    ticker: str
    name: Optional[str] = None
    matched_by: str


class MacroRegimeResponse(BaseModel):
    success: bool = True
    market: Optional[str] = Field("US", description="所屬市場 (TW 或 US)")
    regime_name: str = Field(..., description="大盤/宏觀環境 (全面多頭, 多頭回調, 避險防禦, 極度恐慌)")
    exposure: Optional[float] = Field(None, description="建議投資曝險比例 (0.0 ~ 1.0；資料不足時為 null)")
    vix: Optional[float] = Field(None, description="VIX 恐慌指數")
    spy_above_ma60: Optional[bool] = Field(None, description="S&P500 是否站穩季線")
    sox_above_ma60: Optional[bool] = Field(None, description="費城半導體是否站穩季線")
    twii_above_ma60: Optional[bool] = Field(None, description="台灣加權指數是否站穩季線")
    warnings: List[str] = Field(default_factory=list, description="風控警示清單")
    fed_rate: Optional[Dict[str, Any]] = Field(None, description="聯準會降息機率 (Investing.com CME FedWatch)")
    commodities: Optional[Dict[str, Any]] = Field(None, description="關鍵大宗商品 (黃金、銅博士、原油)")
    earnings_calendar: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="美股重量級財報行事曆")
    economic_calendar: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="重磅總經行事曆")
    catalyst_alerts: List[str] = Field(default_factory=list, description="催化劑事件預警")
    polymarket: Optional[Dict[str, Any]] = Field(None, description="Polymarket 真金白銀預測市場宏觀情緒")
    timestamp: Optional[str] = None


class DBStatsResponse(BaseModel):
    success: bool = True
    total_records: int = Field(..., description="DuckDB 總記錄筆數")
    distinct_tickers: int = Field(..., description="涵蓋之獨立股票數量")
    earliest_timestamp: Optional[str] = Field(None, description="最早期預測時間戳記")
    latest_timestamp: Optional[str] = Field(None, description="最新預測時間戳記")
    indices: List[str] = Field(default_factory=list, description="已分析指數清單")
    models: List[str] = Field(default_factory=list, description="已啟用模型清單")


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    version: str = APP_VERSION
    duckdb_records: int


class EarningsCallItem(BaseModel):
    id: str = Field(..., description="法說會唯一識別碼")
    market: str = Field(..., description="市場類別 (TW 或 US)")
    ticker: str = Field(..., description="股票代碼 (如 2330.TW, NVDA)")
    name: str = Field(..., description="公司名稱")
    call_date: str = Field(..., description="召開日期 (YYYY-MM-DD)")
    call_time: Optional[str] = Field("14:00", description="召開時間 (HH:MM)")
    timezone: str = Field("Asia/Taipei", description="原始時區")
    datetime_cst: Optional[str] = Field(None, description="台灣時間 (CST)")
    datetime_est: Optional[str] = Field(None, description="美東時間 (EST/EDT)")
    countdown_days: Optional[int] = Field(None, description="距今剩餘天數 (負數為歷史)")
    status: Optional[str] = Field("upcoming", description="狀態: upcoming / today / past")
    meeting_type: Optional[str] = Field("online", description="會議形式 (online/physical/hybrid)")
    location: Optional[str] = Field(None, description="會議地點或平台")
    webcast_url: Optional[str] = Field(None, description="線上直播/影音回放網址")
    presentation_url: Optional[str] = Field(None, description="法說會簡報下載網址 (PDF)")
    transcript_url: Optional[str] = Field(None, description="逐字稿網址")
    language: Optional[str] = Field("zh-TW", description="會議語言")
    eps_estimate: Optional[float] = Field(None, description="預估 EPS")
    eps_actual: Optional[float] = Field(None, description="實際 EPS")
    revenue_estimate: Optional[float] = Field(None, description="預估營收")
    revenue_actual: Optional[float] = Field(None, description="實際營收")
    is_weight_stock: bool = Field(False, description="是否為重量級權值股")
    remarks: Optional[str] = Field(None, description="備註與摘要")
    source: Optional[str] = Field("TWSE", description="資料來源")


class UpcomingCallsResponse(BaseModel):
    success: bool = True
    market: str
    total_count: int
    data: List[EarningsCallItem] = Field(default_factory=list)


class EarningsCalendarItem(BaseModel):
    call_date: str
    total_calls: int
    weight_stock_calls: int
    summary_companies: Optional[str] = None


class EarningsCalendarResponse(BaseModel):
    success: bool = True
    year: int
    month: int
    market: str
    data: List[EarningsCalendarItem] = Field(default_factory=list)


# ==================== Podcast Radar Schemas ====================

class PodcastTickerPrice(BaseModel):
    ticker: str
    name: Optional[str] = None
    price: Optional[float] = None
    change_pct: Optional[float] = None
    direction: str = "flat"  # up / down / flat
    stance: Optional[str] = "neutral"
    stance_label: Optional[str] = "↔ 中立"
    stance_confidence: Optional[float] = 0.5
    source: Optional[str] = None
    as_of: Optional[str] = None


class PodcastChannelItem(BaseModel):
    channel_id: str
    title: str
    author: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    rss_url: Optional[str] = None
    youtube_channel_url: Optional[str] = None
    follower_count: int = 0
    episode_count: int = 0
    category: Optional[str] = None
    language: str = "en"
    source: Optional[str] = None


class PodcastEpisodeItem(BaseModel):
    id: str
    channel_id: str
    channel_title: Optional[str] = None
    channel_image_url: Optional[str] = None
    channel_author: Optional[str] = None
    title: str
    published_at: str
    audio_url: Optional[str] = None
    youtube_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    duration_formatted: Optional[str] = None
    language: str = "en"
    host: Optional[str] = None
    guest: Optional[str] = None
    tldr_zh_tw: Optional[str] = None
    key_takeaways: List[str] = Field(default_factory=list)
    topic_tags: List[str] = Field(default_factory=list)
    tickers: List[str] = Field(default_factory=list)
    ticker_prices: List[PodcastTickerPrice] = Field(default_factory=list)
    importance_score: Optional[float] = None
    importance_level: Optional[str] = None
    importance_badge: Optional[str] = None
    market_catalyst: Optional[str] = None
    decision_slot: Optional[str] = None
    ticker_sentiments: Optional[Dict[str, Any]] = None
    status: str = "extracted"


class PodcastTopicItem(BaseModel):
    topic: str
    count: int


class PodcastEpisodesResponse(BaseModel):
    success: bool = True
    total_count: int
    data: List[PodcastEpisodeItem] = Field(default_factory=list)


class PodcastChannelsResponse(BaseModel):
    success: bool = True
    total_count: int
    data: List[PodcastChannelItem] = Field(default_factory=list)


class PodcastTopicsResponse(BaseModel):
    success: bool = True
    total_count: int
    data: List[PodcastTopicItem] = Field(default_factory=list)


class PodcastSyncResponse(BaseModel):
    success: bool = True
    message: str
    timestamp: str


