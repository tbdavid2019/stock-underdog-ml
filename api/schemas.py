"""
api/schemas.py - Pydantic Data Models for Quantitative API & MCP Tools
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


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
    tags: Optional[str] = Field(None, description="綜合標籤 (如 🏆三重共振 | 土洋合買)")


class PredictionResponse(BaseModel):
    success: bool = True
    batch_date: Optional[str] = Field(None, description="最新批次分析日期 (YYYY-MM-DD)")
    latest_batch_timestamp: Optional[str] = Field(None, description="最新批次完整時間戳記")
    is_stale: bool = Field(False, description="整個批次資料是否已逾期 (> 48h)")
    count: int = Field(..., description="符合條件的記錄總數")
    data: List[StockPredictionItem] = Field(default_factory=list, description="預測結果清單")


class MacroRegimeResponse(BaseModel):
    success: bool = True
    regime_name: str = Field(..., description="宏觀環境 (全面多頭, 多頭回調, 避險防禦, 極度恐慌)")
    exposure: float = Field(..., description="建議投資曝險比例 (0.0 ~ 1.0)")
    vix: Optional[float] = Field(None, description="VIX 恐慌指數")
    spy_above_ma60: Optional[bool] = Field(None, description="S&P500 是否站穩季線")
    sox_above_ma60: Optional[bool] = Field(None, description="費城半導體是否站穩季線")
    warnings: List[str] = Field(default_factory=list, description="風控警示清單")
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
    version: str = "2.2.0"
    duckdb_records: int
