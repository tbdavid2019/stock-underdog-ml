"""
api/routes/market.py - 台股全市場行情與三大法人籌碼 Endpoints
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel
from data.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/market", tags=["Market Data & Institutional Flows"])


def get_duckdb():
    return DuckDBManager()


class InstitutionalFlowItem(BaseModel):
    date: str
    ticker: str
    raw_code: Optional[str] = None
    name: Optional[str] = None
    foreign_net: Optional[int] = 0
    trust_net: Optional[int] = 0
    dealer_net: Optional[int] = 0
    total_net: Optional[int] = 0
    foreign_ratio: Optional[float] = 0.0
    market: Optional[str] = None


class MarketStatsResponse(BaseModel):
    total_bars: int
    total_institutional_records: int
    total_broker_records: int = 0
    total_predictions: int
    latest_market_date: Optional[str] = None
    latest_institutional_date: Optional[str] = None


@router.get("/stats", response_model=MarketStatsResponse, summary="取得時序庫行情與法人籌碼整體統計")
def get_market_statistics(db: DuckDBManager = Depends(get_duckdb)):
    """
    查詢 DuckDB 中台股全市場日 K 棒、三大法人歷史進出表、券商分點主力與預測記錄之數據量。
    """
    try:
        with db._get_connection() as con:
            bars_cnt = con.execute("SELECT COUNT(*) FROM tw_daily_bars").fetchone()[0]
            inst_cnt = con.execute("SELECT COUNT(*) FROM tw_institutional_daily").fetchone()[0]
            
            # 檢查 tw_broker_trades 是否存在
            broker_cnt = 0
            has_broker = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'tw_broker_trades'").fetchone()[0]
            if has_broker:
                broker_cnt = con.execute("SELECT COUNT(*) FROM tw_broker_trades").fetchone()[0]
                
            pred_cnt = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            
            latest_bar_date = con.execute("SELECT MAX(date) FROM tw_daily_bars").fetchone()[0]
            latest_inst_date = con.execute("SELECT MAX(date) FROM tw_institutional_daily").fetchone()[0]

            return MarketStatsResponse(
                total_bars=bars_cnt,
                total_institutional_records=inst_cnt,
                total_broker_records=broker_cnt,
                total_predictions=pred_cnt,
                latest_market_date=str(latest_bar_date) if latest_bar_date else None,
                latest_institutional_date=str(latest_inst_date) if latest_inst_date else None
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢市場統計失敗: {e}")


@router.get("/institutional/top", response_model=List[InstitutionalFlowItem], summary="取得三大法人買賣超排行榜")
def get_top_institutional_flows(
    date: Optional[str] = Query(None, description="指定日期 (YYYY-MM-DD)，預設為最新交易日"),
    order_by: str = Query("total_net", description="排序依據: total_net, foreign_net, trust_net, dealer_net"),
    sort_dir: str = Query("desc", description="排序方向: desc (買超榜), asc (賣超榜)"),
    market: str = Query("ALL", description="市場篩選: ALL, TWSE, TPEX"),
    limit: int = Query(20, ge=1, le=100, description="回傳數量"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    查詢三大法人（外資、投信、自營商）當日買超與賣超排行焦點股。
    """
    valid_cols = {"total_net", "foreign_net", "trust_net", "dealer_net"}
    if order_by not in valid_cols:
        order_by = "total_net"

    sort_order = "ASC" if sort_dir.lower() == "asc" else "DESC"

    try:
        with db._get_connection() as con:
            # 若未提供日期，自動獲取最新日期
            if not date:
                max_date = con.execute("SELECT MAX(date) FROM tw_institutional_daily").fetchone()[0]
                date = str(max_date) if max_date else None

            if not date:
                return []

            where_clauses = ["date = ?"]
            params: List[Any] = [date]

            if market.upper() in ["TWSE", "TPEX"]:
                where_clauses.append("market = ?")
                params.append(market.upper())

            where_sql = " AND ".join(where_clauses)
            sql = f"""
            SELECT 
                date,
                ticker,
                raw_code,
                name,
                foreign_net,
                trust_net,
                dealer_net,
                total_net,
                foreign_ratio,
                market
            FROM tw_institutional_daily
            WHERE {where_sql}
            ORDER BY {order_by} {sort_order}
            LIMIT ?;
            """
            params.append(limit)

            df = con.execute(sql, params).df()
            if df.empty:
                return []

            return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢法人買賣超排行失敗: {e}")


@router.get("/institutional/dates", response_model=List[str], summary="取得法人籌碼所有可用歷史交易日")
def get_available_institutional_dates(
    limit: int = Query(60, ge=1, le=500, description="回傳天數上限"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    回傳 DuckDB 中 tw_institutional_daily 所有歷史交易日期清單 (降冪排序)。
    """
    try:
        with db._get_connection() as con:
            df = con.execute("SELECT DISTINCT date FROM tw_institutional_daily ORDER BY date DESC LIMIT ?", [limit]).df()
            return [str(d) for d in df["date"].tolist()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢日期清單失敗: {e}")


@router.get("/institutional/{ticker}", response_model=List[InstitutionalFlowItem], summary="取得單一個股歷史法人進出時序")
def get_ticker_institutional_history(
    ticker: str,
    limit: int = Query(60, ge=1, le=250, description="歷史天數"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    查詢特定標的（如 2330.TW）過去 60 個交易日之三大法人連續進出時序。
    """
    df = db.get_institutional_flow_for_ticker(ticker=ticker, limit=limit)
    if df.empty:
        return []
    return df.to_dict(orient="records")


@router.get("/broker/summary/{ticker}", summary="取得個股券商分點主力買賣超摘要")
def get_ticker_broker_summary(
    ticker: str,
    days: int = Query(20, ge=1, le=120, description="統計天數區間"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    查詢特定標的在過去 N 個交易日內，買超前 15 名與賣超前 15 名之券商分點合計統計。
    """
    return db.get_broker_top_summary(ticker=ticker, limit_days=days)


@router.get("/broker/trades/{ticker}", summary="取得個股近期券商分點明細")
def get_ticker_broker_trades(
    ticker: str,
    limit: int = Query(60, ge=1, le=200, description="明細筆數"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    查詢特定標的最近的券商分點每日進出明細紀錄。
    """
    df = db.get_broker_trades_for_ticker(ticker=ticker, limit=limit)
    if df.empty:
        return []
    return df.to_dict(orient="records")


@router.get("/broker/trend/{ticker}", summary="取得個股目標券商分點累計買賣超趨勢時序")
def get_ticker_broker_trend(
    ticker: str,
    days: int = Query(30, ge=5, le=120, description="時序天數"),
    top_n: int = Query(8, ge=2, le=20, description="取買賣超最顯著之分點數量"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    計算特定標的各關鍵券商分點在歷史區間內的每日「累計買賣超 (張)」時間序列，用於繪製分點走勢圖。
    """
    clean_code = ticker.split(".")[0]
    try:
        with db._get_connection() as con:
            # 1. 取得最近 N 天日期
            dates_df = con.execute("""
                SELECT DISTINCT date 
                FROM tw_broker_trades 
                WHERE (ticker = ? OR raw_code = ? OR ticker LIKE ?)
                ORDER BY date DESC 
                LIMIT ?;
            """, [ticker, clean_code, f"{clean_code}.%", days]).df()
            
            if dates_df.empty:
                return {"ticker": ticker, "dates": [], "series": []}

            dates = sorted(dates_df["date"].tolist())
            min_date = dates[0]

            # 2. 找出這段時間內絕對淨量最大的 Top N 券商分點
            top_brokers_df = con.execute("""
                SELECT broker_name, ABS(SUM(net_vol)) as abs_net
                FROM tw_broker_trades
                WHERE (ticker = ? OR raw_code = ? OR ticker LIKE ?) AND date >= ?
                GROUP BY broker_name
                ORDER BY abs_net DESC
                LIMIT ?;
            """, [ticker, clean_code, f"{clean_code}.%", min_date, top_n]).df()

            if top_brokers_df.empty:
                return {"ticker": ticker, "dates": dates, "series": []}

            top_broker_names = top_brokers_df["broker_name"].tolist()

            # 3. 取得所有相關明細
            placeholders = ",".join(["?"] * len(top_broker_names))
            sql = f"""
                SELECT date, broker_name, net_vol
                FROM tw_broker_trades
                WHERE (ticker = ? OR raw_code = ? OR ticker LIKE ?) 
                  AND date >= ?
                  AND broker_name IN ({placeholders})
                ORDER BY date ASC;
            """
            params = [ticker, clean_code, f"{clean_code}.%", min_date] + top_broker_names
            trades_df = con.execute(sql, params).df()

            # 4. 構建每個分點在各日期的累計淨量
            series_result = []
            for bname in top_broker_names:
                b_df = trades_df[trades_df["broker_name"] == bname]
                vol_map = dict(zip(b_df["date"], b_df["net_vol"]))
                
                cum = 0
                cum_data = []
                for d in dates:
                    net = vol_map.get(d, 0)
                    cum += net
                    cum_data.append(cum)

                series_result.append({
                    "broker_name": bname,
                    "final_net": cum,
                    "data": cum_data
                })

            # 依最終累計淨量降冪排序
            series_result.sort(key=lambda x: abs(x["final_net"]), reverse=True)

            return {
                "ticker": ticker,
                "dates": dates,
                "series": series_result
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"計算券商趨勢失敗: {e}")


class CompanyProfileResponse(BaseModel):
    success: bool = True
    ticker: str
    raw_code: str
    name: str
    industry: Optional[str] = "綜合產業"
    business_summary: Optional[str] = ""
    chairman: Optional[str] = ""
    market_cap: Optional[str] = ""
    established: Optional[str] = ""
    recent_news: List[Dict[str, str]] = []
    links: Dict[str, str] = {}
    source: str = "2md"


@router.get("/company-profile/{ticker}", response_model=CompanyProfileResponse, summary="取得個股公司簡介與即時新聞 (2md / Yahoo)")
@router.get("/company-profile", response_model=CompanyProfileResponse, summary="取得個股公司簡介與即時新聞 (2md / Yahoo)")
def get_company_profile_endpoint(
    ticker: Optional[str] = None,
    query: Optional[str] = Query(None, description="股票代號或名稱，如 9945.TW, 2330, 台積電, AAPL"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    透過 2md.aiurl.tw 與 Yahoo 股市實時提取公司繁體中文簡介、核心營運業務與最新新聞。
    """
    from data.company_profile import CompanyProfileService
    target = ticker or query
    if not target:
        raise HTTPException(status_code=400, detail="請提供股票代號或名稱")

    try:
        profile = CompanyProfileService.get_company_profile(target, db=db)
        return CompanyProfileResponse(
            success=True,
            ticker=profile["ticker"],
            raw_code=profile["raw_code"],
            name=profile["name"],
            industry=profile.get("industry", "綜合產業"),
            business_summary=profile.get("business_summary", ""),
            chairman=profile.get("chairman", ""),
            market_cap=profile.get("market_cap", ""),
            established=profile.get("established", ""),
            recent_news=profile.get("recent_news", []),
            links=profile.get("links", {}),
            source=profile.get("source", "2md")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得公司資訊失敗: {e}")


class BatchCompanyProfilesRequest(BaseModel):
    tickers: List[str]


@router.post("/company-profiles/batch", summary="並發批次取得多支股票之公司簡介與即時新聞")
@router.get("/company-profiles", summary="並發批次取得多支股票之公司簡介與即時新聞")
def get_company_profiles_batch_endpoint(
    tickers: Optional[str] = Query(None, description="逗號分隔的股票代號，如 2330.TW,9945.TW,MDT,NEM"),
    payload: Optional[BatchCompanyProfilesRequest] = None,
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    並發非同步取得多支股票之公司簡介、營運摘要與最新新聞，專為前端背景預載設計。
    """
    from data.company_profile import CompanyProfileService
    ticker_list = []
    if payload and payload.tickers:
        ticker_list.extend(payload.tickers)
    if tickers:
        ticker_list.extend([t.strip() for t in tickers.split(",") if t.strip()])

    if not ticker_list:
        raise HTTPException(status_code=400, detail="請提供 tickers 列表")

    # 限制批次上限 80 支
    ticker_list = ticker_list[:80]
    try:
        results = CompanyProfileService.get_company_profiles_batch(ticker_list, db=db)
        return {
            "success": True,
            "count": len(results),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批次取得公司資訊失敗: {e}")


