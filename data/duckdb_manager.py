"""
data/duckdb_manager.py - 本地 DuckDB 量化時序資料庫管理器 (Local DuckDB Manager)

提供單一檔案嵌入式高效能列式儲存，支援預測數據持久化、零拷貝 Pandas 查詢與 Parquet 冷備份導出。
"""

import os
import logging
import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import duckdb

from core.config import config
from data.macro import MacroState

logger = logging.getLogger("stock_app.duckdb")


class DuckDBManager:
    """DuckDB 本地量化時序資料庫管理器"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.db.DUCKDB_PATH
        self.enabled = config.db.ENABLE_DUCKDB
        self._ensure_storage_dir()
        if self.enabled:
            self._init_schema()

    def _ensure_storage_dir(self):
        storage_dir = os.path.dirname(self.db_path)
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """建立或取得 DuckDB 連線"""
        return duckdb.connect(self.db_path)

    def _init_schema(self):
        """初始化表格結構 (若不存在則建立)"""
        try:
            with self._get_connection() as con:
                con.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    index_name VARCHAR,
                    model_name VARCHAR,
                    strategy_type VARCHAR,
                    ticker VARCHAR,
                    current_price DOUBLE,
                    predicted_price DOUBLE,
                    potential DOUBLE,
                    ma5 DOUBLE,
                    ma10 DOUBLE,
                    ma60 DOUBLE,
                    ma120 DOUBLE,
                    ma250 DOUBLE,
                    pullback_type VARCHAR,
                    pe DOUBLE,
                    pb DOUBLE,
                    forward_pe DOUBLE,
                    ev_ebitda DOUBLE,
                    period VARCHAR,
                    timestamp VARCHAR,
                    macro_regime VARCHAR,
                    trust_net_5d BIGINT,
                    foreign_net_5d BIGINT,
                    tags VARCHAR
                );
                """)
                con.execute("""
                CREATE TABLE IF NOT EXISTS macro_regimes (
                    date VARCHAR,
                    regime_name VARCHAR,
                    exposure DOUBLE,
                    vix DOUBLE,
                    spy_above_ma60 BOOLEAN,
                    sox_above_ma60 BOOLEAN,
                    timestamp VARCHAR
                );
                """)
                con.execute("""
                CREATE TABLE IF NOT EXISTS tw_daily_bars (
                    date VARCHAR,
                    ticker VARCHAR,
                    raw_code VARCHAR,
                    name VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    trade_value DOUBLE,
                    transaction_count BIGINT,
                    market VARCHAR,
                    created_at VARCHAR
                );
                """)
                con.execute("""
                CREATE TABLE IF NOT EXISTS tw_institutional_daily (
                    date VARCHAR,
                    ticker VARCHAR,
                    raw_code VARCHAR,
                    name VARCHAR,
                    foreign_net BIGINT,
                    trust_net BIGINT,
                    dealer_net BIGINT,
                    total_net BIGINT,
                    foreign_ratio DOUBLE,
                    total_shares BIGINT,
                    foreign_shares BIGINT,
                    market VARCHAR,
                    created_at VARCHAR
                );
                """)
                con.execute("""
                CREATE TABLE IF NOT EXISTS tw_broker_trades (
                    date VARCHAR,
                    ticker VARCHAR,
                    raw_code VARCHAR,
                    broker_name VARCHAR,
                    broker_id VARCHAR,
                    buy_vol BIGINT,
                    sell_vol BIGINT,
                    net_vol BIGINT,
                    pct DOUBLE,
                    rank INTEGER,
                    side VARCHAR,
                    created_at VARCHAR
                );
                """)
            logger.info(f"✅ DuckDB 資料庫初始化成功: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ DuckDB 初始化失敗: {e}")

    def save_predictions_batch(self, records: List[Dict[str, Any]]) -> int:
        """批次寫入預測與策略記錄"""
        if not self.enabled or not records:
            return 0

        df = pd.DataFrame(records)
        # 確保所有必要欄位存在
        expected_cols = [
            "index_name", "model_name", "strategy_type", "ticker",
            "current_price", "predicted_price", "potential",
            "ma5", "ma10", "ma60", "ma120", "ma250",
            "pullback_type", "pe", "pb", "forward_pe", "ev_ebitda",
            "period", "timestamp", "macro_regime",
            "trust_net_5d", "foreign_net_5d", "tags"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        # 格式化 tags 欄位為字串
        if "tags" in df.columns:
            df["tags"] = df["tags"].apply(lambda x: " | ".join(x) if isinstance(x, list) else str(x) if x is not None else None)

        try:
            with self._get_connection() as con:
                con.register("temp_batch_df", df[expected_cols])
                con.execute("INSERT INTO predictions SELECT * FROM temp_batch_df")
            logger.info(f"💾 成功寫入 {len(df)} 筆量化數據至 DuckDB ({self.db_path})")
            return len(df)
        except Exception as e:
            logger.error(f"❌ DuckDB 批次寫入失敗: {e}", exc_info=True)
            return 0

    def save_dual_strategy_results(
        self, 
        index_name: str, 
        results: dict, 
        period: str = "6mo",
        macro_state: Optional[MacroState] = None
    ) -> int:
        """
        保存雙軌策略與多維量化結果至 DuckDB (與 Supabase 同步雙寫)
        """
        if not self.enabled:
            return 0

        timestamp = datetime.datetime.now().isoformat()
        macro_regime_str = macro_state.regime_name if macro_state else None
        all_data = []

        # 提取候選標的綜合評估與法人籌碼
        candidates_map = results.get("candidates_map", {})
        inst_summaries = results.get("institutional_summaries", {})

        # 1. 玄鐵策略結果
        xuantie_df = results.get("xuantie_results", pd.DataFrame())
        if isinstance(xuantie_df, pd.DataFrame) and not xuantie_df.empty:
            for idx, row in xuantie_df.iterrows():
                t = row["ticker"]
                cand = candidates_map.get(t, {})
                inst = inst_summaries.get(t, {})
                cand_tags = cand.get("tags")
                tags_str = " | ".join(cand_tags) if cand_tags else "玄鐵買點"

                all_data.append({
                    "index_name": index_name,
                    "model_name": "玄鐵重劍",
                    "strategy_type": "玄鐵重劍",
                    "ticker": t,
                    "current_price": float(row["current_price"]),
                    "predicted_price": None,
                    "potential": None,
                    "ma5": float(row.get("ma5")) if row.get("ma5") and not pd.isna(row.get("ma5")) else None,
                    "ma10": float(row.get("ma10")) if row.get("ma10") and not pd.isna(row.get("ma10")) else None,
                    "ma60": float(row.get("ma60")) if row.get("ma60") and not pd.isna(row.get("ma60")) else None,
                    "ma120": float(row.get("ma120")) if row.get("ma120") and not pd.isna(row.get("ma120")) else None,
                    "ma250": float(row.get("ma250")) if row.get("ma250") and not pd.isna(row.get("ma250")) else None,
                    "pullback_type": row.get("pullback_type"),
                    "pe": float(row.get("pe")) if row.get("pe") and not pd.isna(row.get("pe")) else None,
                    "pb": float(row.get("pb")) if row.get("pb") and not pd.isna(row.get("pb")) else None,
                    "forward_pe": float(row.get("forward_pe")) if row.get("forward_pe") and not pd.isna(row.get("forward_pe")) else None,
                    "ev_ebitda": float(row.get("ev_ebitda")) if row.get("ev_ebitda") and not pd.isna(row.get("ev_ebitda")) else None,
                    "period": period,
                    "timestamp": timestamp,
                    "macro_regime": macro_regime_str,
                    "trust_net_5d": inst.get("trust_net_5d"),
                    "foreign_net_5d": inst.get("foreign_net_5d"),
                    "tags": tags_str
                })

        # 2. LSTM 預測結果
        lstm_results = results.get("lstm_results", [])
        for r in lstm_results:
            t = r["ticker"]
            cand = candidates_map.get(t, {})
            inst = inst_summaries.get(t, {})
            cand_tags = cand.get("tags")
            default_lstm_tag = "LSTM看漲" if float(r["potential"]) > 0 else "LSTM看跌"
            tags_str = " | ".join(cand_tags) if cand_tags else default_lstm_tag

            all_data.append({
                "index_name": index_name,
                "model_name": "LSTM",
                "strategy_type": "LSTM預測",
                "ticker": t,
                "current_price": float(r["current_price"]),
                "predicted_price": float(r["predicted_price"]),
                "potential": float(r["potential"]),
                "ma5": None, "ma10": None, "ma60": None, "ma120": None, "ma250": None,
                "pullback_type": None,
                "pe": float(r.get("pe")) if r.get("pe") and not pd.isna(r.get("pe")) else None,
                "pb": float(r.get("pb")) if r.get("pb") and not pd.isna(r.get("pb")) else None,
                "forward_pe": float(r.get("forward_pe")) if r.get("forward_pe") and not pd.isna(r.get("forward_pe")) else None,
                "ev_ebitda": float(r.get("ev_ebitda")) if r.get("ev_ebitda") and not pd.isna(r.get("ev_ebitda")) else None,
                "period": period,
                "timestamp": timestamp,
                "macro_regime": macro_regime_str,
                "trust_net_5d": inst.get("trust_net_5d"),
                "foreign_net_5d": inst.get("foreign_net_5d"),
                "tags": tags_str
            })

        # 3. 雙重/三重符合結果
        overlap_df = results.get("overlap_results", pd.DataFrame())
        if isinstance(overlap_df, pd.DataFrame) and not overlap_df.empty:
            for idx, row in overlap_df.iterrows():
                t = row["ticker"]
                cand = candidates_map.get(t, {})
                inst = inst_summaries.get(t, {})
                cand_tags = cand.get("tags")
                tags_str = " | ".join(cand_tags) if cand_tags else "🏆三重共振"

                all_data.append({
                    "index_name": index_name,
                    "model_name": "多維共振",
                    "strategy_type": "多維共振",
                    "ticker": t,
                    "current_price": float(row["current_price"]),
                    "predicted_price": float(row["predicted_price"]),
                    "potential": float(row["lstm_potential"]),
                    "ma5": float(row.get("ma5")) if row.get("ma5") and not pd.isna(row.get("ma5")) else None,
                    "ma10": float(row.get("ma10")) if row.get("ma10") and not pd.isna(row.get("ma10")) else None,
                    "ma60": float(row.get("ma60")) if row.get("ma60") and not pd.isna(row.get("ma60")) else None,
                    "ma120": float(row.get("ma120")) if row.get("ma120") and not pd.isna(row.get("ma120")) else None,
                    "ma250": float(row.get("ma250")) if row.get("ma250") and not pd.isna(row.get("ma250")) else None,
                    "pullback_type": row.get("pullback_type"),
                    "pe": float(row.get("pe")) if row.get("pe") and not pd.isna(row.get("pe")) else None,
                    "pb": float(row.get("pb")) if row.get("pb") and not pd.isna(row.get("pb")) else None,
                    "forward_pe": float(row.get("forward_pe")) if row.get("forward_pe") and not pd.isna(row.get("forward_pe")) else None,
                    "ev_ebitda": float(row.get("ev_ebitda")) if row.get("ev_ebitda") and not pd.isna(row.get("ev_ebitda")) else None,
                    "period": period,
                    "timestamp": timestamp,
                    "macro_regime": macro_regime_str,
                    "trust_net_5d": inst.get("trust_net_5d"),
                    "foreign_net_5d": inst.get("foreign_net_5d"),
                    "tags": tags_str
                })

        return self.save_predictions_batch(all_data)

    def query(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """執行 SQL 查詢並返回 Pandas DataFrame"""
        with self._get_connection() as con:
            if params:
                return con.execute(sql, params).df()
            return con.execute(sql).df()

    def get_row_count(self, table_name: str = "predictions") -> int:
        """取得指定表格的總記錄數 (排除測試假資料)"""
        try:
            with self._get_connection() as con:
                filter_sql = "WHERE index_name NOT LIKE '%TEST%' AND index_name NOT LIKE '%DEBUG%' AND index_name NOT LIKE '%SERVICE_KEY%' AND model_name NOT LIKE '%DEBUG%'"
                res = con.execute(f"SELECT COUNT(*) FROM {table_name} {filter_sql}").fetchone()
                return res[0] if res else 0
        except Exception:
            return 0

    def clean_test_data(self) -> int:
        """清理歷史混入的測試與除錯假數據 (TEST, DEBUG, SERVICE_KEY 等)"""
        try:
            with self._get_connection() as con:
                res = con.execute("""
                    DELETE FROM predictions 
                    WHERE index_name LIKE '%TEST%' 
                       OR index_name LIKE '%DEBUG%' 
                       OR index_name LIKE '%SERVICE_KEY%'
                       OR model_name LIKE '%DEBUG%'
                       OR ticker LIKE 'TEST%';
                """).fetchone()
                logger.info("🧹 已成功清理非生產測試資料")
                return res[0] if res else 0
        except Exception as e:
            logger.error(f"❌ 清理測試資料失敗: {e}")
            return 0

    def export_to_parquet(self, output_path: str, table_name: str = "predictions") -> str:
        """將資料庫表格導出為極度壓縮的 Parquet 檔案"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with self._get_connection() as con:
            con.execute(f"COPY {table_name} TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD);")
        logger.info(f"📦 已成功導出 {table_name} 至 Parquet: {output_path}")
        return output_path

    @staticmethod
    def _clean_df_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """將 DataFrame 轉為乾淨的字典清單，將 NaN / NaT / inf 轉為 None"""
        if df.empty:
            return []
        import math
        records = []
        for row in df.to_dict(orient="records"):
            clean_row = {}
            for k, v in row.items():
                if v is None:
                    clean_row[k] = None
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean_row[k] = None
                elif pd.isna(v) or str(v).lower() in ("nan", "none", "nat"):
                    clean_row[k] = None
                else:
                    clean_row[k] = v
            records.append(clean_row)
        return records

    # =========================================================================
    # REST API & MCP 專用高吞吐量化分析查詢接口
    # =========================================================================

    def get_latest_predictions(
        self, 
        index_name: Optional[str] = None, 
        model_name: Optional[str] = None, 
        limit: int = 50, 
        offset: int = 0,
        batch_only: bool = True
    ) -> Dict[str, Any]:
        """
        取得量化預測記錄。
        預設 (batch_only=True) 僅鎖定最新執行批次（例如 24 小時內或最新批次日期），
        並計算 analysis_date, data_as_of, age_hours, is_stale。
        """
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)

        where_clauses = [
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'",
            "model_name NOT LIKE '%DEBUG%'"
        ]
        params = []
        if index_name:
            where_clauses.append("index_name = ?")
            params.append(index_name)
        if model_name:
            where_clauses.append("model_name = ?")
            params.append(model_name)

        # 1. 取得最新批次的時間戳記
        latest_ts_sql = f"""
            SELECT MAX(timestamp) 
            FROM predictions 
            WHERE {' AND '.join(where_clauses)}
        """
        latest_ts = None
        with self._get_connection() as con:
            res = con.execute(latest_ts_sql, params).fetchone()
            if res and res[0]:
                latest_ts = str(res[0])

        if not latest_ts:
            return {
                "batch_date": None,
                "latest_batch_timestamp": None,
                "is_stale": False,
                "records": []
            }

        try:
            ts_clean = latest_ts.replace("Z", "+00:00")
            batch_dt = datetime.datetime.fromisoformat(ts_clean)
            if batch_dt.tzinfo is None:
                batch_dt = batch_dt.replace(tzinfo=datetime.timezone.utc)
            age_hours = round((now - batch_dt).total_seconds() / 3600.0, 1)
            is_stale = age_hours > 48.0
            batch_date = batch_dt.strftime("%Y-%m-%d")
        except Exception:
            age_hours = 0.0
            is_stale = False
            batch_date = str(latest_ts)[:10]

        if batch_only:
            if index_name:
                where_clauses.append("TRY_CAST(timestamp AS TIMESTAMP) >= (TRY_CAST(? AS TIMESTAMP) - INTERVAL 12 HOUR)")
                params.append(latest_ts)
            else:
                where_clauses.append("TRY_CAST(timestamp AS TIMESTAMP) >= (SELECT MAX(TRY_CAST(p2.timestamp AS TIMESTAMP)) - INTERVAL 12 HOUR FROM predictions p2 WHERE p2.index_name = predictions.index_name)")

        where_sql = f"WHERE {' AND '.join(where_clauses)}"
        sql = f"""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY ticker, strategy_type ORDER BY timestamp DESC) as rn
            FROM predictions
            {where_sql}
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
        ORDER BY potential DESC NULLS LAST, ticker ASC
        LIMIT ? OFFSET ?;
        """
        params.extend([limit, offset])
        df = self.query(sql, params)
        records = self._clean_df_records(df)

        for r in records:
            r_ts = r.get("timestamp")
            if r_ts:
                r["data_as_of"] = str(r_ts)
                r["analysis_date"] = str(r_ts)[:10]
                try:
                    r_dt = datetime.datetime.fromisoformat(str(r_ts).replace("Z", "+00:00"))
                    if r_dt.tzinfo is None:
                        r_dt = r_dt.replace(tzinfo=datetime.timezone.utc)
                    r_age = round((now - r_dt).total_seconds() / 3600.0, 1)
                    r["age_hours"] = r_age
                    r["is_stale"] = r_age > 48.0
                except Exception:
                    r["age_hours"] = age_hours
                    r["is_stale"] = is_stale
            else:
                r["analysis_date"] = batch_date
                r["data_as_of"] = latest_ts
                r["age_hours"] = age_hours
                r["is_stale"] = is_stale

        return {
            "batch_date": batch_date,
            "latest_batch_timestamp": latest_ts,
            "is_stale": is_stale,
            "records": records
        }

    def get_resonance_candidates(self, index_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """取得多策略重合 / 雙重符合 / 🏆 三重共振股票"""
        where_clauses = [
            "(model_name = '多維共振' OR strategy_type = '多維共振' OR tags LIKE '%共振%' OR tags LIKE '%雙重符合%' OR tags LIKE '%3重符合%')",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        if index_name:
            where_clauses.append("index_name = ?")
            params.append(index_name)

        where_sql = f"WHERE {' AND '.join(where_clauses)}"
        sql = f"""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) as rn
            FROM predictions
            {where_sql}
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
        ORDER BY potential DESC NULLS LAST, ticker ASC
        LIMIT ?;
        """
        params.append(limit)
        df = self.query(sql, params)
        return self._clean_df_records(df)

    def get_xuantie_candidates(
        self, 
        index_name: Optional[str] = None, 
        pullback_type: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """取得玄鐵重劍技術買點 (MA60/120 回調) 股票"""
        where_clauses = [
            "(model_name = '玄鐵重劍' OR strategy_type = '玄鐵重劍' OR pullback_type IS NOT NULL)",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        if index_name:
            where_clauses.append("index_name = ?")
            params.append(index_name)
        if pullback_type:
            where_clauses.append("pullback_type LIKE ?")
            params.append(f"%{pullback_type}%")

        where_sql = f"WHERE {' AND '.join(where_clauses)}"
        sql = f"""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) as rn
            FROM predictions
            {where_sql}
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
        ORDER BY pe ASC NULLS LAST, ticker ASC
        LIMIT ?;
        """
        params.append(limit)
        df = self.query(sql, params)
        return self._clean_df_records(df)

    def get_top_bullish(self, index_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """取得 LSTM 預測漲幅最大 TOP N 標的"""
        where_clauses = [
            "potential IS NOT NULL", 
            "model_name = 'LSTM'",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        if index_name:
            where_clauses.append("index_name = ?")
            params.append(index_name)

        where_sql = f"WHERE {' AND '.join(where_clauses)}"
        sql = f"""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) as rn
            FROM predictions
            {where_sql}
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
        ORDER BY potential DESC
        LIMIT ?;
        """
        params.append(limit)
        df = self.query(sql, params)
        return self._clean_df_records(df)

    def get_top_bearish(self, index_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """取得 LSTM 預測跌幅最大 / 潛在做空 TOP N 標的"""
        where_clauses = [
            "potential IS NOT NULL", 
            "model_name = 'LSTM'",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        if index_name:
            where_clauses.append("index_name = ?")
            params.append(index_name)

        where_sql = f"WHERE {' AND '.join(where_clauses)}"
        sql = f"""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) as rn
            FROM predictions
            {where_sql}
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
        ORDER BY potential ASC
        LIMIT ?;
        """
        params.append(limit)
        df = self.query(sql, params)
        return self._clean_df_records(df)

    def get_ticker_history(self, ticker: str, limit: int = 30) -> List[Dict[str, Any]]:
        """取得特定股票代碼之歷史預測軌跡"""
        sql = """
        SELECT *
        FROM predictions
        WHERE ticker = ?
          AND index_name NOT LIKE '%TEST%'
          AND index_name NOT LIKE '%DEBUG%'
          AND index_name NOT LIKE '%SERVICE_KEY%'
        ORDER BY timestamp DESC
        LIMIT ?;
        """
        df = self.query(sql, [ticker, limit])
        return self._clean_df_records(df)

    def get_db_stats(self) -> Dict[str, Any]:
        """取得 DuckDB 全局時序庫統計資訊 (已排除內部路徑與測試資料)"""
        try:
            with self._get_connection() as con:
                filter_sql = "WHERE index_name NOT LIKE '%TEST%' AND index_name NOT LIKE '%DEBUG%' AND index_name NOT LIKE '%SERVICE_KEY%' AND model_name NOT LIKE '%DEBUG%'"
                row_count = con.execute(f"SELECT COUNT(*) FROM predictions {filter_sql}").fetchone()[0]
                distinct_tickers = con.execute(f"SELECT COUNT(DISTINCT ticker) FROM predictions {filter_sql}").fetchone()[0]
                min_time = con.execute(f"SELECT MIN(timestamp) FROM predictions {filter_sql}").fetchone()[0]
                max_time = con.execute(f"SELECT MAX(timestamp) FROM predictions {filter_sql}").fetchone()[0]
                indices = [r[0] for r in con.execute(f"SELECT DISTINCT index_name FROM predictions {filter_sql} AND index_name IS NOT NULL ORDER BY index_name").fetchall()]
                models = [r[0] for r in con.execute(f"SELECT DISTINCT model_name FROM predictions {filter_sql} AND model_name IS NOT NULL ORDER BY model_name").fetchall()]
                
                return {
                    "total_records": row_count,
                    "distinct_tickers": distinct_tickers,
                    "earliest_timestamp": str(min_time) if min_time else None,
                    "latest_timestamp": str(max_time) if max_time else None,
                    "indices": indices,
                    "models": models
                }
        except Exception as e:
            logger.error(f"❌ 查詢 DB 統計失敗: {e}")
            return {"total_records": 0, "error": str(e)}

    def get_latest_macro_regime(self) -> Optional[Dict[str, Any]]:
        """取得最新一筆宏觀市場狀態"""
        try:
            with self._get_connection() as con:
                res = con.execute("SELECT * FROM macro_regimes ORDER BY timestamp DESC LIMIT 1").df()
                if not res.empty:
                    clean = self._clean_df_records(res)
                    return clean[0] if clean else None
        except Exception:
            pass
        return None

    def save_daily_bars_batch(self, records: List[Dict[str, Any]]) -> int:
        """
        批次寫入台股全市場日 K 棒數據，並自動去重 (Upsert on date + ticker)
        """
        if not self.enabled or not records:
            return 0

        df = pd.DataFrame(records)
        expected_cols = [
            "date", "ticker", "raw_code", "name",
            "open", "high", "low", "close", "volume",
            "trade_value", "transaction_count", "market", "created_at"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        try:
            with self._get_connection() as con:
                con.register("temp_daily_bars_df", df[expected_cols])
                # 先刪除當日相同代碼之重複資料以支援冪等覆蓋 (Upsert)
                con.execute("""
                DELETE FROM tw_daily_bars
                WHERE (date, ticker) IN (
                    SELECT date, ticker FROM temp_daily_bars_df
                );
                """)
                con.execute("INSERT INTO tw_daily_bars SELECT * FROM temp_daily_bars_df;")
            logger.info(f"💾 成功寫入/更新 {len(df)} 筆全市場日 K 棒至 DuckDB")
            return len(df)
        except Exception as e:
            logger.error(f"❌ 寫入 tw_daily_bars 失敗: {e}", exc_info=True)
            return 0

    def get_daily_bars_for_ticker(self, ticker: str, limit: int = 120) -> pd.DataFrame:
        """
        從 DuckDB 讀取特定個股的歷史日 K 棒數據，格式對齊 yfinance OHLCV
        """
        if not self.enabled:
            return pd.DataFrame()

        clean_code = ticker.split(".")[0]
        sql = """
        SELECT 
            date AS Date,
            open AS Open,
            high AS High,
            low AS Low,
            close AS Close,
            volume AS Volume
        FROM tw_daily_bars
        WHERE (ticker = ? OR raw_code = ? OR ticker LIKE ?)
          AND close > 0
        ORDER BY date ASC
        LIMIT ?;
        """
        try:
            with self._get_connection() as con:
                df = con.execute(sql, [ticker, clean_code, f"{clean_code}.%", limit]).df()
                if not df.empty:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                return df
        except Exception as e:
            logger.warning(f"⚠️ 從 DuckDB 讀取 {ticker} 日 K 失敗: {e}")
            return pd.DataFrame()

    def save_institutional_flows_batch(self, records: List[Dict[str, Any]]) -> int:
        """
        批次寫入三大法人每日進出與持股數據，並自動去重 (Upsert on date + ticker)
        """
        if not self.enabled or not records:
            return 0

        df = pd.DataFrame(records)
        expected_cols = [
            "date", "ticker", "raw_code", "name",
            "foreign_net", "trust_net", "dealer_net", "total_net",
            "foreign_ratio", "total_shares", "foreign_shares",
            "market", "created_at"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        try:
            with self._get_connection() as con:
                con.register("temp_inst_df", df[expected_cols])
                con.execute("""
                DELETE FROM tw_institutional_daily
                WHERE (date, ticker) IN (
                    SELECT date, ticker FROM temp_inst_df
                );
                """)
                con.execute("INSERT INTO tw_institutional_daily SELECT * FROM temp_inst_df;")
            logger.info(f"💾 成功寫入/更新 {len(df)} 筆法人籌碼數據至 DuckDB")
            return len(df)
        except Exception as e:
            logger.error(f"❌ 寫入 tw_institutional_daily 失敗: {e}", exc_info=True)
            return 0

    def get_institutional_flow_for_ticker(self, ticker: str, limit: int = 60) -> pd.DataFrame:
        """
        取得特定標的之歷史法人籌碼進出時序
        """
        if not self.enabled:
            return pd.DataFrame()

        clean_code = ticker.split(".")[0]
        sql = """
        SELECT 
            date,
            ticker,
            name,
            foreign_net,
            trust_net,
            dealer_net,
            total_net,
            foreign_ratio
        FROM tw_institutional_daily
        WHERE (ticker = ? OR raw_code = ? OR ticker LIKE ?)
        ORDER BY date DESC
        LIMIT ?;
        """
        try:
            with self._get_connection() as con:
                df = con.execute(sql, [ticker, clean_code, f"{clean_code}.%", limit]).df()
                return df
        except Exception as e:
            logger.warning(f"⚠️ 從 DuckDB 讀取 {ticker} 法人時序失敗: {e}")
            return pd.DataFrame()

    def save_broker_trades_batch(self, records: List[Dict[str, Any]]) -> int:
        """批次寫入券商分點進出資料 (tw_broker_trades)"""
        if not self.enabled or not records:
            return 0

        df = pd.DataFrame(records)
        expected_cols = [
            "date", "ticker", "raw_code", "broker_name", "broker_id",
            "buy_vol", "sell_vol", "net_vol", "pct", "rank", "side", "created_at"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        try:
            with self._get_connection() as con:
                con.register("temp_broker_df", df[expected_cols])
                con.execute("""
                DELETE FROM tw_broker_trades
                WHERE (date, raw_code, broker_id, side) IN (
                    SELECT date, raw_code, broker_id, side FROM temp_broker_df
                );
                """)
                con.execute("INSERT INTO tw_broker_trades SELECT * FROM temp_broker_df;")
            logger.info(f"💾 成功寫入/更新 {len(df)} 筆券商分點資料至 DuckDB")
            return len(df)
        except Exception as e:
            logger.error(f"❌ 寫入 tw_broker_trades 失敗: {e}", exc_info=True)
            return 0

    def get_broker_trades_for_ticker(self, ticker: str, limit: int = 100) -> pd.DataFrame:
        """
        取得特定標的最近的券商分點進出明細
        """
        if not self.enabled:
            return pd.DataFrame()

        clean_code = ticker.split(".")[0]
        sql = """
        SELECT 
            date,
            ticker,
            raw_code,
            broker_name,
            broker_id,
            buy_vol,
            sell_vol,
            net_vol,
            pct,
            rank,
            side
        FROM tw_broker_trades
        WHERE (ticker = ? OR raw_code = ? OR ticker LIKE ?)
        ORDER BY date DESC, rank ASC
        LIMIT ?;
        """
        try:
            with self._get_connection() as con:
                df = con.execute(sql, [ticker, clean_code, f"{clean_code}.%", limit]).df()
                return df
        except Exception as e:
            logger.warning(f"⚠️ 從 DuckDB 讀取 {ticker} 券商分點失敗: {e}")
            return pd.DataFrame()

    def get_broker_top_summary(self, ticker: str, limit_days: int = 20) -> Dict[str, Any]:
        """
        取得特定標的在過去 N 個交易日中，累計買超與賣超最多的券商分點摘要
        """
        if not self.enabled:
            return {"top_buyers": [], "top_sellers": []}

        clean_code = ticker.split(".")[0]
        try:
            with self._get_connection() as con:
                # 取得可用日期清單
                dates_df = con.execute("""
                    SELECT DISTINCT date 
                    FROM tw_broker_trades 
                    WHERE (ticker = ? OR raw_code = ? OR ticker LIKE ?)
                    ORDER BY date DESC 
                    LIMIT ?;
                """, [ticker, clean_code, f"{clean_code}.%", limit_days]).df()
                
                if dates_df.empty:
                    return {"top_buyers": [], "top_sellers": []}

                min_date = dates_df["date"].min()

                sql = """
                SELECT 
                    broker_name,
                    SUM(buy_vol) as total_buy,
                    SUM(sell_vol) as total_sell,
                    SUM(net_vol) as total_net,
                    AVG(pct) as avg_pct,
                    COUNT(DISTINCT date) as trade_days
                FROM tw_broker_trades
                WHERE (ticker = ? OR raw_code = ? OR ticker LIKE ?) AND date >= ?
                GROUP BY broker_name
                HAVING total_net != 0
                ORDER BY total_net DESC;
                """
                df = con.execute(sql, [ticker, clean_code, f"{clean_code}.%", min_date]).df()
                if df.empty:
                    return {"top_buyers": [], "top_sellers": []}

                top_buyers = df[df["total_net"] > 0].head(15).to_dict(orient="records")
                top_sellers = df[df["total_net"] < 0].sort_values("total_net", ascending=True).head(15).to_dict(orient="records")

                return {
                    "ticker": ticker,
                    "days": len(dates_df),
                    "start_date": str(min_date),
                    "end_date": str(dates_df["date"].max()),
                    "top_buyers": top_buyers,
                    "top_sellers": top_sellers
                }
        except Exception as e:
            logger.warning(f"⚠️ 從 DuckDB 統計 {ticker} 券商主力摘要失敗: {e}")
            return {"top_buyers": [], "top_sellers": []}



