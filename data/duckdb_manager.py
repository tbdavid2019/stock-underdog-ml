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

        # 1. 玄鐵策略結果
        xuantie_df = results.get("xuantie_results", pd.DataFrame())
        if isinstance(xuantie_df, pd.DataFrame) and not xuantie_df.empty:
            for idx, row in xuantie_df.iterrows():
                all_data.append({
                    "index_name": index_name,
                    "model_name": "玄鐵重劍",
                    "strategy_type": "玄鐵重劍",
                    "ticker": row["ticker"],
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
                    "trust_net_5d": None,
                    "foreign_net_5d": None,
                    "tags": "玄鐵買點"
                })

        # 2. LSTM 預測結果
        lstm_results = results.get("lstm_results", [])
        for r in lstm_results:
            all_data.append({
                "index_name": index_name,
                "model_name": "LSTM",
                "strategy_type": "LSTM預測",
                "ticker": r["ticker"],
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
                "trust_net_5d": None,
                "foreign_net_5d": None,
                "tags": "LSTM看漲" if float(r["potential"]) > 0 else "LSTM看跌"
            })

        # 3. 雙重/三重符合結果
        overlap_df = results.get("overlap_results", pd.DataFrame())
        if isinstance(overlap_df, pd.DataFrame) and not overlap_df.empty:
            for idx, row in overlap_df.iterrows():
                all_data.append({
                    "index_name": index_name,
                    "model_name": "多維共振",
                    "strategy_type": "多維共振",
                    "ticker": row["ticker"],
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
                    "trust_net_5d": None,
                    "foreign_net_5d": None,
                    "tags": "重點推薦"
                })

        return self.save_predictions_batch(all_data)

    def query(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """執行 SQL 查詢並返回 Pandas DataFrame"""
        with self._get_connection() as con:
            if params:
                return con.execute(sql, params).df()
            return con.execute(sql).df()

    def get_row_count(self, table_name: str = "predictions") -> int:
        """取得指定表格的總記錄數"""
        try:
            with self._get_connection() as con:
                res = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                return res[0] if res else 0
        except Exception:
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
        # 將所有型態的 NaN 統一轉為 Python None
        cleaned = df.map(lambda x: None if pd.isna(x) else x)
        return cleaned.to_dict(orient="records")

    # =========================================================================
    # REST API & MCP 專用高吞吐量化分析查詢接口
    # =========================================================================

    def get_latest_predictions(
        self, 
        index_name: Optional[str] = None, 
        model_name: Optional[str] = None, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """取得各股票最新一筆預測與策略記錄"""
        where_clauses = []
        params = []
        if index_name:
            where_clauses.append("index_name = ?")
            params.append(index_name)
        if model_name:
            where_clauses.append("model_name = ?")
            params.append(model_name)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
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
        return self._clean_df_records(df)

    def get_resonance_candidates(self, index_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """取得多策略重合 / 雙重符合 / 🏆 三重共振股票"""
        where_clauses = ["(model_name = '多維共振' OR strategy_type = '多維共振' OR tags LIKE '%共振%' OR tags LIKE '%雙重符合%' OR tags LIKE '%3重符合%')"]
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
        where_clauses = ["(model_name = '玄鐵重劍' OR strategy_type = '玄鐵重劍' OR pullback_type IS NOT NULL)"]
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
        where_clauses = ["potential IS NOT NULL", "model_name = 'LSTM'"]
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
        where_clauses = ["potential IS NOT NULL", "model_name = 'LSTM'"]
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
        ORDER BY timestamp DESC
        LIMIT ?;
        """
        df = self.query(sql, [ticker, limit])
        return self._clean_df_records(df)

    def get_db_stats(self) -> Dict[str, Any]:
        """取得 DuckDB 全局時序庫統計資訊"""
        try:
            with self._get_connection() as con:
                row_count = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                distinct_tickers = con.execute("SELECT COUNT(DISTINCT ticker) FROM predictions").fetchone()[0]
                min_time = con.execute("SELECT MIN(timestamp) FROM predictions").fetchone()[0]
                max_time = con.execute("SELECT MAX(timestamp) FROM predictions").fetchone()[0]
                indices = [r[0] for r in con.execute("SELECT DISTINCT index_name FROM predictions WHERE index_name IS NOT NULL").fetchall()]
                models = [r[0] for r in con.execute("SELECT DISTINCT model_name FROM predictions WHERE model_name IS NOT NULL").fetchall()]
                
                return {
                    "total_records": row_count,
                    "distinct_tickers": distinct_tickers,
                    "earliest_timestamp": str(min_time) if min_time else None,
                    "latest_timestamp": str(max_time) if max_time else None,
                    "indices": indices,
                    "models": models,
                    "db_path": self.db_path
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
