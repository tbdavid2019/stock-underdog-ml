"""
data/duckdb_manager.py - 本地 DuckDB 量化時序資料庫管理器 (Local DuckDB Manager)

提供單一檔案嵌入式高效能列式儲存，支援預測數據持久化、零拷貝 Pandas 查詢與 Parquet 冷備份導出。
"""

import os
import re
import time
import math
import logging
import datetime
import json
import random
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import duckdb

from core.config import config
from core.duckdb_lock import duckdb_write_lock, DuckDBBusyError, get_lock_path
from data.macro import MacroState
from data.ticker_resolver import clean_query, normalize_ticker, resolve_alias

logger = logging.getLogger("stock_app.duckdb")

try:
    import zoneinfo
    TZ_TAIPEI = zoneinfo.ZoneInfo("Asia/Taipei")
except Exception:
    TZ_TAIPEI = datetime.timezone(datetime.timedelta(hours=8))


class DuckDBManager:
    """DuckDB 本地量化時序資料庫管理器"""

    _WRITE_LOCK = threading.RLock()
    _RESOLVER_LOCK = threading.RLock()
    _SCHEMA_INITIALIZED_PATHS = set()
    _INIT_LOCK = threading.RLock()

    @contextmanager
    def _write_lock(self, timeout: float = 15.0):
        """跨執行緒與跨進程安全排他寫入鎖"""
        with self._WRITE_LOCK:
            with duckdb_write_lock(self.db_path, timeout=timeout):
                yield

    def __init__(self, db_path: Optional[str] = None, init_schema: bool = True):
        self.db_path = db_path or config.db.DUCKDB_PATH
        self.enabled = config.db.ENABLE_DUCKDB
        if init_schema:
            self._ensure_storage_dir()
            if self.enabled:
                needs_init = (self.db_path not in self._SCHEMA_INITIALIZED_PATHS) or not os.path.exists(self.db_path)
                if needs_init:
                    with self._INIT_LOCK:
                        if (self.db_path not in self._SCHEMA_INITIALIZED_PATHS) or not os.path.exists(self.db_path):
                            try:
                                if self._init_schema():
                                    self._SCHEMA_INITIALIZED_PATHS.add(self.db_path)
                            except DuckDBBusyError:
                                logger.warning("DuckDB busy during __init__ schema initialization; will retry upon connection")
                            except Exception as e:
                                logger.warning("DuckDB schema init deferred due to exception: %s", e)

    def _ensure_storage_dir(self):
        storage_dir = os.path.dirname(self.db_path)
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)

    def _get_connection(
        self, 
        read_only: bool = False, 
        max_retries: int = 5, 
        retry_delay: float = 0.2,
        allow_downgrade: bool = False
    ) -> duckdb.DuckDBPyConnection:
        """建立或取得 DuckDB 連線，支援唯讀模式與並發鎖衝突/配置衝突指數退避重試"""
        if not read_only and self.db_path != ":memory:" and not os.path.exists(self.db_path):
            with self._INIT_LOCK:
                if not os.path.exists(self.db_path):
                    if self._init_schema():
                        self._SCHEMA_INITIALIZED_PATHS.add(self.db_path)
        for attempt in range(max_retries + 1):
            try:
                return duckdb.connect(self.db_path, read_only=read_only)
            except (duckdb.IOException, duckdb.ConnectionException, Exception) as exc:
                err_msg = str(exc).lower()
                is_diff_config = ("different configuration" in err_msg) or ("different config" in err_msg)
                if is_diff_config:
                    if read_only:
                        # 徹底移除「read_only=True 失敗時反向回退開 read_only=False」之危險邏輯
                        logger.warning(
                            "⚠️ DuckDB 唯讀連線配置衝突: %s", exc
                        )
                    else:
                        if allow_downgrade:
                            logger.warning(
                                "⚠️ DuckDB 寫入連線配置衝突 (已有唯讀連線開啟)，嘗試降級為 read_only=True: %s", exc
                            )
                            try:
                                return duckdb.connect(self.db_path, read_only=True)
                            except Exception as fb_exc:
                                logger.warning("⚠️ DuckDB read_only=True 降級亦失敗: %s", fb_exc)

                is_lock_conflict = any(k in err_msg for k in ("conflicting lock", "could not set lock", "unique file handle", "resource temporarily unavailable"))
                if (not is_lock_conflict and not is_diff_config) or attempt >= max_retries:
                    if not read_only and (is_lock_conflict or is_diff_config) and allow_downgrade:
                        logger.warning(
                            "⚠️ DuckDB 取得寫入連線逾時 (鎖/配置衝突)，嘗試降級為唯讀模式以保障查詢: %s", exc
                        )
                        try:
                            return duckdb.connect(self.db_path, read_only=True)
                        except Exception as downgrade_exc:
                            logger.error("❌ DuckDB 降級唯讀連線亦失敗: %s", downgrade_exc)
                    if is_lock_conflict or is_diff_config:
                        raise DuckDBBusyError(f"DuckDB connection busy (attempt {attempt + 1}/{max_retries + 1}): {exc}") from exc
                    raise
                sleep_sec = retry_delay * (2 ** attempt) + random.uniform(0.01, 0.05)
                logger.warning(
                    "⚠️ DuckDB 連線異常 (attempt %d/%d, read_only=%s)，將於 %.2fs 後重試: %s",
                    attempt + 1, max_retries, read_only, sleep_sec, exc
                )
                time.sleep(sleep_sec)

    def _init_schema(self):
        """初始化表格結構 (若不存在則建立)"""
        try:
            with self._write_lock(), duckdb.connect(self.db_path, read_only=False) as con:
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
                    tags VARCHAR,
                    composite_score DOUBLE,
                    timestral_predicted_price DOUBLE,
                    timestral_potential DOUBLE,
                    timestral_p10 DOUBLE,
                    timestral_p90 DOUBLE,
                    timesfm_predicted_price DOUBLE,
                    timesfm_potential DOUBLE,
                    moirai_predicted_price DOUBLE,
                    moirai_potential DOUBLE,
                    moirai_p10 DOUBLE,
                    moirai_p90 DOUBLE,
                    dual_model_consensus BOOLEAN,
                    tri_model_consensus BOOLEAN
                );
                """)
                for col_name, col_type in [
                    ("composite_score", "DOUBLE"),
                    ("timestral_predicted_price", "DOUBLE"),
                    ("timestral_potential", "DOUBLE"),
                    ("timestral_p10", "DOUBLE"),
                    ("timestral_p90", "DOUBLE"),
                    ("timesfm_predicted_price", "DOUBLE"),
                    ("timesfm_potential", "DOUBLE"),
                    ("moirai_predicted_price", "DOUBLE"),
                    ("moirai_potential", "DOUBLE"),
                    ("moirai_p10", "DOUBLE"),
                    ("moirai_p90", "DOUBLE"),
                    ("dual_model_consensus", "BOOLEAN"),
                    ("tri_model_consensus", "BOOLEAN")
                ]:
                    try:
                        con.execute(f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                    except Exception:
                        pass
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
                CREATE TABLE IF NOT EXISTS market_universe (
                    source_id VARCHAR,
                    market VARCHAR,
                    exchange VARCHAR,
                    snapshot_date VARCHAR,
                    local_symbol VARCHAR,
                    normalized_symbol VARCHAR,
                    name_local VARCHAR,
                    name_en VARCHAR,
                    isin VARCHAR,
                    security_type VARCHAR,
                    listing_status VARCHAR,
                    market_category VARCHAR,
                    financial_status VARCHAR,
                    board_lot BIGINT,
                    source VARCHAR,
                    source_updated_at VARCHAR,
                    updated_at VARCHAR
                );
                """)
                con.execute("""
                CREATE TABLE IF NOT EXISTS universe_sync_runs (
                    source_id VARCHAR,
                    snapshot_date VARCHAR,
                    stale BOOLEAN,
                    record_count BIGINT,
                    error VARCHAR,
                    updated_at VARCHAR
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
                CREATE INDEX IF NOT EXISTS idx_tw_broker_trades_ticker_date ON tw_broker_trades (ticker, date);
                CREATE INDEX IF NOT EXISTS idx_tw_broker_trades_raw_code_date ON tw_broker_trades (raw_code, date);
                """)
                con.execute("""
                CREATE TABLE IF NOT EXISTS earnings_calls (
                    id VARCHAR PRIMARY KEY,
                    market VARCHAR NOT NULL,
                    ticker VARCHAR NOT NULL,
                    name VARCHAR NOT NULL,
                    call_date VARCHAR NOT NULL,
                    call_time VARCHAR,
                    timezone VARCHAR NOT NULL,
                    timestamp_utc TIMESTAMP,
                    meeting_type VARCHAR,
                    location VARCHAR,
                    webcast_url VARCHAR,
                    presentation_url VARCHAR,
                    transcript_url VARCHAR,
                    language VARCHAR,
                    eps_estimate DOUBLE,
                    eps_actual DOUBLE,
                    revenue_estimate DOUBLE,
                    revenue_actual DOUBLE,
                    is_weight_stock BOOLEAN DEFAULT FALSE,
                    remarks VARCHAR,
                    financial_period VARCHAR,
                    guidance_summary VARCHAR,
                    guidance_outlook VARCHAR,
                    guidance_margin VARCHAR,
                    guidance_capex VARCHAR,
                    guidance_qa VARCHAR,
                    source VARCHAR,
                    created_at VARCHAR,
                    updated_at VARCHAR
                );
                CREATE INDEX IF NOT EXISTS idx_earnings_calls_mkt_date ON earnings_calls (market, call_date);
                CREATE INDEX IF NOT EXISTS idx_earnings_calls_ticker_date ON earnings_calls (ticker, call_date);
                """)

                # 確保舊版資料庫自動遷移新欄位 (financial_period, guidance_*)
                for g_col in ["financial_period", "guidance_summary", "guidance_outlook", "guidance_margin", "guidance_capex", "guidance_qa"]:
                    try:
                        con.execute(f"ALTER TABLE earnings_calls ADD COLUMN IF NOT EXISTS {g_col} VARCHAR;")
                    except Exception:
                        pass

                # 若舊版資料庫因 ALTER TABLE 欄位順序錯位，自動清除重抓以確保資料品質
                try:
                    bad_rows = con.execute("SELECT count(*) FROM earnings_calls WHERE source LIKE '%Q%' OR guidance_margin LIKE '%BigGo%'").fetchone()[0]
                    if bad_rows > 0:
                        con.execute("DELETE FROM earnings_calls;")
                except Exception:
                    pass

                # 自動植入官方指標企業基準庫種子 (確保台積電 2330.TW 等關鍵法說會永不缺席)
                try:
                    v_path = os.path.join(os.path.dirname(__file__), "verified_earnings.json")
                    if os.path.exists(v_path):
                        with open(v_path, "r", encoding="utf-8") as vf:
                            v_items = json.load(vf)
                        if v_items:
                            v_df = pd.DataFrame(v_items)
                            v_df = v_df.where(pd.notna(v_df), None)
                            if "timestamp_utc" not in v_df.columns:
                                v_df["timestamp_utc"] = None
                            now_str = datetime.datetime.now().isoformat()
                            if "created_at" not in v_df.columns:
                                v_df["created_at"] = now_str
                            if "updated_at" not in v_df.columns:
                                v_df["updated_at"] = now_str
                            con.register("v_seed_batch", v_df)
                            con.execute("INSERT OR IGNORE INTO earnings_calls BY NAME SELECT * FROM v_seed_batch;")
                            logger.info(f"✅ 自動植入官方基準法說會資料 (共 {len(v_items)} 筆)")
                except Exception as seed_err:
                    logger.debug(f"自動植入基準法說會資料略過: {seed_err}")

                con.execute("""
                CREATE TABLE IF NOT EXISTS macro_snapshots (
                    category VARCHAR PRIMARY KEY,
                    data_json VARCHAR NOT NULL,
                    source VARCHAR,
                    updated_at VARCHAR NOT NULL
                );

                CREATE TABLE IF NOT EXISTS company_profiles (
                    ticker VARCHAR PRIMARY KEY,
                    raw_code VARCHAR,
                    name VARCHAR NOT NULL,
                    registered_name VARCHAR,
                    unified_id VARCHAR,
                    capital VARCHAR,
                    industry VARCHAR,
                    chairman VARCHAR,
                    market_cap VARCHAR,
                    established VARCHAR,
                    business_summary VARCHAR,
                    recent_news_json VARCHAR,
                    data_json VARCHAR NOT NULL,
                    source VARCHAR,
                    updated_at VARCHAR NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_company_profiles_raw_code ON company_profiles (raw_code);
                CREATE INDEX IF NOT EXISTS idx_company_profiles_name ON company_profiles (name);

                CREATE TABLE IF NOT EXISTS company_networks (
                    ticker VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    unified_id VARCHAR,
                    capital DOUBLE,
                    representative VARCHAR,
                    address VARCHAR,
                    data_json VARCHAR NOT NULL,
                    web_url VARCHAR,
                    updated_at VARCHAR NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_company_networks_name ON company_networks (name);
                CREATE INDEX IF NOT EXISTS idx_company_networks_unified ON company_networks (unified_id);

                CREATE TABLE IF NOT EXISTS company_tenders (
                    ticker VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    tax_id VARCHAR,
                    wins INTEGER,
                    participations INTEGER,
                    win_rate DOUBLE,
                    award_total DOUBLE,
                    data_json VARCHAR NOT NULL,
                    web_url VARCHAR,
                    updated_at VARCHAR NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_company_tenders_name ON company_tenders (name);
                CREATE INDEX IF NOT EXISTS idx_company_tenders_tax_id ON company_tenders (tax_id);

                CREATE TABLE IF NOT EXISTS podcast_channels (
                    channel_id VARCHAR PRIMARY KEY,
                    title VARCHAR NOT NULL,
                    author VARCHAR,
                    description VARCHAR,
                    image_url VARCHAR,
                    rss_url VARCHAR,
                    youtube_channel_url VARCHAR,
                    follower_count BIGINT DEFAULT 0,
                    episode_count INTEGER DEFAULT 0,
                    category VARCHAR,
                    language VARCHAR,
                    is_subscribed BOOLEAN DEFAULT TRUE,
                    created_at VARCHAR,
                    updated_at VARCHAR
                );

                CREATE TABLE IF NOT EXISTS podcast_episodes (
                    id VARCHAR PRIMARY KEY,
                    channel_id VARCHAR NOT NULL,
                    title VARCHAR NOT NULL,
                    published_at VARCHAR NOT NULL,
                    audio_url VARCHAR,
                    youtube_url VARCHAR,
                    duration_seconds INTEGER,
                    language VARCHAR,
                    host VARCHAR,
                    guest VARCHAR,
                    tldr_zh_tw VARCHAR,
                    key_takeaways VARCHAR,
                    topic_tags VARCHAR,
                    tickers VARCHAR,
                    ticker_prices_json VARCHAR,
                    importance_score DOUBLE,
                    importance_level VARCHAR,
                    ticker_sentiments VARCHAR,
                    decision_slot VARCHAR,
                    status VARCHAR DEFAULT 'extracted',
                    created_at VARCHAR,
                    updated_at VARCHAR
                );
                CREATE INDEX IF NOT EXISTS idx_podcast_episodes_pub ON podcast_episodes (published_at DESC);
                CREATE INDEX IF NOT EXISTS idx_podcast_episodes_channel ON podcast_episodes (channel_id);

                CREATE TABLE IF NOT EXISTS decision_verdicts (
                    ticker VARCHAR NOT NULL,
                    as_of_date VARCHAR NOT NULL,
                    action_decision VARCHAR NOT NULL,
                    action_confidence DOUBLE,
                    action_probabilities VARCHAR,
                    conviction_score DOUBLE,
                    is_favorable_entry DOUBLE,
                    tag VARCHAR,
                    source_slot VARCHAR,
                    raw_verdict VARCHAR,
                    created_at VARCHAR NOT NULL,
                    expires_at VARCHAR NOT NULL,
                    PRIMARY KEY (ticker, as_of_date)
                );
                CREATE INDEX IF NOT EXISTS idx_decision_verdicts_ticker ON decision_verdicts (ticker);
                CREATE INDEX IF NOT EXISTS idx_decision_verdicts_expires ON decision_verdicts (expires_at);

                CREATE TABLE IF NOT EXISTS company_fundamentals (
                    ticker VARCHAR PRIMARY KEY,
                    pe DOUBLE,
                    forward_pe DOUBLE,
                    pb DOUBLE,
                    ev_ebitda DOUBLE,
                    data_json VARCHAR,
                    updated_at VARCHAR NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_company_fundamentals_ticker ON company_fundamentals (ticker);

                -- ================= Discovery Engine Tables =================
                CREATE TABLE IF NOT EXISTS discovery_candidates (
                    ticker VARCHAR,
                    market VARCHAR,
                    name VARCHAR,
                    theme_id VARCHAR,
                    theme_name VARCHAR,
                    total_score DOUBLE,
                    recommendation VARCHAR,
                    review_status VARCHAR,
                    in_pool BOOLEAN,
                    source_score DOUBLE,
                    purity_score DOUBLE,
                    paper_score DOUBLE,
                    novelty_score DOUBLE,
                    sentiment_score DOUBLE,
                    setup_score DOUBLE,
                    risk_score DOUBLE,
                    catalyst_summary VARCHAR,
                    us_anchor VARCHAR,
                    paper_count BIGINT,
                    signal_count BIGINT,
                    run_id VARCHAR,
                    as_of VARCHAR,
                    created_at VARCHAR,
                    updated_at VARCHAR,
                    PRIMARY KEY (ticker, theme_id, as_of)
                );
                CREATE INDEX IF NOT EXISTS idx_disc_cand_rec ON discovery_candidates (recommendation);
                CREATE INDEX IF NOT EXISTS idx_disc_cand_asof ON discovery_candidates (as_of);

                CREATE TABLE IF NOT EXISTS discovery_papers (
                    paper_id VARCHAR,
                    title VARCHAR,
                    authors VARCHAR,
                    abstract VARCHAR,
                    category VARCHAR,
                    published_date VARCHAR,
                    bottleneck_tag VARCHAR,
                    related_tickers VARCHAR,
                    url VARCHAR,
                    significance_score DOUBLE,
                    run_id VARCHAR,
                    as_of VARCHAR,
                    created_at VARCHAR,
                    PRIMARY KEY (paper_id, as_of)
                );
                CREATE INDEX IF NOT EXISTS idx_disc_papers_date ON discovery_papers (published_date);

                CREATE TABLE IF NOT EXISTS discovery_signals (
                    signal_id VARCHAR,
                    ticker VARCHAR,
                    theme_id VARCHAR,
                    source_type VARCHAR,
                    title VARCHAR,
                    content VARCHAR,
                    url VARCHAR,
                    sentiment_score DOUBLE,
                    detected_at VARCHAR,
                    run_id VARCHAR,
                    as_of VARCHAR,
                    created_at VARCHAR,
                    PRIMARY KEY (signal_id, as_of)
                );
                CREATE INDEX IF NOT EXISTS idx_disc_signals_ticker ON discovery_signals (ticker);

                CREATE TABLE IF NOT EXISTS discovery_history (
                    run_id VARCHAR PRIMARY KEY,
                    as_of VARCHAR,
                    total_candidates BIGINT,
                    propose_add_count BIGINT,
                    observe_count BIGINT,
                    already_in_pool_count BIGINT,
                    paper_count BIGINT,
                    signal_count BIGINT,
                    avg_total_score DOUBLE,
                    created_at VARCHAR
                );
                CREATE INDEX IF NOT EXISTS idx_disc_hist_asof ON discovery_history (as_of);

                CREATE TABLE IF NOT EXISTS watchlist_items (
                    profile_id VARCHAR DEFAULT 'default',
                    ticker VARCHAR NOT NULL,
                    canonical_ticker VARCHAR NOT NULL,
                    market VARCHAR,
                    note VARCHAR,
                    sort_order INTEGER,
                    created_at TIMESTAMP,
                    PRIMARY KEY (profile_id, canonical_ticker)
                );
                CREATE INDEX IF NOT EXISTS idx_watchlist_profile ON watchlist_items (profile_id);

                CREATE TABLE IF NOT EXISTS quant_price_alerts (
                    alert_id VARCHAR PRIMARY KEY,
                    profile_id VARCHAR DEFAULT 'default',
                    canonical_ticker VARCHAR NOT NULL,
                    rule_type VARCHAR NOT NULL,
                    threshold_value DOUBLE,
                    reference_model VARCHAR DEFAULT 'auto',
                    enabled BOOLEAN DEFAULT true,
                    cooldown_minutes INTEGER DEFAULT 720,
                    last_triggered_at TIMESTAMP,
                    last_triggered_price DOUBLE,
                    created_at TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_profile_ticker ON quant_price_alerts (profile_id, canonical_ticker);

                CREATE TABLE IF NOT EXISTS global_daily_bars (
                    ticker VARCHAR,
                    market VARCHAR,
                    currency VARCHAR,
                    date VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    is_adjusted BOOLEAN DEFAULT TRUE,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (ticker, date)
                );
                CREATE INDEX IF NOT EXISTS idx_global_daily_bars_ticker_date ON global_daily_bars (ticker, date);
                """);

                # Schema migration for existing installations
                try:
                    con.execute("ALTER TABLE podcast_episodes ADD COLUMN IF NOT EXISTS importance_score DOUBLE;")
                    con.execute("ALTER TABLE podcast_episodes ADD COLUMN IF NOT EXISTS importance_level VARCHAR;")
                    con.execute("ALTER TABLE podcast_episodes ADD COLUMN IF NOT EXISTS ticker_sentiments VARCHAR;")
                    con.execute("ALTER TABLE podcast_episodes ADD COLUMN IF NOT EXISTS decision_slot VARCHAR;")
                except Exception as alter_e:
                    logger.debug(f"DuckDB schema migration note: {alter_e}")
            logger.info(f"✅ DuckDB 資料庫初始化成功: {self.db_path}")
            self.init_industry_chain_tables()
            return True
        except DuckDBBusyError:
            logger.warning(f"⚠️ DuckDB 初始化遇鎖衝突 busy: {self.db_path}")
            raise
        except Exception as e:
            logger.error(f"❌ DuckDB 初始化失敗: {e}")
            raise

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
            "trust_net_5d", "foreign_net_5d", "tags", "composite_score",
            "timestral_predicted_price", "timestral_potential",
            "timestral_p10", "timestral_p90",
            "timesfm_predicted_price", "timesfm_potential",
            "moirai_predicted_price", "moirai_potential",
            "moirai_p10", "moirai_p90",
            "dual_model_consensus", "tri_model_consensus"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        # 格式化 tags 欄位為字串
        if "tags" in df.columns:
            df["tags"] = df["tags"].apply(lambda x: " | ".join(x) if isinstance(x, list) else str(x) if x is not None else None)

        col_names = ", ".join(expected_cols)
        with self._write_lock():
            for attempt in range(3):
                try:
                    with self._get_connection(read_only=False) as con:
                        con.register("temp_batch_df", df[expected_cols])
                        con.execute(f"INSERT INTO predictions ({col_names}) SELECT {col_names} FROM temp_batch_df")
                    logger.info(f"💾 成功寫入 {len(df)} 筆量化數據至 DuckDB ({self.db_path})")
                    return len(df)
                except duckdb.IOException as e:
                    err_msg = str(e).lower()
                    if ("conflicting lock" in err_msg or "could not set lock" in err_msg or "unique file handle" in err_msg) and attempt < 2:
                        time.sleep(0.15 * (attempt + 1))
                        continue
                    logger.error(f"❌ DuckDB 批次寫入失敗: {e}", exc_info=True)
                    return 0
                except Exception as e:
                    logger.error(f"❌ DuckDB 批次寫入失敗: {e}", exc_info=True)
                    return 0
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

        timestamp = datetime.datetime.now(TZ_TAIPEI).isoformat()
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
                    "tags": tags_str,
                    "composite_score": float(cand.get("composite_score")) if cand.get("composite_score") is not None and not pd.isna(cand.get("composite_score")) else None
                })

        # 2. Timestral 預測結果 (Mistral Foundation Model 209M)
        timestral_results = results.get("timestral_results", [])
        for r in timestral_results:
            t = r["ticker"]
            cand = candidates_map.get(t, {})
            inst = inst_summaries.get(t, {})
            cand_tags = cand.get("tags")
            default_timestral_tag = "Timestral看漲" if float(r["potential"]) > 0 else "Timestral看跌"
            tags_str = " | ".join(cand_tags) if cand_tags else default_timestral_tag

            all_data.append({
                "index_name": index_name,
                "model_name": "Timestral",
                "strategy_type": "Timestral預測",
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
                "tags": tags_str,
                "composite_score": float(cand.get("composite_score")) if cand.get("composite_score") is not None and not pd.isna(cand.get("composite_score")) else None,
                "timestral_predicted_price": float(r["predicted_price"]),
                "timestral_potential": float(r["potential"]),
                "timestral_p10": float(r.get("p10")) if r.get("p10") is not None and not pd.isna(r.get("p10")) else None,
                "timestral_p90": float(r.get("p90")) if r.get("p90") is not None and not pd.isna(r.get("p90")) else None,
                "timesfm_predicted_price": None,
                "timesfm_potential": None,
                "moirai_predicted_price": None,
                "moirai_potential": None,
                "moirai_p10": None,
                "moirai_p90": None,
                "dual_model_consensus": False,
                "tri_model_consensus": False,
            })

        # 2.5 TimesFM 預測結果 (Google Foundation Model 200M)
        timesfm_results = results.get("timesfm_results", [])
        for r in timesfm_results:
            t = r["ticker"]
            cand = candidates_map.get(t, {})
            inst = inst_summaries.get(t, {})
            cand_tags = cand.get("tags")
            default_tfm_tag = "TimesFM看漲" if float(r["potential"]) > 0 else "TimesFM看跌"
            tags_str = " | ".join(cand_tags) if cand_tags else default_tfm_tag

            all_data.append({
                "index_name": index_name,
                "model_name": "TimesFM",
                "strategy_type": "TimesFM預測",
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
                "tags": tags_str,
                "composite_score": float(cand.get("composite_score")) if cand.get("composite_score") is not None and not pd.isna(cand.get("composite_score")) else None,
                "timestral_predicted_price": None,
                "timestral_potential": None,
                "timestral_p10": None,
                "timestral_p90": None,
                "timesfm_predicted_price": float(r["predicted_price"]),
                "timesfm_potential": float(r["potential"]),
                "moirai_predicted_price": None,
                "moirai_potential": None,
                "moirai_p10": None,
                "moirai_p90": None,
                "dual_model_consensus": False,
                "tri_model_consensus": False,
            })

        # 2.6 MOIRAI-2 預測結果 (Salesforce Foundation Model)
        moirai_results = results.get("moirai_results", [])
        for r in moirai_results:
            t = r["ticker"]
            cand = candidates_map.get(t, {})
            inst = inst_summaries.get(t, {})
            cand_tags = cand.get("tags")
            default_moi_tag = "MOIRAI看漲" if float(r["potential"]) > 0 else "MOIRAI看跌"
            tags_str = " | ".join(cand_tags) if cand_tags else default_moi_tag

            all_data.append({
                "index_name": index_name,
                "model_name": "MOIRAI-2",
                "strategy_type": "MOIRAI預測",
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
                "tags": tags_str,
                "composite_score": float(cand.get("composite_score")) if cand.get("composite_score") is not None and not pd.isna(cand.get("composite_score")) else None,
                "timestral_predicted_price": None,
                "timestral_potential": None,
                "timestral_p10": None,
                "timestral_p90": None,
                "timesfm_predicted_price": None,
                "timesfm_potential": None,
                "moirai_predicted_price": float(r["predicted_price"]),
                "moirai_potential": float(r["potential"]),
                "moirai_p10": float(r.get("p10")) if r.get("p10") is not None and not pd.isna(r.get("p10")) else None,
                "moirai_p90": float(r.get("p90")) if r.get("p90") is not None and not pd.isna(r.get("p90")) else None,
                "dual_model_consensus": False,
                "tri_model_consensus": False,
            })

        # 3. 雙重/多維共振符合結果 (Multi-Model Consensus & Resonance)
        overlap_df = results.get("overlap_results", pd.DataFrame())
        if isinstance(overlap_df, pd.DataFrame) and not overlap_df.empty:
            for idx, row in overlap_df.iterrows():
                t = row["ticker"]
                cand = candidates_map.get(t, {})
                inst = inst_summaries.get(t, {})
                cand_tags = cand.get("tags")
                tags_str = " | ".join(cand_tags) if cand_tags else "🏆多維共振"

                # Defensive target price & potential synthesis: prioritize synthesized values
                pred_price = float(row["predicted_price"]) if row.get("predicted_price") is not None and not pd.isna(row.get("predicted_price")) else float(row["current_price"])
                pot_val = float(row["potential"]) if row.get("potential") is not None and not pd.isna(row.get("potential")) else (
                    float(row["timestral_potential"]) if row.get("timestral_potential") is not None and not pd.isna(row.get("timestral_potential")) else (
                        float(row["timesfm_potential"]) if row.get("timesfm_potential") is not None and not pd.isna(row.get("timesfm_potential")) else (
                            float(row["moirai_potential"]) if row.get("moirai_potential") is not None and not pd.isna(row.get("moirai_potential")) else 0.0
                        )
                    )
                )

                all_data.append({
                    "index_name": index_name,
                    "model_name": "多維共振",
                    "strategy_type": "多維共振",
                    "ticker": t,
                    "current_price": float(row["current_price"]),
                    "predicted_price": pred_price,
                    "potential": pot_val,
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
                    "tags": tags_str,
                    "composite_score": float(cand.get("composite_score")) if cand.get("composite_score") is not None and not pd.isna(cand.get("composite_score")) else None,
                    "timestral_predicted_price": float(row.get("timestral_predicted_price")) if row.get("timestral_predicted_price") is not None and not pd.isna(row.get("timestral_predicted_price")) else None,
                    "timestral_potential": float(row.get("timestral_potential")) if row.get("timestral_potential") is not None and not pd.isna(row.get("timestral_potential")) else None,
                    "timestral_p10": float(row.get("timestral_p10")) if row.get("timestral_p10") is not None and not pd.isna(row.get("timestral_p10")) else None,
                    "timestral_p90": float(row.get("timestral_p90")) if row.get("timestral_p90") is not None and not pd.isna(row.get("timestral_p90")) else None,
                    "timesfm_predicted_price": float(row.get("timesfm_predicted_price")) if row.get("timesfm_predicted_price") is not None and not pd.isna(row.get("timesfm_predicted_price")) else None,
                    "timesfm_potential": float(row.get("timesfm_potential")) if row.get("timesfm_potential") is not None and not pd.isna(row.get("timesfm_potential")) else None,
                    "moirai_predicted_price": float(row.get("moirai_predicted_price")) if row.get("moirai_predicted_price") is not None and not pd.isna(row.get("moirai_predicted_price")) else None,
                    "moirai_potential": float(row.get("moirai_potential")) if row.get("moirai_potential") is not None and not pd.isna(row.get("moirai_potential")) else None,
                    "moirai_p10": float(row.get("moirai_p10")) if row.get("moirai_p10") is not None and not pd.isna(row.get("moirai_p10")) else None,
                    "moirai_p90": float(row.get("moirai_p90")) if row.get("moirai_p90") is not None and not pd.isna(row.get("moirai_p90")) else None,
                    "dual_model_consensus": bool(row.get("dual_model_consensus", False)),
                    "tri_model_consensus": bool(row.get("tri_model_consensus", False)),
                })

        return self.save_predictions_batch(all_data)

    def query(self, sql: str, params: Optional[List[Any]] = None, read_only: bool = True) -> pd.DataFrame:
        """執行 SQL 查詢並返回 Pandas DataFrame，預設為唯讀模式以防並發寫入鎖"""
        with self._get_connection(read_only=read_only, allow_downgrade=True) as con:
            if params:
                return con.execute(sql, params).df()
            return con.execute(sql).df()

    def get_row_count(self, table_name: str = "predictions") -> int:
        """取得指定表格的總記錄數 (排除測試假資料)"""
        try:
            with self._get_connection(read_only=True) as con:
                filter_sql = "WHERE index_name NOT LIKE '%TEST%' AND index_name NOT LIKE '%DEBUG%' AND index_name NOT LIKE '%SERVICE_KEY%' AND model_name NOT LIKE '%DEBUG%'"
                res = con.execute(f"SELECT COUNT(*) FROM {table_name} {filter_sql}").fetchone()
                return res[0] if res else 0
        except Exception:
            return 0

    def clean_test_data(self) -> int:
        """清理歷史混入的測試與除錯假數據 (TEST, DEBUG, SERVICE_KEY 等)"""
        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
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
        with self._get_connection(read_only=True) as con:
            con.execute(f"COPY {table_name} TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD);")
        logger.info(f"📦 已成功導出 {table_name} 至 Parquet: {output_path}")
        return output_path

    @staticmethod
    def _clean_df_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """將 DataFrame 轉為乾淨的字典清單，將 NaN / NaT / inf 轉為 None，並對齊隔日 (T+1) 目標價與漲跌幅"""
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

            # 若包含現價與隔日預測目標價，嚴格校正 potential 對齊「隔日 (T+1) 目標價 vs 現價」
            curr_p = clean_row.get("current_price")
            if isinstance(curr_p, (int, float)) and curr_p > 0:
                pred_p = clean_row.get("predicted_price")
                if isinstance(pred_p, (int, float)) and pred_p > 0:
                    clean_row["potential"] = round(((pred_p - curr_p) / curr_p) * 100.0, 2)

                tm_p = clean_row.get("timestral_predicted_price")
                if isinstance(tm_p, (int, float)) and tm_p > 0:
                    clean_row["timestral_potential"] = round(((tm_p - curr_p) / curr_p) * 100.0, 2)

                tf_p = clean_row.get("timesfm_predicted_price")
                if isinstance(tf_p, (int, float)) and tf_p > 0:
                    clean_row["timesfm_potential"] = round(((tf_p - curr_p) / curr_p) * 100.0, 2)

                moi_p = clean_row.get("moirai_predicted_price")
                if isinstance(moi_p, (int, float)) and moi_p > 0:
                    clean_row["moirai_potential"] = round(((moi_p - curr_p) / curr_p) * 100.0, 2)

            records.append(clean_row)
        return records

    # =========================================================================
    # REST API & MCP 專用高吞吐量化分析查詢接口
    # =========================================================================

    @staticmethod
    def _clean_index_name(index_name: Optional[str]) -> Optional[str]:
        """清洗並標準化市場指數名稱，將 ALL/全部市場安全視為無篩選"""
        if not index_name:
            return None
        val = str(index_name).strip()
        if not val or val.upper() in ("ALL", "NONE", "NULL", "全部", "全部市場"):
            return None
        return val

    @classmethod
    def _parse_index_names(cls, index_name: Optional[str]) -> List[str]:
        """清洗並標準化市場指數名稱，支援逗號分隔多選（例如 '台灣50,SP500'），將 ALL/全部市場安全排除"""
        if not index_name:
            return []
        cleaned_str = str(index_name).replace("，", ",")
        parts = [p.strip() for p in cleaned_str.split(",") if p.strip()]
        valid = []
        for p in parts:
            if p.upper() not in ("ALL", "NONE", "NULL", "全部", "全部市場") and p not in valid:
                valid.append(p)
        return valid

    @classmethod
    def _build_index_filter_clause(cls, index_names: List[str], prefix: str = "") -> Tuple[str, List[str]]:
        """動態生成 index_name 篩選 SQL 與參數"""
        if not index_names:
            return "", []
        col = f"{prefix}index_name" if prefix else "index_name"
        if len(index_names) == 1:
            return f"{col} = ?", [index_names[0]]
        placeholders = ", ".join(["?"] * len(index_names))
        return f"{col} IN ({placeholders})", list(index_names)

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
        parsed_indices = self._parse_index_names(index_name)
        now = datetime.datetime.now(TZ_TAIPEI)

        where_clauses = [
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'",
            "model_name NOT LIKE '%DEBUG%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            where_clauses.append(idx_clause)
            params.extend(idx_params)
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
        with self._get_connection(read_only=True) as con:
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
                batch_dt = batch_dt.replace(tzinfo=TZ_TAIPEI)
            else:
                batch_dt = batch_dt.astimezone(TZ_TAIPEI)
            age_hours = max(0.0, round((now - batch_dt).total_seconds() / 3600.0, 1))
            batch_date = batch_dt.strftime("%Y-%m-%d")
            from core.market_calendar import MarketCalendar
            freshness = MarketCalendar.check_market_freshness(market="TW", last_data_date=batch_date, now_dt=now)
            is_stale = freshness.get("stale", False)
        except Exception:
            age_hours = 0.0
            is_stale = False
            batch_date = str(latest_ts)[:10]

        if batch_only:
            if len(parsed_indices) == 1:
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
                try:
                    r_dt = datetime.datetime.fromisoformat(str(r_ts).replace("Z", "+00:00"))
                    if r_dt.tzinfo is None:
                        r_dt = r_dt.replace(tzinfo=TZ_TAIPEI)
                    else:
                        r_dt = r_dt.astimezone(TZ_TAIPEI)
                    r_age = max(0.0, round((now - r_dt).total_seconds() / 3600.0, 1))
                    r["analysis_date"] = r_dt.strftime("%Y-%m-%d")
                    r["age_hours"] = r_age
                    r["is_stale"] = r_age > 48.0
                except Exception:
                    r["analysis_date"] = str(r_ts)[:10]
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
        """取得多策略重合 / 雙重符合 / 🏆 三重共振股票 (鎖定指定指數之最新批次快照，排序前嚴格過濾共振列並排除看跌防守)"""
        parsed_indices = self._parse_index_names(index_name)
        base_where = [
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            base_where.append(idx_clause)
            params.extend(idx_params)

        base_where_sql = f"WHERE {' AND '.join(base_where)}"
        p_idx_clause, p_idx_params = self._build_index_filter_clause(parsed_indices, prefix="p.")
        p_where_clauses = [
            "p.index_name NOT LIKE '%TEST%'",
            "p.index_name NOT LIKE '%DEBUG%'",
            "p.index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        if p_idx_clause:
            p_where_clauses.append(p_idx_clause)
        p_where_sql = f"AND {' AND '.join(p_where_clauses)}"
        full_params = list(params) + list(p_idx_params) + [limit]

        sql = f"""
        WITH latest_ts AS (
            SELECT ticker, index_name, MAX(timestamp) as max_ts
            FROM predictions
            {base_where_sql}
            GROUP BY ticker, index_name
        ),
        resonance_rows AS (
            SELECT p.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.ticker 
                       ORDER BY CASE WHEN p.model_name = '多維共振' OR p.strategy_type = '多維共振' THEN 0 ELSE 1 END,
                                p.potential DESC NULLS LAST
                   ) as rn
            FROM predictions p
            JOIN latest_ts l ON p.ticker = l.ticker AND p.index_name = l.index_name AND p.timestamp = l.max_ts
            WHERE (p.model_name = '多維共振' OR p.strategy_type = '多維共振' OR p.tags LIKE '%共振%' OR p.tags LIKE '%雙重符合%' OR p.tags LIKE '%3重符合%' OR p.dual_model_consensus = TRUE OR p.tri_model_consensus = TRUE)
              AND (COALESCE(p.tags, '') NOT LIKE '%防守%' AND COALESCE(p.tags, '') NOT LIKE '%深度防守%')
              AND (p.potential IS NULL OR p.potential > -5.0)
              {p_where_sql}
        )
        SELECT * EXCLUDE (rn)
        FROM resonance_rows
        WHERE rn = 1
        ORDER BY potential DESC NULLS LAST, ticker ASC
        LIMIT ?;
        """
        df = self.query(sql, full_params)
        return self._clean_df_records(df)

    def get_tri_bullish_candidates(self, index_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """取得 3 模型同步看漲標的 (TimesFM ∩ Timestral ∩ MOIRAI-2 全數潛力 > 0)"""
        parsed_indices = self._parse_index_names(index_name)
        base_where = [
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            base_where.append(idx_clause)
            params.extend(idx_params)

        base_where_sql = f"WHERE {' AND '.join(base_where)}"
        p_idx_clause, p_idx_params = self._build_index_filter_clause(parsed_indices, prefix="p.")
        p_where_clauses = [
            "p.index_name NOT LIKE '%TEST%'",
            "p.index_name NOT LIKE '%DEBUG%'",
            "p.index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        if p_idx_clause:
            p_where_clauses.append(p_idx_clause)
        p_where_sql = f"AND {' AND '.join(p_where_clauses)}"
        full_params = list(params) + list(p_idx_params) + [limit]

        sql = f"""
        WITH latest_ts AS (
            SELECT ticker, index_name, MAX(timestamp) as max_ts
            FROM predictions
            {base_where_sql}
            GROUP BY ticker, index_name
        ),
        candidates AS (
            SELECT p.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.ticker 
                       ORDER BY CASE WHEN p.model_name = '多維共振' OR p.strategy_type = '多維共振' THEN 0 ELSE 1 END,
                                p.potential DESC NULLS LAST
                   ) as rn
            FROM predictions p
            JOIN latest_ts l ON p.ticker = l.ticker AND p.index_name = l.index_name AND p.timestamp = l.max_ts
            WHERE (
                p.tri_model_consensus = TRUE
                OR (p.timesfm_potential > 0 AND p.timestral_potential > 0 AND p.moirai_potential > 0)
                OR (p.tags LIKE '%三模型%' OR p.tags LIKE '%三大模型%' OR p.tags LIKE '%3重符合%')
            )
            AND (COALESCE(p.tags, '') NOT LIKE '%防守%' AND COALESCE(p.tags, '') NOT LIKE '%深度防守%')
            {p_where_sql}
        )
        SELECT * EXCLUDE (rn)
        FROM candidates
        WHERE rn = 1
        ORDER BY potential DESC NULLS LAST, ticker ASC
        LIMIT ?;
        """
        df = self.query(sql, full_params)
        return self._clean_df_records(df)

    def get_dual_bullish_candidates(self, index_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """取得雙模型以上看漲標的 (任 2 個時序大模型潛力 > 0)"""
        parsed_indices = self._parse_index_names(index_name)
        base_where = [
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            base_where.append(idx_clause)
            params.extend(idx_params)

        base_where_sql = f"WHERE {' AND '.join(base_where)}"
        p_idx_clause, p_idx_params = self._build_index_filter_clause(parsed_indices, prefix="p.")
        p_where_clauses = [
            "p.index_name NOT LIKE '%TEST%'",
            "p.index_name NOT LIKE '%DEBUG%'",
            "p.index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        if p_idx_clause:
            p_where_clauses.append(p_idx_clause)
        p_where_sql = f"AND {' AND '.join(p_where_clauses)}"
        full_params = list(params) + list(p_idx_params) + [limit]

        sql = f"""
        WITH latest_ts AS (
            SELECT ticker, index_name, MAX(timestamp) as max_ts
            FROM predictions
            {base_where_sql}
            GROUP BY ticker, index_name
        ),
        candidates AS (
            SELECT p.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.ticker 
                       ORDER BY CASE WHEN p.model_name = '多維共振' OR p.strategy_type = '多維共振' THEN 0 ELSE 1 END,
                                p.potential DESC NULLS LAST
                   ) as rn
            FROM predictions p
            JOIN latest_ts l ON p.ticker = l.ticker AND p.index_name = l.index_name AND p.timestamp = l.max_ts
            WHERE (
                p.dual_model_consensus = TRUE
                OR p.tri_model_consensus = TRUE
                OR (
                    CASE WHEN p.timesfm_potential > 0 THEN 1 ELSE 0 END +
                    CASE WHEN p.timestral_potential > 0 THEN 1 ELSE 0 END +
                    CASE WHEN p.moirai_potential > 0 THEN 1 ELSE 0 END
                ) >= 2
                OR (p.tags LIKE '%雙模型%' OR p.tags LIKE '%雙AI%' OR p.tags LIKE '%雙時序%' OR p.tags LIKE '%雙ML%')
            )
            AND (COALESCE(p.tags, '') NOT LIKE '%防守%' AND COALESCE(p.tags, '') NOT LIKE '%深度防守%')
            {p_where_sql}
        )
        SELECT * EXCLUDE (rn)
        FROM candidates
        WHERE rn = 1
        ORDER BY potential DESC NULLS LAST, ticker ASC
        LIMIT ?;
        """
        df = self.query(sql, full_params)
        return self._clean_df_records(df)

    def get_dual_bearish_candidates(self, index_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """取得綜合避險標的 (≥2 個時序大模型預測跌幅 < 0 或共振防守破底)"""
        parsed_indices = self._parse_index_names(index_name)
        base_where = [
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            base_where.append(idx_clause)
            params.extend(idx_params)

        base_where_sql = f"WHERE {' AND '.join(base_where)}"
        p_idx_clause, p_idx_params = self._build_index_filter_clause(parsed_indices, prefix="p.")
        p_where_clauses = [
            "p.index_name NOT LIKE '%TEST%'",
            "p.index_name NOT LIKE '%DEBUG%'",
            "p.index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        if p_idx_clause:
            p_where_clauses.append(p_idx_clause)
        p_where_sql = f"AND {' AND '.join(p_where_clauses)}"
        full_params = list(params) + list(p_idx_params) + [limit]

        sql = f"""
        WITH latest_ts AS (
            SELECT ticker, index_name, MAX(timestamp) as max_ts
            FROM predictions
            {base_where_sql}
            GROUP BY ticker, index_name
        ),
        candidates AS (
            SELECT p.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.ticker 
                       ORDER BY CASE WHEN p.model_name = '多維共振' OR p.strategy_type = '多維共振' THEN 0 ELSE 1 END,
                                p.potential ASC NULLS LAST
                   ) as rn
            FROM predictions p
            JOIN latest_ts l ON p.ticker = l.ticker AND p.index_name = l.index_name AND p.timestamp = l.max_ts
            WHERE (
                (
                    CASE WHEN p.timesfm_potential < 0 THEN 1 ELSE 0 END +
                    CASE WHEN p.timestral_potential < 0 THEN 1 ELSE 0 END +
                    CASE WHEN p.moirai_potential < 0 THEN 1 ELSE 0 END
                ) >= 2
                OR (p.potential < 0 AND (
                    (p.timesfm_potential IS NOT NULL AND p.timesfm_potential < 0) OR
                    (p.timestral_potential IS NOT NULL AND p.timestral_potential < 0) OR
                    (p.moirai_potential IS NOT NULL AND p.moirai_potential < 0)
                ))
                OR (p.potential < 0 AND p.model_name IN ('Timestral', 'TimesFM', 'MOIRAI-2'))
                OR p.tags LIKE '%看跌%'
                OR p.tags LIKE '%防守%'
                OR p.tags LIKE '%避險%'
            )
            {p_where_sql}
        )
        SELECT * EXCLUDE (rn)
        FROM candidates
        WHERE rn = 1
        ORDER BY potential ASC NULLS LAST, ticker ASC
        LIMIT ?;
        """
        df = self.query(sql, full_params)
        return self._clean_df_records(df)

    def screen_candidates(
        self,
        direction: str = "bullish",
        models: Optional[List[str]] = None,
        pe_range: Optional[str] = None,
        min_potential: Optional[float] = None,
        risk_reward: bool = False,
        ma60: bool = False,
        institutional: bool = False,
        index_name: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        多維量化自訂篩選器 (Multi-Factor Quant Screener)
        依交易方向、AI 時序預測模型多選、本益比估值、預期空間與核心量化因子進行高效率列式篩選。
        """
        parsed_indices = self._parse_index_names(index_name)
        base_where = [
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        ts_params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            base_where.append(idx_clause)
            ts_params.extend(idx_params)

        base_where_sql = f"WHERE {' AND '.join(base_where)}"
        p_idx_clause, p_idx_params = self._build_index_filter_clause(parsed_indices, prefix="p.")
        p_where_clauses = [
            "p.index_name NOT LIKE '%TEST%'",
            "p.index_name NOT LIKE '%DEBUG%'",
            "p.index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        if p_idx_clause:
            p_where_clauses.append(p_idx_clause)
        p_params = list(p_idx_params)

        is_bearish = (direction or "").strip().lower() == "bearish"

        # 方向條件
        if is_bearish:
            p_where_clauses.append("p.potential < 0")
            if min_potential is not None:
                p_where_clauses.append("p.potential <= ?")
                p_params.append(-abs(float(min_potential)))
        else:
            p_where_clauses.append("p.potential > 0")
            if min_potential is not None:
                p_where_clauses.append("p.potential >= ?")
                p_params.append(abs(float(min_potential)))

        # 模型多選條件 (所有選中模型皆須符合方向，且僅限支援的模型欄位以防 SQL 異常)
        MODEL_COL_MAP = {
            "timesfm": "timesfm_potential",
            "timestral": "timestral_potential",
            "moirai": "moirai_potential",
            "moirai-2": "moirai_potential",
            "moirai2": "moirai_potential",
        }
        if models:
            for m in models:
                m_clean = m.strip().lower()
                col_name = MODEL_COL_MAP.get(m_clean) or MODEL_COL_MAP.get(re.sub(r'[^a-zA-Z0-9_]', '', m_clean))
                if not col_name:
                    continue
                if is_bearish:
                    p_where_clauses.append(f"(p.{col_name} IS NOT NULL AND p.{col_name} < 0)")
                else:
                    p_where_clauses.append(f"(p.{col_name} IS NOT NULL AND p.{col_name} > 0)")

        # 本益比估值區間
        if pe_range:
            pe_clean = pe_range.strip().lower()
            if pe_clean in ("low", "undervalued", "低估"):
                p_where_clauses.append("(p.pe IS NOT NULL AND p.pe > 0 AND p.pe < 15)")
            elif pe_clean in ("reasonable", "fair", "合理"):
                p_where_clauses.append("(p.pe IS NOT NULL AND p.pe >= 15 AND p.pe <= 25)")
            elif pe_clean in ("growth", "overvalued", "高估", "成長"):
                p_where_clauses.append("(p.pe IS NOT NULL AND p.pe > 25)")

        # 核心量化因子
        if risk_reward:
            p_where_clauses.append("(p.tags LIKE '%高盈虧比%' OR p.tags LIKE '%高風報比%' OR p.tags LIKE '%盈虧比%')")

        if ma60:
            p_where_clauses.append("((p.pullback_type IS NOT NULL AND p.pullback_type != '') OR p.tags LIKE '%季線%' OR p.tags LIKE '%MA60%')")

        if institutional:
            p_where_clauses.append("((p.trust_net_5d > 0 AND p.foreign_net_5d > 0) OR p.tags LIKE '%法人合買%' OR p.tags LIKE '%外資投信合買%' OR p.tags LIKE '%土洋合買%')")

        order_direction = "ASC" if is_bearish else "DESC"
        p_where_sql = f"AND {' AND '.join(p_where_clauses)}"

        full_params = ts_params + p_params + [limit, offset]

        sql = f"""
        WITH latest_ts AS (
            SELECT ticker, index_name, MAX(timestamp) as max_ts
            FROM predictions
            {base_where_sql}
            GROUP BY ticker, index_name
        ),
        candidates AS (
            SELECT p.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.ticker 
                       ORDER BY p.potential {order_direction} NULLS LAST, p.composite_score DESC NULLS LAST
                   ) as rn
            FROM predictions p
            JOIN latest_ts l ON p.ticker = l.ticker AND p.index_name = l.index_name AND p.timestamp = l.max_ts
            WHERE 1=1
            {p_where_sql}
        )
        SELECT * EXCLUDE (rn)
        FROM candidates
        WHERE rn = 1
        ORDER BY potential {order_direction} NULLS LAST, ticker ASC
        LIMIT ? OFFSET ?;
        """
        df = self.query(sql, full_params)
        return self._clean_df_records(df)

    def get_xuantie_candidates(
        self, 
        index_name: Optional[str] = None, 
        pullback_type: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """取得玄鐵重劍技術買點 (MA60/120 回調) 股票"""
        parsed_indices = self._parse_index_names(index_name)
        where_clauses = [
            "(model_name = '玄鐵重劍' OR strategy_type = '玄鐵重劍' OR pullback_type IS NOT NULL)",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            where_clauses.append(idx_clause)
            params.extend(idx_params)
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

    def get_timestral_top_bullish(self, index_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """取得 Timestral 預測漲幅最大 TOP N 標的 (Mistral AI 209M)"""
        parsed_indices = self._parse_index_names(index_name)
        where_clauses = [
            "potential IS NOT NULL", 
            "model_name = 'Timestral'",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            where_clauses.append(idx_clause)
            params.extend(idx_params)

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

    def get_timestral_top_bearish(self, index_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """取得 Timestral 預測跌幅最大 / 潛在做空 TOP N 標的"""
        parsed_indices = self._parse_index_names(index_name)
        where_clauses = [
            "potential IS NOT NULL", 
            "model_name = 'Timestral'",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            where_clauses.append(idx_clause)
            params.extend(idx_params)

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

    def get_top_bullish(self, index_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """取得預測漲幅最大 TOP N 標的 (以 Timestral 為主)"""
        # 優先查詢 Timestral
        timestral_res = self.get_timestral_top_bullish(index_name=index_name, limit=limit)
        if timestral_res:
            return timestral_res

        parsed_indices = self._parse_index_names(index_name)
        where_clauses = [
            "potential IS NOT NULL", 
            "model_name = 'Timestral'",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            where_clauses.append(idx_clause)
            params.extend(idx_params)

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
        """取得預測跌幅最大 / 潛在做空 TOP N 標的 (以 Timestral 為主)"""
        timestral_res = self.get_timestral_top_bearish(index_name=index_name, limit=limit)
        if timestral_res:
            return timestral_res

        parsed_indices = self._parse_index_names(index_name)
        where_clauses = [
            "potential IS NOT NULL", 
            "model_name = 'Timestral'",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            where_clauses.append(idx_clause)
            params.extend(idx_params)

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

    def get_timesfm_top_bullish(self, index_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """取得 TimesFM 預測漲幅最大 TOP N 標的"""
        parsed_indices = self._parse_index_names(index_name)
        where_clauses = [
            "potential IS NOT NULL", 
            "model_name = 'TimesFM'",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            where_clauses.append(idx_clause)
            params.extend(idx_params)

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

    def get_timesfm_top_bearish(self, index_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """取得 TimesFM 預測跌幅最大 / 潛在做空 TOP N 標的"""
        parsed_indices = self._parse_index_names(index_name)
        where_clauses = [
            "potential IS NOT NULL", 
            "model_name = 'TimesFM'",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            where_clauses.append(idx_clause)
            params.extend(idx_params)

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

    def get_moirai_top_bullish(self, index_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """取得 MOIRAI-2 預測漲幅最大 TOP N 標的"""
        parsed_indices = self._parse_index_names(index_name)
        where_clauses = [
            "(potential IS NOT NULL OR moirai_potential IS NOT NULL)", 
            "(model_name = 'MOIRAI-2' OR LOWER(model_name) LIKE '%moirai%' OR moirai_potential IS NOT NULL)",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            where_clauses.append(idx_clause)
            params.extend(idx_params)

        where_sql = f"WHERE {' AND '.join(where_clauses)}"
        sql = f"""
        WITH ranked AS (
            SELECT *,
                   COALESCE(moirai_potential, potential) as effective_pot,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) as rn
            FROM predictions
            {where_sql}
        )
        SELECT * EXCLUDE (rn, effective_pot, potential),
               effective_pot as potential
        FROM ranked
        WHERE rn = 1
        ORDER BY effective_pot DESC
        LIMIT ?;
        """
        params.append(limit)
        df = self.query(sql, params, read_only=True)
        return self._clean_df_records(df)

    def get_moirai_top_bearish(self, index_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """取得 MOIRAI-2 預測跌幅最大 / 潛在做空 TOP N 標的"""
        parsed_indices = self._parse_index_names(index_name)
        where_clauses = [
            "(potential IS NOT NULL OR moirai_potential IS NOT NULL)", 
            "(model_name = 'MOIRAI-2' OR LOWER(model_name) LIKE '%moirai%' OR moirai_potential IS NOT NULL)",
            "index_name NOT LIKE '%TEST%'",
            "index_name NOT LIKE '%DEBUG%'",
            "index_name NOT LIKE '%SERVICE_KEY%'"
        ]
        params = []
        idx_clause, idx_params = self._build_index_filter_clause(parsed_indices)
        if idx_clause:
            where_clauses.append(idx_clause)
            params.extend(idx_params)

        where_sql = f"WHERE {' AND '.join(where_clauses)}"
        sql = f"""
        WITH ranked AS (
            SELECT *,
                   COALESCE(moirai_potential, potential) as effective_pot,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) as rn
            FROM predictions
            {where_sql}
        )
        SELECT * EXCLUDE (rn, effective_pot, potential),
               effective_pot as potential
        FROM ranked
        WHERE rn = 1
        ORDER BY effective_pot ASC
        LIMIT ?;
        """
        params.append(limit)
        df = self.query(sql, params, read_only=True)
        return self._clean_df_records(df)

    def get_ticker_history(self, ticker: str, limit: int = 30) -> List[Dict[str, Any]]:
        """
        取得特定股票代碼之歷史預測軌跡。
        依日期與模型去重，同天若因多指數重疊或多次執行產生多筆記錄，優先保留主要指數 (台灣50) 或最新之預測。
        """
        sql = """
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY SUBSTRING(timestamp, 1, 10), COALESCE(model_name, strategy_type, 'UNKNOWN')
                       ORDER BY CASE WHEN index_name IN ('台灣50', 'TW0050') THEN 0 ELSE 1 END, timestamp DESC
                   ) as rn
            FROM predictions
            WHERE ticker = ?
              AND index_name NOT LIKE '%TEST%'
              AND index_name NOT LIKE '%DEBUG%'
              AND index_name NOT LIKE '%SERVICE_KEY%'
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
        ORDER BY timestamp DESC
        LIMIT ?;
        """
        df = self.query(sql, [ticker, limit], read_only=True)
        return self._clean_df_records(df)

    def get_latest_candidate_snapshot(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        取得特定股票最新批次之完整策略合併快照。
        將同批次中的玄鐵、TimesFM、Timestral、MOIRAI-2、多維共振與法人籌碼資料合併為單一完整候選字典。
        """
        import re
        ts_sql = """
        SELECT timestamp
        FROM predictions
        WHERE ticker = ?
          AND index_name NOT LIKE '%TEST%'
          AND index_name NOT LIKE '%DEBUG%'
          AND index_name NOT LIKE '%SERVICE_KEY%'
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        ts_df = self.query(ts_sql, [ticker])
        if ts_df.empty:
            return None

        latest_ts = ts_df.iloc[0]["timestamp"]

        batch_sql = """
        SELECT *
        FROM predictions
        WHERE ticker = ?
          AND timestamp = ?
          AND index_name NOT LIKE '%TEST%'
          AND index_name NOT LIKE '%DEBUG%'
          AND index_name NOT LIKE '%SERVICE_KEY%'
        """
        batch_df = self.query(batch_sql, [ticker, latest_ts])
        if batch_df.empty:
            return None

        records = self._clean_df_records(batch_df)
        if not records:
            return None

        merged_tags = set()
        cand: Dict[str, Any] = {
            "ticker": ticker,
            "current_price": 0.0,
            "timestral_potential": None,
            "timestral_predicted_price": None,
            "timestral_p10": None,
            "timestral_p90": None,
            "timesfm_potential": None,
            "timesfm_predicted_price": None,
            "moirai_potential": None,
            "moirai_predicted_price": None,
            "moirai_p10": None,
            "moirai_p90": None,
            "potential": None,
            "predicted_price": None,
            "dual_model_consensus": False,
            "tri_model_consensus": False,
            "pullback_type": None,
            "ma5": None,
            "ma10": None,
            "ma60": None,
            "ma120": None,
            "ma250": None,
            "pe": None,
            "pb": None,
            "forward_pe": None,
            "ev_ebitda": None,
            "foreign_net_5d": None,
            "trust_net_5d": None,
            "composite_score": None,
            "macro_regime": None,
            "timestamp": latest_ts
        }

        for r in records:
            if r.get("current_price") and not cand["current_price"]:
                cand["current_price"] = float(r["current_price"])
            if r.get("macro_regime") and not cand["macro_regime"]:
                cand["macro_regime"] = r["macro_regime"]

            # Technicals
            for ma in ("ma5", "ma10", "ma60", "ma120", "ma250"):
                if r.get(ma) is not None and cand[ma] is None:
                    cand[ma] = float(r[ma])
            if r.get("pullback_type") and not cand["pullback_type"]:
                cand["pullback_type"] = r["pullback_type"]

            # Fundamentals
            for fund in ("pe", "pb", "forward_pe", "ev_ebitda"):
                if r.get(fund) is not None and cand[fund] is None:
                    cand[fund] = float(r[fund])

            # Institutional
            for inst in ("foreign_net_5d", "trust_net_5d"):
                if r.get(inst) is not None and cand[inst] is None:
                    cand[inst] = int(r[inst])

            # Tags
            raw_t = r.get("tags")
            if raw_t:
                if isinstance(raw_t, str):
                    for tag_part in re.split(r"[|,]", raw_t):
                        t_clean = tag_part.strip()
                        if t_clean:
                            merged_tags.add(t_clean)
                elif isinstance(raw_t, list):
                    for t_item in raw_t:
                        t_clean = str(t_item).strip()
                        if t_clean:
                            merged_tags.add(t_clean)

            # Strategy-specific metrics
            m_name = (r.get("model_name") or "").upper()
            s_type = (r.get("strategy_type") or "").upper()
            pot = r.get("potential")
            pred = r.get("predicted_price")

            if "TIMESTRAL" in m_name or "TIMESTRAL" in s_type:
                if pot is not None:
                    cand["timestral_potential"] = float(pot)
                if pred is not None:
                    cand["timestral_predicted_price"] = float(pred)
                    if cand.get("predicted_price") is None:
                        cand["predicted_price"] = float(pred)
                if r.get("timestral_p10") is not None:
                    cand["timestral_p10"] = float(r["timestral_p10"])
                if r.get("timestral_p90") is not None:
                    cand["timestral_p90"] = float(r["timestral_p90"])
            elif "TIMESFM" in m_name or "TIMESFM" in s_type:
                if pot is not None:
                    cand["timesfm_potential"] = float(pot)
                if pred is not None:
                    cand["timesfm_predicted_price"] = float(pred)
            elif "MOIRAI" in m_name or "MOIRAI" in s_type:
                if pot is not None:
                    cand["moirai_potential"] = float(pot)
                if pred is not None:
                    cand["moirai_predicted_price"] = float(pred)
                if r.get("moirai_p10") is not None:
                    cand["moirai_p10"] = float(r["moirai_p10"])
                if r.get("moirai_p90") is not None:
                    cand["moirai_p90"] = float(r["moirai_p90"])
            elif "多維共振" in m_name or "多維共振" in s_type:
                if pot is not None:
                    cand["potential"] = float(pot)
                if pred is not None:
                    cand["predicted_price"] = float(pred)
                if r.get("dual_model_consensus") is not None:
                    cand["dual_model_consensus"] = bool(r["dual_model_consensus"])
                if r.get("tri_model_consensus") is not None:
                    cand["tri_model_consensus"] = bool(r["tri_model_consensus"])
            # Composite score (preserve valid 0.0 scores from macro risk control)
            c_score = r.get("composite_score")
            if c_score is not None and not pd.isna(c_score):
                curr_c = cand["composite_score"]
                cand["composite_score"] = float(c_score) if curr_c is None else max(curr_c, float(c_score))

        # Reconcile potential
        if cand["potential"] is None:
            if cand["timestral_potential"] is not None:
                cand["potential"] = cand["timestral_potential"]
            elif cand["timesfm_potential"] is not None:
                cand["potential"] = cand["timesfm_potential"]
            elif cand["moirai_potential"] is not None:
                cand["potential"] = cand["moirai_potential"]

        cand["score"] = cand["composite_score"]
        cand["tags"] = list(merged_tags)
        return cand

    _RESOLVER_CACHE: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
    _RESOLVER_CACHE_TTL: float = 86400.0
    _RESOLVER_CACHE_MAX_SIZE: int = 2000

    def resolve_ticker(self, value: str) -> Optional[Dict[str, Any]]:
        """Resolve a ticker or company name using explicit aliases, memory cache, and local metadata."""
        query = clean_query(value)
        if not query:
            return None

        cache_key = query.casefold()
        now = time.time()
        with self._RESOLVER_LOCK:
            if cache_key in self._RESOLVER_CACHE:
                cached_time, cached_result = self._RESOLVER_CACHE[cache_key]
                if now - cached_time < self._RESOLVER_CACHE_TTL:
                    return cached_result

        def _cache_and_return(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            with self._RESOLVER_LOCK:
                if len(self._RESOLVER_CACHE) >= self._RESOLVER_CACHE_MAX_SIZE:
                    for k in list(self._RESOLVER_CACHE.keys())[:400]:
                        self._RESOLVER_CACHE.pop(k, None)
                self._RESOLVER_CACHE[cache_key] = (now, result)
            return result

        alias_ticker = resolve_alias(query)
        normalized = alias_ticker or normalize_ticker(query)
        lookup_values = [query, normalized]

        if self.enabled:
            try:
                with self._get_connection(read_only=True) as con:
                    # 1. 優先查 market_universe 全球股票庫 (TW, US, JP, HK 等)
                    try:
                        univ_row = con.execute(
                            """
                            SELECT normalized_symbol, COALESCE(name_local, name_en)
                            FROM market_universe
                            WHERE lower(trim(normalized_symbol)) IN (lower(trim(?)), lower(trim(?)))
                               OR lower(trim(local_symbol)) = lower(trim(?))
                               OR lower(trim(name_local)) = lower(trim(?))
                               OR lower(trim(name_en)) = lower(trim(?))
                            ORDER BY 
                               CASE WHEN lower(trim(normalized_symbol)) IN (lower(trim(?)), lower(trim(?))) THEN 0 ELSE 1 END,
                               CASE WHEN normalized_symbol LIKE '%.TW%' THEN 1 ELSE 2 END,
                               snapshot_date DESC NULLS LAST
                            LIMIT 1
                            """,
                            [query, normalized, query, query, query, query, normalized],
                        ).fetchone()
                        if univ_row and univ_row[0]:
                            return _cache_and_return({
                                "query": query,
                                "ticker": str(univ_row[0]).upper(),
                                "name": univ_row[1],
                                "matched_by": "database"
                            })
                    except Exception as e_univ:
                        logger.debug("market_universe 查詢略過: %s", e_univ)

                    # 2. 查台股歷史行情與籌碼表
                    row = con.execute(
                        """
                        SELECT ticker, name
                        FROM (
                            SELECT ticker, raw_code, name, date FROM tw_daily_bars
                            UNION ALL
                            SELECT ticker, raw_code, name, date FROM tw_institutional_daily
                        ) metadata
                        WHERE lower(trim(name)) = lower(trim(?))
                           OR lower(trim(ticker)) = lower(trim(?))
                           OR lower(trim(ticker)) = lower(trim(?))
                           OR lower(trim(raw_code)) = lower(trim(?))
                        ORDER BY date DESC NULLS LAST
                        LIMIT 1
                        """,
                        [query, query, normalized, query],
                    ).fetchone()
                if row and row[0]:
                    return _cache_and_return({"query": query, "ticker": str(row[0]).upper(), "name": row[1], "matched_by": "database"})
            except Exception as exc:
                logger.warning("股票名稱解析略過本機 metadata 查詢: %s", exc)

        if alias_ticker:
            return _cache_and_return({"query": query, "ticker": alias_ticker, "name": None, "matched_by": "alias"})

        # Ticker-shaped input is accepted, but its history endpoint still
        # requires actual records before returning any quantitative data.
        if (
            re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", normalized)
            or re.fullmatch(r"\d{4,6}\.(TW|TWO)", normalized)
            or re.fullmatch(r"\d{4,5}[A-Z]?\.T", normalized)
            or re.fullmatch(r"\d{4,5}\.HK", normalized)
        ):
            return _cache_and_return({"query": query, "ticker": normalized, "name": None, "matched_by": "ticker"})
        return _cache_and_return(None)

    def autocomplete_tickers(self, query: str, limit: int = 12) -> List[Dict[str, Any]]:
        """
        Cross-market ticker autocomplete search for Command Palette (⌘K) and search bars.
        Matches tickers, local symbols, and Chinese/English company names.
        Returns deduplicated, relevance-sorted list of tickers.
        """
        clean_q = (query or "").strip()[:64]
        if not clean_q:
            return []

        limit = max(1, min(limit, 50))
        q_upper = clean_q.upper()
        like_upper = f"%{q_upper}%"
        prefix_upper = f"{q_upper}%"

        try:
            with self._get_connection(read_only=True) as con:
                rows = con.execute("""
                    WITH latest_preds AS (
                        SELECT ticker, current_price, potential,
                               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) as rn
                        FROM predictions
                        WHERE predicted_price IS NOT NULL OR current_price IS NOT NULL
                    ),
                    tw_names AS (
                        SELECT ticker, name,
                               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
                        FROM tw_daily_bars
                        WHERE name IS NOT NULL AND name != ''
                    ),
                    all_symbols AS (
                        SELECT normalized_symbol as ticker,
                               COALESCE(name_local, name_en) as name,
                               market
                        FROM market_universe
                        UNION
                        SELECT ticker,
                               name,
                               CASE WHEN ticker LIKE '%.TW' OR ticker LIKE '%.TWO' THEN 'TW'
                                    WHEN ticker LIKE '%.T' THEN 'JP'
                                    WHEN ticker LIKE '%.HK' THEN 'HK'
                                    ELSE 'US' END as market
                        FROM tw_names WHERE rn = 1
                        UNION
                        SELECT ticker,
                               NULL as name,
                               CASE WHEN ticker LIKE '%.TW' OR ticker LIKE '%.TWO' THEN 'TW'
                                    WHEN ticker LIKE '%.T' THEN 'JP'
                                    WHEN ticker LIKE '%.HK' THEN 'HK'
                                    ELSE 'US' END as market
                        FROM latest_preds
                    )
                    SELECT s.ticker,
                           COALESCE(MAX(tn.name), MAX(s.name), s.ticker) as name,
                           MAX(s.market) as market,
                           MAX(p.current_price) as current_price,
                           MAX(p.potential) as potential,
                           MIN(CASE
                               WHEN UPPER(s.ticker) = ? OR SPLIT_PART(s.ticker, '.', 1) = ? THEN 1
                               WHEN UPPER(s.ticker) LIKE ? THEN 2
                               WHEN UPPER(s.ticker) LIKE ? THEN 3
                               WHEN UPPER(COALESCE(tn.name, s.name, '')) LIKE ? THEN 4
                               ELSE 5
                           END) as rank_score
                    FROM all_symbols s
                    LEFT JOIN latest_preds p ON s.ticker = p.ticker AND p.rn = 1
                    LEFT JOIN tw_names tn ON s.ticker = tn.ticker AND tn.rn = 1
                    WHERE UPPER(s.ticker) LIKE ?
                       OR UPPER(COALESCE(tn.name, s.name, '')) LIKE ?
                    GROUP BY s.ticker
                    ORDER BY rank_score ASC, (MAX(p.current_price) IS NOT NULL) DESC, MAX(s.market) = 'TW' DESC, s.ticker ASC
                    LIMIT ?
                """, [
                    q_upper, q_upper,
                    f"{q_upper}.%",
                    prefix_upper,
                    prefix_upper,
                    like_upper,
                    like_upper,
                    limit
                ]).fetchall()

                results = []
                seen = set()
                for r in rows:
                    tk = str(r[0])
                    if tk in seen:
                        continue
                    seen.add(tk)
                    results.append({
                        "ticker": tk,
                        "name": r[1] or tk,
                        "market": r[2] or ("TW" if tk.endswith(".TW") else "US"),
                        "current_price": float(r[3]) if r[3] is not None else None,
                        "potential": round(float(r[4]), 2) if r[4] is not None else None,
                    })
                return results
        except DuckDBBusyError:
            raise
        except Exception as exc:
            logger.warning("autocomplete_tickers 查詢異常: %s", exc)
            return []

    def get_db_stats(self) -> Dict[str, Any]:
        """取得 DuckDB 全局時序庫統計資訊 (已排除內部路徑與測試資料)"""
        try:
            with self._get_connection(read_only=True) as con:
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
            with self._get_connection(read_only=True) as con:
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

        with self._write_lock():
            for attempt in range(3):
                try:
                    with self._get_connection(read_only=False) as con:
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
                except duckdb.IOException as e:
                    err_msg = str(e).lower()
                    if ("conflicting lock" in err_msg or "could not set lock" in err_msg or "unique file handle" in err_msg) and attempt < 2:
                        time.sleep(0.15 * (attempt + 1))
                        continue
                    logger.error(f"❌ 寫入 tw_daily_bars 失敗: {e}", exc_info=True)
                    return 0
                except Exception as e:
                    logger.error(f"❌ 寫入 tw_daily_bars 失敗: {e}", exc_info=True)
                    return 0
            return 0

    def upsert_global_daily_bars(self, bars: List[Dict[str, Any]]) -> int:
        """寫入或更新海外離線日 K 棒資料 (global_daily_bars)

        寫入強制走 self._write_lock() + read_only=False。
        """
        if not self.enabled or not bars:
            return 0

        df = pd.DataFrame(bars)
        required_cols = ["ticker", "market", "currency", "date", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"upsert_global_daily_bars 缺少必要欄位: {col}")

        if "is_adjusted" not in df.columns:
            df["is_adjusted"] = True
        if "updated_at" not in df.columns:
            df["updated_at"] = datetime.datetime.now()

        df["date"] = df["date"].astype(str).str.slice(0, 10)
        cols = ["ticker", "market", "currency", "date", "open", "high", "low", "close", "volume", "is_adjusted", "updated_at"]

        with self._write_lock():
            for attempt in range(3):
                try:
                    with self._get_connection(read_only=False) as con:
                        con.register("temp_global_bars_df", df[cols])
                        con.execute("""
                        INSERT OR REPLACE INTO global_daily_bars
                        SELECT 
                            CAST(ticker AS VARCHAR),
                            CAST(market AS VARCHAR),
                            CAST(currency AS VARCHAR),
                            CAST(date AS VARCHAR),
                            CAST(open AS DOUBLE),
                            CAST(high AS DOUBLE),
                            CAST(low AS DOUBLE),
                            CAST(close AS DOUBLE),
                            CAST(volume AS DOUBLE),
                            COALESCE(CAST(is_adjusted AS BOOLEAN), TRUE) AS is_adjusted,
                            COALESCE(TRY_CAST(updated_at AS TIMESTAMP), CURRENT_TIMESTAMP) AS updated_at
                        FROM temp_global_bars_df;
                        """)
                        con.unregister("temp_global_bars_df")
                    return len(df)
                except duckdb.IOException as e:
                    err_msg = str(e).lower()
                    if ("conflicting lock" in err_msg or "could not set lock" in err_msg or "unique file handle" in err_msg) and attempt < 2:
                        time.sleep(0.15 * (attempt + 1))
                        continue
                    logger.error(f"❌ 寫入 global_daily_bars 失敗: {e}", exc_info=True)
                    return 0
                except Exception as e:
                    logger.error(f"❌ 寫入 global_daily_bars 失敗: {e}", exc_info=True)
                    return 0
            return 0

    def load_local_close_series(
        self,
        tickers: List[str],
        as_of_date: Optional[str] = None,
        lookback_bars: int = 252
    ) -> Dict[str, Any]:
        """唯讀載入多市場收盤價歷史序列 (100% 零外部網路 I/O)

        查詢強制使用 `WHERE ... AND date <= ? ORDER BY date DESC LIMIT ?` 取最近視窗後反轉為 `date ASC`。
        依序查:
        1. tw_daily_bars (price_adjustment_mode="raw_close", currency="TWD")
        2. global_daily_bars (price_adjustment_mode="adjusted_close", currency from table)
        3. CacheManager.get_stock_data(ticker, period="1y"/"2y", max_age_hours=None) 本地磁碟快取備援
        資料不足或查無日 K 之標的列入 unavailable_tickers。
        """
        result_series: Dict[str, pd.Series] = {}
        price_adjustment_modes: Dict[str, str] = {}
        currencies: Dict[str, str] = {}
        unavailable_tickers: List[str] = []
        available_tickers: List[str] = []

        clean_as_of = str(as_of_date).strip()[:10] if as_of_date else None

        for raw_ticker in tickers:
            if not raw_ticker:
                continue
            tk = str(raw_ticker).strip().upper()
            loaded = False

            # 1. 查詢 tw_daily_bars
            clean_code = tk.split(".")[0]
            try:
                with self._get_connection(read_only=True) as con:
                    if clean_as_of:
                        sql = """
                        SELECT date, close
                        FROM tw_daily_bars
                        WHERE (ticker = ? OR raw_code = ? OR ticker LIKE ?)
                          AND close > 0
                          AND date <= ?
                        ORDER BY date DESC
                        LIMIT ?;
                        """
                        params = [tk, clean_code, f"{clean_code}.%", clean_as_of, lookback_bars]
                    else:
                        sql = """
                        SELECT date, close
                        FROM tw_daily_bars
                        WHERE (ticker = ? OR raw_code = ? OR ticker LIKE ?)
                          AND close > 0
                        ORDER BY date DESC
                        LIMIT ?;
                        """
                        params = [tk, clean_code, f"{clean_code}.%", lookback_bars]
                    rows = con.execute(sql, params).fetchall()
                    if rows:
                        rows.reverse()
                        dates = [str(r[0])[:10] for r in rows]
                        closes = [float(r[1]) for r in rows]
                        s = pd.Series(data=closes, index=dates, name="close")
                        s = s[~s.index.duplicated(keep="last")]
                        if len(s) > 0:
                            result_series[raw_ticker] = s
                            price_adjustment_modes[raw_ticker] = "raw_close"
                            currencies[raw_ticker] = "TWD"
                            available_tickers.append(raw_ticker)
                            loaded = True
            except Exception as e:
                logger.debug(f"tw_daily_bars 查詢 {tk} 失敗: {e}")

            if loaded:
                continue

            # 2. 查詢 global_daily_bars
            try:
                with self._get_connection(read_only=True) as con:
                    if clean_as_of:
                        sql_g = """
                        SELECT date, close, currency, is_adjusted
                        FROM global_daily_bars
                        WHERE ticker = ?
                          AND close > 0
                          AND date <= ?
                        ORDER BY date DESC
                        LIMIT ?;
                        """
                        params_g = [tk, clean_as_of, lookback_bars]
                    else:
                        sql_g = """
                        SELECT date, close, currency, is_adjusted
                        FROM global_daily_bars
                        WHERE ticker = ?
                          AND close > 0
                        ORDER BY date DESC
                        LIMIT ?;
                        """
                        params_g = [tk, lookback_bars]
                    rows = con.execute(sql_g, params_g).fetchall()
                    if rows:
                        rows.reverse()
                        dates = [str(r[0])[:10] for r in rows]
                        closes = [float(r[1]) for r in rows]
                        curr = str(rows[0][2]) if rows[0][2] else ("JPY" if tk.endswith(".T") else "USD")
                        is_adj = bool(rows[0][3]) if rows[0][3] is not None else True
                        s = pd.Series(data=closes, index=dates, name="close")
                        s = s[~s.index.duplicated(keep="last")]
                        if len(s) > 0:
                            result_series[raw_ticker] = s
                            price_adjustment_modes[raw_ticker] = "adjusted_close" if is_adj else "raw_close"
                            currencies[raw_ticker] = curr
                            available_tickers.append(raw_ticker)
                            loaded = True
            except Exception as e:
                logger.debug(f"global_daily_bars 查詢 {tk} 失敗: {e}")

            if loaded:
                continue

            # 3. 本地磁碟 CacheManager 快取備援 (100% 零外部網路 I/O)
            try:
                from data.cache import CacheManager
                cached_df = CacheManager.get_stock_data(tk, "1y", max_age_hours=None)
                if cached_df is None or cached_df.empty:
                    cached_df = CacheManager.get_stock_data(tk, "2y", max_age_hours=None)
                if cached_df is None or cached_df.empty:
                    cached_df = CacheManager.get_stock_data(raw_ticker, "1y", max_age_hours=None)

                if cached_df is not None and not cached_df.empty:
                    df_c = cached_df.copy()
                    col_close = "Close" if "Close" in df_c.columns else ("close" if "close" in df_c.columns else None)
                    if col_close:
                        if not isinstance(df_c.index, pd.DatetimeIndex) and "date" in df_c.columns:
                            df_c["Date"] = pd.to_datetime(df_c["date"])
                            df_c.set_index("Date", inplace=True)
                        elif not isinstance(df_c.index, pd.DatetimeIndex) and "Date" in df_c.columns:
                            df_c["Date"] = pd.to_datetime(df_c["Date"])
                            df_c.set_index("Date", inplace=True)

                        df_c = df_c.sort_index(ascending=False)
                        if clean_as_of:
                            as_of_ts = pd.to_datetime(clean_as_of)
                            df_c = df_c[df_c.index <= as_of_ts]
                        df_c = df_c.head(lookback_bars)
                        df_c = df_c.sort_index(ascending=True)

                        if not df_c.empty:
                            dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10] for d in df_c.index]
                            closes = df_c[col_close].astype(float).tolist()
                            s = pd.Series(data=closes, index=dates, name="close")
                            s = s[~s.index.duplicated(keep="last")]
                            if len(s) > 0:
                                is_tw = tk.endswith(".TW") or tk.endswith(".TWO")
                                result_series[raw_ticker] = s
                                price_adjustment_modes[raw_ticker] = "raw_close" if is_tw else "adjusted_close"
                                currencies[raw_ticker] = "TWD" if is_tw else ("JPY" if tk.endswith(".T") else "USD")
                                available_tickers.append(raw_ticker)
                                loaded = True
            except Exception as e:
                logger.debug(f"CacheManager 讀取 {tk} 失敗: {e}")

            if not loaded:
                unavailable_tickers.append(raw_ticker)

        return {
            "series": result_series,
            "price_adjustment_modes": price_adjustment_modes,
            "currencies": currencies,
            "unavailable_tickers": unavailable_tickers,
            "available_tickers": available_tickers,
            "as_of_date": clean_as_of,
            "lookback_bars": lookback_bars,
        }

    def save_market_universe_snapshot(self, snapshot_date: str, records: List[Dict[str, Any]]) -> int:
        """Persist one validated source snapshot while retaining previous dates."""
        if not self.enabled or not records:
            return 0

        df = pd.DataFrame(records)
        expected_cols = [
            "source_id", "market", "exchange", "snapshot_date", "local_symbol",
            "normalized_symbol", "name_local", "name_en", "isin", "security_type",
            "listing_status", "market_category", "financial_status", "board_lot",
            "source", "source_updated_at", "updated_at",
        ]
        df["snapshot_date"] = snapshot_date
        df["updated_at"] = datetime.datetime.now(TZ_TAIPEI).isoformat()
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        with self._write_lock():
            for attempt in range(3):
                try:
                    with self._get_connection(read_only=False) as con:
                        con.register("temp_market_universe_df", df[expected_cols])
                        con.execute("""
                        DELETE FROM market_universe
                        WHERE snapshot_date = ?
                          AND source_id IN (SELECT DISTINCT source_id FROM temp_market_universe_df)
                        """, [snapshot_date])
                        con.execute("INSERT INTO market_universe SELECT * FROM temp_market_universe_df")
                    logger.info("💾 成功寫入/更新 %s 筆全球清冊資料至 DuckDB", len(df))
                    return len(df)
                except duckdb.IOException as e:
                    err_msg = str(e).lower()
                    if ("conflicting lock" in err_msg or "could not set lock" in err_msg or "unique file handle" in err_msg) and attempt < 2:
                        time.sleep(0.15 * (attempt + 1))
                        continue
                    logger.error("❌ 寫入 market_universe 失敗: %s", e, exc_info=True)
                    return 0
                except Exception as e:
                    logger.error("❌ 寫入 market_universe 失敗: %s", e, exc_info=True)
                    return 0
            return 0

    def get_latest_market_universe(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the latest persisted snapshot, optionally for one source."""
        if not self.enabled:
            return []
        try:
            with self._get_connection(read_only=True) as con:
                if source_id:
                    query = """
                    SELECT * FROM market_universe
                    WHERE source_id = ?
                      AND snapshot_date = (SELECT MAX(snapshot_date) FROM market_universe WHERE source_id = ?)
                    ORDER BY normalized_symbol
                    """
                    return con.execute(query, [source_id, source_id]).df().to_dict(orient="records")
                return con.execute("""
                    SELECT * FROM market_universe
                    WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM market_universe)
                    ORDER BY source_id, normalized_symbol
                """).df().to_dict(orient="records")
        except Exception as e:
            logger.warning("讀取 market_universe 失敗: %s", e)
            return []

    def get_market_coverage(self) -> Dict[str, Any]:
        """取得當前 market_universe 各市場標的涵蓋統計 (唯讀查詢)"""
        if not self.enabled:
            return {}
        try:
            with self._get_connection(read_only=True) as con:
                df = con.execute("""
                    SELECT market, count(DISTINCT normalized_symbol) as count
                    FROM market_universe
                    WHERE listing_status = 'ACTIVE'
                    GROUP BY market
                """).df()
                return dict(zip(df["market"], df["count"])) if not df.empty else {}
        except Exception as e:
            logger.warning("取得市場涵蓋統計失敗: %s", e)
            return {}

    def get_active_universe(self, market: Optional[str] = None) -> List[Dict[str, Any]]:
        """取得有效追蹤股票清單 (listing_status='ACTIVE'，唯讀查詢)"""
        if not self.enabled:
            return []
        try:
            with self._get_connection(read_only=True) as con:
                if market:
                    df = con.execute("""
                        SELECT * FROM market_universe
                        WHERE listing_status = 'ACTIVE' AND market = ?
                        ORDER BY normalized_symbol
                    """, [market.upper()]).df()
                else:
                    df = con.execute("""
                        SELECT * FROM market_universe
                        WHERE listing_status = 'ACTIVE'
                        ORDER BY market, normalized_symbol
                    """).df()
                return self._clean_df_records(df)
        except Exception as e:
            logger.warning("取得 active_universe 失敗: %s", e)
            return []

    def get_tw_ranked_active_stocks(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        獲取依近期成交金額降序排列之台股有效活躍清冊。
        嚴格鎖定最新 market_universe 快照 (listing_status='ACTIVE') 與近期 30 天累積成交金額 > 0，
        全面排除下市、下櫃與零成交殭屍股；當日 K 無資料時自動降級回傳最新有效清冊。
        """
        if not self.enabled:
            return []
        try:
            with self._get_connection(read_only=True) as con:
                tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
                if "market_universe" not in tables:
                    return []

                # 檢查 tw_daily_bars 是否存在且有筆數
                has_bars = False
                if "tw_daily_bars" in tables:
                    bar_count = con.execute("SELECT count(*) FROM tw_daily_bars").fetchone()[0]
                    has_bars = bar_count > 0

                if not has_bars:
                    # 無日 K 或表為空時降級回傳 market_universe 最新有效快照標的
                    rows = con.execute("""
                        SELECT normalized_symbol, local_symbol, name_local, 0.0 as total_turnover
                        FROM market_universe
                        WHERE market = 'TW' 
                          AND listing_status = 'ACTIVE' 
                          AND security_type = 'EQUITY'
                          AND snapshot_date = (SELECT MAX(snapshot_date) FROM market_universe WHERE market = 'TW')
                        ORDER BY normalized_symbol
                        LIMIT ?
                    """, [limit]).fetchall()
                else:
                    query = """
                    SELECT 
                        u.normalized_symbol as ticker,
                        u.local_symbol,
                        u.name_local,
                        COALESCE(SUM(b.trade_value), 0.0) as total_turnover
                    FROM market_universe u
                    LEFT JOIN tw_daily_bars b ON u.normalized_symbol = b.ticker
                    WHERE u.market = 'TW'
                      AND u.listing_status = 'ACTIVE'
                      AND u.security_type = 'EQUITY'
                      AND u.snapshot_date = (SELECT MAX(snapshot_date) FROM market_universe WHERE market = 'TW')
                      AND (b.date IS NULL OR b.date >= (SELECT strftime('%Y-%m-%d', CAST(MAX(date) AS DATE) - INTERVAL '30 days') FROM tw_daily_bars))
                    GROUP BY u.normalized_symbol, u.local_symbol, u.name_local
                    HAVING total_turnover > 0
                    ORDER BY total_turnover DESC
                    LIMIT ?
                    """
                    rows = con.execute(query, [limit]).fetchall()

                return [
                    {
                        "ticker": r[0],
                        "local_symbol": r[1],
                        "name_local": r[2],
                        "total_turnover": float(r[3]),
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.warning(f"查詢台股活躍清冊失敗: {e}")
            return []

    def get_tw_symbol_map(self) -> Dict[str, str]:
        """獲取台股代號至標準化代號對照 (local_symbol -> normalized_symbol，如 3017 -> 3017.TW, 3324 -> 3324.TWO)"""
        if not self.enabled:
            return {}
        try:
            with self._get_connection(read_only=True) as con:
                tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
                if "market_universe" not in tables:
                    return {}
                rows = con.execute(
                    "SELECT local_symbol, normalized_symbol FROM market_universe WHERE market = 'TW' AND snapshot_date = (SELECT MAX(snapshot_date) FROM market_universe WHERE market = 'TW')"
                ).fetchall()
                return {r[0]: r[1] for r in rows if r[0] and r[1]}
        except Exception as e:
            logger.debug(f"讀取台股代號對照表失敗: {e}")
            return {}

    def get_tw_name_map(self) -> Dict[str, str]:
        """獲取台股代號與中文名稱對照表 (自 market_universe 最新快照提取)"""
        if not self.enabled:
            return {}
        try:
            with self._get_connection(read_only=True) as con:
                tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
                if "market_universe" not in tables:
                    return {}
                rows = con.execute(
                    "SELECT DISTINCT normalized_symbol, name_local FROM market_universe WHERE market = 'TW' AND snapshot_date = (SELECT MAX(snapshot_date) FROM market_universe WHERE market = 'TW')"
                ).fetchall()
                return {r[0]: r[1] for r in rows if r[0] and r[1]}
        except Exception as e:
            logger.debug(f"讀取台股名稱對照表失敗: {e}")
            return {}

    def get_jp_name_map(self) -> Dict[str, str]:
        """獲取日股代號與名稱對照表 (自 market_universe 最新快照提取)"""
        if not self.enabled:
            return {}
        try:
            with self._get_connection(read_only=True) as con:
                tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
                if "market_universe" not in tables:
                    return {}
                rows = con.execute(
                    "SELECT DISTINCT normalized_symbol, COALESCE(name_local, name_en) FROM market_universe WHERE market = 'JP' AND snapshot_date = (SELECT MAX(snapshot_date) FROM market_universe WHERE market = 'JP')"
                ).fetchall()
                return {r[0]: r[1] for r in rows if r[0] and r[1]}
        except Exception as e:
            logger.debug(f"讀取日股名稱對照表失敗: {e}")
            return {}

    def save_universe_sync_run(
        self,
        source_id: str,
        snapshot_date: Optional[str],
        stale: bool,
        record_count: int,
        error: Optional[str],
    ) -> None:
        if not self.enabled:
            return
        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
                con.execute(
                    "INSERT INTO universe_sync_runs VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        source_id,
                        snapshot_date,
                        stale,
                        record_count,
                        error,
                        datetime.datetime.now(TZ_TAIPEI).isoformat(),
                    ],
                )
        except Exception as e:
            logger.warning("寫入 universe_sync_runs 失敗: %s", e)

    def get_latest_universe_sync_status(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        """Return the most recent fresh/stale/error status for a source."""
        if not self.enabled:
            return {}
        try:
            with self._get_connection(read_only=True) as con:
                if source_id:
                    row = con.execute(
                        """
                        SELECT source_id, snapshot_date, stale, record_count, error, updated_at
                        FROM universe_sync_runs
                        WHERE source_id = ?
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        [source_id],
                    ).fetchone()
                else:
                    row = con.execute(
                        """
                        SELECT source_id, snapshot_date, stale, record_count, error, updated_at
                        FROM universe_sync_runs
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """
                    ).fetchone()
                if not row:
                    return {}
                return dict(zip(["source_id", "snapshot_date", "stale", "record_count", "error", "updated_at"], row))
        except Exception as e:
            logger.warning("讀取 universe_sync_runs 失敗: %s", e)
            return {}

    def get_daily_bars_for_ticker(self, ticker: str, limit: int = 120) -> pd.DataFrame:
        """
        從 DuckDB 讀取特定個股的歷史日 K 棒數據，格式對齊 yfinance OHLCV
        """
        if not self.enabled:
            return pd.DataFrame()

        t_clean = str(ticker).strip()
        t_upper = t_clean.upper()
        is_tw = t_upper.endswith(".TW") or t_upper.endswith(".TWO") or (t_upper.isdigit() and len(t_upper) in (4, 5, 6))

        if not is_tw:
            sql = """
            SELECT 
                date AS Date,
                open AS Open,
                high AS High,
                low AS Low,
                close AS Close,
                volume AS Volume
            FROM global_daily_bars
            WHERE ticker = ?
              AND close > 0
            ORDER BY date DESC
            LIMIT ?;
            """
            try:
                with self._get_connection(read_only=True) as con:
                    df = con.execute(sql, [t_upper, limit]).df()
                    if df.empty:
                        # Fallback to tw_daily_bars in case foreign ticker was stored there
                        sql_fb = """
                        SELECT date AS Date, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
                        FROM tw_daily_bars WHERE ticker = ? AND close > 0 ORDER BY date DESC LIMIT ?;
                        """
                        df = con.execute(sql_fb, [t_upper, limit]).df()
                    if not df.empty:
                        df["Date"] = pd.to_datetime(df["Date"])
                        df.sort_values("Date", ascending=True, inplace=True)
                        df.set_index("Date", inplace=True)
                    return df
            except Exception as e:
                logger.warning(f"⚠️ 從 DuckDB 讀取 {ticker} 日 K 失敗: {e}")
                return pd.DataFrame()

        clean_code = t_clean.split(".")[0]
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
        ORDER BY date DESC
        LIMIT ?;
        """
        try:
            with self._get_connection(read_only=True) as con:
                df = con.execute(sql, [ticker, clean_code, f"{clean_code}.%", limit]).df()
                if not df.empty:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.sort_values("Date", ascending=True, inplace=True)
                    df.set_index("Date", inplace=True)
                return df
        except Exception as e:
            logger.warning(f"⚠️ 從 DuckDB 讀取 {ticker} 日 K 失敗: {e}")
            return pd.DataFrame()

    def save_institutional_flows_batch(self, records: Any) -> int:
        """
        批次寫入三大法人每日進出與持股數據，並自動去重 (Upsert on date + ticker)
        支援傳入 List[Dict[str, Any]] 或 pandas.DataFrame
        """
        if not self.enabled or records is None:
            return 0

        if isinstance(records, pd.DataFrame):
            df = records.copy()
        elif isinstance(records, list):
            if not records:
                return 0
            df = pd.DataFrame(records)
        else:
            return 0

        if df.empty:
            return 0

        expected_cols = [
            "date", "ticker", "raw_code", "name",
            "foreign_net", "trust_net", "dealer_net", "total_net",
            "foreign_ratio", "total_shares", "foreign_shares",
            "market", "created_at"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        with self._write_lock():
            for attempt in range(3):
                try:
                    with self._get_connection(read_only=False) as con:
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
                except duckdb.IOException as e:
                    err_msg = str(e).lower()
                    if ("conflicting lock" in err_msg or "could not set lock" in err_msg or "unique file handle" in err_msg) and attempt < 2:
                        time.sleep(0.15 * (attempt + 1))
                        continue
                    logger.error(f"❌ 寫入 tw_institutional_daily 失敗: {e}", exc_info=True)
                    return 0
                except Exception as e:
                    logger.error(f"❌ 寫入 tw_institutional_daily 失敗: {e}", exc_info=True)
                    return 0
            return 0

    def get_institutional_flow_for_ticker(self, ticker: str, limit: int = 60) -> pd.DataFrame:
        """
        取得特定標的之歷史法人籌碼進出時序 (非台股標的直接回傳空 DataFrame)
        """
        if not self.enabled:
            return pd.DataFrame()

        t_clean = str(ticker).strip()
        t_upper = t_clean.upper()
        is_tw = t_upper.endswith(".TW") or t_upper.endswith(".TWO") or (t_upper.isdigit() and len(t_upper) in (4, 5, 6))
        if not is_tw:
            return pd.DataFrame()

        clean_code = t_clean.split(".")[0]
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
            with self._get_connection(read_only=True) as con:
                df = con.execute(sql, [ticker, clean_code, f"{clean_code}.%", limit]).df()
                return df
        except Exception as e:
            logger.warning(f"⚠️ 從 DuckDB 讀取 {ticker} 法人時序失敗: {e}")
            return pd.DataFrame()

    def get_institutional_flow_summary(self, ticker: str, *args, **kwargs) -> Dict[str, Any]:
        """相容性方法：取得三大法人進出摘要 (若非台股直接回傳空字典)"""
        t_clean = str(ticker).strip()
        t_upper = t_clean.upper()
        is_tw = t_upper.endswith(".TW") or t_upper.endswith(".TWO") or (t_upper.isdigit() and len(t_upper) in (4, 5, 6))
        if not is_tw:
            return {}
        return {}

    def save_broker_trades_batch(self, records: Union[List[Dict[str, Any]], pd.DataFrame]) -> int:
        """批次寫入券商分點進出資料 (tw_broker_trades)"""
        if not self.enabled:
            return 0
        if isinstance(records, list) and not records:
            return 0
        if isinstance(records, pd.DataFrame) and records.empty:
            return 0

        if isinstance(records, pd.DataFrame):
            df = records.copy()
        else:
            df = pd.DataFrame(records)

        expected_cols = [
            "date", "ticker", "raw_code", "broker_name", "broker_id",
            "buy_vol", "sell_vol", "net_vol", "pct", "rank", "side", "created_at"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        with self._write_lock():
            for attempt in range(3):
                try:
                    with self._get_connection(read_only=False) as con:
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
                except duckdb.IOException as e:
                    err_msg = str(e).lower()
                    if ("conflicting lock" in err_msg or "could not set lock" in err_msg or "unique file handle" in err_msg) and attempt < 2:
                        time.sleep(0.15 * (attempt + 1))
                        continue
                    logger.error(f"❌ 寫入 tw_broker_trades 失敗: {e}", exc_info=True)
                    return 0
                except Exception as e:
                    logger.error(f"❌ 寫入 tw_broker_trades 失敗: {e}", exc_info=True)
                    return 0
            return 0

    def get_broker_trades_for_ticker(self, ticker: str, limit: int = 100) -> pd.DataFrame:
        """
        取得特定標的最近的券商分點進出明細 (非台股標的直接回傳空 DataFrame)
        """
        if not self.enabled:
            return pd.DataFrame()

        t_clean = str(ticker).strip()
        t_upper = t_clean.upper()
        is_tw = t_upper.endswith(".TW") or t_upper.endswith(".TWO") or (t_upper.isdigit() and len(t_upper) in (4, 5, 6))
        if not is_tw:
            return pd.DataFrame()

        clean_code = t_clean.split(".")[0]
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
            with self._get_connection(read_only=True) as con:
                df = con.execute(sql, [ticker, clean_code, f"{clean_code}.%", limit]).df()
                return df
        except Exception as e:
            logger.warning(f"⚠️ 從 DuckDB 讀取 {ticker} 券商分點失敗: {e}")
            return pd.DataFrame()

    def get_broker_top_summary(self, ticker: str, limit_days: int = 20) -> Dict[str, Any]:
        """
        取得特定標的在過去 N 個交易日中，累計買超與賣超最多的券商分點摘要 (非台股標的直接回傳空結構)
        """
        empty_res = {"top_buyers": [], "top_sellers": [], "buy_top": [], "sell_top": []}
        if not self.enabled:
            return empty_res

        t_clean = str(ticker).strip()
        t_upper = t_clean.upper()
        is_tw = t_upper.endswith(".TW") or t_upper.endswith(".TWO") or (t_upper.isdigit() and len(t_upper) in (4, 5, 6))
        if not is_tw:
            return empty_res

        clean_code = t_clean.split(".")[0]
        try:
            with self._get_connection(read_only=True) as con:
                # 取得可用日期清單
                dates_df = con.execute("""
                    SELECT DISTINCT date 
                    FROM tw_broker_trades 
                    WHERE (ticker = ? OR raw_code = ? OR ticker LIKE ?)
                    ORDER BY date DESC 
                    LIMIT ?;
                """, [ticker, clean_code, f"{clean_code}.%", limit_days]).df()
                
                if dates_df.empty:
                    return empty_res

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
                    return empty_res

                top_buyers = df[df["total_net"] > 0].head(15).to_dict(orient="records")
                top_sellers = df[df["total_net"] < 0].sort_values("total_net", ascending=True).head(15).to_dict(orient="records")

                return {
                    "ticker": ticker,
                    "days": len(dates_df),
                    "start_date": str(min_date),
                    "end_date": str(dates_df["date"].max()),
                    "top_buyers": top_buyers,
                    "top_sellers": top_sellers,
                    "buy_top": top_buyers,
                    "sell_top": top_sellers
                }
        except Exception as e:
            logger.warning(f"⚠️ 從 DuckDB 統計 {ticker} 券商主力摘要失敗: {e}")
            return empty_res

    def save_earnings_calls_batch(self, records: List[Dict[str, Any]]) -> int:
        """批次儲存或更新法說會 (Earnings Calls) 紀錄 (Upsert by id)"""
        if not self.enabled or not records:
            return 0
        try:
            now_str = datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame(records)
            expected_cols = [
                "id", "market", "ticker", "name", "call_date", "call_time", "timezone",
                "timestamp_utc", "meeting_type", "location", "webcast_url",
                "presentation_url", "transcript_url", "language", "eps_estimate",
                "eps_actual", "revenue_estimate", "revenue_actual", "is_weight_stock",
                "remarks", "financial_period", "guidance_summary", "guidance_outlook",
                "guidance_margin", "guidance_capex", "guidance_qa", "source",
                "created_at", "updated_at"
            ]
            for col in expected_cols:
                if col not in df.columns:
                    if col in ("eps_estimate", "eps_actual", "revenue_estimate", "revenue_actual"):
                        df[col] = None
                    elif col == "is_weight_stock":
                        df[col] = False
                    elif col in ("created_at", "updated_at"):
                        df[col] = now_str
                    elif col == "timestamp_utc":
                        df[col] = None
                    elif col in ("financial_period", "guidance_summary", "guidance_outlook", "guidance_margin", "guidance_capex", "guidance_qa"):
                        df[col] = None
                    else:
                        df[col] = ""
            df["updated_at"] = now_str
            df = df[expected_cols]

            with self._write_lock():
                with self._get_connection(read_only=False) as con:
                    con.register("df_earnings_batch", df)
                    con.execute("""
                    INSERT OR REPLACE INTO earnings_calls BY NAME
                    SELECT * FROM df_earnings_batch;
                    """)
            logger.info(f"✅ 成功批次寫入/更新 {len(df)} 筆法說會資料")
            return len(df)
        except Exception as e:
            logger.error(f"❌ 批次寫入法說會失敗: {e}")
            return 0

    def get_earnings_calls(
        self,
        market: str = "ALL",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ticker: Optional[str] = None,
        is_weight_stock: Optional[bool] = None,
        order_desc: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """查詢法說會時程與歷史清單"""
        if not self.enabled:
            return []
        try:
            query = "SELECT * FROM earnings_calls WHERE 1=1"
            params = []
            if isinstance(market, str) and market.strip() and market.upper() != "ALL":
                query += " AND market = ?"
                params.append(market.upper())
            if isinstance(start_date, str) and start_date.strip():
                query += " AND call_date >= ?"
                params.append(start_date.strip())
            if isinstance(end_date, str) and end_date.strip():
                query += " AND call_date <= ?"
                params.append(end_date.strip())
            if isinstance(ticker, str) and ticker.strip():
                clean_t = ticker.strip().upper()
                query += " AND (ticker = ? OR ticker LIKE ?)"
                params.append(clean_t)
                params.append(f"{clean_t}.%")
            if isinstance(is_weight_stock, bool):
                query += " AND is_weight_stock = ?"
                params.append(is_weight_stock)
            
            lim = int(limit) if isinstance(limit, (int, str)) and str(limit).isdigit() else 100
            order_clause = "ORDER BY call_date DESC, call_time DESC" if order_desc else "ORDER BY call_date ASC, call_time ASC"
            query += f" {order_clause} LIMIT ?"
            params.append(lim)

            with self._get_connection(read_only=True) as con:
                df = con.execute(query, params).df()
                if df.empty:
                    return []
                df = df.where(pd.notna(df), None)
                return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"❌ 查詢法說會資料失敗: {e}")
            return []

    def get_earnings_calls_count(self) -> int:
        """取得法說會總記錄數 (用於冷啟動判定)"""
        if not self.enabled:
            return 0
        try:
            with self._get_connection(read_only=True) as con:
                val = con.execute("SELECT COUNT(*) FROM earnings_calls;").fetchone()
                return int(val[0]) if val else 0
        except Exception:
            return 0

    def get_earnings_call_counts(
        self,
        market: str = "ALL",
        today_str: str = "",
        future_end_str: str = "",
        past_start_str: str = "",
        is_weight_stock: Optional[bool] = None
    ) -> Dict[str, int]:
        """單次查詢計算今天、未來與過去法說會精確總數 (不受 LIMIT 截斷影響)"""
        if not self.enabled:
            return {"today": 0, "upcoming": 0, "past": 0}
        try:
            query = """
            SELECT
                COUNT(CASE WHEN call_date = ? THEN 1 END) as today_cnt,
                COUNT(CASE WHEN call_date > ? AND call_date <= ? THEN 1 END) as upcoming_cnt,
                COUNT(CASE WHEN call_date < ? AND call_date >= ? THEN 1 END) as past_cnt
            FROM earnings_calls
            WHERE 1=1
            """
            params = [today_str, today_str, future_end_str, today_str, past_start_str]
            if isinstance(market, str) and market.strip() and market.upper() != "ALL":
                query += " AND market = ?"
                params.append(market.upper())
            if isinstance(is_weight_stock, bool):
                query += " AND is_weight_stock = ?"
                params.append(is_weight_stock)

            with self._get_connection(read_only=True) as con:
                row = con.execute(query, params).fetchone()
                if row:
                    return {
                        "today": int(row[0] or 0),
                        "upcoming": int(row[1] or 0),
                        "past": int(row[2] or 0)
                    }
                return {"today": 0, "upcoming": 0, "past": 0}
        except Exception as e:
            logger.error(f"❌ 計算法說會類別總數失敗: {e}")
            return {"today": 0, "upcoming": 0, "past": 0}

    def get_earnings_calendar(
        self,
        year: int,
        month: int,
        market: str = "ALL"
    ) -> List[Dict[str, Any]]:
        """取得指定月份之每日法說會場次計數、星期幾與結構化個股明細 (對齊 BigGo 設計)"""
        if not self.enabled:
            return []
        try:
            month_prefix = f"{year:04d}-{month:02d}-%"
            query = "SELECT * FROM earnings_calls WHERE call_date LIKE ?"
            params = [month_prefix]
            if market and market.upper() != "ALL":
                query += " AND market = ?"
                params.append(market.upper())
            query += " ORDER BY call_date DESC, call_time ASC;"

            with self._get_connection(read_only=True) as con:
                df = con.execute(query, params).df()
                if df.empty:
                    return []
                df = df.where(pd.notna(df), None)
                records = df.to_dict(orient="records")

            def _clean_str(val):
                if val is None or pd.isna(val):
                    return None
                s = str(val).strip()
                return s if s and s.lower() not in ("nan", "none", "nat") else None

            def _clean_bool(val):
                if val is None or pd.isna(val):
                    return False
                if isinstance(val, bool):
                    return val
                if isinstance(val, (int, float)):
                    return bool(val) and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val)))
                return str(val).strip().lower() in ("true", "1", "yes", "y", "t")

            weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            grouped: Dict[str, Dict[str, Any]] = {}

            for r in records:
                c_date = str(r.get("call_date") or "").strip()
                if not c_date:
                    continue
                if c_date not in grouped:
                    wk = ""
                    try:
                        p = [int(x) for x in c_date.split("-")]
                        dt = datetime.date(p[0], p[1], p[2])
                        wk = weekday_names[dt.weekday()]
                    except Exception:
                        pass
                    grouped[c_date] = {
                        "call_date": c_date,
                        "weekday": wk,
                        "total_calls": 0,
                        "weight_stock_calls": 0,
                        "summary_companies": "",
                        "calls": []
                    }

                call_time_val = str(r.get("call_time") or "14:00").strip()
                call_time_clean = call_time_val + ":00" if len(call_time_val) == 5 else call_time_val
                tz_val = str(r.get("timezone") or ("America/New_York" if str(r.get("market", "")).upper() == "US" else "Asia/Taipei"))
                time_cst = call_time_val
                time_est = call_time_val
                try:
                    from zoneinfo import ZoneInfo
                    dt_naive = datetime.datetime.strptime(f"{c_date} {call_time_clean[:8]}", "%Y-%m-%d %H:%M:%S")
                    dt_loc = dt_naive.replace(tzinfo=ZoneInfo(tz_val))
                    time_cst = dt_loc.astimezone(ZoneInfo("Asia/Taipei")).strftime("%H:%M")
                    time_est = dt_loc.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M")
                except Exception:
                    pass

                clean_call = {
                    "id": _clean_str(r.get("id")) or "",
                    "market": _clean_str(r.get("market")) or "TW",
                    "ticker": _clean_str(r.get("ticker")) or "",
                    "name": _clean_str(r.get("name")) or "",
                    "call_date": c_date,
                    "call_time": call_time_val,
                    "call_time_cst": time_cst,
                    "call_time_est": time_est,
                    "timezone": tz_val,
                    "meeting_type": _clean_str(r.get("meeting_type")) or "online",
                    "location": _clean_str(r.get("location")),
                    "webcast_url": _clean_str(r.get("webcast_url")),
                    "presentation_url": _clean_str(r.get("presentation_url")),
                    "transcript_url": _clean_str(r.get("transcript_url")),
                    "is_weight_stock": _clean_bool(r.get("is_weight_stock")),
                    "remarks": _clean_str(r.get("remarks")),
                    "financial_period": _clean_str(r.get("financial_period")),
                    "guidance_summary": _clean_str(r.get("guidance_summary")),
                    "guidance_outlook": _clean_str(r.get("guidance_outlook")),
                    "guidance_margin": _clean_str(r.get("guidance_margin")),
                    "guidance_capex": _clean_str(r.get("guidance_capex")),
                    "guidance_qa": _clean_str(r.get("guidance_qa")),
                    "source": _clean_str(r.get("source")) or "BigGo"
                }
                grouped[c_date]["calls"].append(clean_call)
                grouped[c_date]["total_calls"] += 1
                if clean_call["is_weight_stock"]:
                    grouped[c_date]["weight_stock_calls"] += 1

            today_str = datetime.date.today().strftime("%Y-%m-%d")
            result: List[Dict[str, Any]] = []

            for c_date in sorted(grouped.keys(), reverse=True):
                day_entry = grouped[c_date]
                names = [f"{c['name']} ({c['ticker']})" for c in day_entry["calls"]]
                day_entry["summary_companies"] = "、".join(names[:10]) + (" 等" if len(names) > 10 else "")
                result.append(day_entry)

            # 智慧排序：今天最前 ➔ 未來日程 (由近到遠 ASC) ➔ 過去日程 (由近到遠 DESC，昨天優先)
            def _date_sort_priority(entry: Dict[str, Any]):
                d = entry["call_date"]
                if d == today_str:
                    return (0, 0)
                elif d > today_str:
                    # 未來日期，越早越前
                    return (1, int(d.replace("-", "")))
                else:
                    # 過去日期，越晚越前 (昨天 > 前天)
                    return (2, -int(d.replace("-", "")))

            result.sort(key=_date_sort_priority)
            return result
        except Exception as e:
            logger.error(f"❌ 取得法說會日曆彙總失敗: {e}")
            return []

    # ==================== Macro Snapshots Methods ====================

    def save_macro_snapshot(self, category: str, data: Any, source: str = "Investing.com") -> bool:
        """持久化宏觀數據與行事曆快照至 DuckDB (以 category 為主鍵 Upsert)"""
        if not self.enabled or not category or data is None:
            return False
        try:
            now_str = datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S")
            data_json = json.dumps(data, ensure_ascii=False)
            with self._write_lock():
                with self._get_connection(read_only=False) as con:
                    con.execute("""
                    INSERT OR REPLACE INTO macro_snapshots (category, data_json, source, updated_at)
                    VALUES (?, ?, ?, ?);
                    """, [category, data_json, source, now_str])
            return True
        except Exception as e:
            logger.warning(f"⚠️ 寫入宏觀快照至 DuckDB 失敗 ({category}): {e}")
            return False

    def get_macro_snapshot(self, category: str, max_age_seconds: Optional[int] = None) -> Optional[Any]:
        """從 DuckDB 讀取宏觀數據或行事曆快照，支援效期檢查"""
        if not self.enabled or not category:
            return None
        try:
            with self._get_connection(read_only=True) as con:
                row = con.execute("""
                SELECT data_json, updated_at FROM macro_snapshots WHERE category = ?;
                """, [category]).fetchone()
                if not row:
                    return None
                data_json, updated_at = row
                if max_age_seconds is not None:
                    try:
                        up_time = datetime.datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ_TAIPEI)
                        now_time = datetime.datetime.now(TZ_TAIPEI)
                        if (now_time - up_time).total_seconds() > max_age_seconds:
                            return None
                    except Exception:
                        pass
                return json.loads(data_json)
        except Exception as e:
            logger.debug(f"讀取 DuckDB 宏觀快照失敗 ({category}): {e}")
            return None

    # ==================== Company Profile Methods ====================

    def save_company_profile(self, ticker: str, data: Dict[str, Any]) -> bool:
        """寫入公司基本資料與新聞快取至 DuckDB (支援使用者點擊後跨使用者持久化共享)"""
        if not self.enabled or not ticker or not data:
            return False
        try:
            now_str = datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S")
            data_json = json.dumps(data, ensure_ascii=False)
            t_upper = str(ticker).strip().upper()
            raw_code = str(data.get("raw_code") or re.sub(r"\.(TW|TWO)$", "", t_upper))
            name = str(data.get("name") or ticker)
            registered_name = str(data.get("registered_name") or "")
            unified_id = str(data.get("unified_id") or "")
            capital = str(data.get("capital") or "")
            industry = str(data.get("industry") or "綜合產業")
            chairman = str(data.get("chairman") or "")
            market_cap = str(data.get("market_cap") or "")
            established = str(data.get("established") or "")
            business_summary = str(data.get("business_summary") or "")
            recent_news_json = json.dumps(data.get("recent_news") or [], ensure_ascii=False)
            source = str(data.get("source") or "2md")

            is_tw = t_upper.endswith(".TW") or t_upper.endswith(".TWO")

            with self._write_lock():
                with self._get_connection(read_only=False) as con:
                    # 寫入主代碼
                    con.execute("""
                    INSERT OR REPLACE INTO company_profiles 
                    (ticker, raw_code, name, registered_name, unified_id, capital, industry, chairman, market_cap, established, business_summary, recent_news_json, data_json, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, [ticker, raw_code, name, registered_name, unified_id, capital, industry, chairman, market_cap, established, business_summary, recent_news_json, data_json, source, now_str])

                    # 僅限台股 (如 2371.TW) 同步寫入 2371 供純代碼檢索；非台股 (如 6770.T) 嚴禁寫入純數字以免污染
                    if is_tw and raw_code and raw_code != ticker and raw_code.isdigit():
                        con.execute("""
                        INSERT OR REPLACE INTO company_profiles 
                        (ticker, raw_code, name, registered_name, unified_id, capital, industry, chairman, market_cap, established, business_summary, recent_news_json, data_json, source, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, [raw_code, raw_code, name, registered_name, unified_id, capital, industry, chairman, market_cap, established, business_summary, recent_news_json, data_json, source, now_str])
            return True
        except Exception as e:
            logger.warning(f"⚠️ 寫入公司基本資料至 DuckDB 失敗 ({ticker}): {e}")
            return False

    def get_company_profile(self, ticker: str, max_age_seconds: Optional[int] = 86400 * 20) -> Optional[Dict[str, Any]]:
        """從 DuckDB 讀取公司基本資料，支援效期檢查（預設 20 天）與多代碼/名稱/統編多維命中"""
        if not self.enabled or not ticker:
            return None
        try:
            t_clean = str(ticker).strip()
            t_upper = t_clean.upper()
            is_tw = t_upper.endswith(".TW") or t_upper.endswith(".TWO")
            is_digit = t_upper.isdigit()

            with self._get_connection(read_only=True) as con:
                if is_tw or is_digit:
                    raw_code = re.sub(r"\.(TW|TWO)$", "", t_upper)
                    canon_tw = f"{raw_code}.TW"
                    canon_two = f"{raw_code}.TWO"
                    row = con.execute("""
                    SELECT data_json, updated_at FROM company_profiles 
                    WHERE ticker IN (?, ?, ?, ?, ?) OR raw_code = ? OR name = ? OR registered_name = ? OR unified_id = ?
                    ORDER BY updated_at DESC LIMIT 1;
                    """, [t_clean, t_upper, raw_code, canon_tw, canon_two, raw_code, t_clean, t_clean, t_clean]).fetchone()
                else:
                    # 國際市場標的 (如 6770.T, AAPL, 7203.T)：嚴格依完整 ticker 或公司名檢索，絕不可使用 raw_code 截斷比對
                    row = con.execute("""
                    SELECT data_json, updated_at FROM company_profiles 
                    WHERE ticker IN (?, ?) OR lower(name) = lower(?) OR lower(registered_name) = lower(?)
                    ORDER BY updated_at DESC LIMIT 1;
                    """, [t_clean, t_upper, t_clean, t_clean]).fetchone()
                if not row:
                    return None
                data_json, updated_at = row
                if max_age_seconds is not None:
                    try:
                        up_time = datetime.datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ_TAIPEI)
                        now_time = datetime.datetime.now(TZ_TAIPEI)
                        if (now_time - up_time).total_seconds() > max_age_seconds:
                            return None
                    except Exception:
                        pass
                return json.loads(data_json)
        except Exception as e:
            logger.debug(f"DuckDB 讀取公司基本資料快取失敗 ({ticker}): {e}")
            return None

    # ==================== Company Network Methods ====================

    def save_company_network(self, ticker: str, data: Dict[str, Any]) -> bool:
        """寫入企業架構、轉投資與股權網絡資料至 DuckDB"""
        if not self.enabled or not ticker or not data:
            return False
        try:
            now_str = datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S")
            data_json = json.dumps(data, ensure_ascii=False)
            name = str(data.get("name") or ticker)
            unified_id = str(data.get("unified_id") or "")
            capital_val = data.get("capital")
            capital = float(capital_val) if isinstance(capital_val, (int, float)) else 0.0
            representative = str(data.get("representative") or "")
            address = str(data.get("address") or "")
            web_url = str(data.get("web_url") or "")

            clean_code = re.sub(r"\.(TW|TWO)$", "", str(ticker).strip().upper())
            canonical_ticker = str(data.get("ticker") or ticker).strip().upper()

            with self._write_lock():
                with self._get_connection(read_only=False) as con:
                    con.execute("""
                    INSERT OR REPLACE INTO company_networks 
                    (ticker, name, unified_id, capital, representative, address, data_json, web_url, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, [clean_code, name, unified_id, capital, representative, address, data_json, web_url, now_str])
                    if canonical_ticker and canonical_ticker != clean_code:
                        con.execute("""
                        INSERT OR REPLACE INTO company_networks 
                        (ticker, name, unified_id, capital, representative, address, data_json, web_url, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, [canonical_ticker, name, unified_id, capital, representative, address, data_json, web_url, now_str])
            return True
        except Exception as e:
            logger.warning(f"⚠️ 寫入企業網絡至 DuckDB 失敗 ({ticker}): {e}")
            return False

    def get_company_network(self, ticker: str, max_age_seconds: Optional[int] = 86400 * 30) -> Optional[Dict[str, Any]]:
        """從 DuckDB 讀取企業架構與網絡資料，支援效期檢查與多代碼/名稱/統編多維命中"""
        if not self.enabled or not ticker:
            return None
        try:
            t_clean = str(ticker).strip()
            t_upper = t_clean.upper()
            raw_code = re.sub(r"\.(TW|TWO)$", "", t_upper)
            canon_tw = f"{raw_code}.TW" if raw_code.isdigit() else t_upper
            canon_two = f"{raw_code}.TWO" if raw_code.isdigit() else t_upper

            with self._get_connection(read_only=True) as con:
                row = con.execute("""
                SELECT data_json, updated_at FROM company_networks 
                WHERE ticker IN (?, ?, ?, ?, ?) OR name = ? OR unified_id = ?
                ORDER BY updated_at DESC LIMIT 1;
                """, [t_clean, t_upper, raw_code, canon_tw, canon_two, t_clean, t_clean]).fetchone()
                if not row:
                    return None
                data_json, updated_at = row
                if max_age_seconds is not None:
                    try:
                        up_time = datetime.datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ_TAIPEI)
                        now_time = datetime.datetime.now(TZ_TAIPEI)
                        if (now_time - up_time).total_seconds() > max_age_seconds:
                            return None
                    except Exception:
                        pass
                return json.loads(data_json)
        except Exception as e:
            logger.debug(f"DuckDB 讀取企業網絡快取失敗 ({ticker}): {e}")
            return None

    # ==================== Company Tender Methods (ToBid) ====================

    def save_company_tender(self, ticker: str, data: Dict[str, Any]) -> bool:
        """寫入企業政府採購標案統計與情報至 DuckDB"""
        if not self.enabled or not ticker or not data:
            return False
        try:
            now_str = datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S")
            data_json = json.dumps(data, ensure_ascii=False)
            name = str(data.get("name") or ticker)
            tax_id = str(data.get("tax_id") or "")
            wins = int(data.get("wins", 0))
            participations = int(data.get("participations", 0))
            win_rate = float(data.get("win_rate_pct", 0.0))
            award_total = float(data.get("known_award_total", 0.0))
            web_url = str(data.get("web_url") or "")

            clean_code = re.sub(r"\.(TW|TWO)$", "", str(ticker).strip().upper())
            canonical_ticker = str(data.get("ticker") or ticker).strip().upper()

            with self._write_lock():
                with self._get_connection(read_only=False) as con:
                    con.execute("""
                    INSERT OR REPLACE INTO company_tenders 
                    (ticker, name, tax_id, wins, participations, win_rate, award_total, data_json, web_url, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, [clean_code, name, tax_id, wins, participations, win_rate, award_total, data_json, web_url, now_str])
                    if canonical_ticker and canonical_ticker != clean_code:
                        con.execute("""
                        INSERT OR REPLACE INTO company_tenders 
                        (ticker, name, tax_id, wins, participations, win_rate, award_total, data_json, web_url, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, [canonical_ticker, name, tax_id, wins, participations, win_rate, award_total, data_json, web_url, now_str])
            return True
        except Exception as e:
            logger.warning(f"⚠️ 寫入企業標案情報至 DuckDB 失敗 ({ticker}): {e}")
            return False

    def get_company_tender(self, ticker: str, max_age_seconds: Optional[int] = 86400 * 7) -> Optional[Dict[str, Any]]:
        """從 DuckDB 讀取企業標案資料，支援效期檢查（預設 7 天）與多代碼/名稱/統編多維命中"""
        if not self.enabled or not ticker:
            return None
        try:
            t_clean = str(ticker).strip()
            t_upper = t_clean.upper()
            raw_code = re.sub(r"\.(TW|TWO)$", "", t_upper)
            canon_tw = f"{raw_code}.TW" if raw_code.isdigit() else t_upper
            canon_two = f"{raw_code}.TWO" if raw_code.isdigit() else t_upper

            with self._get_connection(read_only=True) as con:
                row = con.execute("""
                SELECT data_json, updated_at FROM company_tenders 
                WHERE ticker IN (?, ?, ?, ?, ?) OR name = ? OR tax_id = ?
                ORDER BY updated_at DESC LIMIT 1;
                """, [t_clean, t_upper, raw_code, canon_tw, canon_two, t_clean, t_clean]).fetchone()
                if not row:
                    return None
                data_json, updated_at = row
                if max_age_seconds is not None:
                    try:
                        up_time = datetime.datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ_TAIPEI)
                        now_time = datetime.datetime.now(TZ_TAIPEI)
                        if (now_time - up_time).total_seconds() > max_age_seconds:
                            return None
                    except Exception:
                        pass
                return json.loads(data_json)
        except Exception as e:
            logger.debug(f"DuckDB 讀取企業標案快取失敗 ({ticker}): {e}")
            return None

    # ==================== Podcast Radar Methods ====================

    def save_podcast_channels_batch(self, channels: List[Dict[str, Any]]) -> int:
        """批次寫入或更新 Podcast 頻道元數據"""
        if not self.enabled or not channels:
            return 0
        now_str = datetime.datetime.now(TZ_TAIPEI).isoformat()
        saved = 0
        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
                for ch in channels:
                    con.execute("""
                        INSERT INTO podcast_channels (
                            channel_id, title, author, description, image_url,
                            rss_url, youtube_channel_url, follower_count, episode_count,
                            category, language, is_subscribed, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (channel_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            author = EXCLUDED.author,
                            description = EXCLUDED.description,
                            image_url = EXCLUDED.image_url,
                            rss_url = EXCLUDED.rss_url,
                            youtube_channel_url = EXCLUDED.youtube_channel_url,
                            follower_count = EXCLUDED.follower_count,
                            episode_count = EXCLUDED.episode_count,
                            category = EXCLUDED.category,
                            language = EXCLUDED.language,
                            updated_at = EXCLUDED.updated_at;
                    """, [
                        str(ch.get("channel_id", "")),
                        str(ch.get("title", "")),
                        ch.get("author"),
                        ch.get("description"),
                        ch.get("image_url") or ch.get("avatar_url"),
                        ch.get("rss_url") or ch.get("rss_feed"),
                        ch.get("youtube_channel_url"),
                        int(ch.get("follower_count") or 0),
                        int(ch.get("episode_count") or 0),
                        ch.get("category"),
                        ch.get("language") or "en",
                        bool(ch.get("is_subscribed", True)),
                        ch.get("created_at") or now_str,
                        now_str
                    ])
                    saved += 1
            return saved
        except Exception as e:
            logger.error(f"❌ 寫入 Podcast 頻道失敗: {e}")
            return 0

    def get_podcast_channels(self, limit: int = 100) -> List[Dict[str, Any]]:
        """取得收錄的 Podcast 頻道列表"""
        if not self.enabled:
            return []
        try:
            with self._get_connection(read_only=True) as con:
                df = con.execute("""
                    SELECT c.channel_id, c.title, c.author, c.description, c.image_url,
                           c.rss_url, c.youtube_channel_url, c.follower_count,
                           CASE WHEN (SELECT COUNT(*) FROM podcast_episodes e WHERE e.channel_id = c.channel_id) > 0 
                                THEN (SELECT COUNT(*) FROM podcast_episodes e WHERE e.channel_id = c.channel_id)
                                ELSE c.episode_count 
                           END as episode_count,
                           c.category, c.language, c.is_subscribed, c.created_at, c.updated_at
                    FROM podcast_channels c 
                    ORDER BY c.follower_count DESC 
                    LIMIT ?;
                """, [limit]).df()
                if df.empty:
                    return []
                return self._clean_df_records(df)
        except Exception as e:
            logger.error(f"❌ 查詢 Podcast 頻道失敗: {e}")
            return []

    def get_podcast_channels_count(self) -> int:
        """取得收錄的 Podcast 頻道總數"""
        if not self.enabled:
            return 0
        try:
            with self._get_connection(read_only=True) as con:
                val = con.execute("SELECT COUNT(*) FROM podcast_channels;").fetchone()
                return int(val[0]) if val else 0
        except Exception:
            return 0

    def get_latest_podcast_published_at(self) -> Optional[str]:
        """取得資料庫中最新單集的發布時間戳記"""
        if not self.enabled:
            return None
        try:
            with self._get_connection(read_only=True) as con:
                res = con.execute("SELECT MAX(published_at) FROM podcast_episodes;").fetchone()
                return str(res[0]) if res and res[0] else None
        except Exception as e:
            logger.warning(f"⚠️ 查詢最新 Podcast 發布時間失敗: {e}")
            return None


    def save_podcast_episodes_batch(self, records: List[Dict[str, Any]]) -> int:
        """批次寫入或更新單集 AI 萃取情報"""
        if not self.enabled or not records:
            return 0
        now_str = datetime.datetime.now(TZ_TAIPEI).isoformat()
        saved = 0
        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
                for r in records:
                    tt = r.get("topic_tags")
                    if isinstance(tt, list):
                        tt_str = json.dumps(tt, ensure_ascii=False)
                    else:
                        tt_str = str(tt) if tt else "[]"

                    kt = r.get("key_takeaways")
                    if isinstance(kt, list):
                        kt_str = json.dumps(kt, ensure_ascii=False)
                    else:
                        kt_str = str(kt) if kt else "[]"

                    tk = r.get("tickers")
                    if isinstance(tk, list):
                        tk_str = json.dumps(tk, ensure_ascii=False)
                    else:
                        tk_str = str(tk) if tk else "[]"

                    imp_score = float(r["importance_score"]) if r.get("importance_score") is not None else None
                    imp_level = r.get("importance_level")
                    ts = r.get("ticker_sentiments")
                    if isinstance(ts, dict):
                        ts_str = json.dumps(ts, ensure_ascii=False)
                    else:
                        ts_str = str(ts) if ts else None
                    dec_slot = r.get("decision_slot")

                    con.execute("""
                        INSERT INTO podcast_episodes (
                            id, channel_id, title, published_at, audio_url,
                            youtube_url, duration_seconds, language, host, guest,
                            tldr_zh_tw, key_takeaways, topic_tags, tickers,
                            ticker_prices_json, importance_score, importance_level,
                            ticker_sentiments, decision_slot, status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (id) DO UPDATE SET
                            title = EXCLUDED.title,
                            published_at = EXCLUDED.published_at,
                            audio_url = EXCLUDED.audio_url,
                            youtube_url = EXCLUDED.youtube_url,
                            duration_seconds = EXCLUDED.duration_seconds,
                            language = EXCLUDED.language,
                            host = EXCLUDED.host,
                            guest = EXCLUDED.guest,
                            tldr_zh_tw = EXCLUDED.tldr_zh_tw,
                            key_takeaways = EXCLUDED.key_takeaways,
                            topic_tags = EXCLUDED.topic_tags,
                            tickers = EXCLUDED.tickers,
                            ticker_prices_json = EXCLUDED.ticker_prices_json,
                            importance_score = COALESCE(EXCLUDED.importance_score, podcast_episodes.importance_score),
                            importance_level = COALESCE(EXCLUDED.importance_level, podcast_episodes.importance_level),
                            ticker_sentiments = COALESCE(EXCLUDED.ticker_sentiments, podcast_episodes.ticker_sentiments),
                            decision_slot = COALESCE(EXCLUDED.decision_slot, podcast_episodes.decision_slot),
                            status = EXCLUDED.status,
                            updated_at = EXCLUDED.updated_at;
                    """, [
                        str(r.get("id", "")),
                        str(r.get("channel_id", "")),
                        str(r.get("title", "")),
                        str(r.get("published_at", "")),
                        r.get("audio_url"),
                        r.get("youtube_url"),
                        int(r["duration_seconds"]) if r.get("duration_seconds") is not None else None,
                        r.get("language") or "unknown",
                        r.get("host"),
                        r.get("guest"),
                        r.get("tldr_zh_tw"),
                        kt_str,
                        tt_str,
                        tk_str,
                        r.get("ticker_prices_json") or "{}",
                        imp_score,
                        imp_level,
                        ts_str,
                        dec_slot,
                        r.get("status") or "extracted",
                        r.get("created_at") or now_str,
                        now_str
                    ])
                    saved += 1
            return saved
        except Exception as e:
            logger.error(f"❌ 批次寫入 Podcast 單集失敗: {e}")
            return 0

    def update_podcast_episode_decision(
        self,
        episode_id: str,
        importance_score: float,
        importance_level: str,
        ticker_sentiments: Dict[str, Any],
        decision_slot: str
    ) -> bool:
        """更新單集之 AI 決策與多空評估結果"""
        if not self.enabled:
            return False
        now_str = datetime.datetime.now(TZ_TAIPEI).isoformat()
        ts_str = json.dumps(ticker_sentiments, ensure_ascii=False) if isinstance(ticker_sentiments, dict) else str(ticker_sentiments or "{}")
        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
                con.execute("""
                    UPDATE podcast_episodes
                    SET importance_score = ?,
                        importance_level = ?,
                        ticker_sentiments = ?,
                        decision_slot = ?,
                        updated_at = ?
                    WHERE id = ?;
                """, [float(importance_score), str(importance_level), ts_str, str(decision_slot), now_str, str(episode_id)])
            return True
        except Exception as e:
            logger.error(f"❌ 更新 Podcast 單集決策失敗 ({episode_id}): {e}")
            return False

    def update_podcast_episodes_decisions_batch(self, updates: List[Dict[str, Any]]) -> int:
        """批量更新 Podcast 單集之 AI 決策重要性、評級與個股多空評估結果至 DuckDB"""
        if not self.enabled or not updates:
            return 0
        now_str = datetime.datetime.now(TZ_TAIPEI).isoformat()
        updated_count = 0
        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
                for r in updates:
                    ep_id = r.get("id") or r.get("episode_id")
                    if not ep_id:
                        continue
                    ts_val = r.get("ticker_sentiments", {})
                    ts_str = json.dumps(ts_val, ensure_ascii=False) if isinstance(ts_val, dict) else str(ts_val or "{}")
                    con.execute("""
                        UPDATE podcast_episodes
                        SET importance_score = ?,
                            importance_level = ?,
                            ticker_sentiments = ?,
                            decision_slot = ?,
                            updated_at = ?
                        WHERE id = ?;
                    """, [
                        float(r.get("importance_score", 3.0)),
                        str(r.get("importance_level", "watch")),
                        ts_str,
                        str(r.get("decision_slot", "local_rules")),
                        now_str,
                        str(ep_id)
                    ])
                    updated_count += 1
            return updated_count
        except Exception as e:
            logger.error(f"❌ 批量更新單集決策失敗: {e}")
            return updated_count

    def get_podcast_episodes(
        self,
        topic: Optional[str] = None,
        ticker: Optional[str] = None,
        channel_id: Optional[str] = None,
        lang: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """多維度查詢 Podcast 單集情報列表"""
        if not self.enabled:
            return []
        try:
            query = """
                SELECT e.*, c.title as channel_title, c.image_url as channel_image_url, c.author as channel_author
                FROM podcast_episodes e
                LEFT JOIN podcast_channels c ON e.channel_id = c.channel_id
                WHERE 1=1
            """
            params = []

            if topic:
                query += " AND (e.topic_tags LIKE ? OR e.title LIKE ? OR e.tldr_zh_tw LIKE ?)"
                topic_pat = f"%{topic}%"
                params.extend([topic_pat, topic_pat, topic_pat])

            if ticker:
                query += " AND e.tickers LIKE ?"
                params.append(f"%{ticker}%")

            if channel_id and channel_id.upper() != "ALL":
                query += " AND e.channel_id = ?"
                params.append(channel_id)

            if lang and lang.upper() != "ALL":
                query += " AND e.language = ?"
                params.append(lang.lower())

            query += " ORDER BY e.published_at DESC LIMIT ? OFFSET ?;"
            params.extend([limit, offset])

            with self._get_connection(read_only=True) as con:
                df = con.execute(query, params).df()
                if df.empty:
                    return []
                return self._clean_df_records(df)
        except Exception as e:
            logger.error(f"❌ 查詢 Podcast 單集失敗: {e}")
            return []

    def get_podcast_topics_aggregation(self, limit: int = 30) -> List[Dict[str, Any]]:
        """取得熱門實體與主題標籤聚合排行榜"""
        if not self.enabled:
            return []
        try:
            with self._get_connection(read_only=True) as con:
                # 統計各 topic_tags 的出現次數
                df = con.execute("SELECT topic_tags FROM podcast_episodes WHERE topic_tags IS NOT NULL;").df()
                if df.empty:
                    return []
                
                counts: Dict[str, int] = {}
                for row in df["topic_tags"].dropna():
                    try:
                        tags = json.loads(row) if isinstance(row, str) else row
                        if isinstance(tags, list):
                            for t in tags:
                                t_clean = str(t).strip()
                                if t_clean and len(t_clean) > 1:
                                    counts[t_clean] = counts.get(t_clean, 0) + 1
                    except Exception:
                        pass

                sorted_topics = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]
                return [{"topic": t, "count": c} for t, c in sorted_topics]
        except Exception as e:
            logger.error(f"❌ 聚合熱門 Podcast 主題失敗: {e}")
            return []

    def get_total_episodes_count(self) -> int:
        """取得收錄的 Podcast 單集總數 (用於冷啟動判定)"""
        if not self.enabled:
            return 0
        try:
            with self._get_connection(read_only=True) as con:
                val = con.execute("SELECT COUNT(*) FROM podcast_episodes;").fetchone()
                return int(val[0]) if val else 0
        except Exception:
            return 0

    def get_latest_quotes_batch(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """批次查詢多個 Ticker 的最新價格與潛力評估，避免 N+1 查詢與全表掃描"""
        if not self.enabled or not tickers:
            return {}
        try:
            clean_tickers = list({str(t).strip().upper() for t in tickers if t})
            if not clean_tickers:
                return {}
            with self._get_connection(read_only=True) as con:
                placeholders = ",".join(["?"] * len(clean_tickers))
                query = f"""
                    WITH target_tickers AS (
                        SELECT UNNEST([{placeholders}]) as t_raw
                    ),
                    ranked AS (
                        SELECT 
                            p.ticker,
                            p.current_price,
                            p.potential,
                            p.timestamp,
                            ROW_NUMBER() OVER (PARTITION BY p.ticker ORDER BY p.timestamp DESC) as rn
                        FROM predictions p
                        WHERE p.ticker IN (SELECT t_raw FROM target_tickers)
                           OR p.ticker IN (SELECT t_raw || '.TW' FROM target_tickers)
                           OR p.ticker IN (SELECT t_raw || '.TWO' FROM target_tickers)
                    )
                    SELECT ticker, current_price, timestamp
                    FROM ranked
                    WHERE rn = 1;
                """
                df = con.execute(query, clean_tickers).df()
                res = {}
                if not df.empty:
                    for _, row in df.iterrows():
                        t = str(row["ticker"])
                        c_price = float(row["current_price"]) if row["current_price"] is not None and not pd.isna(row["current_price"]) else None
                        quote_obj = {
                            "ticker": t,
                            "name": t,
                            "price": c_price,
                            "change_pct": None,
                            "direction": "flat",
                            "source": "quant_prediction_snapshot",
                            "as_of": str(row["timestamp"]) if row["timestamp"] is not None else None,
                        }
                        res[t] = quote_obj
                        if "." in t:
                            res[t.split(".")[0]] = quote_obj
                return res
        except Exception as e:
            logger.debug(f"批次查詢最新行情失敗: {e}")
            return {}

    # ==================== Decision Verdicts Methods (Clef System One) ====================

    def save_decision_verdict(
        self,
        ticker: str,
        verdict: Dict[str, Any],
        as_of_date: Optional[str] = None,
        ttl_seconds: int = 86400
    ) -> bool:
        """寫入 Clef System One AI 機率化決策推論至 DuckDB，支援多代碼自動正規化與過期時間設置"""
        if not self.enabled or not ticker or not verdict:
            return False
        try:
            t_clean = str(ticker).strip().upper()
            date_str = as_of_date or datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d")
            now_dt = datetime.datetime.now(TZ_TAIPEI)
            created_at = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            expires_at = (now_dt + datetime.timedelta(seconds=ttl_seconds)).strftime("%Y-%m-%d %H:%M:%S")

            def _s_float(v, default=None):
                if v is None:
                    return default
                try:
                    fv = float(v)
                    return fv if math.isfinite(fv) else default
                except (ValueError, TypeError):
                    return default

            action = str(verdict.get("action_decision") or "neutral")
            confidence = _s_float(verdict.get("action_confidence"), 0.0)
            probs = verdict.get("action_probabilities") or {}
            if isinstance(probs, dict):
                probs = {k: _s_float(v, 0.0) for k, v in probs.items()}
            probs_str = json.dumps(probs, ensure_ascii=False) if isinstance(probs, dict) else str(probs)
            conviction = _s_float(verdict.get("conviction_score"), 2.5)
            favorable = _s_float(verdict.get("is_favorable_entry"), 0.5)
            tag = str(verdict.get("tag") or "")
            source_slot = str(verdict.get("source_slot") or "clef")
            raw_verdict_str = json.dumps(verdict, ensure_ascii=False)

            with self._write_lock():
                with self._get_connection(read_only=False) as con:
                    con.execute("""
                    INSERT OR REPLACE INTO decision_verdicts
                    (ticker, as_of_date, action_decision, action_confidence, action_probabilities,
                     conviction_score, is_favorable_entry, tag, source_slot, raw_verdict, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, [
                        t_clean, date_str, action, confidence, probs_str,
                        conviction, favorable, tag, source_slot, raw_verdict_str, created_at, expires_at
                    ])
            return True
        except Exception as e:
            logger.warning(f"⚠️ 寫入 AI 決策推論至 DuckDB 失敗 ({ticker}): {e}")
            return False

    def get_decision_verdict(
        self,
        ticker: str,
        as_of_date: Optional[str] = None,
        check_expiry: bool = True
    ) -> Optional[Dict[str, Any]]:
        """從 DuckDB 讀取特定標的與交易日之 AI 決策推論快取，支援過期時間檢查與對稱代碼檢索"""
        if not self.enabled or not ticker:
            return None
        try:
            t_clean = str(ticker).strip().upper()
            raw_code = re.sub(r"\.(TW|TWO)$", "", t_clean)
            date_str = as_of_date or datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d")

            with self._get_connection(read_only=True) as con:
                row = con.execute("""
                SELECT action_decision, action_confidence, action_probabilities, conviction_score,
                       is_favorable_entry, tag, source_slot, raw_verdict, created_at, expires_at
                FROM decision_verdicts
                WHERE (ticker IN (?, ?) OR ticker LIKE ?) AND as_of_date = ?
                ORDER BY created_at DESC LIMIT 1;
                """, [t_clean, raw_code, f"{raw_code}.%", date_str]).fetchone()

                if not row:
                    return None

                (action, conf, probs_s, conv, fav, tag, slot, raw_v, c_at, exp_at) = row

                if check_expiry and exp_at:
                    now_str = datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S")
                    if now_str > exp_at:
                        return None

                try:
                    probs_dict = json.loads(probs_s) if probs_s else {}
                except Exception:
                    probs_dict = {}

                return {
                    "ticker": t_clean,
                    "as_of_date": date_str,
                    "action_decision": action,
                    "action_confidence": conf,
                    "action_probabilities": probs_dict,
                    "conviction_score": conv,
                    "is_favorable_entry": fav,
                    "tag": tag,
                    "source_slot": slot,
                    "created_at": c_at,
                    "expires_at": exp_at,
                    "success": True
                }
        except Exception as e:
            logger.debug(f"DuckDB 讀取 AI 決策快取失敗 ({ticker}): {e}")
            return None

    # ==================== Company Fundamentals Methods ====================

    def save_company_fundamentals(self, ticker: str, data: Dict[str, Any]) -> bool:
        """寫入個股基本面估值指標 (PE, Forward PE, PB, EV/EBITDA) 至 DuckDB"""
        if not self.enabled or not ticker or not data:
            return False
        try:
            t_clean = str(ticker).strip().upper()
            now_str = datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S")

            def _s_float(v):
                if v is None:
                    return None
                try:
                    fv = float(v)
                    return fv if math.isfinite(fv) else None
                except (ValueError, TypeError):
                    return None

            pe = _s_float(data.get("pe"))
            forward_pe = _s_float(data.get("forward_pe"))
            pb = _s_float(data.get("pb"))
            ev_ebitda = _s_float(data.get("ev_ebitda"))
            data_json = json.dumps(data, ensure_ascii=False)

            with self._write_lock():
                with self._get_connection(read_only=False) as con:
                    con.execute("""
                    INSERT OR REPLACE INTO company_fundamentals
                    (ticker, pe, forward_pe, pb, ev_ebitda, data_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """, [t_clean, pe, forward_pe, pb, ev_ebitda, data_json, now_str])
            return True
        except Exception as e:
            logger.warning(f"⚠️ 寫入個股基本面至 DuckDB 失敗 ({ticker}): {e}")
            return False

    def get_company_fundamentals(
        self,
        ticker: str,
        max_age_hours: int = 24
    ) -> Optional[Dict[str, Optional[float]]]:
        """從 DuckDB 讀取個股基本面估值指標，支援 TTL 效期檢查與對稱代碼檢索"""
        if not self.enabled or not ticker:
            return None
        try:
            t_clean = str(ticker).strip().upper()
            raw_code = re.sub(r"\.(TW|TWO)$", "", t_clean)
            with self._get_connection(read_only=True) as con:
                row = con.execute("""
                SELECT pe, forward_pe, pb, ev_ebitda, updated_at
                FROM company_fundamentals
                WHERE ticker IN (?, ?) OR ticker LIKE ?
                ORDER BY updated_at DESC LIMIT 1;
                """, [t_clean, raw_code, f"{raw_code}.%"]).fetchone()

                if not row:
                    return None

                pe, forward_pe, pb, ev_ebitda, updated_at = row
                if max_age_hours is not None and updated_at:
                    try:
                        up_time = datetime.datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ_TAIPEI)
                        now_time = datetime.datetime.now(TZ_TAIPEI)
                        if (now_time - up_time).total_seconds() > max_age_hours * 3600:
                            return None
                    except Exception:
                        pass

                return {
                    "pe": pe,
                    "forward_pe": forward_pe,
                    "pb": pb,
                    "ev_ebitda": ev_ebitda,
                    "updated_at": updated_at
                }
        except Exception as e:
            logger.debug(f"DuckDB 讀取個股基本面快取失敗 ({ticker}): {e}")
            return None

    def save_company_fundamentals_batch(self, records: Dict[str, Dict[str, Any]]) -> int:
        """批次儲存多個股票之基本面指標，共用單一連線交易以提升效能"""
        if not self.enabled or not records:
            return 0
        saved = 0
        now_str = datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S")

        def _s_float(v):
            if v is None:
                return None
            try:
                fv = float(v)
                return fv if math.isfinite(fv) else None
            except (ValueError, TypeError):
                return None

        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
                    for t, data in records.items():
                        if not t or not data:
                            continue
                        t_clean = str(t).strip().upper()
                        pe = _s_float(data.get("pe"))
                        forward_pe = _s_float(data.get("forward_pe"))
                        pb = _s_float(data.get("pb"))
                        ev_ebitda = _s_float(data.get("ev_ebitda"))
                        data_json = json.dumps(data, ensure_ascii=False)
                        con.execute("""
                        INSERT OR REPLACE INTO company_fundamentals
                        (ticker, pe, forward_pe, pb, ev_ebitda, data_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?);
                        """, [t_clean, pe, forward_pe, pb, ev_ebitda, data_json, now_str])
                        saved += 1
            return saved
        except Exception as e:
            logger.warning(f"⚠️ 批次寫入基本面至 DuckDB 失敗: {e}")
            return saved

    # ==================== Discovery Tech Radar Operations ====================

    def save_discovery_snapshot(
        self,
        run_id: str,
        as_of: str,
        candidates: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
        signals: List[Dict[str, Any]],
        history_summary: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> bool:
        """
        交易式儲存 Discovery 前沿挖掘機之快照資料。
        具備指數退避重試機制，保證 DuckDB 寫入原子性。
        """
        if not self.enabled:
            return False

        now_str = datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S")

        for attempt in range(max_retries):
            try:
                with self._write_lock():
                    with self._get_connection(read_only=False) as con:
                        con.begin()
                        try:
                            # 確保同日重跑之完全冪等性：清除當日舊快照
                            con.execute("DELETE FROM discovery_candidates WHERE as_of = ?", [as_of])
                            con.execute("DELETE FROM discovery_papers WHERE as_of = ?", [as_of])
                            con.execute("DELETE FROM discovery_signals WHERE as_of = ?", [as_of])
                            con.execute("DELETE FROM discovery_history WHERE as_of = ?", [as_of])

                            # 1. 寫入候選標的
                            for c in candidates:
                                con.execute("""
                                INSERT OR REPLACE INTO discovery_candidates (
                                    ticker, market, name, theme_id, theme_name,
                                    total_score, recommendation, review_status, in_pool,
                                    source_score, purity_score, paper_score, novelty_score,
                                    sentiment_score, setup_score, risk_score,
                                    catalyst_summary, us_anchor, paper_count, signal_count,
                                    run_id, as_of, created_at, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                                """, [
                                    str(c.get("ticker", "")),
                                    str(c.get("market", "")),
                                    c.get("name"),
                                    str(c.get("theme_id", "")),
                                    str(c.get("theme_name", "")),
                                    float(c.get("total_score", 0.0)),
                                    str(c.get("recommendation", "reject")),
                                    str(c.get("review_status", "pending")),
                                    bool(c.get("in_pool", False)),
                                    float(c.get("source_score", 0.0)),
                                    float(c.get("purity_score", 0.0)),
                                    float(c.get("paper_score", 0.0)),
                                    float(c.get("novelty_score", 0.0)),
                                    float(c.get("sentiment_score", 0.0)),
                                    float(c.get("setup_score", 0.0)),
                                    float(c.get("risk_score", 0.0)),
                                    c.get("catalyst_summary"),
                                    c.get("us_anchor"),
                                    int(c.get("paper_count", 0)),
                                    int(c.get("signal_count", 0)),
                                    run_id,
                                    as_of,
                                    c.get("created_at") or now_str,
                                    now_str
                                ])

                            # 2. 寫入學術論文
                            for p in papers:
                                authors_str = json.dumps(p.get("authors", []), ensure_ascii=False) if isinstance(p.get("authors"), list) else str(p.get("authors") or "")
                                rel_tickers_str = json.dumps(p.get("related_tickers", []), ensure_ascii=False) if isinstance(p.get("related_tickers"), list) else str(p.get("related_tickers") or "")
                                con.execute("""
                                INSERT OR REPLACE INTO discovery_papers (
                                    paper_id, title, authors, abstract, category,
                                    published_date, bottleneck_tag, related_tickers, url,
                                    significance_score, run_id, as_of, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                                """, [
                                    str(p.get("paper_id", "")),
                                    str(p.get("title", "")),
                                    authors_str,
                                    str(p.get("abstract", "")),
                                    str(p.get("category", "")),
                                    str(p.get("published_date", "")),
                                    str(p.get("bottleneck_tag", "")),
                                    rel_tickers_str,
                                    p.get("url"),
                                    float(p.get("significance_score", 0.0)),
                                    run_id,
                                    as_of,
                                    p.get("created_at") or now_str
                                ])

                            # 3. 寫入產業訊號
                            for s in signals:
                                con.execute("""
                                INSERT OR REPLACE INTO discovery_signals (
                                    signal_id, ticker, theme_id, source_type, title,
                                    content, url, sentiment_score, detected_at,
                                    run_id, as_of, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                                """, [
                                    str(s.get("signal_id", "")),
                                    str(s.get("ticker", "")),
                                    str(s.get("theme_id", "")),
                                    str(s.get("source_type", "")),
                                    str(s.get("title", "")),
                                    s.get("content"),
                                    s.get("url"),
                                    float(s.get("sentiment_score", 0.0)),
                                    str(s.get("detected_at", now_str)),
                                    run_id,
                                    as_of,
                                    s.get("created_at") or now_str
                                ])

                            # 4. 寫入歷史時序記錄
                            if history_summary:
                                con.execute("""
                                INSERT OR REPLACE INTO discovery_history (
                                    run_id, as_of, total_candidates, propose_add_count,
                                    observe_count, already_in_pool_count, paper_count,
                                    signal_count, avg_total_score, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                                """, [
                                    run_id,
                                    as_of,
                                    int(history_summary.get("total_candidates", len(candidates))),
                                    int(history_summary.get("propose_add_count", 0)),
                                    int(history_summary.get("observe_count", 0)),
                                    int(history_summary.get("already_in_pool_count", 0)),
                                    int(history_summary.get("paper_count", len(papers))),
                                    int(history_summary.get("signal_count", len(signals))),
                                    float(history_summary.get("avg_total_score", 0.0)),
                                    now_str
                                ])

                            con.commit()
                            logger.info(f"✅ 成功交易式寫入 Discovery 快照 (run_id={run_id}, as_of={as_of})")
                            return True
                        except Exception as inner_e:
                            con.rollback()
                            raise inner_e
            except Exception as e:
                backoff = 2 ** attempt
                logger.warning(f"⚠️ DuckDB 儲存 Discovery 快照重試中 ({attempt + 1}/{max_retries}), 等待 {backoff}s: {e}")
                time.sleep(backoff)

        logger.error(f"❌ DuckDB 儲存 Discovery 快照已達最大重試次數 ({max_retries})，放棄寫入。")
        return False

    def prune_discovery_expired_records(self, paper_ttl_days: int = 180, signal_ttl_days: int = 90) -> Dict[str, int]:
        """
        依 TTL 清理過期記錄：
        - discovery_papers: 保留 180 天
        - discovery_signals: 保留 90 天
        - discovery_history: 永久保留
        """
        if not self.enabled:
            return {"pruned_papers": 0, "pruned_signals": 0}

        now = datetime.datetime.now(TZ_TAIPEI)
        paper_cutoff = (now - datetime.timedelta(days=paper_ttl_days)).strftime("%Y-%m-%d")
        signal_cutoff = (now - datetime.timedelta(days=signal_ttl_days)).strftime("%Y-%m-%d")

        pruned_papers = 0
        pruned_signals = 0

        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
                    # 清理論文
                    del_papers = con.execute("DELETE FROM discovery_papers WHERE as_of < ? OR published_date < ?", [paper_cutoff, paper_cutoff])
                    pruned_papers = del_papers.fetchall()[0][0] if del_papers else 0

                    # 清理快訊
                    del_signals = con.execute("DELETE FROM discovery_signals WHERE as_of < ? OR detected_at < ?", [signal_cutoff, signal_cutoff])
                    pruned_signals = del_signals.fetchall()[0][0] if del_signals else 0

            logger.info(f"🧹 Discovery TTL 清理完成: 已刪除 {pruned_papers} 篇過期論文, {pruned_signals} 筆過期快訊")
            return {"pruned_papers": pruned_papers, "pruned_signals": pruned_signals}
        except Exception as e:
            logger.warning(f"⚠️ Discovery TTL 清理異常: {e}")
            return {"pruned_papers": 0, "pruned_signals": 0}

    def get_discovery_candidates(
        self,
        recommendation: Optional[str] = None,
        theme_id: Optional[str] = None,
        as_of: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        return_total: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], int]]:
        """以 read_only 模式查詢 Discovery 候選標的"""
        if not self.enabled:
            return ([], 0) if return_total else []

        try:
            with self._get_connection(read_only=True) as con:
                target_as_of = as_of
                if not target_as_of:
                    latest = con.execute("SELECT MAX(as_of) FROM discovery_candidates").fetchone()
                    target_as_of = latest[0] if latest and latest[0] else None

                if not target_as_of:
                    return ([], 0) if return_total else []

                base_where = "WHERE as_of = ?"
                params: List[Any] = [target_as_of]

                if recommendation:
                    base_where += " AND recommendation = ?"
                    params.append(recommendation)

                if theme_id:
                    base_where += " AND theme_id = ?"
                    params.append(theme_id)

                total_count = 0
                if return_total:
                    count_row = con.execute(f"SELECT COUNT(*) FROM discovery_candidates {base_where}", params).fetchone()
                    total_count = count_row[0] if count_row else 0

                query = f"SELECT * FROM discovery_candidates {base_where} ORDER BY total_score DESC LIMIT ? OFFSET ?"
                query_params = list(params) + [limit, offset]

                df = con.execute(query, query_params).df()
                records = df.to_dict(orient="records")
                return (records, total_count) if return_total else records
        except Exception as e:
            logger.warning(f"DuckDB 讀取 discovery_candidates 失敗: {e}")
            return ([], 0) if return_total else []

    def get_discovery_papers(
        self,
        theme_id: Optional[str] = None,
        category: Optional[str] = None,
        as_of: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        return_total: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], int]]:
        """以 read_only 模式查詢前沿學術論文"""
        if not self.enabled:
            return ([], 0) if return_total else []

        try:
            with self._get_connection(read_only=True) as con:
                target_as_of = as_of
                if not target_as_of:
                    latest = con.execute("SELECT MAX(as_of) FROM discovery_papers").fetchone()
                    target_as_of = latest[0] if latest and latest[0] else None

                if not target_as_of:
                    return ([], 0) if return_total else []

                base_where = "WHERE as_of = ?"
                params: List[Any] = [target_as_of]

                if theme_id:
                    base_where += " AND bottleneck_tag = ?"
                    params.append(theme_id)

                if category:
                    base_where += " AND category = ?"
                    params.append(category)

                total_count = 0
                if return_total:
                    count_row = con.execute(f"SELECT COUNT(*) FROM discovery_papers {base_where}", params).fetchone()
                    total_count = count_row[0] if count_row else 0

                query = f"SELECT * FROM discovery_papers {base_where} ORDER BY significance_score DESC, published_date DESC LIMIT ? OFFSET ?"
                query_params = list(params) + [limit, offset]

                df = con.execute(query, query_params).df()
                records = df.to_dict(orient="records")
                for r in records:
                    if isinstance(r.get("authors"), str):
                        try:
                            r["authors"] = json.loads(r["authors"])
                        except Exception:
                            r["authors"] = [r["authors"]]
                    if isinstance(r.get("related_tickers"), str):
                        try:
                            r["related_tickers"] = json.loads(r["related_tickers"])
                        except Exception:
                            r["related_tickers"] = [r["related_tickers"]]
                return (records, total_count) if return_total else records
        except Exception as e:
            logger.warning(f"DuckDB 讀取 discovery_papers 失敗: {e}")
            return ([], 0) if return_total else []

    def get_discovery_signals(
        self,
        ticker: Optional[str] = None,
        theme_id: Optional[str] = None,
        as_of: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        return_total: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], int]]:
        """以 read_only 模式查詢產業訊號，限定最新日期或指定日期"""
        if not self.enabled:
            return ([], 0) if return_total else []

        try:
            with self._get_connection(read_only=True) as con:
                target_as_of = as_of
                if not target_as_of:
                    latest = con.execute("SELECT MAX(as_of) FROM discovery_signals").fetchone()
                    target_as_of = latest[0] if latest and latest[0] else None

                base_where = "WHERE 1=1"
                params: List[Any] = []
                if target_as_of:
                    base_where += " AND as_of = ?"
                    params.append(target_as_of)

                if ticker:
                    base_where += " AND ticker = ?"
                    params.append(ticker)

                if theme_id:
                    base_where += " AND theme_id = ?"
                    params.append(theme_id)

                total_count = 0
                if return_total:
                    count_row = con.execute(f"SELECT COUNT(*) FROM discovery_signals {base_where}", params).fetchone()
                    total_count = count_row[0] if count_row else 0

                query = f"SELECT * FROM discovery_signals {base_where} ORDER BY detected_at DESC LIMIT ? OFFSET ?"
                query_params = list(params) + [limit, offset]

                df = con.execute(query, query_params).df()
                records = df.to_dict(orient="records")
                return (records, total_count) if return_total else records
        except Exception as e:
            logger.warning(f"DuckDB 讀取 discovery_signals 失敗: {e}")
            return ([], 0) if return_total else []
        except Exception as e:
            logger.warning(f"DuckDB 讀取 discovery_signals 失敗: {e}")
            return []

    def get_discovery_history(self, limit: int = 30) -> List[Dict[str, Any]]:
        """以 read_only 模式查詢 Discovery 歷史趨勢"""
        if not self.enabled:
            return []

        try:
            with self._get_connection(read_only=True) as con:
                query = "SELECT * FROM discovery_history ORDER BY as_of DESC LIMIT ?"
                df = con.execute(query, [limit]).df()
                return df.to_dict(orient="records")
        except Exception as e:
            logger.warning(f"DuckDB 讀取 discovery_history 失敗: {e}")
            return []

    def get_discovery_status(self) -> Dict[str, Any]:
        """以 read_only 模式獲取 Discovery 最新狀態與統計資訊"""
        if not self.enabled:
            return {
                "status": "error",
                "stale": True,
                "as_of": "1970-01-01",
                "candidate_count": 0,
                "paper_count": 0,
                "signal_count": 0
            }

        try:
            with self._get_connection(read_only=True) as con:
                latest_hist = con.execute("""
                    SELECT run_id, as_of, total_candidates, paper_count, signal_count, created_at
                    FROM discovery_history
                    ORDER BY as_of DESC, created_at DESC
                    LIMIT 1
                """).fetchone()

                if not latest_hist:
                    return {
                        "status": "empty",
                        "stale": False,
                        "as_of": datetime.date.today().isoformat(),
                        "candidate_count": 0,
                        "paper_count": 0,
                        "signal_count": 0
                    }

                run_id, as_of, total_candidates, paper_count, signal_count, created_at = latest_hist

                # 判斷是否 stale (> 24 小時)
                stale = False
                try:
                    as_of_date = datetime.date.fromisoformat(as_of)
                    today = datetime.datetime.now(TZ_TAIPEI).date()
                    if (today - as_of_date).days > 1:
                        stale = True
                except Exception:
                    pass

                return {
                    "status": "success",
                    "stale": stale,
                    "as_of": as_of,
                    "run_id": run_id,
                    "last_run_timestamp": created_at,
                    "candidate_count": int(total_candidates or 0),
                    "paper_count": int(paper_count or 0),
                    "signal_count": int(signal_count or 0),
                    "message": "Discovery pipeline normal"
                }
        except Exception as e:
            logger.warning(f"DuckDB 讀取 discovery_status 失敗: {e}")
            return {
                "status": "error",
                "stale": True,
                "as_of": "1970-01-01",
                "candidate_count": 0,
                "paper_count": 0,
                "signal_count": 0,
                "message": str(e)
            }

    # =========================================================================
    # 產業價值鏈與實體關係圖譜持久化 (Industry Supply Chain Knowledge Graph)
    # =========================================================================

    def init_industry_chain_tables(self):
        """初始化產業價值鏈與實體關係圖譜表格"""
        if not self.enabled:
            return
        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
                con.execute("""
                CREATE TABLE IF NOT EXISTS tw_industry_chains (
                    chain_id VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    name_en VARCHAR,
                    category VARCHAR,
                    description VARCHAR,
                    updated_at VARCHAR
                );
                ALTER TABLE tw_industry_chains ADD COLUMN IF NOT EXISTS name_en VARCHAR;
                ALTER TABLE tw_industry_chains ADD COLUMN IF NOT EXISTS description VARCHAR;

                CREATE TABLE IF NOT EXISTS tw_industry_nodes (
                    ticker VARCHAR,
                    company VARCHAR,
                    market VARCHAR,
                    chain_id VARCHAR,
                    layer VARCHAR,
                    role VARCHAR,
                    key_focus VARCHAR,
                    theme_tags VARCHAR,
                    PRIMARY KEY (ticker, chain_id, layer)
                );

                CREATE TABLE IF NOT EXISTS tw_industry_relations (
                    source_ticker VARCHAR,
                    target_ticker VARCHAR,
                    relation_type VARCHAR,
                    description VARCHAR,
                    weight DOUBLE,
                    PRIMARY KEY (source_ticker, target_ticker, relation_type)
                );

                CREATE INDEX IF NOT EXISTS idx_ind_nodes_ticker ON tw_industry_nodes (ticker);
                CREATE INDEX IF NOT EXISTS idx_ind_nodes_chain ON tw_industry_nodes (chain_id);
                CREATE INDEX IF NOT EXISTS idx_ind_rel_src ON tw_industry_relations (source_ticker);
                CREATE INDEX IF NOT EXISTS idx_ind_rel_tgt ON tw_industry_relations (target_ticker);
                """)
            logger.info("✅ DuckDB 產業鏈圖譜表格 (tw_industry_chains / nodes / relations) 初始化成功")
        except Exception as e:
            logger.error(f"❌ 初始化產業鏈圖譜表格失敗: {e}")

    def save_industry_chains_batch(self, records: List[Dict[str, Any]]) -> int:
        """
        批次寫入產業鏈主題定義，自動去重 (Upsert on chain_id)
        """
        if not self.enabled or not records:
            return 0
        df = pd.DataFrame(records)
        expected_cols = ["chain_id", "name", "name_en", "category", "description"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
                    con.register("temp_industry_chains_df", df[expected_cols])
                    con.execute("""
                    DELETE FROM tw_industry_chains
                    WHERE chain_id IN (SELECT chain_id FROM temp_industry_chains_df);
                    """)
                    con.execute("""
                    INSERT INTO tw_industry_chains (chain_id, name, name_en, category, description, updated_at)
                    SELECT chain_id, name, name_en, category, description, CURRENT_TIMESTAMP
                    FROM temp_industry_chains_df;
                    """)
            logger.info(f"💾 成功寫入/更新 {len(df)} 筆產業鏈主題至 DuckDB")
            return len(df)
        except Exception as e:
            logger.error(f"❌ 寫入 tw_industry_chains 失敗: {e}", exc_info=True)
            return 0

    def save_industry_nodes_batch(self, records: List[Dict[str, Any]], clear_existing: bool = False) -> int:
        """
        批次寫入產業鏈節點，自動去重 (Upsert on ticker + chain_id + layer)
        若 clear_existing=True 則在寫入前清空全表以確保完全同步最新節點定義。
        使用事務保證原子性：若寫入發生異常則自動 ROLLBACK，絕不破壞既有資料。
        """
        if not self.enabled or not records:
            return 0
        df = pd.DataFrame(records)
        expected_cols = ["ticker", "company", "market", "chain_id", "layer", "role", "key_focus", "theme_tags"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        df["theme_tags"] = df["theme_tags"].apply(
            lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, set)) else str(v or "[]")
        )
        # 記憶體預先去重，杜絕批次內部重複鍵違反 PRIMARY KEY 約束
        df = df.drop_duplicates(subset=["ticker", "chain_id", "layer"], keep="last")

        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
                    con.register("temp_industry_nodes_df", df[expected_cols])
                    con.execute("BEGIN TRANSACTION;")
                    try:
                        if clear_existing:
                            con.execute("DELETE FROM tw_industry_nodes;")
                        else:
                            con.execute("""
                            DELETE FROM tw_industry_nodes
                            WHERE (ticker, chain_id, layer) IN (
                                SELECT ticker, chain_id, layer FROM temp_industry_nodes_df
                            );
                            """)
                        con.execute("INSERT INTO tw_industry_nodes SELECT * FROM temp_industry_nodes_df;")
                        con.execute("COMMIT;")
                    except Exception:
                        con.execute("ROLLBACK;")
                        raise
            logger.info(f"💾 成功寫入/更新 {len(df)} 筆產業鏈節點至 DuckDB")
            return len(df)
        except Exception as e:
            logger.error(f"❌ 寫入 tw_industry_nodes 失敗: {e}", exc_info=True)
            return 0

    def save_industry_relations_batch(self, records: List[Dict[str, Any]], clear_existing: bool = False) -> int:
        """
        批次寫入產業鏈關係邊，自動去重 (Upsert on source_ticker + target_ticker + relation_type)
        若 clear_existing=True 則在寫入前清空全表以確保完全同步最新關係定義。
        使用事務保證原子性：若寫入發生異常則自動 ROLLBACK，絕不破壞既有資料。
        """
        if not self.enabled or not records:
            return 0
        df = pd.DataFrame(records)
        expected_cols = ["source_ticker", "target_ticker", "relation_type", "description", "weight"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
        # 記憶體預先去重，杜絕批次內部重複鍵違反 PRIMARY KEY 約束
        df = df.drop_duplicates(subset=["source_ticker", "target_ticker", "relation_type"], keep="last")

        try:
            with self._write_lock(), self._get_connection(read_only=False) as con:
                    con.register("temp_industry_rels_df", df[expected_cols])
                    con.execute("BEGIN TRANSACTION;")
                    try:
                        if clear_existing:
                            con.execute("DELETE FROM tw_industry_relations;")
                        else:
                            con.execute("""
                            DELETE FROM tw_industry_relations
                            WHERE (source_ticker, target_ticker, relation_type) IN (
                                SELECT source_ticker, target_ticker, relation_type FROM temp_industry_rels_df
                            );
                            """)
                        con.execute("INSERT INTO tw_industry_relations SELECT * FROM temp_industry_rels_df;")
                        con.execute("COMMIT;")
                    except Exception:
                        con.execute("ROLLBACK;")
                        raise
            logger.info(f"💾 成功寫入/更新 {len(df)} 筆產業鏈關係邊至 DuckDB")
            return len(df)
        except Exception as e:
            logger.error(f"❌ 寫入 tw_industry_relations 失敗: {e}", exc_info=True)
            return 0

    def get_industry_nodes(
        self,
        chain_id: Optional[str] = None,
        layer: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        查詢產業鏈節點，支援依 chain_id 或 layer 篩選
        """
        if not self.enabled:
            return []
        try:
            with self._get_connection(read_only=True) as con:
                query = "SELECT ticker, company, market, chain_id, layer, role, key_focus, theme_tags FROM tw_industry_nodes WHERE 1=1"
                params = []
                if chain_id:
                    query += " AND chain_id = ?"
                    params.append(chain_id)
                if layer:
                    query += " AND layer = ?"
                    params.append(layer)
                query += " ORDER BY layer, ticker"
                df = con.execute(query, params).df()
                if df.empty:
                    return []
                records = df.to_dict(orient="records")
                for r in records:
                    tt = r.get("theme_tags")
                    if isinstance(tt, str):
                        try:
                            r["theme_tags"] = json.loads(tt)
                        except Exception:
                            r["theme_tags"] = [t.strip() for t in tt.split(",") if t.strip()]
                    elif not isinstance(tt, list):
                        r["theme_tags"] = []
                return records
        except Exception as e:
            logger.warning(f"DuckDB 查詢 industry_nodes 失敗: {e}")
            return []

    def get_industry_relations(
        self,
        source_ticker: Optional[str] = None,
        target_ticker: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        查詢產業鏈關係邊，支援依 source_ticker 或 target_ticker 篩選
        """
        if not self.enabled:
            return []
        try:
            with self._get_connection(read_only=True) as con:
                query = "SELECT source_ticker, target_ticker, relation_type, description, weight FROM tw_industry_relations WHERE 1=1"
                params = []
                if source_ticker:
                    query += " AND source_ticker = ?"
                    params.append(source_ticker)
                if target_ticker:
                    query += " AND target_ticker = ?"
                    params.append(target_ticker)
                df = con.execute(query, params).df()
                if df.empty:
                    return []
                return df.to_dict(orient="records")
        except Exception as e:
            logger.warning(f"DuckDB 查詢 industry_relations 失敗: {e}")
            return []

    def get_enriched_industry_quotes(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        為圖譜節點批次關聯最新收盤價、漲跌幅、Timestral/TimesFM 目標價與法人連買天數
        """
        if not self.enabled or not tickers:
            return {}

        clean_map = {}
        for t in tickers:
            if not t:
                continue
            raw = str(t).strip().upper()
            clean_map[raw] = raw
            if "." in raw:
                clean_map[raw.split(".")[0]] = raw

        unique_tickers = list(set(clean_map.keys()))
        enrichment: Dict[str, Dict[str, Any]] = {}

        try:
            with self._get_connection(read_only=True) as con:
                # 1. 批次查詢 predictions 表中最新時序大模型預測與現價
                placeholders = ",".join(["?"] * len(unique_tickers))
                pred_query = f"""
                    WITH targets AS (SELECT UNNEST([{placeholders}]) AS t_raw),
                    ranked_pred AS (
                        SELECT
                            p.ticker,
                            p.current_price,
                            p.potential,
                            p.model_name,
                            p.timestral_potential,
                            p.timesfm_potential,
                            p.dual_model_consensus,
                            p.trust_net_5d,
                            p.foreign_net_5d,
                            p.timestamp,
                            DENSE_RANK() OVER (PARTITION BY p.ticker ORDER BY p.timestamp DESC) as rank_time
                        FROM predictions p
                        WHERE p.ticker IN (SELECT t_raw FROM targets)
                           OR p.ticker IN (SELECT t_raw || '.TW' FROM targets)
                           OR p.ticker IN (SELECT t_raw || '.TWO' FROM targets)
                    )
                    SELECT
                        ticker,
                        MAX(current_price) as current_price,
                        MAX(COALESCE(timestral_potential, CASE WHEN LOWER(model_name) LIKE '%timestral%' THEN potential END)) as timestral_potential,
                        MAX(COALESCE(timesfm_potential, CASE WHEN LOWER(model_name) LIKE '%timesfm%' THEN potential END)) as timesfm_potential,
                        BOOL_OR(COALESCE(dual_model_consensus, FALSE)) as dual_model_consensus,
                        MAX(trust_net_5d) as trust_net_5d,
                        MAX(foreign_net_5d) as foreign_net_5d
                    FROM ranked_pred
                    WHERE rank_time = 1
                    GROUP BY ticker;
                """
                pred_df = con.execute(pred_query, unique_tickers).df()
                if not pred_df.empty:
                    for _, row in pred_df.iterrows():
                        t_key = str(row["ticker"]).split(".")[0]
                        enrichment[t_key] = {
                            "current_price": float(row["current_price"]) if row["current_price"] is not None and not pd.isna(row["current_price"]) else None,
                            "change_pct": None,
                            "timestral_potential": float(row["timestral_potential"]) if row["timestral_potential"] is not None and not pd.isna(row["timestral_potential"]) else None,
                            "timesfm_potential": float(row["timesfm_potential"]) if row["timesfm_potential"] is not None and not pd.isna(row["timesfm_potential"]) else None,
                            "dual_model_consensus": bool(row["dual_model_consensus"]) if row["dual_model_consensus"] is not None and not pd.isna(row["dual_model_consensus"]) else None,
                            "institutional_streak": None,
                        }
                        # 若有 5 日投信買超記錄，可作初步籌碼標記
                        t_net = row.get("trust_net_5d")
                        if t_net is not None and not pd.isna(t_net) and int(t_net) > 0:
                            enrichment[t_key]["institutional_streak"] = f"投信5D+{int(t_net):,}張"

                # 2. 批次從 tw_daily_bars 取得最新兩日收盤價以計算真實當日漲跌幅
                try:
                    bars_query = f"""
                        WITH targets AS (SELECT UNNEST([{placeholders}]) AS t_raw),
                        ranked_bars AS (
                            SELECT
                                b.ticker,
                                b.close,
                                b.date,
                                ROW_NUMBER() OVER (PARTITION BY b.ticker ORDER BY b.date DESC) as rn
                            FROM tw_daily_bars b
                            WHERE b.ticker IN (SELECT t_raw FROM targets)
                               OR b.ticker IN (SELECT t_raw || '.TW' FROM targets)
                               OR b.ticker IN (SELECT t_raw || '.TWO' FROM targets)
                        )
                        SELECT ticker, close, date, rn FROM ranked_bars WHERE rn <= 2;
                    """
                    bars_df = con.execute(bars_query, unique_tickers).df()
                    if not bars_df.empty:
                        for (raw_t,), group in bars_df.groupby(["ticker"]):
                            t_key = str(raw_t).split(".")[0]
                            sorted_group = group.sort_values(by="rn")
                            if len(sorted_group) >= 1:
                                latest_close = float(sorted_group.iloc[0]["close"])
                                if t_key not in enrichment:
                                    enrichment[t_key] = {"current_price": latest_close}
                                else:
                                    enrichment[t_key]["current_price"] = latest_close

                                if len(sorted_group) >= 2:
                                    prev_close = float(sorted_group.iloc[1]["close"])
                                    if prev_close > 0:
                                        pct = round(((latest_close - prev_close) / prev_close) * 100, 2)
                                        enrichment[t_key]["change_pct"] = pct
                except Exception as bar_e:
                    logger.debug(f"tw_daily_bars 查詢補充略過: {bar_e}")

                # 3. 批次查詢 tw_institutional_daily 連買天數
                try:
                    inst_query = f"""
                        WITH targets AS (SELECT UNNEST([{placeholders}]) AS t_raw),
                        ranked_inst AS (
                            SELECT
                                i.ticker,
                                i.trust_net,
                                i.foreign_net,
                                i.date,
                                ROW_NUMBER() OVER (PARTITION BY i.ticker ORDER BY i.date DESC) as rn
                            FROM tw_institutional_daily i
                            WHERE i.ticker IN (SELECT t_raw FROM targets)
                               OR i.ticker IN (SELECT t_raw || '.TW' FROM targets)
                               OR i.ticker IN (SELECT t_raw || '.TWO' FROM targets)
                        )
                        SELECT ticker, trust_net, foreign_net, date, rn FROM ranked_inst WHERE rn <= 5;
                    """
                    inst_df = con.execute(inst_query, unique_tickers).df()
                    if not inst_df.empty:
                        for (raw_t,), group in inst_df.groupby(["ticker"]):
                            t_key = str(raw_t).split(".")[0]
                            sorted_group = group.sort_values(by="rn")
                            # 計算投信連買
                            trust_streak = 0
                            for _, r in sorted_group.iterrows():
                                if r["trust_net"] is not None and r["trust_net"] > 0:
                                    trust_streak += 1
                                else:
                                    break
                            if trust_streak >= 2:
                                if t_key in enrichment:
                                    enrichment[t_key]["institutional_streak"] = f"投信連買{trust_streak}D"
                            elif trust_streak == 1:
                                if t_key in enrichment and not enrichment[t_key].get("institutional_streak"):
                                    enrichment[t_key]["institutional_streak"] = "投信買超"
                except Exception as inst_e:
                    logger.debug(f"tw_institutional_daily 連買查詢略過: {inst_e}")

        except Exception as e:
            logger.warning(f"DuckDB 批量補強行情失敗: {e}")

        return enrichment

    # ==================== Watchlist Methods ====================

    def normalize_watchlist_ticker(self, ticker: str) -> Tuple[str, str, str]:
        """
        Normalize input ticker and deduce market.
        Returns (ticker, canonical_ticker, market).
        """
        raw_ticker = str(ticker).strip()
        if not raw_ticker:
            raise ValueError("Ticker cannot be empty")

        canonical = normalize_ticker(raw_ticker)
        canonical_upper = canonical.upper()

        if canonical_upper.endswith(".TW") or canonical_upper.endswith(".TWO"):
            market = "TW"
        elif canonical_upper.endswith(".T") or canonical_upper.endswith(".JP"):
            market = "JP"
        else:
            market = "US"

        return raw_ticker, canonical, market

    def add_watchlist_item(
        self,
        profile_id: str = "default",
        ticker: str = "",
        note: str = "",
        sort_order: Optional[int] = None
    ) -> Dict[str, Any]:
        """新增或更新自選股，限制 note <= 500 字元且單一 profile 上限 100 檔"""
        profile_id = str(profile_id or "default").strip()
        raw_ticker, canonical, market = self.normalize_watchlist_ticker(ticker)
        note = str(note or "").strip()
        if len(note) > 500:
            raise ValueError("Note exceeds 500 characters maximum")

        now_dt = datetime.datetime.now(datetime.timezone.utc)

        with self._write_lock(), self._get_connection(read_only=False) as con:
            existing = con.execute(
                "SELECT COUNT(*) FROM watchlist_items WHERE profile_id = ? AND canonical_ticker = ?",
                [profile_id, canonical]
            ).fetchone()[0]

            if existing == 0:
                total_cnt = con.execute(
                    "SELECT COUNT(*) FROM watchlist_items WHERE profile_id = ?",
                    [profile_id]
                ).fetchone()[0]
                if total_cnt >= 100:
                    raise ValueError(f"Watchlist limit of 100 items reached for profile '{profile_id}'")

            con.execute("""
                INSERT INTO watchlist_items (profile_id, ticker, canonical_ticker, market, note, sort_order, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (profile_id, canonical_ticker) DO UPDATE SET
                    note = EXCLUDED.note,
                    sort_order = COALESCE(EXCLUDED.sort_order, watchlist_items.sort_order)
            """, [profile_id, raw_ticker, canonical, market, note, sort_order, now_dt])

        return {
            "profile_id": profile_id,
            "ticker": raw_ticker,
            "canonical_ticker": canonical,
            "market": market,
            "note": note,
            "sort_order": sort_order,
            "created_at": now_dt.isoformat()
        }

    def remove_watchlist_item(self, profile_id: str = "default", ticker: str = "") -> bool:
        """自選股移除"""
        profile_id = str(profile_id or "default").strip()
        raw_ticker, canonical, _ = self.normalize_watchlist_ticker(ticker)

        with self._write_lock(), self._get_connection(read_only=False) as con:
            con.execute(
                "DELETE FROM watchlist_items WHERE profile_id = ? AND canonical_ticker = ?",
                [profile_id, canonical]
            )
            return True

    def get_watchlist_items(self, profile_id: str = "default") -> List[Dict[str, Any]]:
        """唯讀連線查詢自選股清單並附帶最新預測摘要"""
        profile_id = str(profile_id or "default").strip()
        items = []

        try:
            with self._get_connection(read_only=True) as con:
                rows = con.execute("""
                    SELECT profile_id, ticker, canonical_ticker, market, note, sort_order, created_at
                    FROM watchlist_items
                    WHERE profile_id = ?
                    ORDER BY sort_order ASC NULLS LAST, created_at DESC
                """, [profile_id]).fetchall()

                if not rows:
                    return []

                canonical_tickers = [r[2] for r in rows]
                pred_map = {}
                try:
                    pred_query = f"""
                        WITH ranked_preds AS (
                            SELECT ticker, current_price, predicted_price, potential,
                                   model_name, strategy_type, macro_regime, timestamp, tags,
                                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC, (predicted_price IS NOT NULL) DESC) as rn
                            FROM predictions
                            WHERE ticker IN ({','.join(['?'] * len(canonical_tickers))})
                        )
                        SELECT ticker, current_price, predicted_price, potential,
                               model_name, strategy_type, macro_regime, timestamp, tags
                        FROM ranked_preds WHERE rn = 1;
                    """
                    pred_rows = con.execute(pred_query, canonical_tickers).fetchall()
                    for pr in pred_rows:
                        pred_map[pr[0]] = {
                            "current_price": pr[1],
                            "predicted_price": pr[2],
                            "potential": pr[3],
                            "model_name": pr[4],
                            "strategy_type": pr[5],
                            "macro_regime": pr[6],
                            "timestamp": pr[7],
                            "tags": pr[8]
                        }
                except Exception as pred_e:
                    logger.debug("自選股預測摘要補強略過: %s", pred_e)

                for r in rows:
                    item_dict = {
                        "profile_id": r[0],
                        "ticker": r[1],
                        "canonical_ticker": r[2],
                        "market": r[3],
                        "note": r[4],
                        "sort_order": r[5],
                        "created_at": r[6].isoformat() if r[6] else None
                    }
                    if r[2] in pred_map:
                        item_dict.update(pred_map[r[2]])
                    items.append(item_dict)
        except DuckDBBusyError:
            raise
        except Exception as e:
            logger.error("查詢自選股清單失敗: %s", e)

        return items

    def batch_import_watchlist(
        self,
        profile_id: str = "default",
        tickers: List[str] = []
    ) -> Dict[str, Any]:
        """批次匯入自選股，白名單格式驗證、去重並強制 <= 100 檔上限"""
        profile_id = str(profile_id or "default").strip()
        added_count = 0
        valid_items = []

        import re
        TICKER_REGEX = re.compile(r"^[A-Za-z0-9.\-^]{1,12}$")

        seen = set()
        clean_tickers = []
        for t in tickers:
            cand = str(t).strip().upper()
            if cand and TICKER_REGEX.match(cand) and cand not in seen:
                seen.add(cand)
                clean_tickers.append(cand)

        with self._write_lock(), self._get_connection(read_only=False) as con:
            existing_count = con.execute(
                "SELECT COUNT(*) FROM watchlist_items WHERE profile_id = ?",
                [profile_id]
            ).fetchone()[0]

            for idx, raw_t in enumerate(clean_tickers):
                try:
                    _, canonical, market = self.normalize_watchlist_ticker(raw_t)
                except Exception:
                    continue

                already_in = con.execute(
                    "SELECT COUNT(*) FROM watchlist_items WHERE profile_id = ? AND canonical_ticker = ?",
                    [profile_id, canonical]
                ).fetchone()[0]

                if not already_in and existing_count >= 100:
                    break

                now_dt = datetime.datetime.now(datetime.timezone.utc)
                con.execute("""
                    INSERT INTO watchlist_items (profile_id, ticker, canonical_ticker, market, note, sort_order, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (profile_id, canonical_ticker) DO UPDATE SET
                        sort_order = COALESCE(EXCLUDED.sort_order, watchlist_items.sort_order)
                """, [profile_id, raw_t, canonical, market, "", existing_count + 1, now_dt])

                if not already_in:
                    existing_count += 1
                    added_count += 1

                valid_items.append(canonical)

        return {
            "profile_id": profile_id,
            "added": added_count,
            "total": existing_count,
            "canonical_tickers": valid_items
        }

    # ==================== QUANT PRICE ALERTS ====================

    def add_quant_alert(
        self,
        canonical_ticker: str,
        rule_type: str,
        threshold_value: Optional[float] = None,
        reference_model: str = "auto",
        cooldown_minutes: int = 720,
        profile_id: str = "default"
    ) -> Dict[str, Any]:
        """
        新增量化價位或分位數動態告警規則。
        驗證門檻值與分位數可用性：
        - 靜態門檻 (above_price, below_price) 必須帶有正數 threshold_value。
        - 動態門檻 (below_p10) 必須驗證該標的是否存在有效 P10 分位數預測。
        - 動態均線 (pullback_ma60, pullback_ma120) 驗證是否有最新均線數據。
        """
        import uuid
        profile_id = str(profile_id or "default").strip()
        rule_type = str(rule_type or "").strip().lower()
        reference_model = str(reference_model or "auto").strip().lower()

        valid_rules = {
            "above_price", "below_price",
            "below_p10", "above_p50",
            "pullback_ma60", "pullback_ma120"
        }
        if rule_type not in valid_rules:
            raise ValueError(f"不支援的告警規則類型: {rule_type}，合法範圍: {sorted(list(valid_rules))}")

        _, canonical, _ = self.normalize_watchlist_ticker(canonical_ticker)

        # 靜態規則驗證
        if rule_type in ("above_price", "below_price"):
            if threshold_value is None or float(threshold_value) <= 0:
                raise ValueError("靜態價位告警必須提供大於 0 之 threshold_value 門檻價格")
            threshold_val = float(threshold_value)
        else:
            threshold_val = float(threshold_value) if threshold_value is not None else None

        # 動態分位數規則驗證
        with self._get_connection(read_only=True) as con:
            max_ts_row = con.execute("""
                SELECT MAX(timestamp) FROM predictions WHERE ticker = ? OR ticker = ?
            """, [canonical, canonical.split(".")[0]]).fetchone()
            latest_ts = max_ts_row[0] if max_ts_row else None

            pred_rows = []
            if latest_ts:
                pred_rows = con.execute("""
                    SELECT timestral_p10, moirai_p10, predicted_price, ma60, ma120, current_price,
                           model_name, timestral_predicted_price, timesfm_predicted_price, moirai_predicted_price
                    FROM predictions
                    WHERE (ticker = ? OR ticker = ?) AND timestamp = ?
                """, [canonical, canonical.split(".")[0], latest_ts]).fetchall()

            has_p10 = False
            has_p50 = False
            for pr in pred_rows:
                if pr[0] is not None or pr[1] is not None:
                    has_p10 = True
                if reference_model == "timestral":
                    if pr[7] is not None or (pr[6] and "timestral" in pr[6].lower() and pr[2] is not None):
                        has_p50 = True
                elif reference_model == "timesfm":
                    if pr[8] is not None or (pr[6] and "timesfm" in pr[6].lower() and pr[2] is not None):
                        has_p50 = True
                elif reference_model == "moirai":
                    if pr[9] is not None or (pr[6] and "moirai" in pr[6].lower() and pr[2] is not None):
                        has_p50 = True
                else:  # auto
                    if pr[2] is not None or pr[7] is not None or pr[8] is not None or pr[9] is not None:
                        has_p50 = True

            if rule_type == "below_p10":
                if not pred_rows or not has_p10:
                    raise ValueError(f"標的 {canonical} 最新預測資料庫無有效 P10 分位數（timestral_p10 / moirai_p10），無法建立 P10 動態告警")
            elif rule_type == "above_p50":
                if not pred_rows or not has_p50:
                    raise ValueError(f"標的 {canonical} 最新預測資料庫無有效目標價（predicted_price / P50），無法建立 P50 動態告警")

        alert_id = str(uuid.uuid4())
        now_dt = datetime.datetime.now(datetime.timezone.utc)

        with self._write_lock(), self._get_connection(read_only=False) as con:
            count = con.execute(
                "SELECT COUNT(*) FROM quant_price_alerts WHERE profile_id = ?",
                [profile_id]
            ).fetchone()[0]
            if count >= 50:
                raise ValueError(f"該設定檔 {profile_id} 告警規則已達上限（50 筆）")

            con.execute("""
                INSERT INTO quant_price_alerts (
                    alert_id, profile_id, canonical_ticker, rule_type, threshold_value,
                    reference_model, enabled, cooldown_minutes, last_triggered_at,
                    last_triggered_price, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
            """, [
                alert_id, profile_id, canonical, rule_type, threshold_val,
                reference_model, True, int(cooldown_minutes or 720), now_dt
            ])

        return {
            "alert_id": alert_id,
            "profile_id": profile_id,
            "canonical_ticker": canonical,
            "ticker": canonical,
            "rule_type": rule_type,
            "threshold_value": threshold_val,
            "reference_model": reference_model,
            "cooldown_minutes": cooldown_minutes,
            "enabled": True,
            "created_at": now_dt.isoformat()
        }

    def get_quant_alerts(
        self,
        profile_id: str = "default",
        enabled_only: bool = False
    ) -> List[Dict[str, Any]]:
        """查詢量化價位告警清單"""
        profile_id = str(profile_id or "default").strip()
        query = "SELECT alert_id, profile_id, canonical_ticker, rule_type, threshold_value, reference_model, enabled, cooldown_minutes, last_triggered_at, last_triggered_price, created_at FROM quant_price_alerts WHERE profile_id = ?"
        params = [profile_id]
        if enabled_only:
            query += " AND enabled = true"
        query += " ORDER BY created_at DESC"

        alerts = []
        with self._get_connection(read_only=True) as con:
            rows = con.execute(query, params).fetchall()
            for r in rows:
                alerts.append({
                    "alert_id": r[0],
                    "profile_id": r[1],
                    "canonical_ticker": r[2],
                    "ticker": r[2],
                    "rule_type": r[3],
                    "threshold_value": r[4],
                    "reference_model": r[5],
                    "enabled": bool(r[6]),
                    "cooldown_minutes": r[7],
                    "last_triggered_at": r[8].isoformat() if r[8] else None,
                    "last_triggered_price": r[9],
                    "created_at": r[10].isoformat() if r[10] else None
                })
        return alerts

    def delete_quant_alert(
        self,
        alert_id: str,
        profile_id: str = "default"
    ) -> bool:
        """刪除指定量化價位告警"""
        profile_id = str(profile_id or "default").strip()
        alert_id = str(alert_id or "").strip()
        if not alert_id:
            return False

        with self._write_lock(), self._get_connection(read_only=False) as con:
            count = con.execute(
                "SELECT COUNT(*) FROM quant_price_alerts WHERE alert_id = ? AND profile_id = ?",
                [alert_id, profile_id]
            ).fetchone()[0]
            if count == 0:
                return False
            con.execute(
                "DELETE FROM quant_price_alerts WHERE alert_id = ? AND profile_id = ?",
                [alert_id, profile_id]
            )
            return True

    def record_alert_trigger(
        self,
        alert_id: str,
        triggered_price: float,
        profile_id: str = "default",
        cooldown_minutes: Optional[int] = None
    ) -> bool:
        """原子 CAS 更新告警觸發時間與觸發價格（防洗版與並發重複推播）"""
        profile_id = str(profile_id or "default").strip()
        alert_id = str(alert_id or "").strip()
        now_dt = datetime.datetime.now(datetime.timezone.utc)

        with self._write_lock(), self._get_connection(read_only=False) as con:
            if cooldown_minutes is not None and cooldown_minutes > 0:
                cutoff_dt = now_dt - datetime.timedelta(minutes=cooldown_minutes)
                row = con.execute("""
                    UPDATE quant_price_alerts
                    SET last_triggered_at = ?, last_triggered_price = ?
                    WHERE alert_id = ? AND profile_id = ?
                      AND (last_triggered_at IS NULL OR last_triggered_at <= ?)
                    RETURNING alert_id
                """, [now_dt, float(triggered_price), alert_id, profile_id, cutoff_dt]).fetchone()
            else:
                row = con.execute("""
                    UPDATE quant_price_alerts
                    SET last_triggered_at = ?, last_triggered_price = ?
                    WHERE alert_id = ? AND profile_id = ?
                    RETURNING alert_id
                """, [now_dt, float(triggered_price), alert_id, profile_id]).fetchone()

            return row is not None



