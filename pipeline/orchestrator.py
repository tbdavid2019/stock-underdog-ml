"""
Two-Stage Execution Pipeline Orchestrator.
Coordinates concurrent I/O data prefetching, strategy computation, composite evaluation,
AI narrative commentary synthesis, and sinks routing (database persistence & notifications).
"""
import logging
from typing import Dict, List, Optional, Any
import datetime
import pandas as pd

from core.config import config
from core.device import DeviceManager
from data.fetcher import StockFetcher
from data.fundamentals import FundamentalProvider
from data.macro import MacroRegimeAnalyzer, MacroState
from strategies.base import StockContext, StrategyResult
from strategies.registry import StrategyRegistry
# Import strategy modules to trigger registration
import strategies.xuantie
import strategies.lstm
import strategies.sector_rotation
import strategies.institutional

from evaluators.composite_evaluator import CompositeEvaluator, EvaluationReport
from evaluators.ai_narrative import AINarrativeEngine
from evaluators.formatter import print_evaluation_report
from database import SupabaseManager
from data.duckdb_manager import DuckDBManager
from notifier_dual import send_dual_strategy_results

logger = logging.getLogger("stock_app.pipeline")


class PipelineOrchestrator:
    """Orchestrator for executing quantitative stock selection pipelines"""

    def __init__(
        self, 
        enabled_strategies: Optional[List[str]] = None,
        strategy_weights: Optional[Dict[str, float]] = None,
        db_manager: Optional[SupabaseManager] = None,
        duckdb_manager: Optional[DuckDBManager] = None
    ):
        self.enabled_strategy_names = enabled_strategies or config.pipeline.ENABLED_STRATEGIES
        # 自動納入 sector_rotation 與 institutional 策略若未指定
        if "sector_rotation" not in self.enabled_strategy_names:
            self.enabled_strategy_names.append("sector_rotation")
        if "institutional" not in self.enabled_strategy_names:
            self.enabled_strategy_names.append("institutional")

        self.strategies = StrategyRegistry.create_strategies(self.enabled_strategy_names)
        self.evaluator = CompositeEvaluator(weights=strategy_weights or config.pipeline.STRATEGY_WEIGHTS)
        self.ai_narrative_engine = AINarrativeEngine()
        self.db_manager = db_manager or SupabaseManager()
        self.duckdb_manager = duckdb_manager or DuckDBManager()

    def run_index_analysis(
        self, 
        index_name: str, 
        stock_list: List[str], 
        period: str = "6mo",
        macro_state: Optional[MacroState] = None,
        persist_db: bool = True,
        send_notify: bool = True
    ) -> EvaluationReport:
        """
        Execute full pipeline for a single stock index.
        
        Stages:
        1. Stage 1 (I/O): Concurrent download of OHLCV and fundamental metrics.
        2. Stage 2 (Compute): Polymorphic strategy execution across all preloaded data.
        3. Stage 3 (Evaluation): Multi-factor composite scoring, intersection, and AI narrative synthesis.
        4. Stage 4 (Sinks): Output formatted console tables, send notifications, and save to DB.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 開始分析指數: {index_name} (共 {len(stock_list)} 支成分股)")
        logger.info(f"⚙️ 啟用策略: {', '.join([s.name for s in self.strategies])}")
        logger.info(f"{'='*70}\n")

        # ============= Stage 1: Batch I/O Prefetching =============
        logger.info("🌐 [Stage 1/4] 並發預載行情數據與基本面指標...")
        
        # 1.1 批次下載歷史行情
        data_map = StockFetcher.get_stock_data_batch(stock_list, period=period)
        
        # 1.2 批次下載基本面指標
        fundamentals_map = FundamentalProvider.get_fundamentals_batch(stock_list)

        # 1.3 建立 StockContext 字典
        contexts: Dict[str, StockContext] = {}
        for ticker in stock_list:
            if ticker in data_map and not data_map[ticker].empty:
                contexts[ticker] = StockContext(
                    ticker=ticker,
                    df=data_map[ticker],
                    fundamentals=fundamentals_map.get(ticker, {}),
                    metadata={"index_name": index_name}
                )

        logger.info(f"✅ 成功載入 {len(contexts)}/{len(stock_list)} 支股票數據\n")

        # ============= Stage 2: Strategy Execution =============
        logger.info("🤖 [Stage 2/4] 執行策略運算...")
        strategy_outputs: Dict[str, List[StrategyResult]] = {}

        for strat in self.strategies:
            strat_key = strat.name
            # Map canonical keys
            if "玄鐵" in strat.name or strat.name == "xuantie":
                strat_key = "xuantie"
            elif "LSTM" in strat.name or strat.name == "lstm":
                strat_key = "lstm"
            elif "板塊" in strat.name or strat.name == "sector_rotation":
                strat_key = "sector_rotation"
            elif "籌碼" in strat.name or strat.name == "institutional":
                strat_key = "institutional"

            logger.info(f"   ▶ 正在執行策略: {strat.name} ({strat.category})...")
            results = strat.evaluate_batch(contexts)
            strategy_outputs[strat_key] = results
            hits = [r for r in results if r.is_hit]
            logger.info(f"   ✓ {strat.name} 計算完成 (符合條件: {len(hits)} 支)")

        logger.info("")

        # ============= Stage 3: Composite Evaluation & AI Narrative =============
        logger.info("⭐ [Stage 3/4] 進行多策略綜合評分與交集分析...")
        report = self.evaluator.evaluate(
            index_name, 
            strategy_outputs, 
            fundamentals_map, 
            macro_state=macro_state
        )
        
        # 3.1 構造相容舊版結構字典，並附帶多維標籤與法人籌碼
        candidates_map = {}
        for s in getattr(report, "ranked_stocks", []):
            t = s.get("ticker") if isinstance(s, dict) else getattr(s, "ticker", None)
            if t:
                tags = s.get("tags", []) if isinstance(s, dict) else getattr(s, "tags", [])
                score = s.get("composite_score", 0.0) if isinstance(s, dict) else getattr(s, "composite_score", 0.0)
                signals = s.get("signals", []) if isinstance(s, dict) else getattr(s, "signals", [])
                candidates_map[t] = {
                    "tags": tags,
                    "composite_score": score,
                    "signals": signals
                }
        
        inst_summaries = {}
        for t, ctx in stock_contexts.items():
            if ctx.institutional_summary:
                inst_summaries[t] = ctx.institutional_summary.to_dict()

        report_dict = {
            "xuantie_results": report.xuantie_results,
            "lstm_results": report.lstm_results,
            "overlap_results": report.overlap_results,
            "macro_state": macro_state,
            "candidates_map": candidates_map,
            "institutional_summaries": inst_summaries
        }

        # 3.2 生成 AI 操盤解讀 (3-Tier Fallback)
        report.ai_summary = self.ai_narrative_engine.generate_narrative(
            index_name, 
            macro_state, 
            report_dict
        )
        report_dict["ai_summary"] = report.ai_summary

        # 輸出終端機美化報告
        print_evaluation_report(report, logger)

        # ============= Stage 4: Sinks (Notifications & DB) =============
        # 4.1 發送通知 (Telegram, Discord, Email)
        if send_notify:
            try:
                name_map = StockFetcher.get_index_name_map(index_name)
                send_dual_strategy_results(
                    index_name, 
                    report_dict, 
                    name_map=name_map,
                    macro_state=macro_state,
                    ai_summary=report.ai_summary
                )
            except Exception as e:
                logger.error(f"❌ 發送通知失敗 ({index_name}): {e}")

        # 4.2 寫入 Supabase 雲端資料庫
        if persist_db and self.db_manager and self.db_manager.enabled:
            try:
                self.db_manager.save_dual_strategy_results(index_name, report_dict, period=period)
            except Exception as e:
                logger.error(f"❌ Supabase 資料庫寫入失敗 ({index_name}): {e}")

        # 4.3 雙備份：寫入本地 DuckDB 時序資料庫
        if persist_db and self.duckdb_manager and self.duckdb_manager.enabled:
            try:
                self.duckdb_manager.save_dual_strategy_results(
                    index_name, 
                    report_dict, 
                    period=period, 
                    macro_state=macro_state
                )
            except Exception as e:
                logger.error(f"❌ DuckDB 本地寫入失敗 ({index_name}): {e}")

        return report

    def run_all_indices(
        self, 
        indices: Optional[Dict[str, List[str]]] = None,
        period: str = "6mo",
        persist_db: bool = True,
        send_notify: bool = True
    ) -> Dict[str, EvaluationReport]:
        """
        Execute analysis across all major supported stock indices with pre-flight Macro check.
        """
        # ============= Pre-flight Stage: US Macro Regime Gate =============
        logger.info("🌍 [Pre-flight] 評估美股宏觀環境與全球風控門檻...")
        macro_state = MacroRegimeAnalyzer.evaluate_us_market(period=period)
        logger.info(f"   • 當前市場狀態: {macro_state.regime_name} (建議曝險: {int(macro_state.exposure*100)}%)\n")

        if indices is None:
            indices = {
                "台灣50": StockFetcher.get_tw0050_stocks(),
                "台灣中型100": StockFetcher.get_tw0051_stocks(),
                "SP500": StockFetcher.get_sp500_stocks()
            }

        all_reports = {}
        for index_name, stock_list in indices.items():
            if not stock_list:
                logger.warning(f"⚠️ 指數 {index_name} 成分股為空，跳過")
                continue
            rep = self.run_index_analysis(
                index_name, 
                stock_list, 
                period=period, 
                macro_state=macro_state,
                persist_db=persist_db, 
                send_notify=send_notify
            )
            all_reports[index_name] = rep

        return all_reports
