"""
data/macro.py - 美股宏觀環境與風控分析器 (Macro Regime Analyzer)

監控美股指標 (SPY, ^VIX, ^SOX)，動態評估全球市場風險等級與建議投資曝險比例 (0.0 ~ 1.0)。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import yfinance as yf

logger = logging.getLogger("stock_app.macro")


@dataclass
class MacroState:
    """宏觀市場狀態數據結構"""
    market: str = "US"  # "TW" or "US"
    regime_name: str = "資料不足"
    exposure: Optional[float] = None  # 0.0 ~ 1.0; None means unavailable
    vix: Optional[float] = None
    spy_price: Optional[float] = None
    spy_ma60: Optional[float] = None
    spy_above_ma60: Optional[bool] = None
    spy_above_ma20: Optional[bool] = None
    sox_price: Optional[float] = None
    sox_ma60: Optional[float] = None
    sox_above_ma60: Optional[bool] = None
    sox_mom_20d: Optional[float] = None
    twii_price: Optional[float] = None
    twii_ma60: Optional[float] = None
    twii_above_ma60: Optional[bool] = None
    twii_above_ma20: Optional[bool] = None
    tech_exposure_cap: Optional[float] = None  # None means unavailable
    data_available: bool = False
    warnings: List[str] = field(default_factory=list)
    summary: str = ""
    fed_rate: Optional[Dict[str, Any]] = None
    commodities: Optional[Dict[str, Any]] = None
    earnings_calendar: List[Dict[str, Any]] = field(default_factory=list)
    economic_calendar: List[Dict[str, Any]] = field(default_factory=list)
    catalyst_alerts: List[str] = field(default_factory=list)
    polymarket: Optional[Dict[str, Any]] = None

    def mark_unavailable(self, reason: str) -> None:
        """清除推導值，避免上游缺資料看起來像正常市場狀態。"""
        self.regime_name = "資料不足"
        self.exposure = None
        self.vix = None
        self.spy_price = None
        self.spy_ma60 = None
        self.spy_above_ma60 = None
        self.spy_above_ma20 = None
        self.sox_price = None
        self.sox_ma60 = None
        self.sox_above_ma60 = None
        self.sox_mom_20d = None
        self.twii_price = None
        self.twii_ma60 = None
        self.twii_above_ma60 = None
        self.twii_above_ma20 = None
        self.tech_exposure_cap = None
        self.data_available = False
        self.warnings = [reason]
        self.summary = f"{reason}，暫停產生建議曝險"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market": self.market,
            "regime_name": self.regime_name,
            "exposure": round(self.exposure, 2) if self.exposure is not None else None,
            "vix": round(self.vix, 2) if self.vix is not None else None,
            "spy_above_ma60": self.spy_above_ma60,
            "sox_above_ma60": self.sox_above_ma60,
            "twii_above_ma60": self.twii_above_ma60,
            "tech_exposure_cap": round(self.tech_exposure_cap, 2) if self.tech_exposure_cap is not None else None,
            "data_available": self.data_available,
            "warnings": self.warnings,
            "summary": self.summary,
            "fed_rate": self.fed_rate,
            "commodities": self.commodities,
            "earnings_calendar": self.earnings_calendar,
            "economic_calendar": self.economic_calendar,
            "catalyst_alerts": self.catalyst_alerts,
            "polymarket": self.polymarket,
        }


class MacroRegimeAnalyzer:
    """全球與台美股宏觀市場環境判斷器"""

    US_MACRO_TICKERS = ["SPY", "^VIX", "^SOX"]
    TW_MACRO_TICKERS = ["^TWII", "^SOX", "^VIX"]
    JP_MACRO_TICKERS = ["^N225", "^SOX", "^VIX", "JPY=X"]

    _cache: Dict[str, Tuple[float, Any]] = {}
    CACHE_TTL_SECONDS: float = 300.0

    @classmethod
    def evaluate_market(cls, index_name: str, period: str = "6mo", force_refresh: bool = False) -> MacroState:
        """根據目標指數名稱自動路由台股、美股或日股大盤風控評估，支援 5 分鐘記憶體快取加速"""
        norm_name = index_name.strip()
        val_upper = norm_name.upper()
        is_jp = val_upper in ("JP", "JPX", "TSE", "NIKKEI", "NIKKEI225", "N225", "JP225", "JAPAN") or any(k in norm_name for k in ("日經", "日股")) or norm_name.endswith(".T")
        is_tw = val_upper in ("TW", "TWSE", "TPEX", "TAIWAN", "0050", "0051", "TW0050", "TW0051", "TW1000") or any(k in norm_name for k in ("台股", "台灣")) or norm_name.endswith(".TW")
        market_code = "JP" if is_jp else ("TW" if is_tw else "US")
        cache_key = f"{market_code}_{period}"
        now = time.time()

        if not force_refresh and cache_key in cls._cache:
            ts, cached_state = cls._cache[cache_key]
            if now - ts < cls.CACHE_TTL_SECONDS and cached_state.data_available:
                return cached_state

        if is_jp:
            state = cls.evaluate_jp_market(period=period)
        elif is_tw:
            state = cls.evaluate_tw_market(period=period)
        else:
            state = cls.evaluate_us_market(period=period)

        if state.data_available:
            cls._cache[cache_key] = (now, state)
        elif cache_key in cls._cache:
            return cls._cache[cache_key][1]
        return state

    @classmethod
    def evaluate_tw_market(cls, period: str = "6mo") -> MacroState:
        """
        抓取台股加權指數 (^TWII)、費城半導體 (^SOX) 與 VIX 恐慌指數，
        以台股大盤自身趨勢為主，國際美股連動為輔，精準評估台股大盤風向與建議曝險。
        """
        state = MacroState(market="TW")
        try:
            logger.info("🌐 正在下載台股大盤與國際連動數據 (^TWII, ^SOX, ^VIX)...")
            data = yf.download(cls.TW_MACRO_TICKERS, period=period, progress=False, group_by="ticker")
            
            if data is None or data.empty:
                logger.warning("⚠️ 未能下載台股大盤數據，暫停產生曝險建議")
                state.mark_unavailable("台股大盤資料缺失")
                return state

            # 1. 解析 VIX 恐慌指數
            vix_val = cls._extract_latest_close(data, "^VIX")
            state.vix = vix_val if vix_val is not None else 20.0

            # 2. 解析 SOX (費城半導體)
            sox_series = cls._extract_close_series(data, "^SOX")
            if sox_series is not None and len(sox_series) >= 60:
                state.sox_price = float(sox_series.iloc[-1])
                state.sox_ma60 = float(sox_series.rolling(60).mean().iloc[-1])
                state.sox_above_ma60 = state.sox_price >= state.sox_ma60
                if len(sox_series) >= 20:
                    state.sox_mom_20d = float((state.sox_price / sox_series.iloc[-20] - 1.0) * 100)
            else:
                state.sox_above_ma60 = True

            # 3. 解析 ^TWII (台灣加權指數)
            twii_series = cls._extract_close_series(data, "^TWII")
            if (
                vix_val is None
                or sox_series is None
                or len(sox_series) < 60
                or twii_series is None
                or len(twii_series) < 60
            ):
                logger.warning("⚠️ 台股宏觀資料不完整，暫停產生曝險建議")
                state.mark_unavailable("台股宏觀資料不完整")
                return state

            state.data_available = True
            if twii_series is not None and len(twii_series) >= 60:
                state.twii_price = float(twii_series.iloc[-1])
                state.twii_ma60 = float(twii_series.rolling(60).mean().iloc[-1])
                twii_ma20 = float(twii_series.rolling(20).mean().iloc[-1])
                state.twii_above_ma60 = state.twii_price >= state.twii_ma60
                state.twii_above_ma20 = state.twii_price >= twii_ma20
            else:
                state.twii_above_ma60 = True
                state.twii_above_ma20 = True

            warnings = []

            # (A) 國際恐慌熔斷機制
            if state.vix >= 28.0:
                state.regime_name = "全球恐慌避險"
                state.exposure = 0.0
                warnings.append(f"國際 VIX 恐慌指數過高 ({state.vix:.1f} > 28)，全球避險熔斷，建議台股多單暫停進場")
            # (B) 加權指數跌破季線 MA60
            elif not state.twii_above_ma60:
                if not state.sox_above_ma60:
                    state.regime_name = "台美雙破季線"
                    state.exposure = 0.3
                    warnings.append("加權指數與費城半導體同步跌破季線 MA60，建議降至 30% 輕倉防禦")
                else:
                    state.regime_name = "加權弱勢整理"
                    state.exposure = 0.5
                    warnings.append("加權指數處於季線 (MA60) 之下，防範假突破，嚴格控制部位")
            # (C) 加權指數站穩季線 MA60
            else:
                if state.twii_above_ma20:
                    if state.sox_above_ma60:
                        state.regime_name = "台股全面多頭"
                        state.exposure = 1.0
                    else:
                        state.regime_name = "多頭回調"
                        state.exposure = 0.85
                        warnings.append("加權指數多頭排列，但費半處於季線之下，半導體與科技股留意回測震盪")
                else:
                    state.regime_name = "台股多頭回調"
                    state.exposure = 0.85
                    warnings.append("加權指數回測月線支撐，順應季線多頭趨勢分批佈局")

            # (D) 費半科技股曝險上限
            if not state.sox_above_ma60:
                state.tech_exposure_cap = 0.5
                warnings.append("費城半導體處於季線之下，電子科技股部位建議上限 50%")
            else:
                state.tech_exposure_cap = 1.0

            state.warnings = warnings

            summary_parts = [
                f"大盤: {state.regime_name}",
                f"建議曝險: {int(state.exposure * 100)}%",
                f"加權指數: {'站穩MA60' if state.twii_above_ma60 else '跌破MA60'}",
                f"國際連動: 費半{'站穩MA60' if state.sox_above_ma60 else '破季線'} (VIX {state.vix:.1f})"
            ]
            state.summary = " | ".join(summary_parts)
            cls._enrich_with_investing(state)
            cls._enrich_with_polymarket(state)
            logger.info(f"✅ 台股大盤風控評估完成: {state.summary}")

        except Exception as e:
            logger.error(f"❌ 計算台股大盤狀態失敗: {e}", exc_info=True)
            state.mark_unavailable(f"台股大盤計算異常 ({e})")

        return state

    @classmethod
    def evaluate_jp_market(cls, period: str = "6mo") -> MacroState:
        """
        抓取日經 225 (^N225)、費城半導體 (^SOX)、VIX 恐慌指數與美元兌日圓 (JPY=X)，
        評估日股大盤趨勢、半導體共振風向與日圓匯率動向，給出建議曝險與風控狀態。
        """
        state = MacroState(market="JP")
        try:
            logger.info("🌐 正在下載日股大盤與國際連動數據 (^N225, ^SOX, ^VIX, JPY=X)...")
            data = yf.download(cls.JP_MACRO_TICKERS, period=period, progress=False, group_by="ticker")

            if data is None or data.empty:
                logger.warning("⚠️ 未能下載日股大盤數據，暫停產生曝險建議")
                state.mark_unavailable("日股大盤資料缺失")
                return state

            # 1. 解析 VIX 恐慌指數
            vix_val = cls._extract_latest_close(data, "^VIX")
            state.vix = vix_val if vix_val is not None else 20.0

            # 2. 解析 SOX (費城半導體)
            sox_series = cls._extract_close_series(data, "^SOX")
            if sox_series is not None and len(sox_series) >= 60:
                state.sox_price = float(sox_series.iloc[-1])
                state.sox_ma60 = float(sox_series.rolling(60).mean().iloc[-1])
                state.sox_above_ma60 = state.sox_price >= state.sox_ma60
                if len(sox_series) >= 20:
                    state.sox_mom_20d = float((state.sox_price / sox_series.iloc[-20] - 1.0) * 100)
            else:
                state.sox_above_ma60 = True

            # 3. 解析 ^N225 (日經 225 指數)
            n225_series = cls._extract_close_series(data, "^N225")
            if (
                vix_val is None
                or sox_series is None
                or len(sox_series) < 60
                or n225_series is None
                or len(n225_series) < 60
            ):
                logger.warning("⚠️ 日股宏觀資料不完整，暫停產生曝險建議")
                state.mark_unavailable("日股宏觀資料不完整")
                return state

            state.data_available = True
            n225_price = float(n225_series.iloc[-1])
            n225_ma60 = float(n225_series.rolling(60).mean().iloc[-1])
            n225_ma20 = float(n225_series.rolling(20).mean().iloc[-1])
            n225_above_ma60 = n225_price >= n225_ma60
            n225_above_ma20 = n225_price >= n225_ma20

            # 借用 twii_price 儲存日經收盤價方便渲染
            state.twii_price = n225_price
            state.twii_ma60 = n225_ma60
            state.twii_above_ma60 = n225_above_ma60
            state.twii_above_ma20 = n225_above_ma20

            # 4. 解析美元兌日圓 (JPY=X)
            jpy_val = cls._extract_latest_close(data, "JPY=X")

            warnings = []

            # (A) 國際恐慌熔斷
            if state.vix >= 28.0:
                state.regime_name = "全球恐慌避險"
                state.exposure = 0.0
                warnings.append(f"國際 VIX 恐慌指數過高 ({state.vix:.1f} > 28)，全球避險熔斷，建議日股多單暫停進場")
            # (B) 日經指數跌破季線 MA60
            elif not n225_above_ma60:
                if not state.sox_above_ma60:
                    state.regime_name = "日美雙破季線"
                    state.exposure = 0.3
                    warnings.append("日經225與費城半導體同步跌破季線 MA60，建議降至 30% 輕倉防禦")
                else:
                    state.regime_name = "日經弱勢整理"
                    state.exposure = 0.5
                    warnings.append("日經225處於季線 (MA60) 之下，防範假突破，嚴格控制部位")
            # (C) 日經指數站穩季線 MA60
            else:
                if n225_above_ma20:
                    if state.sox_above_ma60:
                        state.regime_name = "日股全面多頭"
                        state.exposure = 1.0
                    else:
                        state.regime_name = "多頭回調"
                        state.exposure = 0.85
                        warnings.append("日經指數多頭排列，但費半處於季線之下，半導體設備與出口類股留意回測震盪")
                else:
                    state.regime_name = "日股多頭回調"
                    state.exposure = 0.85
                    warnings.append("日經指數回測月線支撐，順應季線多頭趨勢分批佈局")

            # (D) 匯率提醒
            if jpy_val is not None:
                if jpy_val > 155.0:
                    warnings.append(f"日圓匯率目前處於偏弱區間 (1 USD = {jpy_val:.1f} JPY)，出口權值股受惠但須防範日本央行 (BOJ) 干預")
                elif jpy_val < 145.0:
                    warnings.append(f"日圓強勢升值 (1 USD = {jpy_val:.1f} JPY)，留意日股獲利了結與套利交易 (Carry Trade) 平倉波動")

            state.warnings = warnings
            summary_parts = [
                f"大盤: {state.regime_name}",
                f"建議曝險: {int(state.exposure * 100)}%",
                f"日經225: {'站穩MA60' if n225_above_ma60 else '跌破MA60'}",
                f"國際連動: 費半{'站穩MA60' if state.sox_above_ma60 else '破季線'} (VIX {state.vix:.1f})"
            ]
            if jpy_val is not None:
                summary_parts.append(f"日圓 {jpy_val:.1f}")
            state.summary = " | ".join(summary_parts)
            cls._enrich_with_investing(state)
            cls._enrich_with_polymarket(state)
            logger.info(f"✅ 日股大盤風控評估完成: {state.summary}")

        except Exception as e:
            logger.error(f"❌ 計算日股大盤狀態失敗: {e}", exc_info=True)
            state.mark_unavailable(f"日股大盤計算異常 ({e})")

        return state

    @classmethod
    def evaluate_us_market(cls, period: str = "6mo") -> MacroState:
        """
        抓取 SPY, ^VIX, ^SOX 並計算美股宏觀風控狀態
        """
        state = MacroState(market="US")
        try:
            logger.info("🌐 正在下載美股宏觀數據 (SPY, ^VIX, ^SOX)...")
            data = yf.download(cls.US_MACRO_TICKERS, period=period, progress=False, group_by="ticker")
            
            if data is None or data.empty:
                logger.warning("⚠️ 未能下載宏觀數據，暫停產生曝險建議")
                state.mark_unavailable("美股宏觀資料缺失")
                return state

            # 1. 解析 VIX
            vix_val = cls._extract_latest_close(data, "^VIX")
            if vix_val is not None:
                state.vix = vix_val
            else:
                state.vix = 20.0

            # 2. 解析 SPY (S&P 500)
            spy_series = cls._extract_close_series(data, "SPY")
            # 3. 解析 SOX (費城半導體)
            sox_series = cls._extract_close_series(data, "^SOX")
            if (
                vix_val is None
                or spy_series is None
                or len(spy_series) < 60
                or sox_series is None
                or len(sox_series) < 60
            ):
                logger.warning("⚠️ 美股宏觀資料不完整，暫停產生曝險建議")
                state.mark_unavailable("美股宏觀資料不完整")
                return state

            state.data_available = True
            if spy_series is not None and len(spy_series) >= 60:
                state.spy_price = float(spy_series.iloc[-1])
                state.spy_ma60 = float(spy_series.rolling(60).mean().iloc[-1])
                spy_ma20 = float(spy_series.rolling(20).mean().iloc[-1])
                state.spy_above_ma60 = state.spy_price >= state.spy_ma60
                state.spy_above_ma20 = state.spy_price >= spy_ma20
            else:
                state.spy_above_ma60 = True
                state.spy_above_ma20 = True

            if sox_series is not None and len(sox_series) >= 60:
                state.sox_price = float(sox_series.iloc[-1])
                state.sox_ma60 = float(sox_series.rolling(60).mean().iloc[-1])
                state.sox_above_ma60 = state.sox_price >= state.sox_ma60
                if len(sox_series) >= 20:
                    state.sox_mom_20d = float((state.sox_price / sox_series.iloc[-20] - 1.0) * 100)
            else:
                state.sox_above_ma60 = True

            # 4. 綜合風控邏輯判定 (參考 tw_stocker v9.0 模型)
            warnings = []
            
            # (A) VIX 恐慌斷路器 (最高優先級)
            if state.vix >= 28.0:
                state.regime_name = "極度恐慌"
                state.exposure = 0.0
                warnings.append(f"VIX 恐慌指數過高 ({state.vix:.1f} > 28)，觸發避險熔斷，建議完全空倉停止買進")
            elif state.vix >= 25.0:
                if state.spy_above_ma60:
                    state.regime_name = "劇烈震盪"
                    state.exposure = 0.5
                    warnings.append(f"VIX 偏高 ({state.vix:.1f})，降至半倉 50%")
                else:
                    state.regime_name = "空頭恐慌"
                    state.exposure = 0.2
                    warnings.append(f"SPY 破季線且 VIX 偏高 ({state.vix:.1f})，降至 20% 輕倉防禦")
            elif state.vix >= 22.0:
                if state.spy_above_ma60:
                    state.regime_name = "輕微震盪"
                    state.exposure = 0.7
                else:
                    state.regime_name = "溫和空頭"
                    state.exposure = 0.4
                    warnings.append("SPY 跌破季線 MA60，建議降至 40% 倉位")
            else:
                # VIX < 22.0
                if state.spy_above_ma60 and state.spy_above_ma20:
                    state.regime_name = "全面多頭"
                    state.exposure = 1.0
                elif state.spy_above_ma60:
                    state.regime_name = "多頭回調"
                    state.exposure = 0.85
                else:
                    state.regime_name = "弱勢整理"
                    state.exposure = 0.5
                    warnings.append("SPY 處於 MA60 之下，防範假突破")

            # (B) SOX 費半科技門檻判定
            if not state.sox_above_ma60:
                if state.sox_mom_20d < -3.0:
                    state.tech_exposure_cap = 0.3
                    warnings.append(f"費城半導體破季線且動量疲弱 ({state.sox_mom_20d:+.1f}%)，科技/半導體股上限 30%")
                else:
                    state.tech_exposure_cap = 0.5
                    warnings.append("費城半導體處於季線之下，科技股部位建議半倉 (50%)")
            else:
                state.tech_exposure_cap = 1.0

            # (C) SPY + SOX 雙空極端風險
            if not state.spy_above_ma60 and not state.sox_above_ma60:
                state.exposure = min(state.exposure, 0.1)
                warnings.append("SPY 與 SOX 同步破季線，整體市場曝險降至最低 10%")

            state.warnings = warnings
            
            # 組裝文字摘要
            summary_parts = [
                f"環境: {state.regime_name}",
                f"建議曝險: {int(state.exposure * 100)}%",
                f"VIX: {state.vix:.1f}",
                f"SPY: {'站穩MA60' if state.spy_above_ma60 else '跌破MA60'}",
                f"SOX: {'站穩MA60' if state.sox_above_ma60 else '破季線'}"
            ]
            state.summary = " | ".join(summary_parts)
            cls._enrich_with_investing(state)
            cls._enrich_with_polymarket(state)
            logger.info(f"✅ 美股宏觀評估完成: {state.summary}")

        except Exception as e:
            logger.error(f"❌ 計算宏觀風控狀態失敗: {e}", exc_info=True)
            state.mark_unavailable(f"宏觀計算異常 ({e})")

        return state

    @classmethod
    def _enrich_with_investing(cls, state: MacroState):
        """以 Investing.com 宏觀數據與行事曆豐富風控指標 (優雅降級)"""
        try:
            from data.investing_service import InvestingService
            inv_summary = InvestingService.get_investing_macro_summary()
            state.fed_rate = inv_summary.get("fed_rate")
            state.commodities = inv_summary.get("commodities")
            state.earnings_calendar = inv_summary.get("earnings_calendar", [])
            state.economic_calendar = inv_summary.get("economic_calendar", [])
            state.catalyst_alerts = inv_summary.get("catalyst_alerts", [])

            # 若有今日/明日重磅催化劑事件 (如 CPI / FOMC)，自動加入風控警示
            if state.catalyst_alerts:
                for alert in state.catalyst_alerts[:2]:
                    if alert not in state.warnings:
                        state.warnings.append(alert)

            # 若有聯準會利率預期，自動於摘要標註
            if state.fed_rate and state.fed_rate.get("highest_prob_rate"):
                rate_str = state.fed_rate["highest_prob_rate"]
                prob_str = state.fed_rate.get("highest_probability", "")
                date_str = state.fed_rate.get("meeting_date", "")
                fed_note = f"FOMC ({date_str}) 預期利率 {rate_str} ({prob_str})"
                if fed_note not in state.summary:
                    state.summary += f" | {fed_note}"
        except Exception as e:
            logger.debug(f"InvestingService enrichment skipped: {e}")

    @classmethod
    def _enrich_with_polymarket(cls, state: MacroState):
        """以 Polymarket 真金白銀預測市場情緒豐富風控指標 (優雅降級)"""
        try:
            from data.polymarket_service import PolymarketService
            poly_data = PolymarketService.get_macro_sentiment()
            state.polymarket = poly_data
        except Exception as e:
            logger.debug(f"PolymarketService enrichment skipped: {e}")

    @staticmethod
    def _extract_close_series(data: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.levels[0]:
                sub = data[ticker]
                if "Close" in sub.columns:
                    return sub["Close"].dropna()
        elif "Close" in data.columns:
            return data["Close"].dropna()
        return None

    @staticmethod
    def _extract_latest_close(data: pd.DataFrame, ticker: str) -> Optional[float]:
        s = MacroRegimeAnalyzer._extract_close_series(data, ticker)
        if s is not None and not s.empty:
            return float(s.iloc[-1])
        return None
