"""
data/macro.py - 美股宏觀環境與風控分析器 (Macro Regime Analyzer)

監控美股指標 (SPY, ^VIX, ^SOX)，動態評估全球市場風險等級與建議投資曝險比例 (0.0 ~ 1.0)。
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
import yfinance as yf

logger = logging.getLogger("stock_app.macro")


@dataclass
class MacroState:
    """宏觀市場狀態數據結構"""
    market: str = "US"  # "TW" or "US"
    regime_name: str = "中性 (Neutral)"
    exposure: float = 1.0  # 0.0 ~ 1.0
    vix: float = 20.0
    spy_price: float = 0.0
    spy_ma60: float = 0.0
    spy_above_ma60: bool = True
    spy_above_ma20: bool = True
    sox_price: float = 0.0
    sox_ma60: float = 0.0
    sox_above_ma60: bool = True
    sox_mom_20d: float = 0.0
    twii_price: float = 0.0
    twii_ma60: float = 0.0
    twii_above_ma60: bool = True
    twii_above_ma20: bool = True
    tech_exposure_cap: float = 1.0  # 科技股曝險上限
    warnings: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market": self.market,
            "regime_name": self.regime_name,
            "exposure": round(self.exposure, 2),
            "vix": round(self.vix, 2),
            "spy_above_ma60": self.spy_above_ma60,
            "sox_above_ma60": self.sox_above_ma60,
            "twii_above_ma60": self.twii_above_ma60,
            "tech_exposure_cap": round(self.tech_exposure_cap, 2),
            "warnings": self.warnings,
            "summary": self.summary,
        }


class MacroRegimeAnalyzer:
    """全球與台美股宏觀市場環境判斷器"""

    US_MACRO_TICKERS = ["SPY", "^VIX", "^SOX"]
    TW_MACRO_TICKERS = ["^TWII", "^SOX", "^VIX"]

    @classmethod
    def evaluate_market(cls, index_name: str, period: str = "6mo") -> MacroState:
        """根據目標指數名稱自動路由台股或美股大盤風控評估"""
        norm_name = index_name.strip()
        if norm_name in ("台灣50", "台灣中型100", "TW0050", "TW0051", "0050", "0051") or norm_name.startswith("TW") or norm_name.endswith(".TW"):
            return cls.evaluate_tw_market(period=period)
        return cls.evaluate_us_market(period=period)

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
                logger.warning("⚠️ 未能下載台股大盤數據，採用預設中性配置 (曝險 100%)")
                state.summary = "大盤數據缺失，預設全額曝險 (100%)"
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
                state.regime_name = "🚨 全球恐慌避險 (Global Panic)"
                state.exposure = 0.0
                warnings.append(f"國際 VIX 恐慌指數過高 ({state.vix:.1f} > 28)，全球避險熔斷，建議台股多單暫停進場")
            # (B) 加權指數跌破季線 MA60
            elif not state.twii_above_ma60:
                if not state.sox_above_ma60:
                    state.regime_name = "🛡️ 台美雙破季線 (Bearish Defense)"
                    state.exposure = 0.3
                    warnings.append("加權指數與費城半導體同步跌破季線 MA60，建議降至 30% 輕倉防禦")
                else:
                    state.regime_name = "📉 加權弱勢整理 (Weak Consolidation)"
                    state.exposure = 0.5
                    warnings.append("加權指數處於季線 (MA60) 之下，防範假突破，嚴格控制部位")
            # (C) 加權指數站穩季線 MA60
            else:
                if state.twii_above_ma20:
                    if state.sox_above_ma60:
                        state.regime_name = "🟢 台股全面多頭 (Strong Bull)"
                        state.exposure = 1.0
                    else:
                        state.regime_name = "🌱 多頭回調 (費半整理)"
                        state.exposure = 0.85
                        warnings.append("加權指數多頭排列，但費半處於季線之下，半導體與科技股留意回測震盪")
                else:
                    state.regime_name = "🌱 台股多頭回調 (Bull Pullback)"
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
            logger.info(f"✅ 台股大盤風控評估完成: {state.summary}")

        except Exception as e:
            logger.error(f"❌ 計算台股大盤狀態失敗: {e}", exc_info=True)
            state.summary = f"台股大盤計算異常 ({e})，採用中性 100% 曝險"

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
                logger.warning("⚠️ 未能下載宏觀數據，採用預設中性配置 (曝險 100%)")
                state.summary = "宏觀數據缺失，預設全額曝險 (100%)"
                return state

            # 1. 解析 VIX
            vix_val = cls._extract_latest_close(data, "^VIX")
            if vix_val is not None:
                state.vix = vix_val
            else:
                state.vix = 20.0

            # 2. 解析 SPY (S&P 500)
            spy_series = cls._extract_close_series(data, "SPY")
            if spy_series is not None and len(spy_series) >= 60:
                state.spy_price = float(spy_series.iloc[-1])
                state.spy_ma60 = float(spy_series.rolling(60).mean().iloc[-1])
                spy_ma20 = float(spy_series.rolling(20).mean().iloc[-1])
                state.spy_above_ma60 = state.spy_price >= state.spy_ma60
                state.spy_above_ma20 = state.spy_price >= spy_ma20
            else:
                state.spy_above_ma60 = True
                state.spy_above_ma20 = True

            # 3. 解析 SOX (費城半導體)
            sox_series = cls._extract_close_series(data, "^SOX")
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
                state.regime_name = "🚨 極度恐慌 (Extreme Panic)"
                state.exposure = 0.0
                warnings.append(f"VIX 恐慌指數過高 ({state.vix:.1f} > 28)，觸發避險熔斷，建議完全空倉停止買進")
            elif state.vix >= 25.0:
                if state.spy_above_ma60:
                    state.regime_name = "⚠️ 劇烈震盪 (High Volatility)"
                    state.exposure = 0.5
                    warnings.append(f"VIX 偏高 ({state.vix:.1f})，降至半倉 50%")
                else:
                    state.regime_name = "🛡️ 空頭恐慌 (Bearish Panic)"
                    state.exposure = 0.2
                    warnings.append(f"SPY 破季線且 VIX 偏高 ({state.vix:.1f})，降至 20% 輕倉防禦")
            elif state.vix >= 22.0:
                if state.spy_above_ma60:
                    state.regime_name = "🌤️ 輕微震盪 (Mild Volatility)"
                    state.exposure = 0.7
                else:
                    state.regime_name = "📉 溫和空頭 (Mild Bear)"
                    state.exposure = 0.4
                    warnings.append("SPY 跌破季線 MA60，建議降至 40% 倉位")
            else:
                # VIX < 22.0
                if state.spy_above_ma60 and state.spy_above_ma20:
                    state.regime_name = "🟢 全面多頭 (Strong Bull)"
                    state.exposure = 1.0
                elif state.spy_above_ma60:
                    state.regime_name = "🌱 多頭回調 (Bull Pullback)"
                    state.exposure = 0.85
                else:
                    state.regime_name = "📉 弱勢整理 (Weak Consolidation)"
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
            logger.info(f"✅ 美股宏觀評估完成: {state.summary}")

        except Exception as e:
            logger.error(f"❌ 計算宏觀風控狀態失敗: {e}", exc_info=True)
            state.summary = f"宏觀計算異常 ({e})，採用中性 100% 曝險"

        return state

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
