"""
evaluators/ai_narrative.py - 3-Tier Fallback LLM 量化研報解讀引擎

支援主力 (Primary) ➔ 備援 1 (Fallback 1) ➔ 備援 2 (Fallback 2) ➔ 規則模板 (Template)
的 4 級容錯架構，使用標準 OpenAI 相容介面生成每日盤勢操盤洞察。
"""

import logging
import json
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd

from core.config import config
from data.macro import MacroState

logger = logging.getLogger("stock_app.evaluators.ai_narrative")


@dataclass
class LLMSlot:
    name: str
    base_url: str
    model: str
    api_key: Optional[str]
    timeout: int = 10


class AINarrativeEngine:
    """3 級 Fallback LLM 研報解讀生成器"""

    def __init__(self):
        self.enabled = config.llm.ENABLE_LLM_SUMMARY
        self.timeout = config.llm.TIMEOUT
        self.slots: List[LLMSlot] = [
            LLMSlot(
                name=config.llm.PRIMARY_NAME,
                base_url=config.llm.PRIMARY_BASE_URL,
                model=config.llm.PRIMARY_MODEL,
                api_key=config.llm.PRIMARY_API_KEY,
                timeout=self.timeout
            ),
            LLMSlot(
                name=config.llm.FALLBACK1_NAME,
                base_url=config.llm.FALLBACK1_BASE_URL,
                model=config.llm.FALLBACK1_MODEL,
                api_key=config.llm.FALLBACK1_API_KEY,
                timeout=self.timeout
            ),
            LLMSlot(
                name=config.llm.FALLBACK2_NAME,
                base_url=config.llm.FALLBACK2_BASE_URL,
                model=config.llm.FALLBACK2_MODEL,
                api_key=config.llm.FALLBACK2_API_KEY,
                timeout=self.timeout
            )
        ]

    def generate_narrative(
        self, 
        index_name: str, 
        macro_state: Optional[MacroState], 
        report_data: Dict[str, Any]
    ) -> str:
        """
        生成量化分析之自然語言解讀
        若 LLM 呼叫全數失敗或未啟用，自動回退純程式規則模板。
        """
        if not self.enabled:
            return self._generate_template_narrative(index_name, macro_state, report_data)

        # 構造嚴格遵循事實的 Prompt Context
        context_prompt = self._build_context_prompt(index_name, macro_state, report_data)
        
        # 依序嘗試各級 LLM
        for slot_idx, slot in enumerate(self.slots, start=1):
            if not slot.api_key or not slot.base_url or not slot.model:
                continue

            try:
                logger.info(f"🧠 [LLM Slot {slot_idx}] 正在請求 {slot.name} ({slot.model})...")
                content = self._call_openai_compatible(slot, context_prompt)
                if content and len(content.strip()) > 20:
                    logger.info(f"✅ [LLM Slot {slot_idx}] {slot.name} 研報生成成功！")
                    return content.strip()
            except Exception as e:
                logger.warning(f"⚠️ [LLM Slot {slot_idx}] {slot.name} 呼叫失敗 ({e})，切換下一備援...")

        logger.info("ℹ️ 所有 LLM 備援皆不可用或無 API Key，降級為純程式規則模板生成。")
        return self._generate_template_narrative(index_name, macro_state, report_data)

    def _call_openai_compatible(self, slot: LLMSlot, user_prompt: str) -> Optional[str]:
        """呼叫標準 OpenAI /chat/completions API"""
        url = slot.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {slot.api_key}"
        }

        system_prompt = (
            "你是一位頂級避險基金的資深量化分析師。請根據使用者提供的『已確定量化數據』，"
            "撰寫一段 100~150 字精闢、流暢、專業的每日操盤摘要。"
            "【鐵律】：嚴禁捏造數據或猜測未提及的指標，純粹基於提供的數據給出宏觀風險、強勢族群與焦點個股的具體操作解讀。"
        )

        payload = {
            "model": slot.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 350,
            "temperature": 0.2
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=slot.timeout)
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "")
        else:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:150]}")
        return None

    def _build_context_prompt(
        self, 
        index_name: str, 
        macro_state: Optional[MacroState], 
        report_data: Dict[str, Any]
    ) -> str:
        """構造傳入 LLM 的數據文本"""
        is_tw = index_name.strip() in ("台灣50", "台灣中型100", "TW0050", "TW0051", "0050", "0051") or index_name.endswith(".TW")
        
        if is_tw:
            market_type = "🇹🇼 台灣股市 (TWSE/TPEX)"
            if macro_state:
                twii_status = "站穩MA60季線" if getattr(macro_state, "twii_above_ma60", True) else "跌破MA60季線"
                sox_status = "站穩季線" if macro_state.sox_above_ma60 else "破季線"
                macro_info = (
                    f"台股大盤狀態: {macro_state.regime_name}\n"
                    f"建議曝險比例: {int(macro_state.exposure*100)}%\n"
                    f"加權指數 (^TWII): {twii_status}\n"
                    f"國際美股連動: 費城半導體{sox_status}, 國際VIX恐慌指數 {macro_state.vix:.1f}"
                )
            else:
                macro_info = "台股大盤數據缺失"
        else:
            market_type = "🇺🇸 美國股市 (US Markets)"
            if macro_state:
                spy_status = "站穩MA60" if macro_state.spy_above_ma60 else "跌破MA60"
                sox_status = "站穩MA60" if macro_state.sox_above_ma60 else "破季線"
                macro_info = (
                    f"美股宏觀狀態: {macro_state.regime_name}\n"
                    f"建議曝險比例: {int(macro_state.exposure*100)}%\n"
                    f"VIX 恐慌指數: {macro_state.vix:.1f}\n"
                    f"S&P 500 (SPY): {spy_status}\n"
                    f"費城半導體 (SOX): {sox_status}"
                )
            else:
                macro_info = "美股宏觀環境數據缺失"

        overlap_raw = report_data.get("overlap_results")
        if isinstance(overlap_raw, pd.DataFrame):
            overlap_list = overlap_raw.to_dict(orient="records") if not overlap_raw.empty else []
        elif isinstance(overlap_raw, list):
            overlap_list = overlap_raw
        else:
            overlap_list = []

        overlap_summary = []
        for r in overlap_list[:3]:
            ticker = r.get("ticker", "")
            pot = r.get("potential") or r.get("lstm_potential", 0.0)
            raw_tags = r.get("tags", [])
            tags = ",".join(raw_tags) if isinstance(raw_tags, list) else str(raw_tags)
            pe = r.get("pe", "N/A")
            pb = r.get("pb", "N/A")
            pullback = r.get("pullback_type", "MA60")
            overlap_summary.append(f"{ticker}: 預測漲幅 {pot:+.2f}%, 回調支撐 {pullback}, 估值 PE:{pe}/PB:{pb}, 標籤:[{tags}]")

        xuantie_raw = report_data.get("xuantie_results")
        xuantie_hits = len(xuantie_raw) if xuantie_raw is not None else 0
        lstm_raw = report_data.get("lstm_results")
        lstm_predictions = len(lstm_raw) if lstm_raw is not None else 0

        prompt = f"""
目標市場: {market_type} - {index_name}
【市場大盤與風控背景】
{macro_info}

【量化數據概況】
- 技術買點 (玄鐵重劍符合): {xuantie_hits} 支
- LSTM 短線預測完成: {lstm_predictions} 支
- ⭐ 重點交集/多維共振推薦標的:
{chr(10).join(['  • ' + s for s in overlap_summary]) if overlap_summary else '  • 本期無雙重/三重共振股票，建議維持防禦觀望'}

請針對以上數據，以專業、精準、客觀的視角撰寫操盤總評（100~150字）。
【語意要求】：若為台股指數請以台股大盤與籌碼為主視角，美股連動為輔；若為美股指數請以美股總體經濟與S&P500/VIX為主視角。
"""
        return prompt.strip()

    def _generate_template_narrative(
        self, 
        index_name: str, 
        macro_state: Optional[MacroState], 
        report_data: Dict[str, Any]
    ) -> str:
        """規則式純文字模板生成 (零依賴終極備援)"""
        is_tw = index_name.strip() in ("台灣50", "台灣中型100", "TW0050", "TW0051", "0050", "0051") or index_name.endswith(".TW")
        market_label = "台股大盤" if is_tw else "美股宏觀環境"

        overlap_raw = report_data.get("overlap_results")
        if isinstance(overlap_raw, pd.DataFrame):
            overlap_list = overlap_raw.to_dict(orient="records") if not overlap_raw.empty else []
        elif isinstance(overlap_raw, list):
            overlap_list = overlap_raw
        else:
            overlap_list = []

        exposure_str = f"{int(macro_state.exposure * 100)}%" if macro_state else "100%"
        regime_str = macro_state.regime_name if macro_state else "正常"

        if overlap_list:
            top_stock = overlap_list[0].get("ticker", "")
            top_pot = overlap_list[0].get("potential") or overlap_list[0].get("lstm_potential", 0.0)
            raw_tags = overlap_list[0].get("tags", [])
            tags_str = "、".join(raw_tags[:3]) if isinstance(raw_tags, list) else str(raw_tags)
            return (
                f"【量化操盤觀點】當前{market_label}處於「{regime_str}」，整體建議曝險為 {exposure_str}。"
                f"今日 {index_name} 優先聚焦 {top_stock}（預期潛力 {top_pot:+.2f}%），"
                f"符合 {tags_str or '多維共振'} 等條件，建議順應大盤水位分批佈局。"
            )
        else:
            return (
                f"【量化操盤觀點】當前{market_label}處於「{regime_str}」，建議曝險比例維持 {exposure_str}。"
                f"今日 {index_name} 無雙重或三重共振之重點交集標的，短線操作宜保持耐心，控制倉位。"
            )
