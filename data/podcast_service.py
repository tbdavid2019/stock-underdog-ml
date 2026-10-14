"""
data/podcast_service.py - AI Financial & Tech Podcast Intelligence Radar Pipeline
負責頻道清單管理、RSS 串流解析、實體與股票 Ticker 映射、行情增強與 AI 繁中 TL;DR 萃取。
"""

import os
import re
import json
import logging
import datetime
from typing import Dict, List, Optional, Any
import urllib.request
import xml.etree.ElementTree as ET

from core.config import config
from data.duckdb_manager import DuckDBManager

logger = logging.getLogger("stock_app.podcast")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "podcast")
SEED_CHANNELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "podcast_seed_channels.json")

# 金融與科技關鍵字至股票代號映射白名單 (Entity-to-Ticker Whitelist)
ENTITY_TICKER_MAP = {
    "NVIDIA": "NVDA",
    "NVDA": "NVDA",
    "輝達": "NVDA",
    "黃仁勳": "NVDA",
    "JENSEN HUANG": "NVDA",
    "OPENAI": "MSFT",  # 核心投資與獨家雲端
    "MICROSOFT": "MSFT",
    "微軟": "MSFT",
    "MSFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "GEMINI": "GOOGL",
    "DEEPMIND": "GOOGL",
    "GOOGL": "GOOGL",
    "CLAUDE": "AMZN",  # 主要策略股東與 Compute
    "ANTHROPIC": "AMZN",
    "AMAZON": "AMZN",
    "亞馬遜": "AMZN",
    "AWS": "AMZN",
    "AMZN": "AMZN",
    "META": "META",
    "FACEBOOK": "META",
    "ZUCKERBERG": "META",
    "祖克柏": "META",
    "LLAMA": "META",
    "APPLE": "AAPL",
    "蘋果": "AAPL",
    "AAPL": "AAPL",
    "TIM COOK": "AAPL",
    "TESLA": "TSLA",
    "特斯拉": "TSLA",
    "TSLA": "TSLA",
    "ELON MUSK": "TSLA",
    "馬斯克": "TSLA",
    "TSMC": "2330.TW",
    "台積電": "2330.TW",
    "2330": "2330.TW",
    "TSM": "TSM",
    "ASML": "ASML",
    "艾司摩爾": "ASML",
    "AMD": "AMD",
    "超微": "AMD",
    "LISA SU": "AMD",
    "BROADCOM": "AVGO",
    "博通": "AVGO",
    "AVGO": "AVGO",
    "ORACLE": "ORCL",
    "甲骨文": "ORCL",
    "ORCL": "ORCL",
    "MICRON": "MU",
    "美光": "MU",
    "QUALCOMM": "QCOM",
    "高通": "QCOM",
    "INTEL": "INTC",
    "英特爾": "INTC",
    "CURSOR": "NVDA",
    "鴻海": "2317.TW",
    "FOXCONN": "2317.TW",
    "聯發科": "2454.TW",
    "MEDIATEK": "2454.TW",
    "廣達": "2382.TW",
    "QUANTA": "2382.TW",
    "美團": "3690.HK",
    "3690": "3690.HK",
    "騰訊": "0700.HK",
    "TENCENT": "0700.HK",
    "0700": "0700.HK",
    "阿里巴巴": "9988.HK",
    "ALIBABA": "BABA",
    "9988": "9988.HK",
    "拼多多": "PDD",
    "PDD": "PDD",
    "TEMU": "PDD",
    "MODERNA": "MRNA",
    "莫德納": "MRNA",
    "MRNA": "MRNA",
    "ASTRAZENECA": "AZN",
    "阿斯特捷利康": "AZN",
    "AZN": "AZN",
    "NOVARTIS": "NVS",
    "諾華": "NVS",
    "NVS": "NVS",
    "ARGENX": "ARGX",
    "ARGX": "ARGX",
    "EXXON": "XOM",
    "艾克森美孚": "XOM",
    "CHEVRON": "CVX",
    "雪佛龍": "CVX",
    "VISTRA": "VST",
    "CONSTELLATION": "CEG",
    "星座能源": "CEG",
    "CTA": "QLENX",
    "QLENX": "QLENX",
    "PFIZER": "PFE",
    "輝瑞": "PFE"
}

def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def format_duration(seconds: Optional[int]) -> str:
    """格式化秒數為時分秒字串 (例如 5820 -> 1:37:00)"""
    if not seconds or seconds <= 0:
        return "未知長度"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class PodcastService:
    """Podcast 財經與科技智慧情報雷達核心服務"""

    def __init__(self, db_manager: Optional[DuckDBManager] = None):
        self.db = db_manager or DuckDBManager()
        _ensure_cache_dir()

    def get_seed_channels(self) -> List[Dict[str, Any]]:
        """讀取預置種子頻道庫"""
        if os.path.exists(SEED_CHANNELS_PATH):
            try:
                with open(SEED_CHANNELS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ 讀取種子頻道配置失敗: {e}")
        return []

    def sync_seed_channels(self) -> int:
        """同步種子頻道清單至 DuckDB"""
        channels = self.get_seed_channels()
        if not channels:
            return 0
        saved = self.db.save_podcast_channels_batch(channels)
        logger.info(f"✅ 成功同步 {saved} 個 Podcast 種子頻道至資料庫")
        return saved

    def extract_entities_and_tickers(self, text: str) -> tuple[List[str], List[str]]:
        """從文本中萃取科技實體與對應的上市股票代碼"""
        if not text:
            return [], []

        upper_text = text.upper()
        found_entities = set()
        found_tickers = set()

        # 熱門實體標籤識別
        entity_keywords = [
            "Claude", "OpenAI", "NVIDIA", "Google", "Cursor", "Meta",
            "Gemini", "Apple", "Tesla", "TSMC", "ASML", "Microsoft",
            "Amazon", "DeepMind", "AMD", "Broadcom", "Oracle", "聯發科",
            "台積電", "鴻海", "美團", "騰訊", "阿里巴巴"
        ]

        for ent in entity_keywords:
            if ent.upper() in upper_text or ent in text:
                found_entities.add(ent)
                ticker = ENTITY_TICKER_MAP.get(ent.upper())
                if ticker:
                    found_tickers.add(ticker)

        # 額外比對英文代號獨立單詞
        for token, ticker in ENTITY_TICKER_MAP.items():
            pattern = r'\b' + re.escape(token) + r'\b'
            if re.search(pattern, upper_text):
                found_tickers.add(ticker)

        return sorted(list(found_entities)), sorted(list(found_tickers))

    def get_ticker_quote_badge(self, ticker: str, cached_quotes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """組裝關聯標的資料快照 Badge；缺少可驗證行情時保持空值。"""
        if cached_quotes and ticker in cached_quotes:
            return cached_quotes[ticker]

        return {
            "ticker": ticker,
            "name": ticker,
            "price": None,
            "change_pct": None,
            "direction": "flat",
            "source": "unavailable",
            "as_of": None,
        }

    def enrich_episodes_batch(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批次為多個單集情報附加即時股票行情 Badge，一次性查詢 DuckDB 消除 N+1 連線開銷"""
        if not episodes:
            return []

        # 1. 收集整頁所有單集的 distinct tickers
        all_tickers = set()
        for ep in episodes:
            tickers_val = ep.get("tickers")
            if isinstance(tickers_val, list):
                all_tickers.update(tickers_val)
            elif isinstance(tickers_val, str):
                try:
                    loaded = json.loads(tickers_val)
                    if isinstance(loaded, list):
                        all_tickers.update(loaded)
                except Exception:
                    all_tickers.update([t.strip() for t in tickers_val.split(",") if t.strip()])

        # 2. 一次性批量查詢 DuckDB 最新預測行情
        cached_quotes = self.db.get_latest_quotes_batch(list(all_tickers)) if hasattr(self.db, "get_latest_quotes_batch") else {}

        # 3. 在記憶體中進行組裝與時長格式化
        for ep in episodes:
            # 展開 JSON 欄位
            for fld in ("key_takeaways", "topic_tags", "tickers"):
                val = ep.get(fld)
                if isinstance(val, str):
                    try:
                        ep[fld] = json.loads(val)
                    except Exception:
                        ep[fld] = []

            tickers_list = ep.get("tickers") or []
            badges = [self.get_ticker_quote_badge(t, cached_quotes) for t in tickers_list if t]
            ep["ticker_prices"] = badges

            # 格式化時長
            dur_sec = ep.get("duration_seconds")
            ep["duration_formatted"] = format_duration(dur_sec)

        return episodes

    def enrich_episode_quotes(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """單集行情增強（相容舊介面）"""
        return self.enrich_episodes_batch([episode])[0]

    def seed_initial_curated_episodes(self) -> int:
        """
        初始化高價值精煉單集（對齊 BigGo Finance 推薦核心內容）
        讓系統在冷啟動時立即具備豐富情報與實體連動。
        """
        seed_episodes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "podcast_seed_episodes.json")
        if os.path.exists(seed_episodes_path):
            try:
                with open(seed_episodes_path, "r", encoding="utf-8") as f:
                    episodes_cfg = json.load(f)
                    if episodes_cfg:
                        saved = self.db.save_podcast_episodes_batch(episodes_cfg)
                        logger.info(f"🎉 成功寫入 {saved} 筆種子 Podcast 情報單集 (來自配置檔)")
                        return saved
            except Exception as e:
                logger.error(f"讀取 seed_episodes 失敗: {e}")

        logger.warning("⚠️ 種子 Podcast 配置檔缺失，略過未驗證的預置單集")
        return 0

        now = datetime.datetime.now(datetime.timezone.utc)
        episodes = [
            {
                "id": "allin_e204_sam_altman",
                "channel_id": "all-in",
                "title": "All-In E178: Sam Altman on OpenAI, GPT-5, Scaling Laws, AI Agents & Compute",
                "published_at": (now - datetime.timedelta(days=1, hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://dts.podtrac.com/redirect.mp3/traffic.libsyn.com/secure/allinchamathjason/ALLIN-E178_1.mp3?dest-id=1928300",
                "youtube_url": "https://www.youtube.com/watch?v=nSM0xd8xHUM",
                "duration_seconds": 5840,
                "language": "en",
                "host": "Chamath Palihapitiya, Jason Calacanis, David Sacks, David Friedberg",
                "guest": "Sam Altman (CEO of OpenAI)",
                "tldr_zh_tw": "Sam Altman 與 All-In 四人幫深入探討 OpenAI 次世代模型突破、算力集群指數級擴張對全球能源基礎建設的重壓，以及自主 AI Agents 如何在未來 24 個月內重塑全球軟體價值鏈與晶片資本支出賽局。",
                "key_takeaways": [
                    "模型透過強化學習的思考鏈（CoT）解決了傳統 LLM 數學與邏輯推理瓶頸，算力擴展規律（Scaling Laws）在推論期依然持續生效。",
                    "下一階段 AI 訓練集群最大的制約不再是晶片供應，而是千兆瓦級（GW）的電網承載與核能冷卻設施。",
                    "微軟、亞馬遜與 Google 正在發起史上最大的資料中心 Capex 競賽，直接支撐 NVIDIA 晶片高利潤率。"
                ],
                "topic_tags": ["OpenAI", "Claude", "NVIDIA", "Google", "Cursor", "AI Agent"],
                "tickers": ["MSFT", "NVDA", "GOOGL", "AMZN"],
                "status": "extracted"
            },
            {
                "id": "lex_440_mark_zuckerberg",
                "channel_id": "lex-fridman",
                "title": "#383 – Mark Zuckerberg: Llama, Open Source AI, Smart Glasses & Future of Human Connection",
                "published_at": (now - datetime.timedelta(days=2, hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://media.blubrry.com/takeituneasy/content.blubrry.com/takeituneasy/lex_ai_mark_zuckerberg_3.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=Ff4fRgnuFgQ",
                "duration_seconds": 8120,
                "language": "en",
                "host": "Lex Fridman",
                "guest": "Mark Zuckerberg (Founder & CEO of Meta)",
                "tldr_zh_tw": "祖克柏親述 Meta 斥資百億美元押注開源 Llama 模型生態系的背後戰略，強調 Ray-Ban 智慧眼鏡與全息顯示器的突破，並分析晶片自研架構（MTIA）與 NVIDIA H100/Blackwell 集群之互補路徑。",
                "key_takeaways": [
                    "開源 Llama 模型大幅降低全球開發者部署成本，建立起由 Meta 主導的底層軟體架構標準防線。",
                    "多模態 AI 智慧眼鏡出貨量遠超內部預期，正成為智慧手機之後最具潛力的邊緣運算硬體載體。",
                    "Meta 持續擴大資本支出以建構超過 60 萬顆 H100 等效算力池，對營收廣告轉換率帶來顯著 ROI 提升。"
                ],
                "topic_tags": ["Meta", "NVIDIA", "OpenAI", "Llama", "Gemini", "Apple"],
                "tickers": ["META", "NVDA", "AAPL"],
                "status": "extracted"
            },
            {
                "id": "odd_lots_tsmc_dominance",
                "channel_id": "odd-lots",
                "title": "Why Taiwan Became the World's Most Perilous Geopolitical Chokepoint & TSMC's Moat",
                "published_at": (now - datetime.timedelta(days=3, hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://podtrac.com/pts/redirect.mp3/tracking.swap.fm/track/UVBrz8bN8aM2Xe47PEPu/traffic.omny.fm/d/clips/e73c998e-6e60-432f-8610-ae210140c5b1/8a94442e-5a74-4fa2-8b8d-ae27003a8d6b/f7904791-f77c-421b-a3d3-b43b015542d2/audio.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=lxmt7rz4DLQ",
                "duration_seconds": 3406,
                "language": "en",
                "host": "Joe Weisenthal & Tracy Alloway",
                "guest": "Eyck Freymann & Dan Nystedt",
                "tldr_zh_tw": "深入剖析台積電在先進製程與 CoWoS 先進封裝的絕對壟斷地位。地緣政治博弈下台灣『矽盾』實力，以及 NVIDIA、蘋果、超微、聯發科全面依賴台積電台灣產能的戰略現實。",
                "key_takeaways": [
                    "CoWoS 先進封裝產能吃緊是限制全球 AI 伺服器出貨的物理瓶頸，台積電連續擴充封裝產能以滿足晶片巨頭。",
                    "台積電毛利率穩步邁向 55% 以上，具備對晶片設計巨頭強大之定價轉嫁能力。",
                    "ASML High-NA EUV 光刻機陸續到位，奠定埃米世代技術代差優勢。"
                ],
                "topic_tags": ["TSMC", "台積電", "NVIDIA", "ASML", "AMD", "聯發科"],
                "tickers": ["2330.TW", "TSM", "NVDA", "ASML", "AMD", "2454.TW"],
                "status": "extracted"
            },
            {
                "id": "dwarkesh_dario_amodei",
                "channel_id": "dwarkesh",
                "title": "Dario Amodei — We are near the end of the exponential: Scaling Laws & Renaissance of Science",
                "published_at": (now - datetime.timedelta(days=4, hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://www.dwarkesh.com/p/dario-amodei",
                "youtube_url": "https://www.youtube.com/watch?v=n1E9IZfvGMA",
                "duration_seconds": 6480,
                "language": "en",
                "host": "Dwarkesh Patel",
                "guest": "Dario Amodei (CEO of Anthropic)",
                "tldr_zh_tw": "Anthropic 執行長詳解 Claude 系列模型在架構與邏輯推理上的躍進。他預測強大 AI 將在生物醫藥、材料科學與晶片設計領域引發百年一遇的科學文藝復興，並剖析各大雲端運算集群的指數擴張。",
                "key_takeaways": [
                    "AI 編程能力正在從『程式碼補齊』躍遷為『獨立架構師』，軟體產出邊際成本正朝趨近於零飛馳。",
                    "亞馬遜 AWS 與 Google 雲端算力聯盟為 Anthropic 提供每年數十億美元級別之超級叢集支援。",
                    "國家級前瞻模型安全測試（Constitutional AI）將成為各國監管法規准入的核心標準。"
                ],
                "topic_tags": ["Claude", "Cursor", "Amazon", "Google", "OpenAI", "NVIDIA"],
                "tickers": ["AMZN", "GOOGL", "NVDA", "MSFT"],
                "status": "extracted"
            },
            {
                "id": "xiaolin_semiconductor_war",
                "channel_id": "xiao-lin-shuo",
                "title": "小Lin說：人類會被困在1nm嗎？深度解析光刻機與晶片製程的未來（ASML、台積電與晶片商戰）",
                "published_at": (now - datetime.timedelta(days=5, hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://www.youtube.com/watch?v=Tzp0EF-sysI",
                "youtube_url": "https://www.youtube.com/watch?v=Tzp0EF-sysI",
                "duration_seconds": 2180,
                "language": "zh",
                "host": "小Lin",
                "guest": "無 (單人深度講解)",
                "tldr_zh_tw": "以生動易懂的語言梳理全球半導體權力更迭：從美日晶片大戰、台灣聯電台積電雙雄並起，到荷蘭飛利浦小實驗室如何聯手德國蔡司與台積電張忠謀，成功用浸潤式光刻技術擊敗日本光學雙雄 Nikon 與 Canon。",
                "key_takeaways": [
                    "現代光刻機是人類工業精密機械、超精密光學與極端流體力學的集大成者，單台售價高達 3.5 億美元。",
                    "全球前沿晶片供應鏈環環相扣：台積電製造、ASML 設備、NVIDIA 設計，三者構成現代科技金三角。",
                    "美日韓歐各國各自補貼數百億美元建廠，但全球最核心高效產能依然緊緊綁定在台灣西海岸。"
                ],
                "topic_tags": ["台積電", "TSMC", "ASML", "NVIDIA", "Intel", "鴻海"],
                "tickers": ["2330.TW", "TSM", "ASML", "NVDA", "INTC", "2317.TW"],
                "status": "extracted"
            },
            {
                "id": "sv101_cursor_revolution",
                "channel_id": "sv101",
                "title": "硅谷101 E191：小而美的機會來了，聊聊這輪 AI Agent 進化新範式與 Cursor/Claude 實戰",
                "published_at": (now - datetime.timedelta(days=6, hours=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://aphid.fireside.fm/d/1437767933/f0f20376-8faf-4940-b920-84af6c734e2d/5bc119f9-a795-4016-9431-025a536c6941.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=JEXUta6_oaQ",
                "duration_seconds": 3840,
                "language": "zh",
                "host": "泓君",
                "guest": "矽谷資深全端工程師 & AI Agent 創業者",
                "tldr_zh_tw": "訪談多位在矽谷一線使用 Cursor AI 編輯器與 Claude 3.5 Sonnet 的核心開發者，深度討論 AI Native IDE 如何將個人生產力放大五到十倍，一人獨角獸公司是否即將成真，以及微軟 GitHub Copilot 的應對之道。",
                "key_takeaways": [
                    "Cursor 透過深度整合代碼庫語義索引與 Diff 套用，極大壓縮了工程師閱讀舊代碼與撰寫樣板程式的認知負載。",
                    "軟體工程師的核心競爭力迅速轉移為：問題精準定義能力、架構審查敏銳度與高階邊界條件設計。",
                    "底層基礎設施（NVIDIA H100 GPU、各大雲端推論 Token API）持續呈現指數級調用量爆發。"
                ],
                "topic_tags": ["Cursor", "Claude", "OpenAI", "Google", "NVIDIA", "Microsoft"],
                "tickers": ["NVDA", "MSFT", "GOOGL", "AMZN"],
                "status": "extracted"
            },
            {
                "id": "training_data_david_silver",
                "channel_id": "training-data",
                "title": "Ineffable Intelligence 的 David Silver 與 Lasse Espeholt",
                "published_at": (now - datetime.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://feeds.transistor.fm/training-data/david-silver.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=silver_espeholt",
                "duration_seconds": 4020,
                "language": "en",
                "host": "Sonya Hua",
                "guest": "Lasse Espeholt (Co-founder, Ineffable Intelligence)",
                "tldr_zh_tw": "DeepMind 時期主導 AlphaGo 與 AlphaZero 的研究員 David Silver，與人共同創辦了一家名為 Ineffable Intelligence 的新實驗室，要走一條刻意揚棄當前主流大型語言模型（LLM）典範的超智慧（superintelligence）路線。在這場前瞻對話中，深入剖析強化學習突破與世界模型。",
                "key_takeaways": [
                    "超級學習者的定義：從經驗與博弈環境中持續自我演進，擺脫對人類靜態標註數據的硬性依賴。",
                    "從經驗中進行強化學習：透過環境回饋探索超越人類策略空間的最優解。",
                    "移除人類先驗知識：奠定下一代通用推論與具身智慧模型的全新數學基石。"
                ],
                "topic_tags": ["超級學習者的定義", "從經驗中進行強化學習", "移除人類先驗知識", "Google", "NVIDIA", "DeepMind"],
                "tickers": ["GOOGL", "NVDA"],
                "status": "extracted"
            },
            {
                "id": "fengtouquan_149_meituan_douyin",
                "channel_id": "fengtouquan",
                "title": "149 抖省省突襲美團到店業務：本地生活怎麼卷",
                "published_at": (now - datetime.timedelta(hours=2, minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://crazy.capital/episodes/149.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=fengtouquan149",
                "duration_seconds": 5100,
                "language": "zh",
                "host": "黃海, Rio",
                "guest": "無",
                "tldr_zh_tw": "抖音在 2026 年 2 月 10 日上線獨立 App「抖省省」，用半年時間把日活做到 2,000 萬以上，逼近大眾點評做了二十年才累積的 3,000 萬出頭日活。這意味著美團利潤率最高的到店團購業務，第一次遇到一個專門衝著它最後「付錢那一步」來的對手——而這塊利潤池，正深刻影響本地生活估值。",
                "key_takeaways": [
                    "美團到店團購利潤池高達數百億人民幣，成為各家電商與短影音巨頭必爭之地。",
                    "抖音抖省省以高額補貼切入「付錢立折」，直擊大眾點評核心心智防線。",
                    "本地生活競爭已從內容種草延伸至極致履約與支付分潤，各方毛利率面臨重構考驗。"
                ],
                "topic_tags": ["美團到店業務", "抖音抖省省", "本地生活團購券大戰", "阿里巴巴", "拼多多"],
                "tickers": ["3690.HK", "9988.HK", "PDD"],
                "status": "extracted"
            },
            {
                "id": "citeline_scrip_5_things_20261013",
                "channel_id": "citeline",
                "title": "Scrip 五件必知要聞 - 2026年10月13日",
                "published_at": (now - datetime.timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://feeds.buzzsprout.com/citeline/scrip-oct13.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=citeline_scrip_oct13",
                "duration_seconds": 960,
                "language": "en",
                "host": "Ian Haydock",
                "guest": "無",
                "tldr_zh_tw": "生技製藥業的併購交易在 2026 年第三季降溫，但仍維持在歷史上的強勁水準；與此同時，在 Mesa 舉行的細胞與基因治療會議（Cell & Gene Meeting on the Mesa）上，一場由美國投資人組成的小組點名中國生技產業是競爭威脅——但也附帶說明其帶來的全球合作與授權機會。",
                "key_takeaways": [
                    "中國生技競爭加劇：跨國藥企正加速引進中國創新型 ADC 與雙抗研發管線。",
                    "生技製藥併購趨勢：專注於後期臨床具備重磅炸彈潛力之專利藥。",
                    "肺炎鏈球菌疫苗競賽：次世代結合型疫苗進入臨床三期白熱化階段。"
                ],
                "topic_tags": ["中國生技競爭", "生技製藥併購趨勢", "肺炎鏈球菌疫苗競賽", "輝瑞", "諾華"],
                "tickers": ["PFE", "AZN", "NVS"],
                "status": "extracted"
            },
            {
                "id": "citeline_strategic_intent_marketing",
                "channel_id": "citeline",
                "title": "策略意圖：促銷監管趨嚴時代的名人代言行銷",
                "published_at": (now - datetime.timedelta(hours=4, minutes=26)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://feeds.buzzsprout.com/citeline/strategic-intent.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=citeline_strategic_intent",
                "duration_seconds": 1560,
                "language": "en",
                "host": "Manas Mishra",
                "guest": "Frank Watanabe (CEO of Arcutis)",
                "tldr_zh_tw": "Arcutis Biotherapeutics 執行長 Frank Watanabe 表示，公司第三度找名人合作——這次是體操選手 Simone Biles，於本集錄製前一週對外公布——這是一套刻意設計的疾病認知推廣劇本的一環，瞄準的是一個大多數乾癬、脂漏性皮膚炎與異位性皮膚炎患者未被充分治療的龐大市場。",
                "key_takeaways": [
                    "名人品牌合作：透過頂級體育明星親身經歷提升患者對新型皮膚外用藥之依從性。",
                    "疾病衛教宣導活動：打破傳統處方藥行銷刻板框架，建立直接面向消費者之數位信任渠道。",
                    "FDA 直接對消費者廣告規範：在日趨嚴格的宣傳規範下尋求合規與轉換率最佳平衡點。"
                ],
                "topic_tags": ["名人品牌合作", "疾病衛教宣導活動", "FDA 直接對消費者廣告規範", "莫德納"],
                "tickers": ["MRNA", "PFE"],
                "status": "extracted"
            },
            {
                "id": "top_traders_cta_mainstream_421",
                "channel_id": "top-traders-unplugged",
                "title": "讓 CTA 走向主流 | Systematic Investor | 第421集",
                "published_at": (now - datetime.timedelta(hours=4, minutes=18)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://feeds.megaphone.fm/toptradersunplugged/ep421.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=toptraders421",
                "duration_seconds": 4680,
                "language": "en",
                "host": "Niels Kaastrup-Larsen",
                "guest": "Andrew Beer (Managing Member at Dynamic Beta investments)",
                "tldr_zh_tw": "Andrew Beer，一家管理期貨領域 ETF 發行商的執行長，在 Systematic Investor 系列節目中與主持人 Niels Kaastrup-Larsen 對談，主張趨勢跟蹤型 CTA 是資產配置者手上最低估的分散工具之一，而 60/40 股債配置模型的霸主地位正受到結構性通膨與流動性波動的強烈挑戰。",
                "key_takeaways": [
                    "60/40 投資組合的批判：在股債同跌的高利率周期中，傳統股債平衡策略出現顯著回撤。",
                    "管理期貨 CTA：透過多空動能模型捕捉跨大宗商品、外匯與指數期貨的非線性趨勢。",
                    "危機 alpha：CTA 策略在流動性黑天鵝與劇烈熊市中具備極強的反向凸性保護收益。"
                ],
                "topic_tags": ["60/40 投資組合的批判", "管理期貨 CTA", "危機 alpha", "量化交易", "避險策略"],
                "tickers": ["QLENX"],
                "status": "extracted"
            },
            {
                "id": "biocentury_ep392_calm_before_storm",
                "channel_id": "biocentury",
                "title": "Ep. 392 - 生技市場：暴風雨前的寧靜？",
                "published_at": (now - datetime.timedelta(hours=9, minutes=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://feeds.buzzsprout.com/biocentury/ep392.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=biocentury_ep392",
                "duration_seconds": 1440,
                "language": "en",
                "host": "Steve Usdin, Selena Koch",
                "guest": "無",
                "tldr_zh_tw": "生技類股進入 2026 年第四季時，上半年的動能已經熄火。本集節目的核心發現是：第三季的回檔，主因與其說是該季機構備受矚目的臨床失敗，不如說是 Moderna 的一場軋空，逼得多空策略的專業基金出清手上其他無關、但被廣泛持有的生技持股。",
                "key_takeaways": [
                    "生技第四季市場展望：機構對併購與重磅三期臨床數據的預期轉趨務實。",
                    "併購放緩的衝擊：大藥廠更偏向授權與合作開發，而非動輒溢價 100% 收購。",
                    "Moderna 軋空帶動多空部位平倉連鎖效應，為優質管線生技股創造低接買點。"
                ],
                "topic_tags": ["生技第四季市場展望", "併購放緩的衝擊", "Moderna 軋空", "諾華", "阿斯特捷利康"],
                "tickers": ["MRNA", "AZN", "NVS", "ARGX"],
                "status": "extracted"
            },
            {
                "id": "42zhangjing_personal_agent_ambi",
                "channel_id": "42zhangjing",
                "title": "一個 Personal Agent 創業者的思考 | 對談 Ambi 創始人子皓",
                "published_at": (now - datetime.timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://feed.xyz/42zhangjing/ambi-zihao.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=42zhangjing_ambi",
                "duration_seconds": 5160,
                "language": "zh",
                "host": "曲凱",
                "guest": "莫子皓 (Ambi 創始人)",
                "tldr_zh_tw": "Ambi 創始人莫子皓（子皓）在這集 86 分鐘的對談中，給出了一個與當前輿論相反的判斷：Personal AI 這一波之所以在 2026 年下半年真正具備爆發條件，不是因為模型突然變強，而是因為「產出交付層」找到了與原生手機操作系統融合的入口，不再要求用戶適應聊天框。",
                "key_takeaways": [
                    "Personal Agent 的關鍵非聊天對話，而在於跨應用上下文捕捉與背景自主執行任務的能力。",
                    "原生 OS 融合：蘋果 Apple Intelligence 與 Google Gemini 正在重新定義終端觸控與語音操作界限。",
                    "AI 交付層革命：軟體價值正在從程式碼編寫轉移至用戶個人工作流的精準客製與數據沈澱。"
                ],
                "topic_tags": ["Personal Agent", "原生OS融合", "AI交付層革命", "Apple", "Google", "微軟"],
                "tickers": ["AAPL", "GOOGL", "MSFT", "NVDA"],
                "status": "extracted"
            },
            {
                "id": "bloomberg_diesel_deal_trump_putin",
                "channel_id": "bloomberg-businessweek",
                "title": "川普與普丁的柴油交易讓共和黨如坐針氈",
                "published_at": (now - datetime.timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://feeds.bloomberg.fm/BLM8818814777/diesel-deal.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=bloomberg_diesel_deal",
                "duration_seconds": 3240,
                "language": "en",
                "host": "Tim Stenovec, Carol Massar",
                "guest": "Bloomberg Energy Analysts",
                "tldr_zh_tw": "本集節目的頭條話題——川普總統據傳將達成協議，讓俄羅斯柴油流入全球市場——在 54 分鐘的節目中只占了幾分鐘。對話的主體是一份三部分的簡報：(1) 兩國仍在進行中的地緣政治博弈，(2) 由於原油提煉成柴油的裂解價差，導致這項潛在交易在政治上格外敏感。",
                "key_takeaways": [
                    "俄羅斯柴油重返西方供應鏈傳聞引發大宗商品交易員極大震動，油輪航運指數應聲震盪。",
                    "裂解價差與通膨：精煉產能瓶頸依然是維持美國柴油與航空燃油高價位的主要推手。",
                    "地緣政治石油交易的潛在重塑，將對傳統能源巨頭埃克森美孚與雪佛龍下游煉油獲利帶來重新定價。"
                ],
                "topic_tags": ["俄羅斯柴油", "地緣政治石油交易", "裂解價差與通膨", "能源行情"],
                "tickers": ["XOM", "CVX"],
                "status": "extracted"
            },
            {
                "id": "bloomberg_datacenter_astronomical_cost",
                "channel_id": "bloomberg-businessweek",
                "title": "單座資料中心的天價成本：從土地到晶片",
                "published_at": (now - datetime.timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://feeds.bloomberg.fm/BLM8818814777/datacenter-cost.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=bloomberg_datacenter_cost",
                "duration_seconds": 3600,
                "language": "en",
                "host": "Ian King",
                "guest": "Lynn Doan (Bloomberg News 科技總編輯)",
                "tldr_zh_tw": "一座 AI 資料中心造價高得驚人，這是本集的切入點，但真正的論點在於：超大規模者的擴建已經演變成一場廣泛的原材料與通膨故事，而不只是晶片的故事。Bloomberg News 科技總編輯 Lynn Doan 解釋，微軟、Google、亞馬遜與 Meta 正鎖定電網與冷卻水資源。",
                "key_takeaways": [
                    "資料中心建設成本：單座 GW 級 AI 運算中心總投資動輒突破 500 億美元，建築、變壓器與電網成本佔比迅速攀升。",
                    "AI 電力荒與核能電網：微軟與星座能源重啟三哩島核電廠、亞馬遜收購核電資料中心，引發公用事業股估值重估。",
                    "超大規模雲端資本支出：四大雲端巨頭年化 Capex 超過 2,000 億美元，直接帶動 NVIDIA Blackwell 與博通 ASIC 晶片需求。"
                ],
                "topic_tags": ["資料中心建設成本", "AI 電力荒與核能電網", "超大規模雲端資本支出", "NVIDIA", "微軟", "亞馬遜"],
                "tickers": ["NVDA", "MSFT", "GOOGL", "AMZN", "CEG", "VST"],
                "status": "extracted"
            },
            {
                "id": "acquired_tsmc_epic",
                "channel_id": "acquired",
                "title": "Acquired: Taiwan Semiconductor Manufacturing Company (TSMC) Epic History",
                "published_at": (now - datetime.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://feeds.transistor.fm/acquired/tsmc.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=acquired_tsmc",
                "duration_seconds": 12840,
                "language": "en",
                "host": "Ben Gilbert & David Rosenthal",
                "guest": "無",
                "tldr_zh_tw": "歷時三個多小時的史詩級商業巨作，還原張忠謀如何在 1987 年無中生有創立專業晶圓代工（Pure-play Foundry）商業模式，以極致良率、巨額資本支出與不與客戶競爭的信譽，締造當今全球掌控超過 90% 最先進製程的半導體帝國。",
                "key_takeaways": [
                    "晶圓代工模式的破壞式創新：讓全球 Fabless 晶片設計公司（如 NVIDIA、Qualcomm、Apple）得以百花齊放。",
                    "台積電護城河：製程研發速度（Tick-Tock）、客戶黏著度、CoWoS 先進封裝與台灣西海岸高密度工程師聚落。",
                    "地緣政治與資本支出：台積電每年超過 300 億美元 Capex，拉動 ASML、應用材料與東京威力科創等設備巨頭。"
                ],
                "topic_tags": ["TSMC", "台積電", "ASML", "NVIDIA", "Apple", "AMD"],
                "tickers": ["2330.TW", "TSM", "ASML", "NVDA", "AAPL", "AMD"],
                "status": "extracted"
            },
            {
                "id": "gooaye_ep480_cowos_capex",
                "channel_id": "gooaye",
                "title": "股癌 EP480：半導體高階封裝 CoWoS 與美股大型科技股財報多空解讀",
                "published_at": (now - datetime.timedelta(days=2, hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "audio_url": "https://feeds.soundon.fm/podcasts/ep480.mp3",
                "youtube_url": "https://www.youtube.com/watch?v=gooaye_ep480",
                "duration_seconds": 3120,
                "language": "zh",
                "host": "謝孟恭",
                "guest": "無",
                "tldr_zh_tw": "主委謝孟恭犀利評析台積電法說會透露出的強勁展望：CoWoS 先進封裝產能持續翻倍仍供不應求，大型雲端服務商（CSP）對 AI 基礎設施投資絲毫未見縮手，並深入剖析伺服器 ODM 代工廠（鴻海、廣達）的毛利結構與長線籌碼分佈。",
                "key_takeaways": [
                    "台積電 CoWoS 產能吃緊預計延續至 2026 年底，推升周邊設備廠與封測供應鏈營運動能。",
                    "CSP 廠商的 AI Capex 已經從單純的晶片採購延伸至水冷散熱、高壓供電與高速光通訊傳輸模組。",
                    "散戶在震盪行情中切忌追高殺低，應專注於獲利能見度達兩年以上的產業龍頭標的。"
                ],
                "topic_tags": ["台積電", "CoWoS", "鴻海", "廣達", "聯發科", "NVIDIA"],
                "tickers": ["2330.TW", "NVDA", "2317.TW", "2382.TW", "2454.TW"],
                "status": "extracted"
            }
        ]

        saved = self.db.save_podcast_episodes_batch(episodes)
        logger.info(f"🎉 成功寫入 {saved} 筆種子 Podcast 情報單集")
        return saved

    def fetch_rss_episodes(self, rss_url: str, channel_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        從真實 Podcast RSS XML 串流解析最新單集並萃取實體與股票標籤
        """
        if not rss_url:
            return []

        episodes = []
        try:
            req = urllib.request.Request(
                rss_url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; StockUnderdogRadar/1.0; +https://stock.example.com)"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            channel = root.find("channel")
            if channel is None:
                return []

            items = channel.findall("item")[:limit]
            for item in items:
                title_elem = item.find("title")
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else "未命名單集"

                link_elem = item.find("link")
                link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""

                desc_elem = item.find("description")
                desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""
                clean_desc = re.sub(r'<[^>]+>', ' ', desc).strip()

                pub_elem = item.find("pubDate")
                pub_date_str = ""
                if pub_elem is not None and pub_elem.text:
                    try:
                        import email.utils
                        parsed_tuple = email.utils.parsedate_to_datetime(pub_elem.text)
                        pub_date_str = parsed_tuple.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        pub_date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    pub_date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                audio_url = ""
                enclosure = item.find("enclosure")
                if enclosure is not None:
                    audio_url = enclosure.attrib.get("url", "")

                duration_seconds = None
                for child in item.iter():
                    if child.tag.rsplit("}", 1)[-1] == "duration" and child.text:
                        raw_duration = child.text.strip()
                        try:
                            parts = [int(part) for part in raw_duration.split(":")]
                            if len(parts) == 3:
                                duration_seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
                            elif len(parts) == 2:
                                duration_seconds = parts[0] * 60 + parts[1]
                            else:
                                duration_seconds = int(raw_duration)
                        except ValueError:
                            duration_seconds = None
                        break

                language = (
                    item.findtext("language")
                    or channel.findtext("language")
                    or "unknown"
                ).strip()

                combined_text = f"{title} {clean_desc}"
                entities, tickers = self.extract_entities_and_tickers(combined_text)

                import hashlib
                ep_hash = hashlib.md5(f"{channel_id}_{title}_{pub_date_str}".encode()).hexdigest()[:12]
                ep_id = f"{channel_id}_{ep_hash}"

                episodes.append({
                    "id": ep_id,
                    "channel_id": channel_id,
                    "title": title,
                    "published_at": pub_date_str,
                    "audio_url": audio_url or link,
                    "youtube_url": link if ("youtube.com" in link or "youtu.be" in link) else None,
                    "duration_seconds": duration_seconds,
                    "language": language,
                    "host": None,
                    "guest": None,
                    "tldr_zh_tw": clean_desc[:250] + ("..." if len(clean_desc) > 250 else "") if clean_desc else None,
                    "key_takeaways": [],
                    "topic_tags": entities,
                    "tickers": tickers,
                    "status": "raw"
                })
        except Exception as e:
            logger.debug(f"RSS 解析失敗 ({rss_url}): {e}")
            return []

        return episodes

    def sync_all(self, fetch_live_rss: bool = False) -> int:
        """執行全量頻道與情報單集同步"""
        ch_count = self.sync_seed_channels()
        ep_count = self.seed_initial_curated_episodes()
        rss_ep_count = 0
        if fetch_live_rss:
            channels = self.db.get_podcast_channels(limit=5)
            for ch in channels:
                rss_url = ch.get("rss_url")
                if rss_url:
                    fetched = self.fetch_rss_episodes(rss_url, ch.get("channel_id"), limit=3)
                    if fetched:
                        rss_ep_count += self.db.save_podcast_episodes_batch(fetched)
        return ch_count + ep_count + rss_ep_count


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    service = PodcastService()
    total = service.sync_all()
    print(f"Podcast sync complete: {total} items synchronized.")

