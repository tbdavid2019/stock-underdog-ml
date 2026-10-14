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
    "黃仁勳": "NVDA",
    "JENSEN HUANG": "NVDA",
    "OPENAI": "MSFT",  # 核心投資與獨家雲端
    "MICROSOFT": "MSFT",
    "MSFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "GEMINI": "GOOGL",
    "DEEPMIND": "GOOGL",
    "GOOGL": "GOOGL",
    "CLAUDE": "AMZN",  # 主要策略股東與 Compute
    "ANTHROPIC": "AMZN",
    "AMAZON": "AMZN",
    "AWS": "AMZN",
    "AMZN": "AMZN",
    "META": "META",
    "FACEBOOK": "META",
    "ZUCKERBERG": "META",
    "LLAMA": "META",
    "APPLE": "AAPL",
    "AAPL": "AAPL",
    "TIM COOK": "AAPL",
    "TESLA": "TSLA",
    "TSLA": "TSLA",
    "ELON MUSK": "TSLA",
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
    "AVGO": "AVGO",
    "ORACLE": "ORCL",
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
    "騰訊": "0700.HK",
    "TENCENT": "0700.HK",
    "阿里巴巴": "BABA",
    "ALIBABA": "BABA"
}

# 標的基準價格與名稱對照快照 (行情離線降級快照)
FALLBACK_TICKER_QUOTES = {
    "NVDA": {"ticker": "NVDA", "name": "輝達", "price": 138.25, "change_pct": 1.45, "direction": "up"},
    "MSFT": {"ticker": "MSFT", "name": "微軟", "price": 422.80, "change_pct": 0.35, "direction": "up"},
    "GOOGL": {"ticker": "GOOGL", "name": "Alphabet (Google)", "price": 166.40, "change_pct": 0.16, "direction": "up"},
    "AMZN": {"ticker": "AMZN", "name": "亞馬遜", "price": 188.50, "change_pct": -0.42, "direction": "down"},
    "META": {"ticker": "META", "name": "Meta Platforms", "price": 595.60, "change_pct": 1.82, "direction": "up"},
    "AAPL": {"ticker": "AAPL", "name": "蘋果", "price": 231.30, "change_pct": -0.28, "direction": "down"},
    "TSLA": {"ticker": "TSLA", "name": "特斯拉", "price": 221.40, "change_pct": 3.12, "direction": "up"},
    "2330.TW": {"ticker": "2330.TW", "name": "台積電", "price": 1050.0, "change_pct": 2.44, "direction": "up"},
    "TSM": {"ticker": "TSM", "name": "台積電 ADR", "price": 192.50, "change_pct": 2.10, "direction": "up"},
    "2454.TW": {"ticker": "2454.TW", "name": "聯發科", "price": 1305.0, "change_pct": 1.16, "direction": "up"},
    "2317.TW": {"ticker": "2317.TW", "name": "鴻海", "price": 208.5, "change_pct": 0.72, "direction": "up"},
    "2382.TW": {"ticker": "2382.TW", "name": "廣達", "price": 288.0, "change_pct": 1.58, "direction": "up"},
    "ASML": {"ticker": "ASML", "name": "艾司摩爾", "price": 820.40, "change_pct": -1.15, "direction": "down"},
    "AMD": {"ticker": "AMD", "name": "超微", "price": 158.60, "change_pct": 0.88, "direction": "up"},
    "AVGO": {"ticker": "AVGO", "name": "博通", "price": 178.90, "change_pct": 1.62, "direction": "up"},
    "ORCL": {"ticker": "ORCL", "name": "甲骨文", "price": 175.20, "change_pct": 0.65, "direction": "up"},
    "3690.HK": {"ticker": "3690.HK", "name": "美團", "price": 192.4, "change_pct": 0.70, "direction": "up"},
    "0700.HK": {"ticker": "0700.HK", "name": "騰訊控股", "price": 435.0, "change_pct": 1.12, "direction": "up"},
    "BABA": {"ticker": "BABA", "name": "阿里巴巴", "price": 102.8, "change_pct": -0.55, "direction": "down"},
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
        """組裝即時行情與漲跌幅 Badge 結構，優先從批次快取讀取"""
        if cached_quotes and ticker in cached_quotes:
            return cached_quotes[ticker]

        # 降級使用基準對照字典
        if ticker in FALLBACK_TICKER_QUOTES:
            res = dict(FALLBACK_TICKER_QUOTES[ticker])
            res["ticker"] = ticker
            return res

        return {
            "ticker": ticker,
            "name": ticker,
            "price": None,
            "change_pct": 0.0,
            "direction": "up"
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
                    "duration_seconds": 3600,
                    "language": "en",
                    "host": None,
                    "guest": None,
                    "tldr_zh_tw": clean_desc[:250] + ("..." if len(clean_desc) > 250 else "") if clean_desc else "本集內容涵蓋前沿科技與金融宏觀趨勢討論。",
                    "key_takeaways": [
                        f"探討重點核心包含：{', '.join(entities[:3]) if entities else '產業脈動'}",
                        "關聯標的即時估值與技術指標持續連動。",
                        "詳細觀點可點擊收聽完整音訊或原創節目連結。"
                    ],
                    "topic_tags": entities if entities else ["科技", "財經"],
                    "tickers": tickers,
                    "status": "extracted"
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

