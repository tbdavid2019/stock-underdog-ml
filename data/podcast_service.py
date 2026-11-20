"""
data/podcast_service.py - AI Financial & Tech Podcast Intelligence Radar Pipeline
負責頻道清單管理、RSS 串流解析、實體與股票 Ticker 映射、行情增強與 AI 繁中 TL;DR 萃取。
"""

import os
import re
import json
import logging
import datetime
import time
from typing import Dict, List, Optional, Any
import urllib.request
import xml.etree.ElementTree as ET

from core.config import config
from data.duckdb_manager import DuckDBManager
from data.decision_service import DecisionService

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
        self.decision_service = DecisionService()
        self.last_live_sync_at: float = 0.0
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
        """同步種子頻道清單至 DuckDB，並清理過期或重複的舊頻道 ID"""
        channels = self.get_seed_channels()
        if not channels:
            return 0
        saved = self.db.save_podcast_channels_batch(channels)

        # 清理不在種子清單中的過期/重複頻道 ID，並將關聯的舊 episode channel_id 遷移至標準 ID
        legacy_remap = {
            "all-in": "all-in-with-chamath-jason-sacks-friedberg",
            "lex-fridman": "lex-fridman-podcast",
            "dwarkesh": "dwarkesh-podcast",
            "diary-of-a-ceo": "the-diary-of-a-ceo",
            "y-combinator": "y-combinator-startup-podcast",
            "lennys-podcast": "lenny-s-podcast",
            "xiao-lin-shuo": "lin",
            "sv101": "101",
            "kurzgesagt": "in-a-nutshell",
            "joe-rogan": "the-joe-rogan-experience",
            "cls-tongxue": "cls",
            "theo-t3gg": "theo-t3-gg",
            "pragmatic-engineer": "the-pragmatic-engineer",
            "knowledge-project": "the-knowledge-project",
            "fengtouquan": "biggo-channel-0da8ef6bed3a",
            "citeline": "citeline-podcasts",
            "biocentury": "biocentury-this-week",
            "42zhangjing": "42"
        }
        try:
            with self.db._write_lock(), self.db._get_connection(read_only=False) as con:
                for old_id, new_id in legacy_remap.items():
                    con.execute("UPDATE podcast_episodes SET channel_id = ? WHERE channel_id = ?", [new_id, old_id])
                
                valid_ids = [c["channel_id"] for c in channels]
                placeholders = ",".join(["?"] * len(valid_ids))
                con.execute(f"DELETE FROM podcast_channels WHERE channel_id NOT IN ({placeholders})", valid_ids)
        except Exception as e:
            logger.warning(f"⚠️ 清理舊 Podcast 頻道失敗: {e}")

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

        # 3. 在記憶體中進行組裝、時長格式化與 AI 決策重要性附加
        pending_db_updates = []
        for ep in episodes:
            # 展開 JSON 欄位
            for fld in ("key_takeaways", "topic_tags", "tickers"):
                val = ep.get(fld)
                if isinstance(val, str):
                    try:
                        ep[fld] = json.loads(val)
                    except Exception:
                        ep[fld] = []

            # 4. 決策模型重要性與標的多空推論
            imp_score = ep.get("importance_score")
            ts_val = ep.get("ticker_sentiments")
            if isinstance(ts_val, str):
                try:
                    ticker_sentiments = json.loads(ts_val)
                except Exception:
                    ticker_sentiments = {}
            elif isinstance(ts_val, dict):
                ticker_sentiments = ts_val
            else:
                ticker_sentiments = {}

            if imp_score is None:
                # 尚未評估過：為了保證 GET 查詢在數毫秒內極速響應，直接使用確定性本地啟發式決策規則推論 (<0.2ms)
                # 並寫入 pending_db_updates 批次存入 DuckDB，永不再重複評估
                verdict = self.decision_service.evaluate_podcast_episode(ep, local_only=True)
                ep["importance_score"] = verdict.importance_score
                ep["importance_level"] = verdict.importance_level
                ep["importance_badge"] = verdict.importance_badge
                ep["market_catalyst"] = verdict.market_catalyst
                ep["decision_slot"] = verdict.source_slot
                ticker_sentiments = verdict.ticker_sentiments
                ep["ticker_sentiments"] = ticker_sentiments

                pending_db_updates.append({
                    "id": ep.get("id"),
                    "importance_score": verdict.importance_score,
                    "importance_level": verdict.importance_level,
                    "ticker_sentiments": ticker_sentiments,
                    "decision_slot": verdict.source_slot
                })
            else:
                score_num = float(imp_score)
                ep["importance_score"] = score_num
                if score_num >= 4.2:
                    ep["importance_level"] = "heavyweight"
                    ep["importance_badge"] = "⭐ 5星重磅"
                elif score_num >= 3.4:
                    ep["importance_level"] = "important"
                    ep["importance_badge"] = "⚡ 4星重要"
                else:
                    ep["importance_level"] = "watch"
                    ep["importance_badge"] = "📰 3星關注"
                ep["ticker_sentiments"] = ticker_sentiments

            # 5. 股票行情 Badge 結合決策多空立場
            tickers_list = ep.get("tickers") or []
            badges = []
            for t in tickers_list:
                if not t:
                    continue
                b = self.get_ticker_quote_badge(t, cached_quotes)
                sent = ticker_sentiments.get(str(t).upper(), {}) or ticker_sentiments.get(str(t), {})
                b["stance"] = sent.get("stance", "neutral")
                b["stance_label"] = sent.get("label", "↔ 中立")
                b["stance_confidence"] = sent.get("confidence", 0.5)
                badges.append(b)
            ep["ticker_prices"] = badges

            # 格式化時長
            dur_sec = ep.get("duration_seconds")
            ep["duration_formatted"] = format_duration(dur_sec)

        # 批次持久化決策結果至 DuckDB，避免下次開啟網頁或重新查詢時重複推論
        if pending_db_updates and hasattr(self.db, "update_podcast_episodes_decisions_batch"):
            try:
                self.db.update_podcast_episodes_decisions_batch(pending_db_updates)
                logger.info(f"💾 成功將 {len(pending_db_updates)} 筆單集決策評級持久化存入 DuckDB")
            except Exception as pe:
                logger.warning(f"⚠️ 單集決策持久化存入 DuckDB 失敗: {pe}")

        return episodes

    def evaluate_episode_intelligence(self, episode_id: str, force: bool = False) -> Dict[str, Any]:
        """
        針對特定單集強制或即時執行 5-Tier 決策評估（Clef/JEV/OpenAI/LocalRules），
        並將結果持久化至 DuckDB。
        """
        target = None
        try:
            with self.db._get_connection(read_only=True) as con:
                df = con.execute("SELECT * FROM podcast_episodes WHERE id = ?", [str(episode_id)]).df()
                if not df.empty:
                    target = df.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"❌ 查詢單集失敗 ({episode_id}): {e}")
            return {"success": False, "error": str(e)}

        if not target:
            return {"success": False, "error": f"Episode {episode_id} not found"}

        # 展開欄位
        for fld in ("key_takeaways", "topic_tags", "tickers"):
            val = target.get(fld)
            if isinstance(val, str):
                try:
                    target[fld] = json.loads(val)
                except Exception:
                    target[fld] = []

        verdict = self.decision_service.evaluate_podcast_episode(target)
        self.db.update_podcast_episode_decision(
            episode_id=episode_id,
            importance_score=verdict.importance_score,
            importance_level=verdict.importance_level,
            ticker_sentiments=verdict.ticker_sentiments,
            decision_slot=verdict.source_slot
        )

        return {
            "success": True,
            "episode_id": episode_id,
            "verdict": verdict.to_dict()
        }

    def evaluate_episodes_batch(self, limit: int = 20) -> Dict[str, Any]:
        """批量評估尚未進行決策分析的單集"""
        evaluated_count = 0
        try:
            with self.db._get_connection(read_only=True) as con:
                df = con.execute("""
                    SELECT * FROM podcast_episodes
                    WHERE importance_score IS NULL
                    ORDER BY published_at DESC
                    LIMIT ?;
                """, [limit]).df()
                records = df.to_dict(orient="records") if not df.empty else []
        except Exception as e:
            logger.error(f"❌ 查詢未評估單集失敗: {e}")
            return {"success": False, "error": str(e)}

        for r in records:
            ep_id = str(r["id"])
            res = self.evaluate_episode_intelligence(ep_id, force=True)
            if res.get("success"):
                evaluated_count += 1

        return {
            "success": True,
            "evaluated_count": evaluated_count,
            "total_requested": len(records)
        }

    def enrich_episode_quotes(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """單集行情增強（相容舊介面）"""
        return self.enrich_episodes_batch([episode])[0]

    def seed_initial_curated_episodes(self) -> int:
        """
        初始化高價值精煉單集（對齊全球頂級財經與科技音訊核心內容）
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

    def fetch_biggo_live_episodes(self, max_pages: int = 3, page_size: int = 50) -> int:
        """
        從 BigGo 財經 Podcast API 即時抓取最新單集、自動萃取觀點/標的/情緒，並同步存入 DuckDB。
        """
        seed_channels = self.get_seed_channels()
        source_map = {c.get("source_id"): c.get("channel_id") for c in seed_channels if c.get("source_id")}

        headers = {
            "Site": "finance.biggo.com.tw",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }

        total_saved = 0
        new_channels_map = {}

        for page in range(1, max_pages + 1):
            url = f"https://api.biggo.com/api/v1/finance/podcast/list?page={page}&size={page_size}"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                logger.warning(f"⚠️ BigGo Podcast API 抓取失敗 (頁碼 {page}): {e}")
                break

            episodes_data = resp_data.get("data", {}).get("episodes", [])
            if not episodes_data:
                logger.info(f"ℹ️ BigGo API 第 {page} 頁無更多單集")
                break

            batch_records = []
            for ep in episodes_data:
                ep_id = str(ep.get("id") or "").strip()
                if not ep_id:
                    continue

                src_id = ep.get("source_id")
                ch_id = source_map.get(src_id)
                if not ch_id:
                    ch_id = f"biggo-{src_id}" if src_id else "unknown-channel"
                    if src_id and src_id not in new_channels_map:
                        new_channels_map[src_id] = {
                            "channel_id": ch_id,
                            "title": ep.get("source_name") or f"Podcast {src_id}",
                            "author": ep.get("source_name") or "Unknown",
                            "description": ep.get("source_description") or "",
                            "image_url": ep.get("source_image_url") or "",
                            "rss_url": f"https://api.biggo.com/api/v1/finance/podcast?source={src_id}",
                            "youtube_channel_url": None,
                            "follower_count": int(ep.get("source_follower_count") or 0),
                            "episode_count": int(ep.get("source_episode_count") or 0),
                            "category": "Finance & Technology",
                            "language": ep.get("language") or "zh",
                            "is_subscribed": True
                        }

                # 處理發布時間 (epoch timestamp -> UTC ISO 8601)
                raw_pub = ep.get("published_at")
                if isinstance(raw_pub, (int, float)) and raw_pub > 0:
                    pub_dt_str = datetime.datetime.fromtimestamp(raw_pub, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                elif isinstance(raw_pub, str) and raw_pub:
                    pub_dt_str = raw_pub
                else:
                    pub_dt_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                # 主持人與來賓萃取
                speakers = ep.get("speakers") or []
                hosts = [s.get("name") for s in speakers if isinstance(s, dict) and s.get("role") == "Host" and s.get("name")]
                guests = [s.get("name") for s in speakers if isinstance(s, dict) and s.get("role") == "Guest" and s.get("name")]

                # 關鍵洞察萃取
                highlights = ep.get("business_highlights") or []
                takeaways = [h.get("insight") for h in highlights if isinstance(h, dict) and h.get("insight")]

                # 主題與實體標籤
                topics = ep.get("topics") or []
                entities = ep.get("key_entities") or []
                topic_tags = list(dict.fromkeys([t for t in (topics + entities) if t and isinstance(t, str)]))

                # 上市標的代碼與多空立場
                companies = ep.get("companies") or []
                tickers = [c.get("symbol") for c in companies if isinstance(c, dict) and c.get("symbol")]

                # 文本輔助萃取台美股額外標的
                summary_text = ep.get("summary") or ep.get("quick_take") or ""
                extra_entities, extra_tickers = self.extract_entities_and_tickers(f"{ep.get('title', '')} {summary_text}")
                tickers = list(dict.fromkeys(tickers + extra_tickers))
                topic_tags = list(dict.fromkeys(topic_tags + extra_entities))

                # 多空情緒字典
                ticker_sentiments = {}
                for c in companies:
                    if not isinstance(c, dict):
                        continue
                    sym = c.get("symbol")
                    if not sym:
                        continue
                    raw_sent = (c.get("sentiment") or "neutral").lower()
                    stance = "bullish" if raw_sent == "bullish" else ("bearish" if raw_sent == "bearish" else "neutral")
                    label = "↑ 看多" if stance == "bullish" else ("↓ 看空" if stance == "bearish" else "↔ 中立")
                    m_count = c.get("mentions", 1)
                    ticker_sentiments[str(sym).upper()] = {
                        "stance": stance,
                        "label": label,
                        "confidence": 0.85 if stance != "neutral" else 0.5,
                        "reasoning": f"提及次數: {m_count}, 情感: {raw_sent}"
                    }

                dur = ep.get("duration")
                duration_seconds = int(round(dur)) if isinstance(dur, (int, float)) and dur > 0 else None

                rec = {
                    "id": ep_id,
                    "channel_id": ch_id,
                    "title": ep.get("title") or "未命名單集",
                    "published_at": pub_dt_str,
                    "audio_url": ep.get("url") or "",
                    "youtube_url": None,
                    "duration_seconds": duration_seconds,
                    "language": ep.get("language") or "zh",
                    "host": ", ".join(hosts) if hosts else None,
                    "guest": ", ".join(guests) if guests else None,
                    "tldr_zh_tw": summary_text,
                    "key_takeaways": takeaways,
                    "topic_tags": topic_tags,
                    "tickers": tickers,
                    "ticker_sentiments": ticker_sentiments,
                    "status": "extracted"
                }

                # 即時推論 5-Tier 決策評級 (本地極速規則 <0.02ms)
                verdict = self.decision_service.evaluate_podcast_episode(rec, local_only=True)
                rec["importance_score"] = verdict.importance_score
                rec["importance_level"] = verdict.importance_level
                rec["decision_slot"] = verdict.source_slot
                if verdict.ticker_sentiments:
                    for k, v in verdict.ticker_sentiments.items():
                        if k not in rec["ticker_sentiments"]:
                            rec["ticker_sentiments"][k] = v

                batch_records.append(rec)

            if batch_records:
                saved = self.db.save_podcast_episodes_batch(batch_records)
                total_saved += saved
                logger.info(f"📥 成功同步 BigGo 第 {page} 頁: {saved} 筆單集存入 DuckDB")

        # 若發現新頻道，存入頻道庫
        if new_channels_map:
            ch_saved = self.db.save_podcast_channels_batch(list(new_channels_map.values()))
            logger.info(f"📡 動態收錄 {ch_saved} 個新 Podcast 頻道至資料庫")

        self.last_live_sync_at = time.time()
        logger.info(f"✅ BigGo 即時單集同步完畢: 共寫入/更新 {total_saved} 筆單集")
        return total_saved

    def backfill_unevaluated_episodes(self, limit: int = 500) -> int:
        """為資料庫中尚未評估決策等級的單集進行確定性本地規則回填持久化"""
        try:
            with self.db._get_connection(read_only=True) as con:
                df = con.execute("""
                    SELECT * FROM podcast_episodes
                    WHERE importance_score IS NULL
                    LIMIT ?;
                """, [limit]).df()
                if df.empty:
                    return 0
                records = df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"❌ 查詢待回填單集失敗: {e}")
            return 0

        updates = []
        for r in records:
            # 展開欄位
            for fld in ("key_takeaways", "topic_tags", "tickers"):
                val = r.get(fld)
                if isinstance(val, str):
                    try:
                        r[fld] = json.loads(val)
                    except Exception:
                        r[fld] = []
            verdict = self.decision_service.evaluate_podcast_episode(r, local_only=True)
            updates.append({
                "id": r.get("id"),
                "importance_score": verdict.importance_score,
                "importance_level": verdict.importance_level,
                "ticker_sentiments": verdict.ticker_sentiments,
                "decision_slot": verdict.source_slot
            })

        if updates and hasattr(self.db, "update_podcast_episodes_decisions_batch"):
            saved = self.db.update_podcast_episodes_decisions_batch(updates)
            logger.info(f"✅ 成功為 {saved} 筆歷史單集回填決策評級至 DuckDB")
            return saved
        return 0

    def sync_all(self, fetch_live_api: bool = True, max_pages: int = 3, fetch_live_rss: bool = False) -> int:
        """執行全量頻道、即時 BigGo API 與情報單集同步"""
        ch_count = self.sync_seed_channels()
        ep_count = self.seed_initial_curated_episodes()
        live_count = 0
        if fetch_live_api:
            live_count = self.fetch_biggo_live_episodes(max_pages=max_pages)
        elif fetch_live_rss:
            channels = self.db.get_podcast_channels(limit=5)
            for ch in channels:
                rss_url = ch.get("rss_url")
                if rss_url:
                    fetched = self.fetch_rss_episodes(rss_url, ch.get("channel_id"), limit=3)
                    if fetched:
                        live_count += self.db.save_podcast_episodes_batch(fetched)
        backfilled = self.backfill_unevaluated_episodes()
        total = ch_count + ep_count + live_count + backfilled
        logger.info(f"🎉 Podcast 全量同步完成: 頻道 {ch_count} 筆, 種子 {ep_count} 筆, Live API {live_count} 筆, 回填 {backfilled} 筆 (總計 {total})")
        return total


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    fetch_live = True
    pages = 3
    for arg in sys.argv[1:]:
        if arg == "--no-live":
            fetch_live = False
        elif arg.startswith("--pages="):
            try:
                pages = int(arg.split("=")[1])
            except ValueError:
                pass
    service = PodcastService()
    total = service.sync_all(fetch_live_api=fetch_live, max_pages=pages)
    print(f"Podcast sync complete: {total} items synchronized.")


