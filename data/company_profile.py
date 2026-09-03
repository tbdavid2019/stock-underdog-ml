"""
data/company_profile.py - 公司簡介與即時新聞提取服務 (支援 2md 原生批次、多級 Fallback 與並發節流)

整合 2md URL to Markdown API (https://2md.aiurl.tw / https://2md.glsoft.ai / https://create360.ai)
提供股票代碼 (如 9945.TW, 2330.TW, AAPL) 之繁體中文公司簡介、核心營運業務、董事長/股本與最新新聞摘要。
支援 2md 原生批次 (POST /v1/batch)、全域信號量節流、超時延長與 DuckDB 股票名稱解析。
"""

import re
import time
import logging
import threading
import urllib.parse
from typing import Dict, List, Optional, Any
import requests

from data.ticker_resolver import normalize_ticker, COMPANY_ALIASES
try:
    from data.duckdb_manager import DuckDBManager
except ImportError:
    DuckDBManager = None

logger = logging.getLogger("stock_app.company_profile")


class CompanyProfileService:
    """公司簡介與即時新聞服務 (支援 2md 原生批次、多級 Fallback 與並發節流)"""

    FALLBACK_2MD_HOSTS = [
        "https://2md.aiurl.tw",
        "https://2md.glsoft.ai",
        "https://create360.ai"
    ]

    # 並發信號量：嚴格限制最多同時並發 2 個 2MD 爬蟲連線，杜絕瞬間暴衝與 OOM
    _CRAWLER_SEMAPHORE = threading.Semaphore(2)

    # 超時時間配置 (秒)
    TIMEOUT_PROFILE = 10.0   # Yahoo Profile 需開 Chromium 動態渲染 (通常 4~7 秒)
    TIMEOUT_SEARCH = 5.0     # 2MD 即時新聞 SERP 搜尋
    TIMEOUT_BATCH = 25.0     # 2MD 原生微批次 (2~3 支) 充足安全超時 (避免過早判定失敗)
    TASK_WAIT_TIMEOUT = 12.0 # 外層執行緒等待上限

    # In-memory cache: ticker -> (timestamp, data_dict)
    _CACHE: Dict[str, tuple] = {}
    CACHE_TTL = 3600 * 6  # 6 小時快取

    @classmethod
    def get_company_profile(cls, ticker_raw: str, db: Optional[Any] = None) -> Dict[str, Any]:
        """
        取得公司名稱、簡介、主要業務與最新新聞 (單一標的)
        """
        ticker = normalize_ticker(ticker_raw).upper()
        now = time.time()

        # 1. 檢查快取
        if ticker in cls._CACHE:
            ts, cached_data = cls._CACHE[ticker]
            if now - ts < cls.CACHE_TTL:
                return cached_data

        # 2. 解析公司基本名稱
        stock_name = cls._resolve_stock_name(ticker, db)
        is_tw = ticker.endswith(".TW") or ticker.endswith(".TWO")
        raw_code = ticker.split(".")[0] if is_tw else ticker

        profile_data = {
            "ticker": ticker,
            "raw_code": raw_code,
            "name": stock_name or raw_code,
            "industry": "綜合產業",
            "business_summary": "",
            "chairman": "",
            "market_cap": "",
            "established": "",
            "recent_news": [],
            "links": cls._build_stock_links(ticker, raw_code, is_tw),
            "source": "2md"
        }

        # 3. 並發非同步獲取基本資料與新聞（美股與台股分流）
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            if not is_tw:
                f_prof = executor.submit(cls._fallback_yfinance, profile_data, ticker)
                f_news = executor.submit(cls._fetch_news_from_2md, profile_data, False)
            else:
                f_prof = executor.submit(cls._fetch_profile_from_2md, profile_data, True)
                f_news = executor.submit(cls._fetch_news_from_2md, profile_data, True)
            
            # 等待時間放寬至 12.0 秒，容納 Chromium 渲染完成
            try:
                f_prof.result(timeout=cls.TASK_WAIT_TIMEOUT)
            except Exception as e:
                logger.debug(f"Profile fetch task ended: {e}")
            try:
                f_news.result(timeout=cls.TASK_WAIT_TIMEOUT)
            except Exception as e:
                logger.debug(f"News fetch task ended: {e}")

        # 4. 若未取得足夠簡介，使用 yfinance 終極快速備援
        if not profile_data.get("business_summary"):
            cls._fallback_yfinance(profile_data, ticker)

        # 5. 寫入快取
        cls._CACHE[ticker] = (now, profile_data)
        return profile_data

    @classmethod
    def _resolve_stock_name(cls, ticker: str, db: Optional[Any] = None) -> Optional[str]:
        """從 DuckDB 或別名表解析繁體中文股票名稱"""
        # 1. 別名表反查
        for alias, t in COMPANY_ALIASES.items():
            if t.upper() == ticker.upper():
                return alias

        # 2. DuckDB 查詢 (安全容錯)
        if DuckDBManager is None and db is None:
            return None
        manager = db or (DuckDBManager() if DuckDBManager else None)
        if not manager:
            return None
        try:
            with manager._get_connection() as con:
                row = con.execute("SELECT name FROM tw_daily_bars WHERE ticker = ? AND name IS NOT NULL AND name != '' LIMIT 1", [ticker]).fetchone()
                if row and row[0]:
                    return str(row[0]).strip()
                row_inst = con.execute("SELECT name FROM tw_institutional_daily WHERE ticker = ? AND name IS NOT NULL AND name != '' LIMIT 1", [ticker]).fetchone()
                if row_inst and row_inst[0]:
                    return str(row_inst[0]).strip()
        except Exception as e:
            logger.debug(f"DuckDB stock name lookup failed: {e}")

        return None

    @classmethod
    def _fetch_profile_from_2md(cls, profile: Dict[str, Any], is_tw: bool):
        """利用 2md URL Reader 抓取 Yahoo 股市 Profile 頁面 (受信號量保護與放寬超時)"""
        ticker = profile["ticker"]
        if is_tw:
            target_url = f"https://tw.stock.yahoo.com/quote/{ticker}/profile"
        else:
            target_url = f"https://finance.yahoo.com/quote/{ticker}/profile"

        text = ""
        with cls._CRAWLER_SEMAPHORE:
            for host in cls.FALLBACK_2MD_HOSTS:
                try:
                    reader_url = f"{host.rstrip('/')}/{target_url}"
                    resp = requests.get(reader_url, timeout=cls.TIMEOUT_PROFILE)
                    if resp.status_code == 200:
                        resp.encoding = "utf-8"
                        if len(resp.text) > 20:
                            text = resp.text
                            break
                except requests.exceptions.Timeout:
                    logger.warning(
                        f"2md profile timeout ({cls.TIMEOUT_PROFILE}s) on {host} for {ticker}，切換至下一台備援站..."
                    )
                    continue
                except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as ce:
                    logger.warning(f"2md profile host {host} 連線異常 ({ce})，切換至下一台備援站...")
                    continue
                except Exception as e:
                    logger.debug(f"2md profile fetch error on {host} for {ticker}: {e}")
                    continue

        if not text:
            return

        cls._parse_profile_markdown(profile, text, is_tw=is_tw)

    @classmethod
    def _parse_profile_markdown(cls, profile: Dict[str, Any], text: str, is_tw: bool):
        """解析 Yahoo 股市 Profile Markdown 內容並填入 profile 字典"""
        if not text:
            return

        if is_tw:
            # 公司名稱
            name_m = re.search(r"公司名稱\s*\n\s*([^\n]+)", text)
            if name_m and name_m.group(1).strip() and (not profile.get("name") or profile["name"] == profile.get("raw_code")):
                profile["name"] = name_m.group(1).strip()

            # 產業類別
            ind_m = re.search(r"產業類別\s*\n\s*([^\n]+)", text)
            if ind_m and ind_m.group(1).strip():
                cat = ind_m.group(1).strip()
                if cat != "其他":
                    profile["industry"] = cat

            # 董事長
            chair_m = re.search(r"董事長\s*\n\s*([^\n]+)", text)
            if chair_m:
                profile["chairman"] = chair_m.group(1).strip()

            # 市值
            cap_m = re.search(r"市值\s*\(百萬\)\s*\n\s*([\d,\.]+)", text)
            if cap_m:
                try:
                    val = float(cap_m.group(1).replace(",", ""))
                    profile["market_cap"] = f"{val/100:.1f} 億元" if val > 100 else f"{val:.0f} 百萬元"
                except Exception:
                    pass

            # 成立時間
            est_m = re.search(r"成立時間\s*\n\s*([^\n]+)", text)
            if est_m:
                profile["established"] = est_m.group(1).strip()

            # 主要經營業務
            biz_m = re.search(r"主要經營業務\s*\n\s*([^\n#]+)", text)
            if biz_m:
                biz = biz_m.group(1).strip()
                biz = re.sub(r"\s+", " ", biz)
                profile["business_summary"] = biz[:300]

    @classmethod
    def _fetch_news_from_2md(cls, profile: Dict[str, Any], is_tw: bool):
        """利用 2md Live Search 抓取即時新聞與業務概述 (受信號量保護與放寬超時)"""
        ticker = profile["ticker"]
        name = profile.get("name") or ticker
        if is_tw:
            query = f"{ticker} {name} 公司簡介 營運 最新新聞"
        else:
            query = f"{ticker} {name} stock latest news"

        text = ""
        with cls._CRAWLER_SEMAPHORE:
            for host in cls.FALLBACK_2MD_HOSTS:
                try:
                    search_url = f"{host.rstrip('/')}/s/{urllib.parse.quote(query)}"
                    resp = requests.get(search_url, timeout=cls.TIMEOUT_SEARCH)
                    if resp.status_code == 200:
                        resp.encoding = "utf-8"
                        if len(resp.text) > 20:
                            text = resp.text
                            break
                except requests.exceptions.Timeout:
                    logger.warning(f"2md news timeout ({cls.TIMEOUT_SEARCH}s) on {host} for {ticker}")
                    continue
                except Exception as e:
                    logger.debug(f"2md news search error on {host} for {ticker}: {e}")
                    continue

        if not text:
            return

        cls._parse_news_markdown(profile, text)

    @classmethod
    def _parse_news_markdown(cls, profile: Dict[str, Any], text: str):
        """解析 Markdown 搜尋結果"""
        if not text:
            return

        pattern = re.compile(r"\[(\d+)\] Title:\s*(.+?)\n\[\1\] URL Source:\s*(.+?)\n\[\1\] Description:\s*(.+?)(?=\n\[\d+\] Title|\Z)", re.DOTALL)
        matches = pattern.findall(text)
        news_items = []
        for m in matches:
            title = m[1].strip()
            url = m[2].strip()
            desc = m[3].strip()
            if not title or len(title) < 4:
                continue
            if not profile.get("business_summary") and ("業務" in desc or "營運" in desc or "從事" in desc or "提供" in desc):
                profile["business_summary"] = desc[:200]

            news_items.append({
                "title": title,
                "url": url,
                "snippet": desc[:120] + ("..." if len(desc) > 120 else "")
            })
            if len(news_items) >= 4:
                break

        profile["recent_news"] = news_items

    @classmethod
    def _fetch_batch_from_2md(cls, urls: List[str]) -> List[Dict[str, Any]]:
        """
        呼叫 2md 原生批次 API (POST /v1/batch)，依序使用 FALLBACK_2MD_HOSTS 容災輪詢。
        在單一 HTTP 請求內並發取得多個網址之 Markdown 內容。
        """
        if not urls:
            return []

        with cls._CRAWLER_SEMAPHORE:
            for host in cls.FALLBACK_2MD_HOSTS:
                endpoint = f"{host.rstrip('/')}/v1/batch"
                try:
                    resp = requests.post(
                        endpoint,
                        json={"urls": urls},
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        },
                        timeout=cls.TIMEOUT_BATCH
                    )
                    if resp.status_code == 200:
                        resp_json = resp.json()
                        raw_data = resp_json.get("data", {})
                        if isinstance(raw_data, dict):
                            items = raw_data.get("data", [])
                        elif isinstance(raw_data, list):
                            items = raw_data
                        else:
                            items = []
                        if items:
                            return items
                except requests.exceptions.Timeout:
                    logger.warning(
                        f"2md batch timeout ({cls.TIMEOUT_BATCH}s) on {host} for {len(urls)} URLs，切換至下一台備援站..."
                    )
                    continue
                except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as ce:
                    logger.warning(f"2md batch host {host} 連線異常 ({ce})，切換至下一台備援站...")
                    continue
                except Exception as e:
                    logger.debug(f"2md batch error on {host}: {e}")
                    continue

        return []

    @classmethod
    def _fallback_yfinance(cls, profile: Dict[str, Any], ticker: str):
        """使用 yfinance 作為快速解析或最後備援"""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info or {}
            if not profile.get("name") or profile["name"] == profile["raw_code"]:
                profile["name"] = info.get("shortName") or info.get("longName") or profile["raw_code"]
            if profile.get("industry") == "綜合產業" and (info.get("industry") or info.get("sector")):
                profile["industry"] = info.get("industry") or info.get("sector")
            if not profile.get("business_summary") and info.get("longBusinessSummary"):
                profile["business_summary"] = info.get("longBusinessSummary")[:300]
            if not profile.get("market_cap") and info.get("marketCap"):
                mc = info["marketCap"]
                if mc >= 1_000_000_000:
                    profile["market_cap"] = f"${mc/1_000_000_000:.1f}B"
                elif mc > 0:
                    profile["market_cap"] = f"${mc/1_000_000:.0f}M"
        except Exception as e:
            logger.debug(f"yfinance profile fallback failed: {e}")

    @staticmethod
    def _build_stock_links(ticker: str, raw_code: str, is_tw: bool) -> Dict[str, str]:
        """建立常用外部行情鏈結"""
        if is_tw:
            return {
                "yahoo": f"https://tw.stock.yahoo.com/quote/{ticker}",
                "goodinfo": f"https://goodinfo.tw/tw/BasicInfo.asp?STOCK_ID={raw_code}",
                "cnyes": f"https://www.cnyes.com/twstock/{raw_code}",
                "tradingview": f"https://www.tradingview.com/symbols/TWSE-{raw_code}/"
            }
        else:
            return {
                "yahoo": f"https://finance.yahoo.com/quote/{ticker}",
                "finviz": f"https://finviz.com/quote.ashx?t={ticker}",
                "tradingview": f"https://www.tradingview.com/symbols/{ticker}/"
            }

    @classmethod
    def get_company_profiles_batch(cls, tickers: List[str], db: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
        """
        使用 2md 原生批次 API (POST /v1/batch) 批次取得多支股票之公司簡介與即時新聞。
        搭配多站 Fallback 容災與並發信號量保護，避免瞬間暴衝與 OOM。
        """
        results = {}
        unique_tickers = list(dict.fromkeys([t.strip() for t in tickers if t and t.strip()]))
        if not unique_tickers:
            return results

        # 1. 優先回傳已在快取者
        missing = []
        now = time.time()
        for t in unique_tickers:
            norm = normalize_ticker(t).upper()
            if norm in cls._CACHE and (now - cls._CACHE[norm][0] < cls.CACHE_TTL):
                results[norm] = cls._CACHE[norm][1]
            else:
                missing.append(norm)

        if not missing:
            return results

        # 2. 為缺少快取的標的初始化 profile 結構
        profiles_to_fetch = {}
        for norm in missing:
            stock_name = cls._resolve_stock_name(norm, db)
            is_tw = norm.endswith(".TW") or norm.endswith(".TWO")
            raw_code = norm.split(".")[0] if is_tw else norm
            profiles_to_fetch[norm] = {
                "ticker": norm,
                "raw_code": raw_code,
                "name": stock_name or raw_code,
                "industry": "綜合產業",
                "business_summary": "",
                "chairman": "",
                "market_cap": "",
                "established": "",
                "recent_news": [],
                "links": cls._build_stock_links(norm, raw_code, is_tw),
                "source": "2md_batch"
            }

        # 3. 台股標的：使用 2md 原生批次 API (POST /v1/batch) 抓取 Yahoo Profile 頁面
        tw_tickers = [t for t in missing if (t.endswith(".TW") or t.endswith(".TWO"))]
        batch_chunk_size = 3  # 每批 2~3 支微批次，兼顧伺服端 Chromium 資源與低延遲回應

        for i in range(0, len(tw_tickers), batch_chunk_size):
            chunk = tw_tickers[i:i + batch_chunk_size]
            urls_map = {
                f"https://tw.stock.yahoo.com/quote/{t}/profile": t
                for t in chunk
            }
            batch_items = cls._fetch_batch_from_2md(list(urls_map.keys()))

            for item in batch_items:
                item_url = item.get("url", "")
                content = item.get("content") or item.get("markdown") or ""
                matched_ticker = urls_map.get(item_url)
                if not matched_ticker:
                    for u, t in urls_map.items():
                        if u.rstrip("/") == item_url.rstrip("/"):
                            matched_ticker = t
                            break
                if matched_ticker and content:
                    cls._parse_profile_markdown(profiles_to_fetch[matched_ticker], content, is_tw=True)

        # 4. 補齊未取得業務簡介者 (使用 yfinance 作為快速補底)
        for norm in missing:
            prof = profiles_to_fetch[norm]
            if not prof.get("business_summary"):
                cls._fallback_yfinance(prof, norm)
            # 寫入快取與回傳結果
            cls._CACHE[norm] = (now, prof)
            results[norm] = prof

        return results

    @classmethod
    def warmup_cache(cls, tickers: List[str], db: Optional[Any] = None):
        """背景預熱快取"""
        t = threading.Thread(target=cls.get_company_profiles_batch, args=(tickers, db), daemon=True)
        t.start()
