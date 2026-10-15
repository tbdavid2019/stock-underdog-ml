"""
data/company_network.py - Taiwan Company Network 集團股權網絡與關係企業調度服務
串接 https://taiwan-company-network.david888.com 公開靜態分片 CDN API
支援本機磁碟快取、DuckDB 快取、Gzip 壓縮傳輸、反向上市櫃代號補齊與高可用容錯
"""

import os
import re
import json
import time
import logging
import urllib.parse
from typing import Dict, List, Optional, Any, Tuple
import requests

logger = logging.getLogger("stock_app.company_network")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CompanyNetworkService:
    BASE_URL = "https://taiwan-company-network.david888.com/api/v1"
    CACHE_DIR = os.path.join(BASE_DIR, "cache", "company_network")
    SHARDS_DIR = os.path.join(CACHE_DIR, "shards")
    NETWORKS_DIR = os.path.join(CACHE_DIR, "networks")

    # 快取效期配置 (公司股權與組織架構變動極少)
    ALIASES_TTL = 86400 * 7       # 別名表快取 7 天
    SHARD_TTL = 86400 * 7         # 分片檔案快取 7 天
    NETWORK_TTL = 86400 * 30      # 個股網絡結果快取 30 天
    TIMEOUT_REQUEST = 8.0         # 靜態 CDN 請求超時 8 秒

    # 記憶體 L1 快取
    _ALIASES_DATA: Optional[Dict[str, Any]] = None
    _ALIASES_TIMESTAMP: float = 0.0
    _CODE_TO_INFO: Dict[str, Dict[str, Any]] = {}
    _NAME_TO_INFO: Dict[str, Dict[str, Any]] = {}
    _MEM_NETWORKS: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    @classmethod
    def clear_in_memory_cache(cls):
        """清除記憶體快取（測試或重載用）"""
        cls._ALIASES_DATA = None
        cls._ALIASES_TIMESTAMP = 0.0
        cls._CODE_TO_INFO.clear()
        cls._NAME_TO_INFO.clear()
        cls._MEM_NETWORKS.clear()

    @classmethod
    def _ensure_dirs(cls):
        os.makedirs(cls.SHARDS_DIR, exist_ok=True)
        os.makedirs(cls.NETWORKS_DIR, exist_ok=True)

    @classmethod
    def _load_aliases(cls) -> Optional[Dict[str, Any]]:
        """
        載入別名表 aliases.json (L1 記憶體 -> L2 磁碟 -> L3 遠端 CDN)
        """
        now = time.time()
        # 1. 檢查 L1 記憶體快取
        if cls._ALIASES_DATA and (now - cls._ALIASES_TIMESTAMP < cls.ALIASES_TTL):
            return cls._ALIASES_DATA

        cls._ensure_dirs()
        alias_file = os.path.join(cls.CACHE_DIR, "aliases.json")

        # 2. 檢查 L2 磁碟快取
        if os.path.exists(alias_file):
            try:
                mtime = os.path.getmtime(alias_file)
                if now - mtime < cls.ALIASES_TTL:
                    with open(alias_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        cls._build_alias_index(data)
                        cls._ALIASES_DATA = data
                        cls._ALIASES_TIMESTAMP = mtime
                        return data
            except Exception as e:
                logger.warning(f"讀取磁碟 aliases.json 失敗: {e}")

        # 3. 請求 L3 遠端 CDN
        try:
            url = f"{cls.BASE_URL}/aliases.json"
            headers = {"Accept-Encoding": "gzip, deflate", "User-Agent": "888-Stock-Quant-Engine/2.4"}
            resp = requests.get(url, headers=headers, timeout=cls.TIMEOUT_REQUEST)
            if resp.status_code == 200:
                data = resp.json()
                cls._build_alias_index(data)
                cls._ALIASES_DATA = data
                cls._ALIASES_TIMESTAMP = now
                # 寫入磁碟快取
                try:
                    tmp_file = f"{alias_file}.tmp"
                    with open(tmp_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                    os.replace(tmp_file, alias_file)
                except Exception as e:
                    logger.debug(f"持久化 aliases.json 失敗: {e}")
                return data
        except Exception as e:
            logger.warning(f"從 Taiwan Company Network 下載 aliases.json 失敗: {e}")

        # 若遠端失敗但磁碟有舊資料，降級使用
        if os.path.exists(alias_file):
            try:
                with open(alias_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cls._build_alias_index(data)
                    cls._ALIASES_DATA = data
                    cls._ALIASES_TIMESTAMP = os.path.getmtime(alias_file)
                    return data
            except Exception:
                pass

        return None

    @classmethod
    def _build_alias_index(cls, data: Dict[str, Any]):
        """建立代號與名稱的雙向索引"""
        aliases_map = data.get("aliases") if isinstance(data, dict) else None
        if not aliases_map or not isinstance(aliases_map, dict):
            return

        code_map: Dict[str, Dict[str, Any]] = {}
        name_map: Dict[str, Dict[str, Any]] = {}

        for key, items in aliases_map.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                code = item.get("code")
                name = item.get("name")
                market = item.get("market", "TWSE")
                if code and code not in code_map:
                    code_map[code] = {"name": name, "code": code, "market": market}
                if name and name not in name_map:
                    name_map[name] = {"name": name, "code": code, "market": market}
                # 若 key 為俗稱/別名 (如 '台積電')，亦建立索引
                if key and key not in name_map:
                    name_map[key] = {"name": name, "code": code, "market": market}

        cls._CODE_TO_INFO = code_map
        cls._NAME_TO_INFO = name_map

    @classmethod
    def resolve_company_info(cls, query: str) -> Optional[Dict[str, Any]]:
        """
        從查詢字串 (代號如 '2330', '2330.TW', 或簡稱 '台積電') 解析上市櫃正式名稱與代碼
        """
        if not query:
            return None

        cls._load_aliases()
        q = str(query).strip().upper()
        # 去除 .TW / .TWO 後綴
        raw_code = re.sub(r"\.(TW|TWO)$", "", q)

        # 1. 依代號匹配
        if raw_code in cls._CODE_TO_INFO:
            return cls._CODE_TO_INFO[raw_code]

        # 2. 依名稱匹配
        if query in cls._NAME_TO_INFO:
            return cls._NAME_TO_INFO[query]
        if q in cls._NAME_TO_INFO:
            return cls._NAME_TO_INFO[q]

        return None

    @classmethod
    def _fetch_shard(cls, first_char: str) -> Optional[Dict[str, Any]]:
        """
        下載或讀取快取的 Unicode 字元分片 (companies/{hex}.json)
        """
        if not first_char:
            return None
        cls._ensure_dirs()
        hex_code = hex(ord(first_char[0]))[2:].lower()
        shard_file = os.path.join(cls.SHARDS_DIR, f"{hex_code}.json")
        now = time.time()

        # 1. 檢查磁碟分片快取
        if os.path.exists(shard_file):
            try:
                mtime = os.path.getmtime(shard_file)
                if now - mtime < cls.SHARD_TTL:
                    with open(shard_file, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception as e:
                logger.debug(f"讀取分片快取 {hex_code}.json 失敗: {e}")

        # 2. 遠端抓取
        try:
            url = f"{cls.BASE_URL}/companies/{hex_code}.json"
            headers = {"Accept-Encoding": "gzip, deflate", "User-Agent": "888-Stock-Quant-Engine/2.4"}
            resp = requests.get(url, headers=headers, timeout=cls.TIMEOUT_REQUEST)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    tmp_file = f"{shard_file}.tmp"
                    with open(tmp_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                    os.replace(tmp_file, shard_file)
                except Exception as e:
                    logger.debug(f"持久化分片 {hex_code}.json 失敗: {e}")
                return data
            elif resp.status_code == 404:
                logger.debug(f"分片不存在: {hex_code}.json")
                return None
        except Exception as e:
            logger.warning(f"下載分片 {hex_code}.json 失敗: {e}")

        # 遠端失敗時回退讀取過期磁碟分片
        if os.path.exists(shard_file):
            try:
                with open(shard_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return None

    @classmethod
    def _enrich_relationship_node(cls, company_name: str) -> Dict[str, Any]:
        """將上/下游公司名稱標註上市櫃資訊"""
        if not company_name:
            return {"name": "", "code": None, "ticker": None, "market": None, "is_listed": False}

        info = cls._NAME_TO_INFO.get(company_name)
        if info and info.get("code"):
            code = info["code"]
            market = info.get("market", "TWSE")
            suffix = ".TWO" if market == "TPEx" else ".TW"
            return {
                "name": company_name,
                "code": code,
                "ticker": f"{code}{suffix}",
                "market": market,
                "is_listed": True
            }
        return {
            "name": company_name,
            "code": None,
            "ticker": None,
            "market": None,
            "is_listed": False
        }

    @classmethod
    def get_company_network(cls, ticker_raw: str, db: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        取得特定台股標的的企業網絡、上下游控股投資與官方登記資料
        """
        if not ticker_raw:
            return None

        ticker_upper = str(ticker_raw).strip().upper()

        # 快速過濾外盤標的（如 AAPL, NVDA, TSLA, SPY, QQQ 等非台股格式）
        is_tw_explicit = ticker_upper.endswith(".TW") or ticker_upper.endswith(".TWO")
        raw_code = re.sub(r"\.(TW|TWO)$", "", ticker_upper)
        if not is_tw_explicit and not raw_code.isdigit() and ticker_upper not in cls._NAME_TO_INFO:
            # 嘗試檢查是否為中文別名 (如 '台積電')
            info = cls.resolve_company_info(ticker_raw)
            if not info:
                return None

        # 1. 檢查 L1 記憶體快取
        now = time.time()
        if raw_code in cls._MEM_NETWORKS:
            ts, cached_net = cls._MEM_NETWORKS[raw_code]
            if now - ts < cls.NETWORK_TTL:
                return cached_net

        # 2. 檢查 DuckDB 快取
        if db is not None and hasattr(db, "get_company_network"):
            try:
                db_data = db.get_company_network(raw_code, max_age_seconds=cls.NETWORK_TTL)
                if db_data:
                    cls._MEM_NETWORKS[raw_code] = (now, db_data)
                    return db_data
            except Exception as e:
                logger.debug(f"DuckDB 讀取企業網絡快取失敗: {e}")

        # 3. 檢查 L2 磁碟快取
        cls._ensure_dirs()
        net_file = os.path.join(cls.NETWORKS_DIR, f"{raw_code}.json")
        if os.path.exists(net_file):
            try:
                mtime = os.path.getmtime(net_file)
                if now - mtime < cls.NETWORK_TTL:
                    with open(net_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        cls._MEM_NETWORKS[raw_code] = (mtime, data)
                        return data
            except Exception as e:
                logger.debug(f"讀取磁碟網絡快取失敗 {raw_code}: {e}")

        # 4. 解析公司正式全名
        company_info = cls.resolve_company_info(ticker_raw)
        if not company_info:
            return None

        registered_name = company_info.get("name")
        if not registered_name:
            return None

        # 5. 下載/讀取對應中文首字之分片
        first_char = registered_name[0]
        shard = cls._fetch_shard(first_char)
        if not shard or not isinstance(shard, dict):
            return None

        shard_data = shard.get("data", {})
        entry = shard_data.get(registered_name)
        if not entry and isinstance(shard_data, dict):
            # 若無精確匹配，嘗試搜尋名稱相符者
            for k, v in shard_data.items():
                if k == registered_name or registered_name in k:
                    entry = v
                    break

        if not entry:
            return None

        # 6. 組裝與豐富化關係節點
        capital_num = entry.get("capital")
        capital_formatted = ""
        if isinstance(capital_num, (int, float)) and capital_num > 0:
            if capital_num >= 1e8:
                capital_formatted = f"{capital_num / 1e8:.2f} 億元"
            else:
                capital_formatted = f"{capital_num / 1e4:.0f} 萬元"

        raw_relationships = entry.get("relationships", {}) or {}
        raw_upstream = raw_relationships.get("upstream", []) or []
        raw_downstream = raw_relationships.get("downstream", []) or []

        enriched_upstream = [cls._enrich_relationship_node(name) for name in raw_upstream if name]
        enriched_downstream = [cls._enrich_relationship_node(name) for name in raw_downstream if name]

        encoded_name = urllib.parse.quote(registered_name)
        default_graph_url = f"https://taiwan-company-network.david888.com/graph?company={encoded_name}"
        web_url = entry.get("webUrl") or default_graph_url

        code = company_info.get("code") or raw_code
        market = company_info.get("market", "TWSE")
        suffix = ".TWO" if market == "TPEx" else ".TW"
        canonical_ticker = f"{code}{suffix}" if code.isdigit() else ticker_upper

        result = {
            "ticker": canonical_ticker,
            "raw_code": code,
            "name": registered_name,
            "unified_id": entry.get("id"),
            "capital": capital_num,
            "capital_formatted": capital_formatted,
            "representative": entry.get("representative"),
            "address": entry.get("address"),
            "relationships": {
                "upstream": enriched_upstream,
                "downstream": enriched_downstream
            },
            "web_url": web_url,
            "source": "taiwan-company-network"
        }

        # 7. 寫入快取 (L1 記憶體 + L2 磁碟 + DuckDB)
        cls._MEM_NETWORKS[raw_code] = (now, result)
        try:
            tmp_f = f"{net_file}.tmp"
            with open(tmp_f, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            os.replace(tmp_f, net_file)
        except Exception as e:
            logger.debug(f"持久化網絡快取失敗 {raw_code}: {e}")

        if db is not None and hasattr(db, "save_company_network"):
            try:
                db.save_company_network(raw_code, result)
            except Exception as e:
                logger.debug(f"DuckDB 寫入網絡資料失敗 {raw_code}: {e}")

        return result
