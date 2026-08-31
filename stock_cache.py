"""
Stock list cache manager
Caches stock index components to reduce API dependency
"""
import json
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from logger import logger

CACHE_FILE = "cache/stock_lists.json"
# Used for reporting cache freshness. Reads always try the upstream API first.
CACHE_DURATION = timedelta(days=5)
MAX_CACHE_AGE = timedelta(days=90)  # 最大快取壽命

# 備份清單（API 完全失敗時使用）
FALLBACK_STOCKS = {
    "TW0050": [
        "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW",
        "2882.TW", "2412.TW", "2891.TW", "2886.TW", "2303.TW"
    ],
    "SP500": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
        "BRK-B", "LLY", "AVGO", "JPM", "WMT", "V", "XOM", "UNH"
    ]
}


class StockListCache:
    """管理股票清單快取"""
    
    def __init__(self):
        self.cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """載入快取檔案"""
        if not os.path.exists(CACHE_FILE):
            return {}
        
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"載入快取失敗: {e}")
            return {}
    
    def _save_cache(self):
        """儲存快取檔案"""
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"儲存快取失敗: {e}")
    
    def _is_expired(self, index_name: str) -> bool:
        """檢查指定指數的快取是否過期"""
        if index_name not in self.cache_data:
            return True
        
        timestamp_str = self.cache_data[index_name].get('timestamp')
        if not timestamp_str:
            return True
        
        try:
            cache_time = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - cache_time
            return age > CACHE_DURATION
        except:
            return True
    
    def _is_too_old(self, index_name: str) -> bool:
        """檢查快取是否太舊（超過最大壽命）"""
        if index_name not in self.cache_data:
            return True
        
        timestamp_str = self.cache_data[index_name].get('timestamp')
        if not timestamp_str:
            return True
        
        try:
            cache_time = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - cache_time
            return age > MAX_CACHE_AGE
        except:
            return True

    def _parse_fetch_result(self, result) -> Tuple[List[str], Dict[str, str]]:
        if isinstance(result, dict) and 'stocks' in result:
            return result.get('stocks', []), result.get('name_map', {})
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        return result, {}
    
    def get(self, index_name: str, fetcher_func) -> List[str]:
        """
        取得股票清單（帶快取和容錯）
        
        Args:
            index_name: 指數名稱（如 "TW0050"）
            fetcher_func: API 抓取函數
        
        Returns:
            股票清單
        """
        # Always try the upstream API first so each run sees the latest list.
        try:
            logger.info(f"🔄 更新 {index_name} 快取...")
            fetch_result = fetcher_func()
            stocks, name_map = self._parse_fetch_result(fetch_result)
            if not stocks:
                raise ValueError("API 回傳空股票清單")
            
            # 更新快取
            payload = {
                'stocks': stocks,
                'timestamp': datetime.now().isoformat()
            }
            if name_map:
                payload['name_map'] = name_map
            self.cache_data[index_name] = payload
            self._save_cache()
            
            logger.info(f"✅ {index_name} 快取已更新 ({len(stocks)} 支股票)")
            return stocks
        
        except Exception as e:
            logger.warning(f"⚠️ API 失敗 ({index_name}): {e}")
            
            # Use the previous cache only when the upstream fetch fails.
            if index_name in self.cache_data and not self._is_too_old(index_name):
                stocks = self.cache_data[index_name].get('stocks', [])
                if stocks:
                    cache_age = self._get_cache_age(index_name)
                    logger.warning(f"⚠️ API 失敗，使用快取 fallback: {index_name} (已 {cache_age} 天)")
                    return stocks
            
            # 第四層：使用備份清單
            if index_name in FALLBACK_STOCKS:
                logger.error(f"🆘 使用備份清單: {index_name}")
                return FALLBACK_STOCKS[index_name]
            
            # 最後手段：回傳空清單
            logger.error(f"❌ 無法取得 {index_name} 股票清單")
            return []

    def get_name_map(self, index_name: str, fetcher_func) -> Dict[str, str]:
        """
        取得股票名稱對照表（帶快取和容錯）
        """
        # Always try the upstream API first so names stay current.
        try:
            logger.info(f"🔄 更新 {index_name} 名稱快取...")
            fetch_result = fetcher_func()
            stocks, name_map = self._parse_fetch_result(fetch_result)
            if not name_map:
                raise ValueError("API 未回傳名稱對照表")

            payload = {
                'stocks': stocks,
                'timestamp': datetime.now().isoformat()
            }
            if name_map:
                payload['name_map'] = name_map
            self.cache_data[index_name] = payload
            self._save_cache()

            if name_map:
                logger.info(f"✅ {index_name} 名稱快取已更新 ({len(name_map)} 支股票)")
            return name_map

        except Exception as e:
            logger.warning(f"⚠️ 名稱 API 失敗 ({index_name}): {e}")

            if index_name in self.cache_data and not self._is_too_old(index_name):
                name_map = self.cache_data[index_name].get('name_map')
                if name_map:
                    cache_age = self._get_cache_age(index_name)
                    logger.warning(f"⚠️ API 失敗，使用名稱快取 fallback: {index_name} (已 {cache_age} 天)")
                    return name_map

            logger.error(f"❌ 無法取得 {index_name} 名稱對照表")
            return {}
    
    def _get_cache_age(self, index_name: str) -> int:
        """取得快取年齡（天數）"""
        if index_name not in self.cache_data:
            return 999
        
        timestamp_str = self.cache_data[index_name].get('timestamp')
        if not timestamp_str:
            return 999
        
        try:
            cache_time = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - cache_time
            return age.days
        except:
            return 999
    
    def force_refresh(self, index_name: str, fetcher_func) -> List[str]:
        """強制更新指定指數的快取"""
        logger.info(f"🔄 強制更新 {index_name}...")
        
        try:
            stocks = fetcher_func()
            self.cache_data[index_name] = {
                'stocks': stocks,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()
            logger.info(f"✅ {index_name} 已強制更新")
            return stocks
        except Exception as e:
            logger.error(f"❌ 強制更新失敗 ({index_name}): {e}")
            return self.cache_data.get(index_name, {}).get('stocks', [])
    
    def get_cache_status(self) -> Dict[str, Any]:
        """取得所有快取狀態"""
        status = {}
        for index_name in self.cache_data:
            age_days = self._get_cache_age(index_name)
            is_expired = self._is_expired(index_name)
            stock_count = len(self.cache_data[index_name].get('stocks', []))
            
            status[index_name] = {
                'age_days': age_days,
                'expired': is_expired,
                'stock_count': stock_count,
                'timestamp': self.cache_data[index_name].get('timestamp')
            }
        
        return status


# 全域快取實例
_cache = StockListCache()


def get_cached_stocks(index_name: str, fetcher_func) -> List[str]:
    """
    取得股票清單（帶快取）
    
    Args:
        index_name: 指數名稱
        fetcher_func: API 抓取函數
    
    Returns:
        股票清單
    """
    return _cache.get(index_name, fetcher_func)


def get_cached_stock_map(index_name: str, fetcher_func) -> Dict[str, str]:
    """
    取得股票名稱對照表（帶快取）
    """
    return _cache.get_name_map(index_name, fetcher_func)


def force_refresh_cache(index_name: str, fetcher_func) -> List[str]:
    """強制更新快取"""
    return _cache.force_refresh(index_name, fetcher_func)


def get_cache_status() -> Dict[str, Any]:
    """取得快取狀態"""
    return _cache.get_cache_status()
