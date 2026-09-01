"""
data/institutional.py - 台股三大法人籌碼數據提供者 (Institutional Flow Provider)

從 TWSE (T86) 與 TPEX (三大法人買賣超) 獲取外資、投信、自營商每日進出數據，
並計算 5日/20日 累計買賣超、投信連續買超天數 (Streak) 與土洋合買等籌碼指標。
"""

import os
import json
import logging
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import requests

logger = logging.getLogger("stock_app.institutional")


@dataclass
class InstitutionalSummary:
    """單一股票法人籌碼指標摘要"""
    ticker: str
    trust_net_5d: int = 0         # 投信 5 日累計淨買賣 (張)
    trust_net_20d: int = 0        # 投信 20 日累計淨買賣 (張)
    trust_streak: int = 0         # 投信連續買超天數
    foreign_net_5d: int = 0       # 外資 5 日累計淨買賣 (張)
    foreign_net_20d: int = 0      # 外資 20 日累計淨買賣 (張)
    dealer_net_5d: int = 0        # 自營商 5 日累計淨買賣 (張)
    total_net_5d: int = 0         # 三大法人 5 日合計淨買賣 (張)
    is_trust_streak: bool = False # 是否投信連買 (>= 3 天)
    is_sync_buy: bool = False     # 是否土洋合買 (外資+投信 5D 雙買超)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "trust_net_5d": self.trust_net_5d,
            "trust_net_20d": self.trust_net_20d,
            "trust_streak": self.trust_streak,
            "foreign_net_5d": self.foreign_net_5d,
            "foreign_net_20d": self.foreign_net_20d,
            "dealer_net_5d": self.dealer_net_5d,
            "total_net_5d": self.total_net_5d,
            "is_trust_streak": self.is_trust_streak,
            "is_sync_buy": self.is_sync_buy,
        }


class InstitutionalProvider:
    """台股三大法人數據提供者"""

    CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "institutional")

    @classmethod
    def _ensure_cache_dir(cls):
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

    @classmethod
    def fetch_twse_t86(cls, date_str: str) -> Dict[str, Dict[str, int]]:
        """
        抓取上市三大法人買賣超 (TWSE T86)
        date_str 格式: YYYYMMDD (例如 20260901)
        回傳: { '2330': {'foreign_net': 1000, 'trust_net': 500, 'dealer_net': 200, 'total_net': 1700} } (單位: 張)
        """
        cls._ensure_cache_dir()
        cache_file = os.path.join(cls.CACHE_DIR, f"twse_t86_{date_str}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALLBUT0999&response=json"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        res_map = {}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("stat") == "OK" and "data" in data:
                    for row in data["data"]:
                        # row 欄位說明:
                        # 0: 證券代號, 1: 證券名稱, 4: 外陸資買賣超股數, 10: 投信買賣超股數, 11: 自營商買賣超股數, 18: 三大法人買賣超股數
                        try:
                            code = str(row[0]).strip()
                            foreign_net = int(str(row[4]).replace(",", "")) // 1000
                            trust_net = int(str(row[10]).replace(",", "")) // 1000
                            dealer_net = int(str(row[11]).replace(",", "")) // 1000
                            total_net = int(str(row[18]).replace(",", "")) // 1000
                            res_map[code] = {
                                "foreign_net": foreign_net,
                                "trust_net": trust_net,
                                "dealer_net": dealer_net,
                                "total_net": total_net,
                            }
                        except (ValueError, IndexError):
                            continue

                    # 寫入快取
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(res_map, f, ensure_ascii=False)
                    logger.info(f"💾 成功抓取並快取 TWSE T86 法人數據 ({date_str}, 共 {len(res_map)} 檔)")
        except Exception as e:
            logger.warning(f"⚠️ 抓取 TWSE T86 ({date_str}) 失敗: {e}")

        return res_map

    @classmethod
    def get_recent_trading_days(cls, count: int = 20) -> List[str]:
        """
        推算最近 count 個工作日 (排除週末) 之日期字串 YYYYMMDD
        """
        days = []
        cur = datetime.date.today()
        while len(days) < count:
            if cur.weekday() < 5:  # 0~4 為週一至週五
                days.append(cur.strftime("%Y%m%d"))
            cur -= datetime.timedelta(days=1)
        return days

    @classmethod
    def get_institutional_summary_batch(
        cls, 
        tickers: List[str], 
        lookback_days: int = 10
    ) -> Dict[str, InstitutionalSummary]:
        """
        批次取得指定股票清單的法人籌碼指標
        """
        # 美股或非台股直接略過
        tw_tickers = [t for t in tickers if t.endswith(".TW") or t.endswith(".TWO") or t.isdigit()]
        if not tw_tickers:
            return {}

        trading_days = cls.get_recent_trading_days(count=lookback_days)
        
        # 依日期收集法人每日淨買賣
        daily_records: List[Dict[str, Dict[str, int]]] = []
        for d in trading_days:
            day_data = cls.fetch_twse_t86(d)
            if day_data:
                daily_records.append(day_data)
            if len(daily_records) >= 5:  # 至少收集到 5 個有交易數據的天數
                break

        summaries: Dict[str, InstitutionalSummary] = {}

        for raw_t in tickers:
            clean_code = raw_t.split(".")[0].strip()
            summary = InstitutionalSummary(ticker=raw_t)

            if not daily_records:
                summaries[raw_t] = summary
                continue

            # 統計 5 日累積
            t_net_5d = 0
            f_net_5d = 0
            d_net_5d = 0
            tot_net_5d = 0

            # 計算投信連買 streak (由近至遠)
            streak = 0
            streak_active = True

            for idx, day_map in enumerate(daily_records[:5]):
                stock_data = day_map.get(clean_code)
                if stock_data:
                    t_val = stock_data.get("trust_net", 0)
                    f_val = stock_data.get("foreign_net", 0)
                    d_val = stock_data.get("dealer_net", 0)
                    tot_val = stock_data.get("total_net", 0)

                    t_net_5d += t_val
                    f_net_5d += f_val
                    d_net_5d += d_val
                    tot_net_5d += tot_val

                    if streak_active:
                        if t_val > 0:
                            streak += 1
                        else:
                            streak_active = False

            summary.trust_net_5d = t_net_5d
            summary.foreign_net_5d = f_net_5d
            summary.dealer_net_5d = d_net_5d
            summary.total_net_5d = tot_net_5d
            summary.trust_streak = streak
            summary.is_trust_streak = streak >= 3
            summary.is_sync_buy = (f_net_5d > 0 and t_net_5d > 0)

            summaries[raw_t] = summary

        return summaries
