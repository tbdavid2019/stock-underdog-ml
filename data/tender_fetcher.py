"""
data/tender_fetcher.py - 台灣政府電子採購網標案與得標情報服務 (ToBid MCP)
整合 https://tobid.tw/guide/mcp 公開標準 MCP 端點 (https://api.tobid.tw/mcp)
支援記憶體 L1、DuckDB L2、磁碟 L3 快取、自動上市櫃實體名稱消歧義 (Entity Resolution) 與高可用降級
"""

import os
import re
import json
import time
import logging
import datetime
import urllib.parse
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple, Union
import requests

logger = logging.getLogger("stock_app.tender")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def format_currency_ntd(amount: Union[int, float]) -> str:
    """格式化新台幣金額為閱讀友善之單位 (元/萬/億)"""
    if not isinstance(amount, (int, float)) or amount <= 0:
        return "0 元"
    if amount >= 1e8:
        val = amount / 1e8
        return f"{val:.2f} 億元" if val < 1000 else f"{val:,.1f} 億元"
    elif amount >= 1e4:
        return f"{amount / 1e4:,.0f} 萬元"
    else:
        return f"{amount:,.0f} 元"


class TenderService:
    BASE_URL = "https://api.tobid.tw/mcp"
    CACHE_DIR = os.path.join(BASE_DIR, "cache", "tender")
    CACHE_TTL = 86400 * 7       # 7 天快取（標案資料每週至每日更新）
    TIMEOUT_SECONDS = 8.0       # 單次 MCP 呼叫超時 8 秒

    # 記憶體 L1 快取: ticker -> (timestamp, data)
    _MEM_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    @classmethod
    def clear_in_memory_cache(cls):
        """清除記憶體快取（測試或重載用）"""
        cls._MEM_CACHE.clear()

    @classmethod
    def _ensure_dirs(cls):
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

    @classmethod
    def _call_mcp_tool(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float = TIMEOUT_SECONDS
    ) -> Tuple[bool, Any]:
        """
        向 ToBid MCP 服務發送 JSON-RPC tools/call 請求
        回傳: (is_success, parsed_data_or_error_message)
        """
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000) % 1000000,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "888-Stock-Quant-Engine/2.4"
        }

        try:
            resp = requests.post(cls.BASE_URL, json=payload, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                return False, f"HTTP Error {resp.status_code}"

            res_json = resp.json()
            if "error" in res_json:
                return False, res_json.get("error", {}).get("message", "Unknown JSON-RPC error")

            result = res_json.get("result", {})
            if result.get("isError"):
                content = result.get("content", [])
                err_text = content[0].get("text", "") if content else "Tool execution error"
                if "查無此廠商" in err_text:
                    return False, "NOT_FOUND"
                return False, err_text

            content = result.get("content", [])
            if not content:
                return False, "Empty tool response content"

            raw_text = content[0].get("text", "")
            if not raw_text:
                return False, "Empty text in tool response"

            # 解析回傳之 JSON 字串
            try:
                parsed = json.loads(raw_text)
                return True, parsed
            except Exception as e:
                return False, f"JSON parse error: {e}"

        except requests.Timeout:
            logger.warning(f"ToBid MCP 呼叫逾時 ({tool_name}): timeout={timeout}s")
            return False, "TIMEOUT"
        except Exception as e:
            logger.warning(f"ToBid MCP 連線失敗 ({tool_name}): {e}")
            return False, str(e)

    @classmethod
    def get_company_tender(
        cls,
        ticker_raw: str,
        db: Optional[Any] = None,
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        查詢特定台股個股之政府採購得標統計、機關客戶、競標對手與大案清單
        支援 L1 記憶體 -> L2 DuckDB -> L3 磁碟 -> 遠端 ToBid MCP
        """
        if not ticker_raw:
            return None

        ticker_clean = str(ticker_raw).strip()
        ticker_upper = ticker_clean.upper()
        raw_code = re.sub(r"\.(TW|TWO)$", "", ticker_upper)

        # 非台股代碼過濾（ToBid 僅支援台灣政府採購）
        is_tw = bool(
            re.search(r"\.(TW|TWO)$", ticker_upper) or
            (raw_code.isdigit() and len(raw_code) in (4, 5))
        )
        if not is_tw and not any(ord(c) > 127 for c in ticker_clean):
            # 若為美股等非中文且非台股代號，直接回傳不支援
            return {
                "ticker": ticker_upper,
                "name": ticker_clean,
                "has_data": False,
                "message": "ToBid 僅收錄台灣政府採購電子公報標案"
            }

        now = time.time()

        # 1. 檢查 L1 記憶體快取
        if not force_refresh and raw_code in cls._MEM_CACHE:
            ts, cached_data = cls._MEM_CACHE[raw_code]
            if now - ts < cls.CACHE_TTL:
                return cached_data

        # 2. 檢查 DuckDB 快取
        if not force_refresh and db is not None and hasattr(db, "get_company_tender"):
            try:
                db_data = db.get_company_tender(raw_code, max_age_seconds=cls.CACHE_TTL)
                if db_data:
                    cls._MEM_CACHE[raw_code] = (now, db_data)
                    return db_data
            except Exception as e:
                logger.debug(f"DuckDB 讀取標案快取失敗 ({raw_code}): {e}")

        # 3. 檢查 L3 磁碟快取
        cls._ensure_dirs()
        cache_file = os.path.join(cls.CACHE_DIR, f"{raw_code}.json")
        if not force_refresh and os.path.exists(cache_file):
            try:
                mtime = os.path.getmtime(cache_file)
                if now - mtime < cls.CACHE_TTL:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        disk_data = json.load(f)
                        cls._MEM_CACHE[raw_code] = (mtime, disk_data)
                        return disk_data
            except Exception as e:
                logger.debug(f"讀取磁碟標案快取失敗 ({raw_code}): {e}")

        # 4. 解析公司正式全名 (Entity Resolution)
        company_name = None
        canonical_ticker = ticker_upper
        try:
            from data.company_network import CompanyNetworkService
            comp_info = CompanyNetworkService.resolve_company_info(ticker_clean)
            if comp_info and comp_info.get("name"):
                company_name = comp_info.get("name")
                code = comp_info.get("code") or raw_code
                market = comp_info.get("market", "TWSE")
                suffix = ".TWO" if market == "TPEx" else ".TW"
                canonical_ticker = f"{code}{suffix}" if code.isdigit() else ticker_upper
        except Exception as e:
            logger.debug(f"解析公司正式名稱失敗: {e}")

        if not company_name:
            # 若為中文名稱輸入，直接使用
            if any(ord(c) > 127 for c in ticker_clean):
                company_name = ticker_clean
            else:
                company_name = ticker_clean

        # 5. 並行呼叫 ToBid MCP: vendor_report, vendor_awards, vendor_compliance
        report_data = None
        awards_data = None
        compliance_data = None
        is_not_found = False

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            fut_report = executor.submit(cls._call_mcp_tool, "vendor_report", {"name": company_name})
            fut_awards = executor.submit(cls._call_mcp_tool, "vendor_awards", {"name": company_name, "limit": 6})
            fut_comp = executor.submit(cls._call_mcp_tool, "vendor_compliance", {"name": company_name})

            ok_rep, res_rep = fut_report.result()
            ok_awa, res_awa = fut_awards.result()
            ok_comp, res_comp = fut_comp.result()

            if ok_rep and isinstance(res_rep, dict):
                report_data = res_rep
            elif res_rep == "NOT_FOUND":
                is_not_found = True

            if ok_awa and isinstance(res_awa, dict):
                awards_data = res_awa

            if ok_comp and isinstance(res_comp, dict):
                compliance_data = res_comp

        # 若查無此廠商，回傳明確標記並寫入快取，避免重複請求
        if is_not_found or not report_data:
            empty_result = {
                "ticker": canonical_ticker,
                "raw_code": raw_code,
                "name": company_name,
                "has_data": False,
                "message": "查無政府採購標案得標紀錄（非政府標案依賴型產業）",
                "web_url": f"https://tobid.tw/?q={urllib.parse.quote(company_name)}",
                "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            # 寫入各級快取
            cls._save_to_all_caches(raw_code, empty_result, db=db)
            return empty_result

        # 6. 組裝豐富化標案情報資料結構
        award_total = report_data.get("known_award_total") or 0
        participations = report_data.get("participations") or 0
        wins = report_data.get("wins") or 0
        win_rate = report_data.get("win_rate_pct") or 0.0
        sole_bid_rate = report_data.get("sole_bid_rate_pct") or 0.0

        # Top 5 採購機關金主
        raw_clients = report_data.get("top_clients") or []
        top_clients = []
        for c in raw_clients[:5]:
            amt = c.get("amount") or 0
            top_clients.append({
                "unit": c.get("unit") or "未具名機關",
                "wins": c.get("wins") or 0,
                "amount": amt,
                "amount_formatted": format_currency_ntd(amt)
            })

        # Top 5 競標對手與勝率
        raw_rivals = report_data.get("top_rivals") or []
        top_rivals = []
        for r in raw_rivals[:5]:
            met = r.get("met") or 0
            my_wins = r.get("this_vendor_wins") or 0
            rival_wins = r.get("rival_wins") or 0
            win_pct = round((my_wins / met * 100), 1) if met > 0 else 0.0
            top_rivals.append({
                "name": r.get("name") or "競爭廠商",
                "met": met,
                "this_vendor_wins": my_wins,
                "rival_wins": rival_wins,
                "win_rate": win_pct
            })

        # 主力採購品項標籤
        main_products = []
        prod_obj = report_data.get("main_products") or {}
        if isinstance(prod_obj, dict):
            terms = prod_obj.get("terms") or []
            main_products = [{"term": t.get("term"), "cases": t.get("cases")} for t in terms[:8] if t.get("term")]

        # 最近未得標案例 (落標原因分析)
        raw_losses = report_data.get("recent_losses") or []
        recent_losses = []
        for l in raw_losses[:4]:
            amt = l.get("award_amount") or 0
            date_raw = str(l.get("date") or "")
            if len(date_raw) == 8:
                date_formatted = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
            else:
                date_formatted = date_raw
            recent_losses.append({
                "date": date_formatted,
                "title": l.get("title") or "採購標案",
                "lost_to": l.get("lost_to") or "其他得標廠商",
                "award_amount": amt,
                "award_amount_formatted": format_currency_ntd(amt) if amt else "未公開",
                "lost_reason": l.get("lost_reason") or "規格或價格未達優勝"
            })

        # 最近得標案清單 (來自 vendor_awards)
        recent_awards = []
        yearly_awards = []
        if awards_data and isinstance(awards_data, dict):
            for a in (awards_data.get("items") or [])[:6]:
                amt = a.get("award_amount") or 0
                recent_awards.append({
                    "date": a.get("date") or "",
                    "unit_name": a.get("unit_name") or "",
                    "title": a.get("title") or "",
                    "job_number": a.get("job_number") or "",
                    "award_amount": amt,
                    "award_amount_formatted": format_currency_ntd(amt) if amt else "未公開",
                    "bidders": a.get("bidders") or 1,
                    "sole_bid": bool(a.get("sole_bid"))
                })

            for y in (awards_data.get("yearly") or []):
                cases = y.get("cases") or 0
                if cases > 0:
                    amt = y.get("amount") or 0
                    yearly_awards.append({
                        "year": y.get("year"),
                        "cases": cases,
                        "amount": amt,
                        "amount_formatted": format_currency_ntd(amt)
                    })
            yearly_awards = yearly_awards[:5]

        # 風控查核 (政府採購法第 101 條停權／黑天鵝查核)
        compliance_obj = {
            "currently_debarred": False,
            "status": "PASSED",
            "label": "🛡️ 採購法第 101 條正常（無停權處分）",
            "debarments": [],
            "debarment_count": 0,
            "article_101_cases_count": 0,
            "revoked_count": 0,
            "quality_awards_count": 0
        }
        if compliance_data and isinstance(compliance_data, dict):
            is_debarred = bool(compliance_data.get("currently_debarred"))
            debarments = compliance_data.get("debarments") or []
            cases = compliance_data.get("article_101_cases") or []
            revoked = compliance_data.get("revoked") or []
            awards = compliance_data.get("quality_awards") or []
            compliance_obj = {
                "currently_debarred": is_debarred,
                "status": "DEBARRED" if is_debarred else "PASSED",
                "label": "🚨 採購法第 101 條停權生效中（全面禁止投標）" if is_debarred else "🛡️ 採購法第 101 條正常（無停權處分）",
                "debarments": debarments,
                "debarment_count": len(debarments),
                "article_101_cases_count": len(cases),
                "revoked_count": len(revoked),
                "quality_awards_count": len(awards),
                "sources": compliance_data.get("sources") or {}
            }

        result_data = {
            "ticker": canonical_ticker,
            "raw_code": raw_code,
            "name": company_name,
            "has_data": True,
            "tax_id": report_data.get("tax_id") or "",
            "participations": participations,
            "wins": wins,
            "win_rate_pct": win_rate,
            "known_award_total": award_total,
            "known_award_total_formatted": format_currency_ntd(award_total),
            "sole_bid_rate_pct": sole_bid_rate,
            "top_clients": top_clients,
            "top_rivals": top_rivals,
            "main_products": main_products,
            "recent_losses": recent_losses,
            "recent_awards": recent_awards,
            "yearly_awards": yearly_awards,
            "compliance": compliance_obj,
            "web_url": report_data.get("web") or f"https://tobid.tw/?vendor={urllib.parse.quote(company_name)}",
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 7. 寫入快取 (L1, L2 DuckDB, L3 磁碟)
        cls._save_to_all_caches(raw_code, result_data, db=db)
        return result_data

    @classmethod
    def _save_to_all_caches(cls, code: str, data: Dict[str, Any], db: Optional[Any] = None):
        """將資料同步持久化至 L1 記憶體、L2 DuckDB 與 L3 本地磁碟"""
        now = time.time()
        # L1
        cls._MEM_CACHE[code] = (now, data)

        # L2 DuckDB
        if db is not None and hasattr(db, "save_company_tender"):
            try:
                db.save_company_tender(code, data)
            except Exception as e:
                logger.debug(f"DuckDB 寫入標案快取失敗: {e}")

        # L3 磁碟
        try:
            cls._ensure_dirs()
            cache_file = os.path.join(cls.CACHE_DIR, f"{code}.json")
            tmp_file = f"{cache_file}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, cache_file)
        except Exception as e:
            logger.debug(f"磁碟寫入標案快取失敗 ({code}): {e}")
