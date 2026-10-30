"""
pipeline/discovery_pipeline.py - AI 產業鏈前沿挖掘機資料管線 (Discovery Pipeline)

實作 arXiv 論文爬取 (cs.AI, cs.RO, cs.LG, cs.CV，3秒間隔與 429 退避重試)、
語意技術瓶頸萃取、代號正規化與 Universe 白名單硬過濾、7 維多因子評分，
以及 DuckDB 交易寫入 + tempfile + os.replace 原子快取替換與異常 Fallback 快照機制。
"""

import datetime
import json
import logging
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Set, Tuple

from core.discovery_scorer import compute_discovery_score
from core.supply_chain_map import default_supply_chain_map
from core.ticker_canonicalizer import canonicalize_ticker, default_validator, detect_market
from data.duckdb_manager import DuckDBManager, TZ_TAIPEI

logger = logging.getLogger("stock_app.discovery.pipeline")


class DiscoveryPipeline:
    """前沿雷達資料收集、評分與雙層原子發布管線"""

    def __init__(
        self,
        db_manager: Optional[DuckDBManager] = None,
        cache_dir: str = "cache/discovery",
    ):
        self.db_manager = db_manager or DuckDBManager()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_arxiv_papers(
        self,
        categories: Optional[List[str]] = None,
        max_results_per_cat: int = 10,
        throttle_seconds: float = 3.0,
        max_retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        自 arXiv 官方 API 擷取最新學術前沿論文。
        包含 3 秒間隔節流與 429 指數退避重試。
        """
        cats = categories or ["cs.AI", "cs.RO", "cs.LG", "cs.CV"]
        papers: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        for idx, cat in enumerate(cats):
            if idx > 0 and throttle_seconds > 0:
                time.sleep(throttle_seconds)

            query_url = (
                f"http://export.arxiv.org/api/query?"
                f"search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending&max_results={max_results_per_cat}"
            )

            req = urllib.request.Request(
                query_url,
                headers={"User-Agent": "888StockQuantDiscoveryEngine/2.4 (mailto:quant@david888.com)"}
            )

            success = False
            for attempt in range(max_retries):
                try:
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        if resp.status == 200:
                            content = resp.read()
                            parsed = self._parse_arxiv_atom(content, cat)
                            for p in parsed:
                                if p["paper_id"] not in seen_ids:
                                    seen_ids.add(p["paper_id"])
                                    papers.append(p)
                            success = True
                            break
                        elif resp.status == 429:
                            backoff = 2 ** attempt * 3
                            logger.warning(f"arXiv API 回傳 429 速率限制，退避等待 {backoff} 秒 (嘗試 {attempt+1}/{max_retries})")
                            time.sleep(backoff)
                        else:
                            logger.warning(f"arXiv API 回應異常 HTTP {resp.status}")
                except Exception as e:
                    backoff = 2 ** attempt * 2
                    logger.warning(f"擷取 arXiv ({cat}) 失敗，退避等待 {backoff} 秒: {e}")
                    time.sleep(backoff)

            if not success:
                logger.error(f"無法自 arXiv 擷取分類 {cat} 論文。")

        return papers

    def _parse_arxiv_atom(self, xml_content: bytes, default_cat: str) -> List[Dict[str, Any]]:
        """解析 arXiv Atom XML 回應"""
        papers: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_content)
            atom_ns = "{http://www.w3.org/2005/Atom}"
            for entry in root.findall(f"{atom_ns}entry"):
                id_elem = entry.find(f"{atom_ns}id")
                title_elem = entry.find(f"{atom_ns}title")
                summary_elem = entry.find(f"{atom_ns}summary")
                published_elem = entry.find(f"{atom_ns}published")

                raw_id = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                paper_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
                title = " ".join((title_elem.text or "").split()) if title_elem is not None else ""
                abstract = " ".join((summary_elem.text or "").split()) if summary_elem is not None else ""
                pub_date = (published_elem.text or "")[:10] if published_elem is not None else ""

                authors = []
                for author in entry.findall(f"{atom_ns}author"):
                    name_elem = author.find(f"{atom_ns}name")
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                # 映射主題瓶頸標籤
                matched_themes = default_supply_chain_map.map_paper_to_themes(title, abstract)
                bottleneck_tag = matched_themes[0] if matched_themes else ""

                # 提取關聯代號
                related_tickers = []
                for t in matched_themes:
                    for item in default_supply_chain_map.get_supply_chain_for_theme(t):
                        related_tickers.append(item["ticker"])
                    related_tickers.extend(default_supply_chain_map.get_anchors_for_theme(t))

                papers.append({
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "category": default_cat,
                    "published_date": pub_date or datetime.date.today().isoformat(),
                    "bottleneck_tag": bottleneck_tag,
                    "related_tickers": list(dict.fromkeys(related_tickers))[:10],
                    "url": f"https://arxiv.org/abs/{paper_id}",
                    "significance_score": 85.0
                })
        except Exception as e:
            logger.error(f"解析 arXiv Atom XML 失敗: {e}")

        return papers

    def get_tracked_pool_tickers(self) -> Set[str]:
        """查詢系統目前追蹤池內之代號 (自 predictions 表讀取)"""
        tracked: Set[str] = set()
        try:
            if os.path.exists(self.db_manager.db_path):
                with self.db_manager._get_connection(read_only=True) as con:
                    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
                    if "predictions" in tables:
                        rows = con.execute("SELECT DISTINCT ticker FROM predictions").fetchall()
                        for r in rows:
                            if r[0]:
                                c = canonicalize_ticker(r[0])
                                if c:
                                    tracked.add(c)
        except Exception as e:
            logger.debug(f"查詢追蹤池股票失敗: {e}")
        return tracked

    def process_and_score(
        self,
        papers: List[Dict[str, Any]],
        as_of: str,
        run_id: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        將收集到的論文與主題供應鏈進行對照、代號正規化與白名單硬過濾，並進行 7 維評分。
        """
        tracked_pool = self.get_tracked_pool_tickers()
        candidates: List[Dict[str, Any]] = []
        signals: List[Dict[str, Any]] = []

        all_themes = default_supply_chain_map.get_all_themes()

        # 彙整各主題關聯的論文數量
        theme_paper_counts: Dict[str, int] = {}
        for p in papers:
            btag = p.get("bottleneck_tag", "")
            theme_paper_counts[btag] = theme_paper_counts.get(btag, 0) + 1

        for theme in all_themes:
            t_id = theme["id"]
            t_name = theme.get("name_zh", "")
            paper_cnt = theme_paper_counts.get(t_id, 0)
            us_anchors = default_supply_chain_map.get_anchors_for_theme(t_id)
            main_anchor = us_anchors[0] if us_anchors else "NVDA"

            for item in theme.get("tw_supply_chain", []):
                raw_ticker = item.get("ticker", "")
                canon_ticker = canonicalize_ticker(raw_ticker)
                if not canon_ticker:
                    continue

                # 白名單校驗
                if not default_validator.is_valid(canon_ticker):
                    logger.debug(f"標的 {canon_ticker} 未在 Universe 白名單中，予以過濾。")
                    continue

                market = detect_market(canon_ticker)
                name = item.get("name", "")
                role = item.get("role", "")
                relevance = float(item.get("relevance_weight", 0.8))
                in_pool = canon_ticker in tracked_pool

                # 7 維因子評估數值建構
                source_score = 90.0 if paper_cnt > 0 else 50.0
                purity_score = relevance * 100.0
                paper_score = min(100.0, 50.0 + paper_cnt * 10.0) if paper_cnt > 0 else 0.0
                novelty_score = 80.0 if not in_pool else 40.0
                sentiment_score = 85.0 if paper_cnt > 0 else 50.0
                setup_score = 75.0
                risk_score = 15.0 if in_pool else 25.0

                score_result = compute_discovery_score(
                    source_score=source_score,
                    purity_score=purity_score,
                    paper_score=paper_score,
                    novelty_score=novelty_score,
                    sentiment_score=sentiment_score,
                    setup_score=setup_score,
                    risk_score=risk_score,
                    in_pool=in_pool,
                )

                candidate_record = {
                    "ticker": canon_ticker,
                    "market": market,
                    "name": name,
                    "theme_id": t_id,
                    "theme_name": t_name,
                    "total_score": score_result.total_score,
                    "recommendation": score_result.recommendation,
                    "review_status": score_result.review_status,
                    "in_pool": in_pool,
                    "source_score": score_result.source_score,
                    "purity_score": score_result.purity_score,
                    "paper_score": score_result.paper_score,
                    "novelty_score": score_result.novelty_score,
                    "sentiment_score": score_result.sentiment_score,
                    "setup_score": score_result.setup_score,
                    "risk_score": score_result.risk_score,
                    "catalyst_summary": f"對齊前沿瓶頸：{role}，美股鏈接 {main_anchor}",
                    "us_anchor": main_anchor,
                    "paper_count": paper_cnt,
                    "signal_count": paper_cnt,
                    "run_id": run_id,
                    "as_of": as_of,
                }
                candidates.append(candidate_record)

                # 生成關聯產業訊號 (僅在有論文訊號時生成，避免假造)
                if paper_cnt > 0:
                    sig_id = f"sig_{as_of.replace('-', '')}_{t_id}_{canon_ticker}"
                    signals.append({
                        "signal_id": sig_id,
                        "ticker": canon_ticker,
                        "theme_id": t_id,
                        "source_type": "arxiv",
                        "title": f"【{t_name}】前沿突破映射標的 {name} ({canon_ticker})",
                        "content": f"{name} 於 {t_name} 領域扮演關鍵角色：{role}。錨點帶動：{main_anchor}。",
                        "url": f"https://arxiv.org/abs/search?query={t_id}",
                        "sentiment_score": sentiment_score,
                        "detected_at": datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S"),
                        "run_id": run_id,
                        "as_of": as_of,
                    })

        # 依 total_score 排序
        candidates.sort(key=lambda x: x["total_score"], reverse=True)
        return candidates, papers, signals

    def publish_atomic_cache(
        self,
        run_id: str,
        as_of: str,
        candidates: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
        signals: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
        status: Dict[str, Any],
    ) -> bool:
        """
        以 tempfile + os.replace 執行兩階段原子快取替換，消除快取不一致與檔案損壞。
        """
        payloads = {
            "candidates.json": {
                "success": True,
                "total_count": len(candidates),
                "as_of": as_of,
                "stale": status.get("stale", False),
                "data": candidates,
            },
            "papers.json": {
                "success": True,
                "total_count": len(papers),
                "as_of": as_of,
                "stale": status.get("stale", False),
                "data": papers,
            },
            "signals.json": {
                "success": True,
                "total_count": len(signals),
                "as_of": as_of,
                "stale": status.get("stale", False),
                "data": signals,
            },
            "history.json": {
                "success": True,
                "total_count": len(history),
                "data": history,
            },
            "status.json": {
                "success": True,
                "data": status,
            },
        }

        staging: List[Tuple[str, str]] = []
        try:
            # 兩階段寫入：先將所有暫存檔寫出並 fsync，確認全部無誤後再原子替換
            for filename, data in payloads.items():
                target_path = os.path.join(self.cache_dir, filename)
                temp_file = tempfile.NamedTemporaryFile(
                    "w",
                    dir=self.cache_dir,
                    delete=False,
                    encoding="utf-8",
                    suffix=".tmp"
                )
                try:
                    json.dump(data, temp_file, ensure_ascii=False, indent=2)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                    temp_file.close()
                    staging.append((temp_file.name, target_path))
                except Exception as file_err:
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                    raise file_err

            # 全部寫入成功，一次原子替換所有檔案
            for temp_name, target_name in staging:
                os.replace(temp_name, target_name)

            logger.info(f"⚡ 成功原子替換靜態 JSON 快照至 {self.cache_dir}")
            return True
        except Exception as e:
            for temp_name, _ in staging:
                if os.path.exists(temp_name):
                    try:
                        os.unlink(temp_name)
                    except Exception:
                        pass
            logger.error(f"❌ 快取原子發布失敗: {e}", exc_info=True)
            return False

    def load_fallback_cache(self) -> Dict[str, Any]:
        """
        當網路或上游完全不可用時，載入既有快照並將所有檔案標記 stale=True。
        """
        fallback_files = ["status.json", "candidates.json", "papers.json", "signals.json"]
        status_data = {}

        for fname in fallback_files:
            fpath = os.path.join(self.cache_dir, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = json.load(f)

                    content["stale"] = True
                    if fname == "status.json":
                        data = content.get("data", {})
                        data["status"] = "fallback"
                        data["stale"] = True
                        data["message"] = "網路或上游逾時，降級載入前次有效快照。"
                        content["data"] = data
                        status_data = data

                    temp_file = tempfile.NamedTemporaryFile("w", dir=self.cache_dir, delete=False, encoding="utf-8", suffix=".tmp")
                    json.dump(content, temp_file, ensure_ascii=False, indent=2)
                    temp_file.flush()
                    temp_file.close()
                    os.replace(temp_file.name, fpath)
                except Exception as e:
                    logger.warning(f"更新 Fallback 檔案 {fname} 失敗: {e}")

        if status_data:
            logger.warning("已啟動 Discovery 快照 Fallback 降級模式，所有快取已同步標記 stale=True。")
            return status_data

        # 無舊快照可用的極端情況
        return {
            "status": "fallback",
            "stale": True,
            "as_of": datetime.date.today().isoformat(),
            "candidate_count": 0,
            "paper_count": 0,
            "signal_count": 0,
            "message": "無可用之 Discovery 歷史快照。"
        }

    def run(
        self,
        as_of: Optional[str] = None,
        mock_papers: Optional[List[Dict[str, Any]]] = None,
        force_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        執行 Discovery Pipeline 完整流程：
        1. 抓取 arXiv 論文 (或使用注入/快照資料)
        2. 語意解析、主題瓶頸映射、代號白名單硬檢驗與 7 維度評分
        3. Phase 1: DuckDB 交易批次寫入
        4. Phase 2: tempfile + os.replace 原子快照發布
        5. TTL 過期資料清理
        """
        target_as_of = as_of or datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d")
        run_id = str(uuid.uuid4())

        if force_fallback:
            return self.load_fallback_cache()

        try:
            # 1. 抓取或使用指定論文
            if mock_papers is not None:
                papers = mock_papers
            else:
                try:
                    papers = self.fetch_arxiv_papers(max_results_per_cat=5, throttle_seconds=0.5)
                except Exception as fetch_err:
                    logger.warning(f"爬取 arXiv 遇到異常: {fetch_err}，嘗試使用 Fallback。")
                    return self.load_fallback_cache()

            # 若未抓到任何論文 (如全數 429 失敗或網路斷線)，觸發 Fallback 降級
            if not papers:
                logger.warning("未自 arXiv 取得任何有效論文，觸發 Fallback 降級模式。")
                return self.load_fallback_cache()

            # 2. 評分與映射
            candidates, papers, signals = self.process_and_score(papers, target_as_of, run_id)

            # 計算各推薦分類統計
            propose_add_count = sum(1 for c in candidates if c["recommendation"] == "propose_add")
            observe_count = sum(1 for c in candidates if c["recommendation"] == "observe")
            already_in_pool_count = sum(1 for c in candidates if c["recommendation"] == "already_in_pool")
            avg_score = sum(c["total_score"] for c in candidates) / len(candidates) if candidates else 0.0

            history_record = {
                "run_id": run_id,
                "as_of": target_as_of,
                "total_candidates": len(candidates),
                "propose_add_count": propose_add_count,
                "observe_count": observe_count,
                "already_in_pool_count": already_in_pool_count,
                "paper_count": len(papers),
                "signal_count": len(signals),
                "avg_total_score": round(avg_score, 2),
            }

            # 3. Phase 1: DuckDB 交易式批次寫入
            db_success = self.db_manager.save_discovery_snapshot(
                run_id=run_id,
                as_of=target_as_of,
                candidates=candidates,
                papers=papers,
                signals=signals,
                history_summary=history_record,
            )

            # 4. Phase 2: 原子快照發布
            status_data = {
                "status": "success" if db_success else "fallback",
                "stale": not db_success,
                "as_of": target_as_of,
                "run_id": run_id,
                "last_run_timestamp": datetime.datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S"),
                "candidate_count": len(candidates),
                "paper_count": len(papers),
                "signal_count": len(signals),
                "message": "管線運作正常" if db_success else "DuckDB 寫入失敗，使用快取降級",
            }

            history_list = self.db_manager.get_discovery_history(limit=30)
            if not history_list:
                history_list = [history_record]

            self.publish_atomic_cache(
                run_id=run_id,
                as_of=target_as_of,
                candidates=candidates,
                papers=papers,
                signals=signals,
                history=history_list,
                status=status_data,
            )

            # 5. TTL 清理
            self.db_manager.prune_discovery_expired_records(paper_ttl_days=180, signal_ttl_days=90)

            return status_data

        except Exception as e:
            logger.error(f"Discovery 管線執行中斷: {e}", exc_info=True)
            return self.load_fallback_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("🚀 啟動 AI 產業鏈前沿挖掘機 (Discovery Pipeline)...")
    pipe = DiscoveryPipeline()
    result = pipe.run()
    logger.info(f"🏁 Discovery Pipeline 執行結果: {result}")
