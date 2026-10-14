"""
api/routes/podcast.py - AI Financial & Tech Podcast Intelligence Radar REST API
提供單集情報檢索、實體快篩、行情增強、收錄頻道列表與熱門話題頻次排行。
"""

import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, Path, HTTPException, BackgroundTasks

from api.schemas import (
    PodcastEpisodesResponse,
    PodcastChannelsResponse,
    PodcastTopicsResponse,
    PodcastSyncResponse,
    PodcastEpisodeItem,
    PodcastChannelItem,
    PodcastTopicItem
)
from data.duckdb_manager import DuckDBManager
from data.podcast_service import PodcastService

router = APIRouter(prefix="/podcast", tags=["Podcast Radar 財經情報雷達"])

duckdb_manager = DuckDBManager()
podcast_service = PodcastService(duckdb_manager)


@router.get("/episodes", response_model=PodcastEpisodesResponse, summary="取得 Podcast 單集情報列表與量化資料快照")
def get_episodes(
    topic: Optional[str] = Query(None, description="依實體或主題篩選 (例如 Claude, OpenAI, NVIDIA, TSMC)"),
    ticker: Optional[str] = Query(None, description="依股票代號篩選 (例如 NVDA, 2330.TW)"),
    channel_id: Optional[str] = Query(None, description="依特定頻道 ID 篩選 (例如 all-in, odd-lots)"),
    lang: Optional[str] = Query("ALL", description="語言篩選: ALL / EN / ZH"),
    limit: int = Query(100, ge=1, le=200, description="每頁筆數"),
    offset: int = Query(0, ge=0, description="分頁偏移量")
):
    """
    查詢單集深度情報，包含主持人/來賓、AI 繁中精煉摘要、關鍵觀點標籤與即時關聯股票 Badge。
    """
    episodes = duckdb_manager.get_podcast_episodes(
        topic=topic,
        ticker=ticker,
        channel_id=channel_id,
        lang=lang,
        limit=limit,
        offset=offset
    )

    # 僅在資料庫完全為空或少於配置種子數時，進行冷啟動同步
    if not any([topic, ticker, channel_id]) and (not lang or lang.upper() == "ALL") and offset == 0:
        if duckdb_manager.get_total_episodes_count() < 50:
            podcast_service.sync_all()
            episodes = duckdb_manager.get_podcast_episodes(
                topic=topic,
                ticker=ticker,
                channel_id=channel_id,
                lang=lang,
                limit=limit,
                offset=offset
            )

    # 批次進行即時行情與時長格式化增強 (一次性 SQL 查詢消除 N+1)
    enriched = podcast_service.enrich_episodes_batch(episodes)

    return PodcastEpisodesResponse(
        success=True,
        total_count=len(enriched),
        data=enriched
    )


@router.get("/channels", response_model=PodcastChannelsResponse, summary="取得收錄之頂級財經與科技 Podcast 頻道清單")
def get_channels(
    limit: int = Query(200, ge=1, le=500, description="最多返回頻道數"),
    refresh: bool = Query(False, description="強制重新同步種子頻道元數據")
):
    """
    取得已收錄的國際頂級科技、宏觀金融與商業頻道庫元數據。
    """
    channels = duckdb_manager.get_podcast_channels(limit=limit)
    seed_channels = podcast_service.get_seed_channels()
    if refresh or len(channels) != len(seed_channels):
        podcast_service.sync_seed_channels()
        channels = duckdb_manager.get_podcast_channels(limit=limit)

    # 嚴格去重頻道標題，確保 UI 下拉選單絕無重複項 (Deduplicate by Title)
    seen_titles = set()
    unique_channels = []
    for ch in channels:
        t_key = ch.get("title", "").strip().lower()
        if t_key and t_key in seen_titles:
            continue
        seen_titles.add(t_key)
        ch["source"] = "seed_snapshot"
        unique_channels.append(ch)

    return PodcastChannelsResponse(
        success=True,
        total_count=len(unique_channels),
        data=unique_channels
    )


@router.get("/topics", response_model=PodcastTopicsResponse, summary="取得熱門實體與話題頻次排行榜")
def get_topics(
    limit: int = Query(30, ge=5, le=100, description="最多返回標籤數")
):
    """
    取得近期各 Podcast 單集最頻繁討論的核心科技實體（如 Claude, OpenAI, NVIDIA 等）。
    """
    topics = duckdb_manager.get_podcast_topics_aggregation(limit=limit)
    if not topics and duckdb_manager.get_total_episodes_count() == 0:
        podcast_service.sync_all()
        topics = duckdb_manager.get_podcast_topics_aggregation(limit=limit)

    return PodcastTopicsResponse(
        success=True,
        total_count=len(topics),
        data=topics
    )


@router.post("/sync", response_model=PodcastSyncResponse, summary="觸發 Podcast 頻道與單集情報非同步同步")
def trigger_podcast_sync(background_tasks: BackgroundTasks):
    """
    手動或排程觸發最新 Podcast RSS Feed 擷取與 AI 萃取。
    """
    background_tasks.add_task(podcast_service.sync_all, fetch_live_rss=True)
    return PodcastSyncResponse(
        success=True,
        message="Podcast 頻道與單集同步任務已於背景排隊執行",
        timestamp=datetime.datetime.now().isoformat()
    )


@router.post("/episodes/{episode_id}/evaluate", summary="對特定 Podcast 單集執行 5-Tier 決策推論評估")
def evaluate_episode(
    episode_id: str = Path(..., description="單集 ID"),
    force: bool = Query(False, description="是否強制重新推論")
):
    """
    針對單集調用 5 級決策鏈（Clef/JEV/OpenAI/LocalRules），
    計算市場重要性評級與提及標的多空立場。即使所有遠端模型故障，
    也能於 2ms 內無縫降級至本地規則返回結果。
    """
    res = podcast_service.evaluate_episode_intelligence(episode_id, force=force)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("error", "單集評估失敗"))
    return res


@router.post("/batch-evaluate", summary="批量執行尚未評估之單集決策推論")
def batch_evaluate_episodes(
    limit: int = Query(20, ge=1, le=100, description="批次處理上限")
):
    """
    批次掃描尚未擁有重要性評級的單集，執行 5-Tier 決策評估並寫入快取。
    """
    return podcast_service.evaluate_episodes_batch(limit=limit)

