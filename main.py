import argparse
import datetime
import sys
from core.config import config
from core.device import DeviceManager
from logger import logger
from pipeline.orchestrator import PipelineOrchestrator


def resolve_market(market_arg: str) -> str:
    """
    智慧解析目標市場:
    若指定為 auto (預設):
      - 依據當前 Asia/Taipei 台北時間判定:
      - 05:00 ~ 13:30 -> tw (台股開盤前 08:00 與盤中)
      - 13:30 ~ 05:00 -> us (美股開盤前 20:30 與美股時段)
      - 嚴格落實「白天跑台股、晚上跑美股」，杜絕夜間誤跑台股浪費運算資源
    """
    if market_arg.lower() != "auto":
        return market_arg.lower()

    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("Asia/Taipei")
        now_taipei = datetime.datetime.now(tz)
    except Exception:
        now_taipei = datetime.datetime.now()

    total_minutes = now_taipei.hour * 60 + now_taipei.minute
    if 300 <= total_minutes < 810:
        return "tw"
    else:
        return "us"


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="股票多維量化策略分析系統 (Pre-Market Quantitative Strategy Engine)"
    )
    parser.add_argument(
        "--market", "-m",
        choices=["auto", "tw", "us", "all"],
        default="auto",
        help="目標市場: auto (依台北時間智慧判定: 白天台股/晚上美股，預設), tw (台股盤前 08:00), us (美股盤前 20:30), all (全市場，僅限手動指定)"
    )
    parser.add_argument(
        "--index", "-i",
        nargs="+",
        help="指定單一或多個指數 (例如: 台灣50 SP500)"
    )
    parser.add_argument(
        "--period", "-p",
        default=config.pipeline.DEFAULT_PERIOD,
        help="歷史行情回溯週期 (預設: 6mo)"
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="略過寫入 Supabase 雲端與 DuckDB 本地資料庫"
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="略過發送 Telegram / Discord / Email 通知"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="乾跑模式 (不寫入 DB 且不發送任何通知)"
    )
    return parser.parse_args()


def main():
    """主程序入口"""
    args = parse_args()
    try:
        start_time = datetime.datetime.now()
        device_info = DeviceManager.get_device_info()

        logger.info("=" * 70)
        logger.info("🚀 啟動股票多策略量化分析系統 (Pre-Market Trading Guide)...")
        logger.info(f"⏰ 執行時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        target_market = resolve_market(args.market)
        market_label = f"{args.market.upper()} (自動解析: {target_market.upper()})" if args.market.lower() == "auto" else args.market.upper()
        target_display = ", ".join(args.index) if args.index else market_label
        logger.info(f"🎯 目標市場/指數: {target_display}")
        logger.info(f"💻 運算設備: {device_info['name']} ({device_info['device']})")
        logger.info(f"📋 啟用策略: {', '.join(config.pipeline.ENABLED_STRATEGIES)}")
        logger.info("=" * 70)

        persist_db = not (args.no_db or args.dry_run)
        send_notify = not (args.no_notify or args.dry_run)

        # 初始化兩階段管線排程器
        orchestrator = PipelineOrchestrator()

        # 執行目標指數分析 (嚴格依據解析後的目標市場執行)
        orchestrator.run_all_indices(
            period=args.period,
            persist_db=persist_db,
            send_notify=send_notify,
            market=target_market,
            index_names=args.index
        )

        elapsed = datetime.datetime.now() - start_time
        logger.info(f"✅ 量化策略分析全數完成！總耗時: {elapsed.total_seconds():.1f} 秒\n")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ 收到使用者中斷信號 (Ctrl+C)，程式終止")
        sys.exit(0)
    except Exception as e:
        logger.error(f"⚠️ 主程序發生未預期錯誤: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
