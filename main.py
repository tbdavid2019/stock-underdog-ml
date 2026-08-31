"""
股票多策略分析系統主程序 (Stock Prediction & Multi-Strategy Application)
採用模組化兩階段管線架構，整合硬體自動偵測、批次 I/O 預載、可擴充策略插件與綜合評分。
"""
import datetime
import sys
from core.config import config
from core.device import DeviceManager
from logger import logger
from pipeline.orchestrator import PipelineOrchestrator


def main():
    """主程序入口"""
    try:
        start_time = datetime.datetime.now()
        device_info = DeviceManager.get_device_info()

        logger.info("=" * 70)
        logger.info("🚀 啟動股票多策略量化分析系統...")
        logger.info(f"⏰ 執行時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"💻 運算設備: {device_info['name']} ({device_info['device']})")
        logger.info(f"📋 啟用策略: {', '.join(config.pipeline.ENABLED_STRATEGIES)}")
        logger.info("=" * 70)

        # 初始化兩階段管線排程器
        orchestrator = PipelineOrchestrator()

        # 執行所有目標指數分析
        orchestrator.run_all_indices(period=config.pipeline.DEFAULT_PERIOD)

        elapsed = datetime.datetime.now() - start_time
        logger.info(f"✅ 雙軌量化策略分析全數完成！總耗時: {elapsed.total_seconds():.1f} 秒\n")

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
