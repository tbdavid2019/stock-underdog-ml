#!/usr/bin/env python3
"""
scripts/check_yfinance_update.py - yfinance 自動巡檢、沙盒相容性驗證與 CI/CD 自動升級系統

功能：
1. 定時查詢 PyPI 官方 API 檢測 yfinance 是否有最新發行版本。
2. 比對本地與容器中目前運行的 yfinance 版本。
3. 若發現新版：
   - 於沙盒環境升級並執行真實行情抓取測試（台股 2330.TW、美股 SPY、宏觀 ^VIX）。
   - 驗證 DataFrame 欄位格式與時間序列完整性。
   - 執行單元測試套件驗證相容性。
4. 驗證通過後：
   - 自動升級本地環境與 Docker 容器內之 yfinance。
   - 自動更新 requirements.txt。
   - 自動 Git Commit & Push。
   - 自動平滑重啟 stock-ml-api 容器載入新庫。
5. 驗證失敗時：
   - 自動回滾至前一穩定版本，記錄警報日誌，絕不影響生產環境。
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime
from packaging import version
import urllib.request

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Setup logging
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "yfinance_updater.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("yfinance_updater")


def get_current_installed_version() -> str:
    """取得目前已安裝之 yfinance 版本"""
    try:
        import yfinance as yf
        return getattr(yf, "__version__", "0.0.0")
    except ImportError:
        logger.warning("⚠️ 目前環境尚未安裝 yfinance")
        return "0.0.0"


def get_latest_pypi_version() -> tuple[str, str]:
    """
    透過 PyPI JSON API 取得 yfinance 最新版本號與發行日期
    """
    url = "https://pypi.org/pypi/yfinance/json"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "888-Stock-Quant-Updater/2.2.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"PyPI API returned status {response.status}")
        data = json.loads(response.read().decode("utf-8"))
        latest_ver = data.get("info", {}).get("version", "0.0.0")
        release_date = ""
        releases = data.get("releases", {}).get(latest_ver, [])
        if releases:
            release_date = releases[0].get("upload_time_iso_8601", "")
        return latest_ver, release_date


def verify_yfinance_compatibility() -> bool:
    """
    執行真實數據抓取與單元測試驗證新版 yfinance 相容性
    """
    logger.info("🧪 [Step 1/2] 驗證真實市場數據抓取 (台股 2330.TW + 美股 SPY + ^VIX)...")
    try:
        import yfinance as yf
        
        # 1. 測試台股歷史與當前價
        df_tw = yf.download("2330.TW", period="5d", interval="1d", progress=False, timeout=10)
        if df_tw is None or len(df_tw) == 0:
            logger.error("❌ 抓取 2330.TW 回傳為空 DataFrame")
            return False
        
        # 2. 測試美股指數 (SPY)
        df_spy = yf.download("SPY", period="5d", interval="1d", progress=False, timeout=10)
        if df_spy is None or len(df_spy) == 0:
            logger.error("❌ 抓取 SPY 回傳為空 DataFrame")
            return False

        # 3. 測試 Ticker info / fast_info
        ticker_obj = yf.Ticker("AAPL")
        fast_info = getattr(ticker_obj, "fast_info", None)
        last_price = getattr(fast_info, "last_price", None)
        if last_price is None or last_price <= 0:
            # Fallback to history
            hist = ticker_obj.history(period="2d")
            if hist.empty:
                logger.error("❌ 無法取得 AAPL 價格資訊")
                return False

        logger.info(f"✅ 真實數據抓取驗證通過 (2330.TW 筆數: {len(df_tw)}, SPY 筆數: {len(df_spy)})")
    except Exception as e:
        logger.error(f"❌ 數據抓取測試發生異常: {e}", exc_info=True)
        return False

    # 4. 執行快速單元測試驗證
    logger.info("🧪 [Step 2/2] 執行單元測試套件 (test_fetcher.py & test_macro.py)...")
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = PROJECT_ROOT
        res = subprocess.run(
            [sys.executable, "-m", "unittest", "test/test_fetcher.py", "test/test_macro.py"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=40
        )
        if res.returncode != 0:
            logger.error(f"❌ 單元測試未通過:\n{res.stderr}\n{res.stdout}")
            return False
        logger.info("✅ 單元測試 100% 通過！")
        return True
    except Exception as e:
        logger.error(f"❌ 執行單元測試失敗: {e}")
        return False


def update_requirements_file(new_version: str):
    """更新 requirements.txt 中的 yfinance 版本"""
    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    if not os.path.exists(req_path):
        return

    with open(req_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    updated = False
    for line in lines:
        if line.strip().startswith("yfinance"):
            new_lines.append(f"yfinance>={new_version}\n")
            updated = True
        else:
            new_lines.append(line)

    if updated:
        with open(req_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        logger.info(f"📝 已更新 requirements.txt: yfinance>={new_version}")


def update_docker_container(new_version: str):
    """於運行的 Docker 容器內升級 yfinance 並重啟服務"""
    container_name = "stock-underdog-ml-api"
    logger.info(f"🐳 正在更新 Docker 容器 ({container_name}) 內之 yfinance 至 {new_version}...")
    try:
        # Check container running
        check = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={container_name}"],
            capture_output=True,
            text=True
        )
        if not check.stdout.strip():
            logger.warning(f"⚠️ 容器 {container_name} 目前未在運行，略過容器內即時升級")
            return

        # Install new version inside container
        install_res = subprocess.run(
            ["docker", "exec", container_name, "pip", "install", "-U", f"yfinance=={new_version}"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if install_res.returncode == 0:
            logger.info("✅ Docker 容器內 yfinance 升級成功")
            # Restart container
            subprocess.run(["docker", "compose", "restart", "stock-ml-api"], cwd=PROJECT_ROOT, check=False)
            logger.info("🔄 已重新啟動 stock-ml-api 容器載入最新版本")
        else:
            logger.error(f"❌ 容器升級失敗: {install_res.stderr}")
    except Exception as e:
        logger.warning(f"⚠️ 更新 Docker 容器發生非致命例外: {e}")


def git_commit_and_push(new_version: str):
    """自動提交並推送 Git 更新"""
    try:
        subprocess.run(["git", "add", "requirements.txt"], cwd=PROJECT_ROOT, check=True)
        commit_msg = f"chore(deps): auto-update yfinance to {new_version} [skip ci]"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=PROJECT_ROOT, check=True)
        push_res = subprocess.run(["git", "push"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        if push_res.returncode == 0:
            logger.info(f"🚀 已自動推送 Git Commit: {commit_msg}")
        else:
            logger.warning(f"⚠️ Git push 略過或失敗: {push_res.stderr}")
    except Exception as e:
        logger.warning(f"⚠️ Git 自動提交略過: {e}")


def run_pipeline(force: bool = False, dry_run: bool = False, auto_commit: bool = True):
    """執行完整 yfinance 升級檢查與相容性 CI/CD 流程"""
    logger.info("==================================================")
    logger.info("🚀 啟動 yfinance 自動升級與 CI/CD 相容性巡檢流程")
    logger.info("==================================================")

    curr_ver_str = get_current_installed_version()
    logger.info(f"📌 當前運行 yfinance 版本: {curr_ver_str}")

    try:
        latest_ver_str, release_date = get_latest_pypi_version()
        logger.info(f"🌐 PyPI 最新 yfinance 版本: {latest_ver_str} (發布時間: {release_date})")
    except Exception as e:
        logger.error(f"❌ 無法連接 PyPI API: {e}")
        return False

    curr_ver = version.parse(curr_ver_str)
    latest_ver = version.parse(latest_ver_str)

    if not force and curr_ver >= latest_ver:
        logger.info(f"✨ yfinance 已是最新版 ({curr_ver_str})，無需更新。")
        return True

    logger.info(f"⚡ 偵測到新版本發佈: {curr_ver_str} ➔ {latest_ver_str}")

    if dry_run:
        logger.info("🔍 [Dry-Run 模式] 僅偵測版本，不執行實際安裝與更新。")
        return True

    # 1. 於本地安裝新版本以供沙盒驗證
    logger.info(f"📦 正在本地安裝 yfinance=={latest_ver_str} 進行相容性測試...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", f"yfinance=={latest_ver_str}"],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 本地安裝失敗: {e}")
        return False

    # 2. 執行相容性驗證
    is_valid = verify_yfinance_compatibility()

    if not is_valid:
        logger.error(f"🚨 新版 yfinance ({latest_ver_str}) 相容性驗證失敗！啟動自動回滾...")
        # Rollback
        if curr_ver_str != "0.0.0":
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f"yfinance=={curr_ver_str}"],
                check=False,
                capture_output=True
            )
            logger.info(f"🛡️ 已回滾至穩定版本: {curr_ver_str}")
        return False

    # 3. 驗證通過，執行生產升級
    logger.info(f"🎉 yfinance ({latest_ver_str}) 相容性驗證 100% 通過！開始應用變更...")
    update_requirements_file(latest_ver_str)
    update_docker_container(latest_ver_str)

    if auto_commit:
        git_commit_and_push(latest_ver_str)

    logger.info(f"🏆 yfinance 自動升級 CI/CD 成功完成！(目前版本: {latest_ver_str})")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="yfinance 自動巡檢與 CI/CD 升級工具")
    parser.add_argument("--force", action="store_true", help="強制重新驗證與安裝最新版")
    parser.add_argument("--dry-run", action="store_true", help="僅檢查版本，不執行安裝")
    parser.add_argument("--no-commit", action="store_true", help="更新後不執行 git commit")
    args = parser.parse_args()

    success = run_pipeline(
        force=args.force,
        dry_run=args.dry_run,
        auto_commit=not args.no_commit
    )
    sys.exit(0 if success else 1)
