#!/usr/bin/env python3
"""
scripts/check_env.py - 888 StockQuant Platform Environment Variables Doctor

分類檢測環境變數（必填、選填降級說明、WATCHLIST_API_KEY 生產保護、已棄用設定），
100% 遮罩所有敏感憑證（僅輸出 [SET] / [MISSING]），並支援 --json 機器可讀輸出。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load local .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def is_set(var_name: str) -> bool:
    val = os.getenv(var_name)
    return bool(val and val.strip())


def run_env_check() -> Dict[str, Any]:
    deployment_mode = os.getenv("DEPLOYMENT_MODE", "production").lower()
    allow_anonymous = os.getenv("ALLOW_ANONYMOUS_LOCAL_WRITES", "false").lower() in ("true", "1", "yes")

    # 1. 必填變數檢查 (Required)
    required_checks = []
    has_required_error = False

    # Supabase URL & Key (至少需要 URL 與一把 Key)
    from urllib.parse import urlparse
    raw_sp_url = (os.getenv("SUPABASE_URL") or "").strip()
    sp_url_valid = False
    sp_url_err = None
    if raw_sp_url:
        try:
            parsed = urlparse(raw_sp_url)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                sp_url_valid = True
            else:
                sp_url_err = f"無效的 URL 格式（scheme 必須為 http 或 https 且具備有效網域名稱，目前 scheme='{parsed.scheme}' netloc='{parsed.netloc}'）"
        except Exception as e:
            sp_url_err = str(e)
    else:
        sp_url_err = "環境變數未設定"

    has_sp_key = is_set("SUPABASE_SERVICE_KEY") or is_set("SUPABASE_KEY")
    
    required_checks.append({
        "name": "SUPABASE_URL",
        "status": "SET" if sp_url_valid else ("INVALID" if raw_sp_url else "MISSING"),
        "description": "Supabase 雲端資料庫 URL",
        "valid": sp_url_valid,
        "error": sp_url_err if not sp_url_valid else None
    })
    if not sp_url_valid:
        has_required_error = True

    required_checks.append({
        "name": "SUPABASE_SERVICE_KEY / SUPABASE_KEY",
        "status": "SET" if has_sp_key else "MISSING",
        "description": "Supabase API 金鑰 (服務端或發布金鑰)",
        "valid": has_sp_key
    })
    if not has_sp_key:
        has_required_error = True

    # DuckDB Path (若啟用)
    enable_duckdb = os.getenv("ENABLE_DUCKDB", "true").lower() in ("true", "1", "yes")
    duckdb_path = os.getenv("DUCKDB_PATH", "data/storage/stock_quant.duckdb")
    required_checks.append({
        "name": "ENABLE_DUCKDB",
        "status": "SET" if enable_duckdb else "DISABLED",
        "description": f"本地 DuckDB 列式儲存 ({duckdb_path})",
        "valid": True
    })

    # 2. 選填變數與平滑降級說明 (Optional with Graceful Degradation)
    optional_checks = [
        {
            "category": "AI LLM 每日操盤解讀",
            "names": ["LLM_PRIMARY_API_KEY", "GEMINI_API_KEY"],
            "status": "SET" if (is_set("LLM_PRIMARY_API_KEY") or is_set("GEMINI_API_KEY")) else "MISSING",
            "degradation": "若未設定，自動平滑降級為 DeepSeek / OpenAI 或本地確定性操盤規則模板"
        },
        {
            "category": "AI LLM 備援 1 (DeepSeek)",
            "names": ["LLM_FALLBACK1_API_KEY", "DEEPSEEK_API_KEY"],
            "status": "SET" if (is_set("LLM_FALLBACK1_API_KEY") or is_set("DEEPSEEK_API_KEY")) else "MISSING",
            "degradation": "選填次級備援；未設定時直接略過"
        },
        {
            "category": "Discord 操盤訊號推播",
            "names": ["DISCORD_WEBHOOK_URL"],
            "status": "SET" if is_set("DISCORD_WEBHOOK_URL") else "MISSING",
            "degradation": "若未設定，停用 Discord 即時推播"
        },
        {
            "category": "Telegram 操盤訊號推播",
            "names": ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHANNEL_ID"],
            "status": "SET" if (is_set("TELEGRAM_BOT_TOKEN") and is_set("TELEGRAM_CHANNEL_ID")) else "MISSING",
            "degradation": "若未設定，停用 Telegram 機器人推播"
        },
        {
            "category": "Email 每日報告通知",
            "names": ["SMTP_SERVER", "SENDER_EMAIL", "EMAIL_PASSWORD", "TO_EMAILS"],
            "status": "SET" if (is_set("SMTP_SERVER") and is_set("SENDER_EMAIL")) else "MISSING",
            "degradation": "若未設定，停用 Email 報告寄送"
        },
        {
            "category": "Clef 系統一決策推論金鑰",
            "names": ["DECISION_API_KEY", "CLEF_API_KEY"],
            "status": "SET" if (is_set("DECISION_API_KEY") or is_set("CLEF_API_KEY")) else "MISSING",
            "degradation": "若未設定，決策推論 API 在本機開放或由全域金鑰保護"
        },
        {
            "category": "外部 2MD 抓取多金鑰池",
            "names": ["INVESTING_2MD_KEYS", "CREATE360_API_KEY"],
            "status": "SET" if (is_set("INVESTING_2MD_KEYS") or is_set("CREATE360_API_KEY")) else "MISSING",
            "degradation": "若未設定，使用預設免金鑰公共節點輪詢"
        }
    ]

    # 3. 安全與自選股防護 (Security & Watchlist Guard)
    has_watchlist_key = is_set("WATCHLIST_API_KEY")
    security_info = {
        "DEPLOYMENT_MODE": deployment_mode,
        "WATCHLIST_API_KEY": "SET" if has_watchlist_key else "MISSING",
        "ALLOW_ANONYMOUS_LOCAL_WRITES": allow_anonymous,
    }

    warnings = []
    if deployment_mode == "production" and not has_watchlist_key:
        warnings.append(
            "⚠️ [Fail-Closed 防護啟動] 生產模式 (DEPLOYMENT_MODE=production) 下未設定 WATCHLIST_API_KEY：\n"
            "   後端自選股與量化告警寫入/巡檢端點 (POST/DELETE /api/v1/watchlist*, /api/v1/alerts*) 將全面鎖定 (HTTP 401)。\n"
            "   公開訪客仍可無縫使用瀏覽器 localStorage (sq_local_watchlist_v1) 享受免登入自選體驗。"
        )
    if deployment_mode == "production" and allow_anonymous:
        warnings.append(
            "⚠️ 檢測到 ALLOW_ANONYMOUS_LOCAL_WRITES=true，但在 DEPLOYMENT_MODE=production 模式下此設定已被強制忽略。"
        )

    # 4. 已棄用設定檢測 (Deprecated variables)
    deprecated_vars = [
        ("MONGODB_URI", "已全面遷移至 DuckDB 列式資料庫，請自 .env 移除"),
        ("MONGO_URI", "已全面遷移至 DuckDB 列式資料庫，請自 .env 移除"),
        ("MYSQL_HOST", "專案無 MySQL 依賴，請自 .env 移除"),
        ("MYSQL_URL", "專案無 MySQL 依賴，請自 .env 移除"),
        ("TENSORFLOW_MODEL_PATH", "已遷移至 TimesFM / Timestral PyTorch 基礎模型，請自 .env 移除"),
        ("USE_LSTM_LEGACY", "舊版 LSTM 模型已被時序基礎模型取代，請自 .env 移除")
    ]
    deprecated_found = []
    for var_name, reason in deprecated_vars:
        if is_set(var_name):
            deprecated_found.append({"name": var_name, "reason": reason})

    overall_status = "error" if has_required_error else ("warning" if warnings else "ok")

    return {
        "status": overall_status,
        "deployment_mode": deployment_mode,
        "required": required_checks,
        "optional": optional_checks,
        "security": security_info,
        "warnings": warnings,
        "deprecated": deprecated_found
    }


def print_cli_report(report: Dict[str, Any]) -> None:
    print("=" * 78)
    print("🩺 888 StockQuant Platform - 環境設定健檢 (Environment Doctor)")
    print("=" * 78)
    print(f"運行環境模式 (DEPLOYMENT_MODE): {report['deployment_mode'].upper()}")
    print("")

    # 必填
    print("📌 [1/4] 核心必填服務設定 (Required Settings)")
    for item in report["required"]:
        icon = "✅" if item["valid"] else "❌"
        print(f"  {icon} {item['name']:<35} : [{item['status']}] - {item['description']}")
    print("")

    # 選填
    print("🧩 [2/4] 選填功能與自動降級策略 (Optional Settings & Graceful Degradation)")
    for item in report["optional"]:
        icon = "✨" if item["status"] == "SET" else "⚪"
        names_str = "/".join(item["names"])
        print(f"  {icon} {item['category']:<24} [{item['status']}]: {names_str}")
        if item["status"] == "MISSING":
            print(f"     ↳ 💡 降級模式: {item['degradation']}")
    print("")

    # 安全防護
    print("🛡️ [3/4] 安全授權與自選股防護 (Security & Watchlist Guard)")
    sec = report["security"]
    wl_icon = "🔒" if sec["WATCHLIST_API_KEY"] == "SET" else "🛡️"
    print(f"  {wl_icon} WATCHLIST_API_KEY                : [{sec['WATCHLIST_API_KEY']}]")
    print(f"  ⚙️ ALLOW_ANONYMOUS_LOCAL_WRITES      : {sec['ALLOW_ANONYMOUS_LOCAL_WRITES']}")
    if report["warnings"]:
        print("")
        for w in report["warnings"]:
            print(f"  {w}")
    print("")

    # 已棄用設定
    print("🧹 [4/4] 已棄用環境變數清理 (Deprecated Settings)")
    if report["deprecated"]:
        for d in report["deprecated"]:
            print(f"  ⚠️ 檢測到已棄用變數 '{d['name']}': {d['reason']}")
    else:
        print("  ✅ 未檢測到任何已棄用環境變數，環境檔案極簡健康。")
    print("=" * 78)

    if report["status"] == "error":
        print("❌ 健檢未通過：請檢查上述缺失之核心必填設定！")
    else:
        print("✅ 健檢完成：核心服務準備就緒！")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check environment variables for 888 StockQuant")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    report = run_env_check()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_cli_report(report)

    return 1 if report["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
