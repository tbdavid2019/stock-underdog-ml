"""
data/alert_evaluator.py
Quant Price Alert Evaluator & Atomic Dispatcher.

Evaluates user quant price alerts against latest predictions and price data:
- above_price: current_price >= threshold_value
- below_price: current_price <= threshold_value
- below_p10: current_price <= P10 (conservative lower quantile support)
- above_p50: current_price >= P50 / predicted_price (fair target price breakout)
- pullback_ma60: current_price <= MA60 (quarterly moving average support pullback)
- pullback_ma120: current_price <= MA120 (half-year moving average support pullback)

Atomic cooldown anti-spam guarantee:
1. Verifies cooldown window (cooldown_minutes).
2. Updates DuckDB (last_triggered_at, last_triggered_price) under write flock FIRST.
3. ONLY after DB update is successfully committed does it call notifier.py.
   If DB write fails or locks out, notification is aborted to prevent spam.
"""

import html
import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from data.duckdb_manager import DuckDBManager
from core.duckdb_lock import DuckDBBusyError

logger = logging.getLogger("stock_app.alert_evaluator")

_EVALUATOR_LOCK = threading.Lock()


def dispatch_alert_notification(alert_info: Dict[str, Any]) -> bool:
    """
    Dispatch short, targeted price alert push notification via notifier.py.
    Does NOT touch the routine daily comprehensive market report.
    Applies html.escape to all variables to protect against formatting breakage / XSS.
    """
    raw_ticker = alert_info.get("canonical_ticker", "")
    raw_rule = alert_info.get("rule_type", "")
    raw_curr = alert_info.get("current_price", "")
    raw_thresh = alert_info.get("evaluated_threshold", "")
    raw_desc = alert_info.get("trigger_description", "")

    ticker_esc = html.escape(str(raw_ticker))
    rule_esc = html.escape(str(raw_rule))
    curr_esc = html.escape(str(raw_curr))
    thresh_esc = html.escape(str(raw_thresh))
    desc_esc = html.escape(str(raw_desc))

    subject = f"🔔 [888 StockQuant 到價告警] {ticker_esc} 觸發 {rule_esc}"
    html_msg = (
        f"🔔 <b>[888 StockQuant 到價告警]</b><br>"
        f"標的代號: <b>{ticker_esc}</b><br>"
        f"觸發規則: <code>{rule_esc}</code><br>"
        f"當前現價: <b>{curr_esc}</b><br>"
        f"門檻價位: <b>{thresh_esc}</b><br>"
        f"觸發說明: {desc_esc}"
    )
    discord_msg = (
        f"🔔 **[888 StockQuant 到價告警]**\n"
        f"標的代號: **{raw_ticker}**\n"
        f"觸發規則: `{raw_rule}`\n"
        f"當前現價: **{raw_curr}**\n"
        f"門檻價位: **{raw_thresh}**\n"
        f"觸發說明: {raw_desc}"
    )

    try:
        from core.config import TelegramConfig, DiscordConfig, EmailConfig
        import notifier

        # Telegram
        if getattr(TelegramConfig, "BOT_TOKEN", None) and (getattr(TelegramConfig, "CHANNEL_ID", None) or getattr(TelegramConfig, "CHAT_ID", None)):
            try:
                notifier.send_to_telegram(html_msg.replace("<br>", "\n"))
            except Exception as e:
                logger.warning(f"Failed to send Telegram alert: {e}")

        # Discord
        if getattr(DiscordConfig, "WEBHOOK_URL", None):
            try:
                notifier.send_to_discord(discord_msg)
            except Exception as e:
                logger.warning(f"Failed to send Discord alert: {e}")

        # Email
        if (
            getattr(EmailConfig, "SENDER_EMAIL", None)
            and getattr(EmailConfig, "EMAIL_PASSWORD", None)
            and getattr(EmailConfig, "TO_EMAILS", None)
        ):
            try:
                notifier.send_email(
                    subject=subject,
                    body=discord_msg,
                    to_emails=EmailConfig.TO_EMAILS,
                    html_body=html_msg
                )
            except Exception as e:
                logger.warning(f"Failed to send Email alert: {e}")

        return True
    except Exception as e:
        logger.error(f"Alert dispatcher error for {raw_ticker}: {e}")
        return False


def evaluate_and_dispatch_alerts(
    profile_id: str = "default",
    db: Optional[DuckDBManager] = None,
    force_dispatch: bool = False
) -> Dict[str, Any]:
    """
    Evaluate all enabled price alerts for profile_id and dispatch notifications.
    Enforces atomic write-before-dispatch and in-process mutual exclusion.
    """
    if not _EVALUATOR_LOCK.acquire(blocking=False):
        logger.info("Alert evaluation already in progress in current process, skipping.")
        return {
            "status": "skipped",
            "reason": "concurrent_evaluation_in_progress",
            "evaluated": 0,
            "triggered": 0,
            "alerts": []
        }

    try:
        if db is None:
            db = DuckDBManager()

        alerts = db.get_quant_alerts(profile_id=profile_id, enabled_only=True)
        if not alerts:
            return {
                "status": "success",
                "evaluated": 0,
                "triggered": 0,
                "alerts": []
            }

        now_utc = datetime.now(timezone.utc)
        triggered_results: List[Dict[str, Any]] = []

        # Read latest prices and quant indicators for all distinct tickers
        distinct_tickers = list(set(a["canonical_ticker"] for a in alerts))
        ticker_data_map: Dict[str, Dict[str, Any]] = {}

        try:
            with db._get_connection(read_only=True) as con:
                for tk in distinct_tickers:
                    raw_c = tk.split(".")[0]
                    max_ts_row = con.execute("""
                        SELECT MAX(timestamp) FROM predictions WHERE ticker = ? OR ticker = ?
                    """, [tk, raw_c]).fetchone()
                    latest_ts = max_ts_row[0] if max_ts_row else None
                    if not latest_ts:
                        continue

                    rows = con.execute("""
                        SELECT current_price, predicted_price, timestral_p10, moirai_p10,
                               ma60, ma120, potential, timestamp, model_name,
                               timestral_predicted_price, timesfm_predicted_price, moirai_predicted_price
                        FROM predictions
                        WHERE (ticker = ? OR ticker = ?) AND timestamp = ?
                    """, [tk, raw_c, latest_ts]).fetchall()

                    if rows:
                        first_valid_price = next((float(r[0]) for r in rows if r[0] is not None), None)
                        if first_valid_price is None:
                            continue

                        timestral_p10 = next((float(r[2]) for r in rows if r[2] is not None), None)
                        moirai_p10 = next((float(r[3]) for r in rows if r[3] is not None), None)
                        ma60 = next((float(r[4]) for r in rows if r[4] is not None), None)
                        ma120 = next((float(r[5]) for r in rows if r[5] is not None), None)
                        potential = next((float(r[6]) for r in rows if r[6] is not None), None)
                        pred_price = next((float(r[1]) for r in rows if r[1] is not None), None)
                        timestral_pred = next(
                            (float(r[9]) for r in rows if r[9] is not None),
                            next((float(r[1]) for r in rows if r[8] and "timestral" in str(r[8]).lower() and r[1] is not None), None),
                        )
                        timesfm_pred = next(
                            (float(r[10]) for r in rows if r[10] is not None),
                            next((float(r[1]) for r in rows if r[8] and "timesfm" in str(r[8]).lower() and r[1] is not None), None),
                        )
                        moirai_pred = next(
                            (float(r[11]) for r in rows if r[11] is not None),
                            next((float(r[1]) for r in rows if r[8] and "moirai" in str(r[8]).lower() and r[1] is not None), None),
                        )

                        ticker_data_map[tk] = {
                            "current_price": first_valid_price,
                            "predicted_price": pred_price,
                            "timestral_p10": timestral_p10,
                            "moirai_p10": moirai_p10,
                            "ma60": ma60,
                            "ma120": ma120,
                            "potential": potential,
                            "timestamp": latest_ts,
                            "timestral_predicted_price": timestral_pred,
                            "timesfm_predicted_price": timesfm_pred,
                            "moirai_predicted_price": moirai_pred,
                            "rows": rows
                        }
        except DuckDBBusyError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch market data for alerts evaluation: {e}")
            return {
                "status": "error",
                "error": str(e),
                "evaluated": 0,
                "triggered": 0,
                "alerts": []
            }

        for alert in alerts:
            tk = alert["canonical_ticker"]
            rule = alert["rule_type"]
            data = ticker_data_map.get(tk)
            if not data or data.get("current_price") is None:
                continue

            curr_price = data["current_price"]
            cooldown_min = int(alert.get("cooldown_minutes") or 720)

            # 1. Cooldown check
            last_trig_str = alert.get("last_triggered_at")
            if last_trig_str:
                try:
                    last_dt = datetime.fromisoformat(str(last_trig_str).replace("Z", "+00:00"))
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    elapsed = (now_utc - last_dt).total_seconds()
                    if elapsed < cooldown_min * 60:
                        # In cooldown, skip
                        continue
                except Exception:
                    pass

            # 2. Rule evaluation
            is_triggered = False
            evaluated_thresh: Optional[float] = None
            desc = ""

            if rule == "above_price":
                evaluated_thresh = alert.get("threshold_value")
                if evaluated_thresh is not None and curr_price >= evaluated_thresh:
                    is_triggered = True
                    desc = f"現價 {curr_price} 已向上突破目標價位 {evaluated_thresh}"

            elif rule == "below_price":
                evaluated_thresh = alert.get("threshold_value")
                if evaluated_thresh is not None and curr_price <= evaluated_thresh:
                    is_triggered = True
                    desc = f"現價 {curr_price} 已跌破警戒底限 {evaluated_thresh}"

            elif rule == "below_p10":
                p10_candidates = [v for v in (data.get("timestral_p10"), data.get("moirai_p10")) if v is not None]
                if p10_candidates:
                    ref_m = (alert.get("reference_model") or "auto").lower()
                    if ref_m == "timestral" and data.get("timestral_p10") is not None:
                        evaluated_thresh = data["timestral_p10"]
                    elif ref_m == "moirai" and data.get("moirai_p10") is not None:
                        evaluated_thresh = data["moirai_p10"]
                    else:
                        # auto mode uses conservative minimum P10
                        evaluated_thresh = min(p10_candidates)

                    if evaluated_thresh is not None and curr_price <= evaluated_thresh:
                        is_triggered = True
                        desc = f"現價 {curr_price} 已跌破量化分位數 P10 支撐位 {round(evaluated_thresh, 2)}"

            elif rule == "above_p50":
                ref_m = (alert.get("reference_model") or "auto").lower()
                p50_val = None
                if ref_m == "timestral":
                    p50_val = data.get("timestral_predicted_price")
                elif ref_m == "timesfm":
                    p50_val = data.get("timesfm_predicted_price")
                elif ref_m == "moirai":
                    p50_val = data.get("moirai_predicted_price")
                if p50_val is None:
                    p50_val = data.get("predicted_price") or data.get("timesfm_predicted_price") or data.get("timestral_predicted_price") or data.get("moirai_predicted_price")

                if p50_val is not None:
                    evaluated_thresh = p50_val
                    if curr_price >= evaluated_thresh:
                        is_triggered = True
                        desc = f"現價 {curr_price} 已達成模型 P50 目標價 {round(evaluated_thresh, 2)}"

            elif rule == "pullback_ma60":
                if data.get("ma60") is not None:
                    evaluated_thresh = data["ma60"]
                    if curr_price <= evaluated_thresh:
                        is_triggered = True
                        desc = f"現價 {curr_price} 回測季線 MA60 支撐價位 {round(evaluated_thresh, 2)}"

            elif rule == "pullback_ma120":
                if data.get("ma120") is not None:
                    evaluated_thresh = data["ma120"]
                    if curr_price <= evaluated_thresh:
                        is_triggered = True
                        desc = f"現價 {curr_price} 回測半年線 MA120 支撐價位 {round(evaluated_thresh, 2)}"

            if is_triggered:
                # 3. ATOMIC WRITE: Update DuckDB BEFORE sending push notification
                alert_id = alert["alert_id"]
                try:
                    write_ok = db.record_alert_trigger(alert_id, curr_price, profile_id=profile_id, cooldown_minutes=cooldown_min)
                except DuckDBBusyError:
                    logger.warning(f"DuckDB lock busy while recording alert {alert_id}, aborting dispatch to prevent spam")
                    raise
                except Exception as e:
                    logger.error(f"Failed to record trigger for alert {alert_id}: {e}")
                    continue

                if not write_ok:
                    logger.info(f"Alert {alert_id} not triggered (CAS check failed or in cooldown)")
                    continue

                # 4. DISPATCH NOTIFICATION
                alert_payload = {
                    "alert_id": alert_id,
                    "canonical_ticker": tk,
                    "rule_type": rule,
                    "reference_model": alert.get("reference_model") or "auto",
                    "current_price": curr_price,
                    "evaluated_threshold": round(evaluated_thresh, 2) if evaluated_thresh is not None else None,
                    "trigger_description": desc,
                    "triggered_at": now_utc.isoformat()
                }

                dispatch_ok = dispatch_alert_notification(alert_payload)
                alert_payload["dispatch_ok"] = bool(dispatch_ok)
                triggered_results.append(alert_payload)

        return {
            "status": "success",
            "evaluated": len(alerts),
            "triggered": len(triggered_results),
            "alerts": triggered_results
        }
    finally:
        _EVALUATOR_LOCK.release()
