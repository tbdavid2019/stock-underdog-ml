"""
data/signal_alignment.py
Deterministic Multi-Source Signal Alignment Synthesis Engine.

Synthesizes:
1. Time-Series AI Models Forecasts (Timestral, TimesFM, MOIRAI)
2. Institutional Flow (Taiwan: Foreign + Investment Trust net 5-day flow; Overseas: applicable=False)
3. Podcast Intelligence (30-day window entity sentiment net stance)
4. Frontier AI Radar Discovery (arXiv / Industry discovery candidates & recommendation)

Outputs:
- bullish_alignment: >= 2 active sources are +1 and no active source is -1
- bearish_alignment: >= 2 active sources are -1 and no active source is +1
- divergence_warning: Model dimension (+1 or -1) conflicts with Flow or Intelligence dimension
- mixed: Active sources >= 2 with mixed non-divergent signals
- insufficient_sources: Active applicable sources < 2
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import json
import logging

from data.duckdb_manager import DuckDBManager

logger = logging.getLogger("stock_app.signal_alignment")


def _normalize_ticker(ticker: str) -> str:
    """Normalize input ticker into canonical ticker."""
    if not ticker:
        return ""
    t = str(ticker).strip().upper()
    if t.isdigit() and len(t) in (4, 5, 6):
        return f"{t}.TW"
    return t


def _detect_market(ticker: str) -> str:
    """Detect market for ticker."""
    t = str(ticker).strip().upper()
    if t.endswith(".TW") or t.endswith(".TWO") or (t.isdigit() and len(t) in (4, 5, 6)):
        return "TW"
    if t.endswith(".T") or t.endswith(".JP"):
        return "JP"
    return "US"


def compute_signal_alignment(ticker: str, db: Optional[DuckDBManager] = None) -> Dict[str, Any]:
    """
    Compute deterministic multi-source signal alignment for a given ticker.
    Uses read-only DuckDB connection.
    """
    if db is None:
        db = DuckDBManager()

    canonical = _normalize_ticker(ticker)
    market = _detect_market(canonical)
    now_utc = datetime.now(timezone.utc)

    # Dimension containers
    dim_model: Dict[str, Any] = {"name": "時序模型預測", "applicable": True, "stale": False, "score": None, "details": {}}
    dim_inst: Dict[str, Any] = {"name": "籌碼法人動向", "applicable": market == "TW", "stale": False, "score": None, "details": {}}
    dim_podcast: Dict[str, Any] = {"name": "Podcast 情報姿態", "applicable": True, "stale": False, "score": None, "details": {}}
    dim_discovery: Dict[str, Any] = {"name": "前沿雷達標選", "applicable": True, "stale": False, "score": None, "details": {}}

    try:
        with db._get_connection(read_only=True) as con:
            raw_code = canonical.split(".")[0]

            # ---------------------------------------------------------
            # 1. Model Dimension
            # ---------------------------------------------------------
            try:
                max_ts_row = con.execute("""
                    SELECT MAX(timestamp) FROM predictions WHERE ticker = ? OR ticker = ?
                """, [canonical, raw_code]).fetchone()
                latest_ts = max_ts_row[0] if max_ts_row else None

                if not latest_ts:
                    dim_model["details"] = {"reason": "no_prediction_record"}
                else:
                    is_model_stale = False
                    try:
                        ts_clean = str(latest_ts).replace("Z", "+00:00")
                        ts_dt = datetime.fromisoformat(ts_clean)
                        if ts_dt.tzinfo is None:
                            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                        if (now_utc - ts_dt).total_seconds() > 14 * 86400:
                            is_model_stale = True
                    except Exception:
                        pass

                    if is_model_stale:
                        dim_model["stale"] = True
                        dim_model["score"] = None
                        dim_model["details"] = {"reason": "model_data_stale_over_10_business_days", "as_of": str(latest_ts)}
                    else:
                        pred_rows = con.execute("""
                            SELECT timestral_potential, timesfm_potential, moirai_potential, potential,
                                   current_price, predicted_price, trust_net_5d, foreign_net_5d, timestamp, model_name
                            FROM predictions
                            WHERE (ticker = ? OR ticker = ?) AND timestamp = ?
                        """, [canonical, raw_code, latest_ts]).fetchall()

                        potentials: List[float] = []
                        for p in pred_rows:
                            for val in (p[0], p[1], p[2]):
                                if val is not None:
                                    try:
                                        potentials.append(float(val))
                                    except (ValueError, TypeError):
                                        pass
                            if p[3] is not None:
                                try:
                                    potentials.append(float(p[3]))
                                except (ValueError, TypeError):
                                    pass

                        if potentials:
                            avg_pot = sum(potentials) / len(potentials)
                            if avg_pot >= 0.5:
                                dim_model["score"] = 1
                            elif avg_pot <= -0.5:
                                dim_model["score"] = -1
                            else:
                                dim_model["score"] = 0
                            dim_model["details"] = {
                                "avg_potential_pct": round(avg_pot, 2),
                                "potentials_sample": [round(x, 2) for x in potentials[:6]],
                                "as_of": str(latest_ts)
                            }
                        else:
                            dim_model["details"] = {"reason": "no_valid_model_potential"}
            except Exception as e:
                logger.warning(f"Signal alignment model evaluation error for {canonical}: {e}")
                dim_model["details"] = {"error": str(e)}

            # ---------------------------------------------------------
            # 2. Institutional Dimension (TW only)
            # ---------------------------------------------------------
            if market != "TW":
                dim_inst["applicable"] = False
                dim_inst["score"] = None
                dim_inst["details"] = {"reason": "overseas_stock_not_applicable"}
            else:
                try:
                    t_net = None
                    f_net = None
                    flow_period = "5d"
                    as_of_date = None

                    if 'pred_rows' in locals() and pred_rows:
                        first_valid_inst = next((p for p in pred_rows if p[6] is not None or p[7] is not None), None)
                        if first_valid_inst:
                            t_net = first_valid_inst[6]
                            f_net = first_valid_inst[7]
                            flow_period = "5d"
                            as_of_date = str(first_valid_inst[8])[:10]

                    if t_net is None and f_net is None:
                        inst_rows = con.execute("""
                            SELECT trust_net, foreign_net, date
                            FROM tw_institutional_daily
                            WHERE ticker = ? OR ticker = ?
                            ORDER BY date DESC
                            LIMIT 1
                        """, [canonical, raw_code]).fetchall()
                        if inst_rows:
                            t_net = inst_rows[0][0]
                            f_net = inst_rows[0][1]
                            flow_period = "1d"
                            as_of_date = str(inst_rows[0][2])[:10]

                    is_inst_stale = False
                    if as_of_date:
                        try:
                            d_clean = as_of_date.replace("/", "-")
                            if len(d_clean) == 8 and d_clean.isdigit():
                                d_clean = f"{d_clean[:4]}-{d_clean[4:6]}-{d_clean[6:]}"
                            d_obj = datetime.fromisoformat(d_clean).date()
                            if (now_utc.date() - d_obj).days > 14:
                                is_inst_stale = True
                        except Exception:
                            pass

                    if is_inst_stale:
                        dim_inst["stale"] = True
                        dim_inst["score"] = None
                        dim_inst["details"] = {
                            "reason": "institutional_data_stale_over_10_business_days",
                            "period": flow_period,
                            "date": as_of_date
                        }
                    elif t_net is None and f_net is None:
                        dim_inst["details"] = {"reason": "no_institutional_data"}
                    else:
                        t_val = int(t_net or 0)
                        f_val = int(f_net or 0)
                        total_net = t_val + f_val

                        if total_net > 0 and t_val >= 0:
                            dim_inst["score"] = 1
                        elif total_net > 0 and t_val < 0:
                            dim_inst["score"] = 1 if f_val > abs(t_val) * 1.5 else 0
                        elif total_net < 0:
                            dim_inst["score"] = -1
                        else:
                            dim_inst["score"] = 0

                        dim_inst["details"] = {
                            "trust_net": t_val,
                            "foreign_net": f_val,
                            "total_net": total_net,
                            "period": flow_period,
                            "date": as_of_date
                        }
                except Exception as e:
                    logger.warning(f"Signal alignment inst evaluation error for {canonical}: {e}")
                    dim_inst["details"] = {"error": str(e)}

            # ---------------------------------------------------------
            # 3. Podcast Dimension (30-day window)
            # ---------------------------------------------------------
            try:
                # Search podcast episodes mentioning ticker or raw_code
                pod_rows = con.execute("""
                    SELECT title, published_at, ticker_sentiments
                    FROM podcast_episodes
                    WHERE ticker_sentiments IS NOT NULL
                      AND (ticker_sentiments LIKE ? OR ticker_sentiments LIKE ?)
                    ORDER BY published_at DESC
                    LIMIT 20
                """, [f"%{canonical}%", f"%{raw_code}%"]).fetchall()

                if not pod_rows:
                    dim_podcast["score"] = None
                    dim_podcast["details"] = {"reason": "no_podcast_mentions"}
                else:
                    recent_stances: List[int] = []
                    most_recent_dt: Optional[datetime] = None

                    for row in pod_rows:
                        pub_str = row[1]
                        sent_str = row[2]
                        if not sent_str:
                            continue

                        # Parse published date
                        pub_dt = None
                        if pub_str:
                            try:
                                # ISO format or YYYY-MM-DD
                                pub_dt = datetime.fromisoformat(str(pub_str).replace("Z", "+00:00"))
                                if pub_dt.tzinfo is None:
                                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                            except Exception:
                                pass

                        if most_recent_dt is None and pub_dt:
                            most_recent_dt = pub_dt

                        # Check 30-day freshness (must be valid parseable date <= 30 days)
                        is_fresh = False
                        if pub_str:
                            try:
                                pub_dt = datetime.fromisoformat(str(pub_str).replace("Z", "+00:00"))
                                if pub_dt.tzinfo is None:
                                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                                diff_days = (now_utc - pub_dt).total_seconds() / 86400.0
                                if diff_days <= 30.0:
                                    is_fresh = True
                            except Exception:
                                is_fresh = False

                        if not is_fresh:
                            continue

                        # Parse ticker_sentiments JSON
                        try:
                            s_data = json.loads(sent_str)
                            target_key = canonical if canonical in s_data else (raw_code if raw_code in s_data else None)
                            if target_key:
                                item = s_data[target_key]
                                stance = item.get("stance", "").lower()
                                if "bull" in stance or "多" in item.get("label", ""):
                                    recent_stances.append(1)
                                elif "bear" in stance or "空" in item.get("label", ""):
                                    recent_stances.append(-1)
                                else:
                                    recent_stances.append(0)
                        except Exception:
                            pass

                    if not recent_stances:
                        # Mentioned previously but stale (> 30 days)
                        dim_podcast["stale"] = True
                        dim_podcast["score"] = None
                        dim_podcast["details"] = {
                            "reason": "stale_over_30d",
                            "last_published": most_recent_dt.isoformat() if most_recent_dt else None
                        }
                    else:
                        net_stance = sum(recent_stances)
                        if net_stance > 0:
                            dim_podcast["score"] = 1
                        elif net_stance < 0:
                            dim_podcast["score"] = -1
                        else:
                            dim_podcast["score"] = 0
                        dim_podcast["details"] = {
                            "recent_mentions_count": len(recent_stances),
                            "net_stance_sum": net_stance
                        }
            except Exception as e:
                logger.warning(f"Signal alignment podcast evaluation error for {canonical}: {e}")
                dim_podcast["details"] = {"error": str(e)}

            # ---------------------------------------------------------
            # 4. AI Frontier Radar Discovery Dimension
            # ---------------------------------------------------------
            try:
                disc_rows = con.execute("""
                    SELECT recommendation, total_score, theme_name, as_of
                    FROM discovery_candidates
                    WHERE ticker = ? OR ticker = ?
                    ORDER BY as_of DESC
                    LIMIT 1
                """, [canonical, raw_code]).fetchall()

                if not disc_rows:
                    dim_discovery["score"] = None
                    dim_discovery["details"] = {"reason": "not_in_discovery_pool"}
                else:
                    d_row = disc_rows[0]
                    rec = (d_row[0] or "").lower()
                    tot_score = float(d_row[1] or 0.0)
                    theme = d_row[2] or ""

                    as_of_val = d_row[3]
                    is_disc_stale = False
                    if as_of_val:
                        try:
                            a_clean = str(as_of_val)[:10].replace("/", "-")
                            if len(a_clean) == 8 and a_clean.isdigit():
                                a_clean = f"{a_clean[:4]}-{a_clean[4:6]}-{a_clean[6:]}"
                            a_date = datetime.fromisoformat(a_clean).date()
                            if (now_utc.date() - a_date).days > 30:
                                is_disc_stale = True
                        except Exception:
                            pass
                    else:
                        is_disc_stale = True

                    if is_disc_stale:
                        dim_discovery["stale"] = True
                        dim_discovery["score"] = None
                        dim_discovery["details"] = {
                            "reason": "discovery_data_stale_over_30d",
                            "as_of": as_of_val
                        }
                    else:
                        # Recommendation criteria:
                        # propose_add / already_in_pool with total_score >= 65 -> +1
                        # reject -> -1
                        # observe / others -> 0
                        if rec in ("propose_add", "already_in_pool") and tot_score >= 65.0:
                            dim_discovery["score"] = 1
                        elif rec == "reject":
                            dim_discovery["score"] = -1
                        else:
                            dim_discovery["score"] = 0

                        dim_discovery["details"] = {
                            "recommendation": rec,
                            "total_score": round(tot_score, 1),
                            "theme_name": theme,
                            "as_of": d_row[3]
                        }
            except Exception as e:
                logger.warning(f"Signal alignment discovery evaluation error for {canonical}: {e}")
                dim_discovery["details"] = {"error": str(e)}

    except Exception as e:
        logger.error(f"Signal alignment connection error for {canonical}: {e}")

    # ---------------------------------------------------------
    # Synthesis & Consensus Logic
    # ---------------------------------------------------------
    dimensions = {
        "model": dim_model,
        "institutional": dim_inst,
        "podcast": dim_podcast,
        "discovery": dim_discovery
    }

    applicable_sources = [k for k, v in dimensions.items() if v.get("applicable", False)]
    active_sources = [
        k for k, v in dimensions.items()
        if v.get("applicable", False) and not v.get("stale", False) and v.get("score") is not None
    ]

    active_scores = [dimensions[k]["score"] for k in active_sources]

    if len(active_scores) < 2:
        alignment_status = "insufficient_sources"
        final_score = None
    else:
        # Check Divergence Warning:
        # Model dimension has a non-zero score AND at least one other active source is opposite
        m_score = dim_model.get("score")
        has_divergence = False
        if m_score in (1, -1):
            other_active = [dimensions[k]["score"] for k in active_sources if k != "model"]
            if any(s == -m_score for s in other_active):
                has_divergence = True

        if has_divergence:
            alignment_status = "divergence_warning"
            final_score = round(sum(active_scores) / len(active_scores), 3)
        else:
            # Bullish Alignment: >= 2 sources are +1 and no active source is -1
            if all(s >= 0 for s in active_scores) and active_scores.count(1) >= 2:
                alignment_status = "bullish_alignment"
            # Bearish Alignment: >= 2 sources are -1 and no active source is +1
            elif all(s <= 0 for s in active_scores) and active_scores.count(-1) >= 2:
                alignment_status = "bearish_alignment"
            else:
                alignment_status = "mixed"
            final_score = round(sum(active_scores) / len(active_scores), 3)

    return {
        "ticker": canonical,
        "market": market,
        "alignment_status": alignment_status,
        "score": final_score,
        "active_sources_count": len(active_scores),
        "total_applicable_sources": len(applicable_sources),
        "dimensions": dimensions,
        "computed_at": now_utc.isoformat()
    }
