"""
data/clef_client.py - Clef-Flash System One Typed Decision Client

Provides 3-Tier Fallback structured probabilistic decision inference for stock quant candidates.
Integrates Cloudflare Clef-Flash model endpoints:
- Primary: Clef-Create360 (https://clef.create360.ai/v1/systemone)
- Fallback 1: Clef-Aiurl (https://clef.aiurl.tw/v1/systemone)
- Fallback 2: Clef-Jev (Jev-compatible endpoint)
"""

import re
import math
import logging
import requests
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import config

logger = logging.getLogger("stock_app.data.clef_client")


@dataclass
class ClefSlot:
    name: str
    url: str
    timeout: int = 6


@dataclass
class ClefDecisionVerdict:
    ticker: str
    action_decision: str = "neutral"
    action_confidence: float = 0.0
    action_probabilities: Dict[str, float] = field(default_factory=dict)
    conviction_score: float = 2.5
    is_favorable_entry: float = 0.5
    tag: str = ""
    source_slot: str = "none"
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "action_decision": self.action_decision,
            "action_confidence": round(self.action_confidence, 3),
            "action_probabilities": {k: round(v, 3) for k, v in self.action_probabilities.items()},
            "conviction_score": round(self.conviction_score, 2),
            "is_favorable_entry": round(self.is_favorable_entry, 3),
            "tag": self.tag,
            "source_slot": self.source_slot,
            "success": self.success,
            "error": self.error
        }


DEFAULT_STOCK_QUESTIONS = {
    "action_decision": {
        "type": "choice",
        "instructions": "What is the recommended trading action for this stock candidate given the holistic quantitative state?",
        "criteria": {
            "strong_buy": "High conviction buy with confluent technical, model, and institutional tailwinds",
            "buy": "Standard swing buy on technical pullback or positive momentum",
            "hold_watch": "Hold or place on watchlist; await clearer technical or market confirmation",
            "avoid": "Avoid entry due to elevated risk, excessive valuation, or conflicting signals"
        }
    },
    "conviction_score": {
        "type": "score",
        "instructions": "Rate the overall trade conviction score from 1 (lowest) to 5 (highest)",
        "criteria": ["1_very_low", "2_low", "3_moderate", "4_high", "5_very_high"]
    },
    "is_favorable_entry": {
        "type": "noul",
        "instructions": "Is this stock currently positioned at a favorable risk-reward entry point for a swing trade?"
    }
}


def _is_finite_num(val: Any) -> bool:
    """Check if value can be converted to a finite float (rejecting nan, inf, -inf)"""
    if val is None or isinstance(val, (dict, list, tuple)):
        return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (ValueError, TypeError):
        return False


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert numeric value or string to float, returning default on None, error, or non-finite values"""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


class ClefDecisionClient:
    """3-Tier Fallback Client for Clef-Flash System One Decision API"""

    def __init__(
        self,
        slots: Optional[List[ClefSlot]] = None,
        enabled: Optional[bool] = None,
        timeout: Optional[int] = None
    ):
        self.enabled = config.clef.ENABLE_CLEF_DECISION if enabled is None else enabled
        default_timeout = config.clef.TIMEOUT if timeout is None else timeout

        raw_slots = slots if slots is not None else [
            ClefSlot(
                name=config.clef.PRIMARY_NAME,
                url=config.clef.PRIMARY_URL,
                timeout=default_timeout
            ),
            ClefSlot(
                name=config.clef.FALLBACK1_NAME,
                url=config.clef.FALLBACK1_URL,
                timeout=default_timeout
            ),
            ClefSlot(
                name=config.clef.FALLBACK2_NAME,
                url=config.clef.FALLBACK2_URL,
                timeout=default_timeout
            )
        ]

        # Filter out empty URLs or duplicate endpoints to avoid redundant timeouts against the same host
        seen_urls = set()
        self.slots = []
        for slot in raw_slots:
            url_str = (slot.url or "").strip()
            if url_str and url_str not in seen_urls:
                seen_urls.add(url_str)
                self.slots.append(ClefSlot(name=slot.name, url=url_str, timeout=slot.timeout))

    def check_health(self) -> Dict[str, Any]:
        """Check health across all configured endpoints"""
        results = {}
        for slot in self.slots:
            health_url = slot.url.replace("/v1/systemone", "/health")
            try:
                resp = requests.get(health_url, timeout=3)
                if resp.status_code == 200:
                    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    results[slot.name] = {
                        "status": "healthy",
                        "status_code": 200,
                        "data": data,
                        "url": health_url
                    }
                else:
                    results[slot.name] = {
                        "status": "degraded",
                        "status_code": resp.status_code,
                        "url": health_url
                    }
            except Exception as e:
                results[slot.name] = {
                    "status": "unreachable",
                    "error": str(e),
                    "url": health_url
                }

        if not any(slot.name == config.clef.FALLBACK2_NAME for slot in self.slots):
            results[config.clef.FALLBACK2_NAME] = {
                "status": "unconfigured",
                "note": "Optional 3rd tier endpoint. Set CLEF_FALLBACK2_URL or JEV_SYSTEMONE_URL to enable.",
                "url": None
            }

        return results

    def _call_systemone(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], str, Optional[str]]:
        """
        Execute request with 3-tier fallback.
        Returns: (success, answers_dict, slot_name, error_msg)
        """
        payload = {
            "model": "clef-flash",
            "state": state,
            "questions": questions
        }
        headers = {"Content-Type": "application/json"}
        last_error = None

        for slot in self.slots:
            if not slot.url:
                continue
            try:
                resp = requests.post(
                    slot.url,
                    json=payload,
                    headers=headers,
                    timeout=slot.timeout
                )
                if resp.status_code == 200:
                    res_json = resp.json()
                    answers = res_json.get("answers", {})
                    if not isinstance(answers, dict) or not answers:
                        last_error = f"Empty or non-dict answers from {slot.name}"
                        logger.warning(f"⚠️ Clef slot {slot.name} invalid answers structure, falling back...")
                        continue

                    # Validate answer format to prevent malformed responses from terminating fallback chain
                    is_valid, validation_err = self._validate_answers(answers)
                    if not is_valid:
                        last_error = f"{slot.name} validation failed: {validation_err}"
                        logger.warning(f"⚠️ Clef slot {slot.name} answer validation failed ({validation_err}), falling back...")
                        continue

                    return True, answers, slot.name, None
                else:
                    last_error = f"{slot.name} HTTP {resp.status_code}: {resp.text[:120]}"
                    logger.warning(f"⚠️ Clef slot {slot.name} failed with {resp.status_code}, falling back...")
            except Exception as e:
                last_error = f"{slot.name} exception: {str(e)}"
                logger.warning(f"⚠️ Clef slot {slot.name} connection error ({e}), falling back...")

        return False, {}, "none", last_error

    @staticmethod
    def _validate_answers(answers: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate structure and essential values of Clef answers"""
        if not isinstance(answers, dict) or not answers:
            return False, "answers is not a non-empty dict"

        # Check required fields
        for req_field in ("action_decision", "conviction_score", "is_favorable_entry"):
            if req_field not in answers:
                return False, f"missing required answer field: '{req_field}'"
            if not isinstance(answers[req_field], dict):
                return False, f"answer field '{req_field}' must be a dict, got {type(answers[req_field]).__name__}"

        # 1. action_decision validation
        act = answers["action_decision"]
        choice = act.get("choice")
        supported_choices = {"strong_buy", "buy", "hold_watch", "avoid"}
        if choice not in supported_choices:
            return False, f"action_decision.choice must be one of {supported_choices}, got {choice!r}"

        conf = act.get("confidence")
        if conf is not None:
            if not _is_finite_num(conf) or not (0.0 <= float(conf) <= 1.0):
                return False, f"confidence must be a finite float in [0.0, 1.0], got {conf}"

        raw_probs = act.get("probabilities")
        if raw_probs is not None:
            if not isinstance(raw_probs, dict):
                return False, f"action_decision.probabilities must be a dict, got {type(raw_probs).__name__}"
            for k, v in raw_probs.items():
                if not _is_finite_num(v) or not (0.0 <= float(v) <= 1.0):
                    return False, f"probability for key '{k}' must be in [0.0, 1.0], got {v}"

        # 2. conviction_score validation
        sc = answers["conviction_score"]
        if "score" not in sc or sc["score"] is None:
            return False, "conviction_score missing required 'score' field"
        if not _is_finite_num(sc["score"]) or not (1.0 <= float(sc["score"]) <= 5.0):
            return False, f"conviction score must be in [1.0, 5.0], got {sc['score']}"

        # 3. is_favorable_entry validation
        fe = answers["is_favorable_entry"]
        if "noul" in fe and fe["noul"] is not None:
            if not _is_finite_num(fe["noul"]) or not (0.0 <= float(fe["noul"]) <= 1.0):
                return False, f"is_favorable_entry noul must be in [0.0, 1.0], got {fe['noul']}"

        return True, None

    @staticmethod
    def format_stock_state(
        candidate: Dict[str, Any],
        macro_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert candidate dict into clean structured state for Clef.
        Seamlessly handles both pipeline nested dicts and flat DuckDB database records.
        """
        fund = candidate.get("fundamentals") or {}
        inst = candidate.get("institutional") or {}
        tech = candidate.get("technical") or {}
        models = candidate.get("models") or {}

        # Tags parsing (handle list, pipe-separated, or comma-separated string from DuckDB)
        raw_tags = candidate.get("tags", [])
        if isinstance(raw_tags, str):
            tags_list = [t.strip() for t in re.split(r"[|,]", raw_tags) if t.strip()]
        elif isinstance(raw_tags, list):
            tags_list = [str(t).strip() for t in raw_tags if str(t).strip()]
        else:
            tags_list = []

        # Valuation metrics (check nested, then flat)
        pe = fund.get("pe") if fund else candidate.get("pe")
        pb = fund.get("pb") if fund else candidate.get("pb")
        f_pe = fund.get("forward_pe") if fund else candidate.get("forward_pe")
        ev_ebitda = fund.get("ev_ebitda") if fund else candidate.get("ev_ebitda")

        # Technical metrics
        ma60 = tech.get("ma60") if tech else candidate.get("ma60")
        pullback_type = tech.get("pullback_type") if tech else candidate.get("pullback_type", "")

        # Model potentials: preserve explicit None and avoid attributing TimesFM potential to LSTM
        lstm_pot = models.get("lstm_potential") if models else candidate.get("lstm_potential")
        timesfm_pot = models.get("timesfm_potential") if models else candidate.get("timesfm_potential")
        rr = models.get("risk_reward_ratio") if models else candidate.get("risk_reward_ratio")

        # Fallback to generic 'potential' ONLY if model-specific keys are absent and record identifies model
        if lstm_pot is None and timesfm_pot is None and candidate.get("potential") is not None:
            m_name = (candidate.get("model_name") or candidate.get("strategy_type") or "").upper()
            if "LSTM" in m_name:
                lstm_pot = candidate.get("potential")
            elif "TIMESFM" in m_name:
                timesfm_pot = candidate.get("potential")
            else:
                # Default generic legacy potential to lstm_potential
                lstm_pot = candidate.get("potential")

        # Institutional flows (check nested, then flat)
        f_net = inst.get("foreign_net_5d") if inst else candidate.get("foreign_net_5d")
        t_net = inst.get("trust_net_5d") if inst else candidate.get("trust_net_5d")
        t_streak = inst.get("trust_streak", 0) if inst else candidate.get("trust_streak", 0)
        is_sync = inst.get("is_sync_buy") if inst else candidate.get("is_sync_buy", False)
        if not is_sync and "土洋合買" in tags_list:
            is_sync = True

        # Composite score
        score = candidate.get("composite_score")
        if score is None or pd.isna(score):
            score = candidate.get("score", 0.0)

        # Current price
        price = candidate.get("current_price", 0.0)

        state = {
            "ticker": candidate.get("ticker", ""),
            "current_price": float(price) if price and pd.notna(price) else 0.0,
            "composite_score": float(score) if score and pd.notna(score) else 0.0,
            "hit_strategies": candidate.get("hit_strategies") or candidate.get("strategy_type", []),
            "tags": tags_list,
            "valuation": {
                "pe": float(pe) if pe and pd.notna(pe) else None,
                "pb": float(pb) if pb and pd.notna(pb) else None,
                "forward_pe": float(f_pe) if f_pe and pd.notna(f_pe) else None,
                "ev_ebitda": float(ev_ebitda) if ev_ebitda and pd.notna(ev_ebitda) else None
            },
            "technical": {
                "ma60": float(ma60) if ma60 and pd.notna(ma60) else None,
                "pullback_type": str(pullback_type) if pullback_type else ""
            },
            "models": {
                "lstm_potential": float(lstm_pot) if lstm_pot is not None and pd.notna(lstm_pot) else None,
                "timesfm_potential": float(timesfm_pot) if timesfm_pot is not None and pd.notna(timesfm_pot) else None,
                "risk_reward_ratio": float(rr) if rr is not None and pd.notna(rr) else None
            },
            "institutional": {
                "foreign_net_5d": float(f_net) if f_net is not None and pd.notna(f_net) else None,
                "trust_net_5d": float(t_net) if t_net is not None and pd.notna(t_net) else None,
                "trust_streak": int(t_streak) if t_streak else 0,
                "is_sync_buy": bool(is_sync)
            }
        }
        if macro_regime:
            state["macro_regime"] = macro_regime
        return state

    def _evaluate_local_rules(self, state: Dict[str, Any], error_msg: Optional[str] = None) -> ClefDecisionVerdict:
        """
        Local heuristic rules fallback when all Clef remote endpoints fail or are disabled.
        Produces a deterministic probabilistic decision based on quantitative metrics and tags.
        """
        ticker = state.get("ticker", "UNKNOWN")
        models = state.get("models") or {}
        valuation = state.get("valuation") or {}
        institutional = state.get("institutional") or {}
        technical = state.get("technical") or {}
        tags = state.get("tags") or []
        composite_score = _safe_float(state.get("composite_score"), 0.0)

        lstm_pot = _safe_float(models.get("lstm_potential"), 0.0)
        tfm_pot = _safe_float(models.get("timesfm_potential"), 0.0)
        rr = _safe_float(models.get("risk_reward_ratio"), 0.0)

        is_sync_buy = bool(institutional.get("is_sync_buy") or "土洋合買" in tags)
        has_pullback = bool(technical.get("pullback_type") or state.get("pullback_type") or "玄鐵買點" in tags)

        if lstm_pot <= -5.0 or (lstm_pot < 0 and tfm_pot < 0 and composite_score < 50):
            action = "avoid"
            conviction = 1.0
            prob = {"strong_buy": 0.0, "buy": 0.05, "hold_watch": 0.25, "avoid": 0.70}
            tag = "🔻避開(本地規則)"
        elif composite_score >= 85.0 and (lstm_pot > 3.0 or tfm_pot > 3.0 or is_sync_buy):
            action = "strong_buy"
            conviction = 4.5
            prob = {"strong_buy": 0.65, "buy": 0.25, "hold_watch": 0.08, "avoid": 0.02}
            tag = "🤖AI強買(本地規則)"
        elif composite_score >= 70.0 and (lstm_pot > 0.0 or has_pullback or is_sync_buy):
            action = "buy"
            conviction = 3.5
            prob = {"strong_buy": 0.20, "buy": 0.60, "hold_watch": 0.15, "avoid": 0.05}
            tag = "🤖AI做多(本地規則)"
        else:
            action = "hold_watch"
            conviction = 2.5
            prob = {"strong_buy": 0.05, "buy": 0.20, "hold_watch": 0.65, "avoid": 0.10}
            tag = "👀觀望(本地規則)"

        favorable_entry = 0.85 if ((action in ("strong_buy", "buy")) and (rr >= 1.5 or has_pullback)) else 0.15

        return ClefDecisionVerdict(
            ticker=ticker,
            action_decision=action,
            action_confidence=prob.get(action, 0.6),
            action_probabilities=prob,
            conviction_score=conviction,
            is_favorable_entry=favorable_entry,
            tag=tag,
            source_slot="local_rules",
            success=True,
            error=f"Degraded to local rules fallback: {error_msg}" if error_msg else "Local rules fallback"
        )

    def evaluate_stock(
        self,
        candidate: Optional[Dict[str, Any]] = None,
        macro_regime: Optional[str] = None,
        custom_questions: Optional[Dict[str, Any]] = None,
        raw_state: Optional[Dict[str, Any]] = None
    ) -> ClefDecisionVerdict:
        """Evaluate a single stock candidate state or arbitrary custom raw_state against Clef decision questions"""
        if raw_state is not None:
            state = dict(raw_state)
            ticker = str(state.get("ticker", "CUSTOM_STATE"))
            if macro_regime and "macro_regime" not in state:
                state["macro_regime"] = macro_regime
        else:
            candidate = candidate or {}
            ticker = candidate.get("ticker", "UNKNOWN")
            state = self.format_stock_state(candidate, macro_regime=macro_regime)

        if not self.enabled:
            return self._evaluate_local_rules(state, error_msg="Clef decision disabled")

        questions = custom_questions or DEFAULT_STOCK_QUESTIONS

        success, answers, slot_name, error_msg = self._call_systemone(state, questions)
        if not success or not answers:
            return self._evaluate_local_rules(state, error_msg=error_msg or "All Clef endpoints failed")

        try:
            # Parse action_decision safely
            action_data = answers.get("action_decision") or {}
            action_choice = str(action_data.get("choice") or "hold_watch")
            action_conf = _safe_float(action_data.get("confidence"), 0.0)
            raw_probs = action_data.get("probabilities") or {}
            action_probs = {k: _safe_float(v, 0.0) for k, v in raw_probs.items()}

            # Parse conviction_score safely
            conviction_data = answers.get("conviction_score") or {}
            conviction_score = _safe_float(conviction_data.get("score"), 2.5)

            # Parse is_favorable_entry safely
            noul_data = answers.get("is_favorable_entry") or {}
            noul_val = _safe_float(noul_data.get("noul"), 0.5)

            # Generate readable decision tag
            prob_pct = int(action_probs.get(action_choice, action_conf) * 100)
            if action_choice == "strong_buy":
                tag = f"🤖AI強買({prob_pct}%)"
            elif action_choice == "buy":
                tag = f"🤖AI看好({prob_pct}%)"
            elif action_choice == "hold_watch":
                tag = "🤖AI建議觀察"
            elif action_choice == "avoid":
                tag = f"⚠️AI建議避開({prob_pct}%)"
            else:
                tag = "🤖AI中立"

            return ClefDecisionVerdict(
                ticker=ticker,
                action_decision=action_choice,
                action_confidence=action_conf,
                action_probabilities=action_probs,
                conviction_score=conviction_score,
                is_favorable_entry=noul_val,
                tag=tag,
                source_slot=slot_name,
                success=True
            )
        except Exception as parse_err:
            logger.warning(f"Error parsing answers from {slot_name}: {parse_err}, degrading to local rules...")
            return self._evaluate_local_rules(state, error_msg=f"Answer parsing error: {parse_err}")

    def evaluate_candidates_batch(
        self,
        candidates: List[Dict[str, Any]],
        macro_regime: Optional[str] = None,
        max_workers: int = 5
    ) -> Dict[str, ClefDecisionVerdict]:
        """
        Evaluate multiple candidates in parallel.
        Returns mapping of ticker -> ClefDecisionVerdict.
        """
        if not candidates:
            return {}

        # When remote inference is disabled, still provide usable local-rules decisions for each candidate
        if not self.enabled:
            return {
                cand.get("ticker", ""): self.evaluate_stock(cand, macro_regime)
                for cand in candidates
                if cand.get("ticker")
            }

        results: Dict[str, ClefDecisionVerdict] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cand = {
                executor.submit(self.evaluate_stock, cand, macro_regime): cand
                for cand in candidates
            }
            for future in as_completed(future_to_cand):
                cand = future_to_cand[future]
                ticker = cand.get("ticker", "")
                try:
                    verdict = future.result()
                    results[ticker] = verdict
                except Exception as e:
                    logger.error(f"Error evaluating {ticker} with Clef: {e}")
                    results[ticker] = self._evaluate_local_rules(
                        self.format_stock_state(cand, macro_regime),
                        error_msg=str(e)
                    )

        return results
