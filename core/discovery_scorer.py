"""
core/discovery_scorer.py - 確定性 7 維多因子評分核心

依據以下權重公式計算 [0, 100] 閉區間之總分，並分流至四級推薦類別：
Total = clamp(0.20*S_src + 0.20*S_purity + 0.20*S_paper + 0.10*S_novelty + 0.10*S_sent + 0.10*S_setup - 0.10*S_risk, 0, 100)

分流閾值：
- propose_add: 未在現有池中，且 Total >= 75 且 S_purity >= 60 (review_status: pending)
- observe: 未在現有池中，且 ((60 <= Total < 75 且 S_purity >= 40) 或 (Total >= 75 且 40 <= S_purity < 60)) (review_status: pending)
- already_in_pool: 已在現有追蹤池中，且具有前沿訊號支持 (Total >= 50 且 S_purity >= 40) (review_status: approved)
- reject: Total < 50 或 S_purity < 40，或不符合上述條件 (review_status: rejected)
"""

import math
from dataclasses import dataclass
from typing import Any, Dict


def clamp(val: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """將數值限制在 [min_val, max_val] 區間，若非有限數值 (如 NaN, Inf) 則回傳 min_val"""
    if val is None:
        return min_val
    try:
        f_val = float(val)
        if not math.isfinite(f_val):
            return min_val
        return max(min_val, min(max_val, f_val))
    except (ValueError, TypeError):
        return min_val


@dataclass
class DiscoveryScoreResult:
    source_score: float
    purity_score: float
    paper_score: float
    novelty_score: float
    sentiment_score: float
    setup_score: float
    risk_score: float
    total_score: float
    recommendation: str
    review_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_score": round(self.source_score, 2),
            "purity_score": round(self.purity_score, 2),
            "paper_score": round(self.paper_score, 2),
            "novelty_score": round(self.novelty_score, 2),
            "sentiment_score": round(self.sentiment_score, 2),
            "setup_score": round(self.setup_score, 2),
            "risk_score": round(self.risk_score, 2),
            "total_score": round(self.total_score, 2),
            "recommendation": self.recommendation,
            "review_status": self.review_status,
        }


def compute_discovery_score(
    source_score: float,
    purity_score: float,
    paper_score: float,
    novelty_score: float,
    sentiment_score: float,
    setup_score: float,
    risk_score: float,
    in_pool: bool = False,
) -> DiscoveryScoreResult:
    """
    計算 7 維多因子綜合評分與推薦分類

    權重向量：
    - source_score: 0.20
    - purity_score: 0.20
    - paper_score: 0.20
    - novelty_score: 0.10
    - sentiment_score: 0.10
    - setup_score: 0.10
    - risk_score: -0.10
    """
    # 確保各單項因子限制在 [0, 100]，若為 NaN 則歸零
    s_src = clamp(source_score)
    s_purity = clamp(purity_score)
    s_paper = clamp(paper_score)
    s_novelty = clamp(novelty_score)
    s_sent = clamp(sentiment_score)
    s_setup = clamp(setup_score)
    s_risk = clamp(risk_score)

    # 線性組合計算
    raw_total = (
        0.20 * s_src
        + 0.20 * s_purity
        + 0.20 * s_paper
        + 0.10 * s_novelty
        + 0.10 * s_sent
        + 0.10 * s_setup
        - 0.10 * s_risk
    )
    total_score = clamp(raw_total, 0.0, 100.0)

    # 4 級分流邏輯
    if in_pool:
        if total_score >= 50.0 and s_purity >= 40.0:
            recommendation = "already_in_pool"
            review_status = "approved"
        else:
            recommendation = "reject"
            review_status = "rejected"
    else:
        if s_purity < 40.0 or total_score < 50.0:
            recommendation = "reject"
            review_status = "rejected"
        elif total_score >= 75.0 and s_purity >= 60.0:
            recommendation = "propose_add"
            review_status = "pending"
        elif (total_score >= 60.0 and s_purity >= 40.0) or (total_score >= 75.0 and s_purity >= 40.0):
            recommendation = "observe"
            review_status = "pending"
        else:
            recommendation = "reject"
            review_status = "rejected"

    return DiscoveryScoreResult(
        source_score=s_src,
        purity_score=s_purity,
        paper_score=s_paper,
        novelty_score=s_novelty,
        sentiment_score=s_sent,
        setup_score=s_setup,
        risk_score=s_risk,
        total_score=total_score,
        recommendation=recommendation,
        review_status=review_status,
    )
