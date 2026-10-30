"""
test/test_discovery_scorer.py - 測試 7 維度評分核心與分流邏輯
"""

import pytest
from core.discovery_scorer import compute_discovery_score, clamp


def test_clamp_function():
    assert clamp(-10.0) == 0.0
    assert clamp(150.0) == 100.0
    assert clamp(72.5) == 72.5
    assert clamp(float('nan')) == 0.0
    assert clamp(float('inf')) == 0.0
    assert clamp(None) == 0.0


def test_exact_weighted_sum():
    # 0.20*100 + 0.20*100 + 0.20*100 + 0.10*100 + 0.10*100 + 0.10*100 - 0.10*0 = 90.0
    res = compute_discovery_score(
        source_score=100.0,
        purity_score=100.0,
        paper_score=100.0,
        novelty_score=100.0,
        sentiment_score=100.0,
        setup_score=100.0,
        risk_score=0.0,
        in_pool=False,
    )
    assert pytest.approx(res.total_score, 0.01) == 90.0
    assert res.recommendation == "propose_add"
    assert res.review_status == "pending"


def test_propose_add_boundary():
    # 恰好達標：Total >= 75.0, purity >= 60.0
    # 0.20*80 + 0.20*70 + 0.20*80 + 0.10*70 + 0.10*80 + 0.10*80 - 0.10*20
    # = 16 + 14 + 16 + 7 + 8 + 8 - 2 = 67 (不足 75)
    # 調整至 75:
    # 0.20*90 (18) + 0.20*80 (16) + 0.20*85 (17) + 0.10*80 (8) + 0.10*80 (8) + 0.10*80 (8) - 0.10*0 (0)
    # = 18 + 16 + 17 + 8 + 8 + 8 = 75.0
    res = compute_discovery_score(
        source_score=90.0,
        purity_score=80.0,
        paper_score=85.0,
        novelty_score=80.0,
        sentiment_score=80.0,
        setup_score=80.0,
        risk_score=0.0,
        in_pool=False,
    )
    assert pytest.approx(res.total_score, 0.01) == 75.0
    assert res.recommendation == "propose_add"

    # 若 total >= 75 但 purity < 60，應被降級 (不能為 propose_add)
    res_low_purity = compute_discovery_score(
        source_score=100.0,
        purity_score=50.0,  # 低於 60
        paper_score=100.0,
        novelty_score=100.0,
        sentiment_score=100.0,
        setup_score=100.0,
        risk_score=0.0,
        in_pool=False,
    )
    # raw_total = 20 + 10 + 20 + 10 + 10 + 10 = 80.0, 但 purity 為 50
    assert res_low_purity.total_score == 80.0
    assert res_low_purity.recommendation == "observe"


def test_observe_category():
    # Total 介於 60 ~ 74.99，且 purity >= 40
    res = compute_discovery_score(
        source_score=70.0,
        purity_score=60.0,
        paper_score=60.0,
        novelty_score=60.0,
        sentiment_score=60.0,
        setup_score=60.0,
        risk_score=10.0,
        in_pool=False,
    )
    # 0.20*70 (14) + 0.20*60 (12) + 0.20*60 (12) + 0.10*60 (6) + 0.10*60 (6) + 0.10*60 (6) - 0.10*10 (1) = 55 (reject)
    # 提升至 65:
    res2 = compute_discovery_score(
        source_score=80.0,
        purity_score=60.0,
        paper_score=70.0,
        novelty_score=60.0,
        sentiment_score=60.0,
        setup_score=60.0,
        risk_score=10.0,
        in_pool=False,
    )
    # 16 + 12 + 14 + 6 + 6 + 6 - 1 = 59 (reject)
    res3 = compute_discovery_score(
        source_score=80.0,
        purity_score=60.0,
        paper_score=80.0,
        novelty_score=70.0,
        sentiment_score=70.0,
        setup_score=70.0,
        risk_score=10.0,
        in_pool=False,
    )
    # 16 + 12 + 16 + 7 + 7 + 7 - 1 = 64.0
    assert pytest.approx(res3.total_score, 0.01) == 64.0
    assert res3.recommendation == "observe"


def test_already_in_pool():
    res = compute_discovery_score(
        source_score=90.0,
        purity_score=80.0,
        paper_score=85.0,
        novelty_score=70.0,
        sentiment_score=80.0,
        setup_score=80.0,
        risk_score=10.0,
        in_pool=True,
    )
    assert res.recommendation == "already_in_pool"
    assert res.review_status == "approved"


def test_reject_category():
    # 題材純度過低 (< 40)
    res1 = compute_discovery_score(
        source_score=90.0,
        purity_score=35.0,
        paper_score=90.0,
        novelty_score=80.0,
        sentiment_score=80.0,
        setup_score=80.0,
        risk_score=10.0,
        in_pool=False,
    )
    assert res1.recommendation == "reject"
    assert res1.review_status == "rejected"

    # 總分過低 (< 50)
    res2 = compute_discovery_score(
        source_score=40.0,
        purity_score=45.0,
        paper_score=40.0,
        novelty_score=30.0,
        sentiment_score=30.0,
        setup_score=30.0,
        risk_score=80.0,
        in_pool=False,
    )
    assert res2.total_score < 50.0
    assert res2.recommendation == "reject"


def test_clamping_limits():
    # 當各項正向因子皆達最大值 100 且 risk 為 0 時，總分為 90.0
    res_high = compute_discovery_score(
        source_score=120.0,
        purity_score=120.0,
        paper_score=120.0,
        novelty_score=120.0,
        sentiment_score=120.0,
        setup_score=120.0,
        risk_score=-50.0,
        in_pool=False,
    )
    assert res_high.total_score == 90.0
    assert res_high.source_score == 100.0
    assert res_high.risk_score == 0.0

    # 極大風險扣分導致為負數時應 clamp 在 0
    res_low = compute_discovery_score(
        source_score=0.0,
        purity_score=0.0,
        paper_score=0.0,
        novelty_score=0.0,
        sentiment_score=0.0,
        setup_score=0.0,
        risk_score=100.0,
        in_pool=False,
    )
    assert res_low.total_score == 0.0
