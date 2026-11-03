from evaluators.composite_evaluator import CompositeEvaluator
from strategies.base import StrategyResult


def result(model, *, role="active", status="ready", price=110, hit=True):
    return StrategyResult(ticker="AAPL", strategy_name=model, is_hit=hit, score=90,
        current_price=100, predicted_price=price, potential=price - 100,
        signals={"risk_reward_ratio": 2}, metadata={"role": role, "status": status})


def row(outputs):
    return CompositeEvaluator().evaluate("SP500", outputs, {}).ranked_stocks[0]


def test_shadow_and_suspended_do_not_influence_target_or_consensus():
    outputs = {"timesfm": [result("TimesFM", status="suspended", price=150)],
               "timestral": [result("Timestral", price=110)],
               "moirai": [result("MOIRAI-2", role="shadow", price=200)]}
    item = row(outputs)
    assert item["predicted_price"] == 110
    assert not item["dual_model_consensus"]
    assert "🔮雙大模型共振" not in item["tags"]


def test_no_eligible_model_has_no_synthetic_target():
    item = row({"timesfm": [result("TimesFM", status="disabled")],
                "timestral": [result("Timestral", status="features_unavailable")]})
    assert item["predicted_price"] is None
    assert item["potential"] is None
    assert not item["dual_model_consensus"]
