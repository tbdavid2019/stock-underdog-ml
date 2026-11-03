"""Regression tests for the verified performance view and history."""
import json
import subprocess
from pathlib import Path


def _computed(name, next_name, context):
    source = Path("api/static/js/app.js").read_text(encoding="utf-8")
    start = source.index(f"{name}() {{") + len(f"{name}() {{")
    end = source.index(f"{next_name}() {{", start)
    body = source[start:end].rstrip().removesuffix(",").rstrip().removesuffix("}")
    script = f"const f=new Function({json.dumps(body)}); console.log(JSON.stringify(f.call({json.dumps(context)})));"
    return json.loads(subprocess.run(["node", "-e", script], text=True, capture_output=True, check=True).stdout)


def test_history_does_not_use_next_available_prediction_as_actual():
    records = [
        {"timestamp": "2026-01-05T22:00:00Z", "current_price": 100, "model_name": "TimesFM", "predicted_price": 101},
        {"timestamp": "2026-01-08T22:00:00Z", "current_price": 120, "model_name": "TimesFM", "predicted_price": 121},
    ]
    rows = _computed("dailyHistoryRows", "modelScorecard", {"historyRecords": records})
    assert all(row["nextActualPrice"] is None for row in rows)
    assert all(row["winner"] is None for row in rows)


def test_history_uses_verified_actual_and_explicit_target_date():
    records = [{"timestamp": "2026-01-05T22:00:00Z", "current_price": 100,
                "model_name": "TimesFM", "predicted_price": 101,
                "verification_status": "settled", "target_date": "2026-01-06", "actual_price": 102}]
    rows = _computed("dailyHistoryRows", "modelScorecard", {"historyRecords": records})
    assert rows[0]["nextDate"] == "2026-01-06"
    assert rows[0]["nextActualPrice"] == 102


def test_history_keeps_per_model_verification_with_its_forecast():
    records = [
        {"timestamp": "2026-01-05T22:00:00Z", "forecast_session": "2026-01-05", "current_price": 100,
         "model_name": "TimesFM", "predicted_price": 101, "verification_status": "settled",
         "target_date": "2026-01-06", "actual_price": 102},
        {"timestamp": "2026-01-05T22:00:00Z", "forecast_session": "2026-01-05", "current_price": 100,
         "model_name": "Timestral", "predicted_price": 103, "verification_status": "pending",
         "target_date": "2026-01-06"},
    ]
    row = _computed("dailyHistoryRows", "modelScorecard", {"historyRecords": records})[0]
    assert row["timesfmErrorPct"] is not None
    assert row["timestralErrorPct"] is None
    assert row["winner"] != "Timestral 勝"


def test_rankings_are_visible_and_single_stock_uses_backend_scorecard():
    html = Path("api/templates/partials/tab_history.html").read_text(encoding="utf-8")
    js = Path("api/static/js/app.js").read_text(encoding="utf-8")
    assert "預測可信度排行榜" in html
    assert "股票可信榜" in html and "模型戰績榜" in html
    assert 'v-model="performanceMarket"' in html and 'v-model.number="performanceWindow"' in html
    assert "/api/v1/performance/stocks" in js and "/api/v1/performance/models" in js
    assert "/api/v1/performance/tickers/" in js
    assert "historyPerformance" in html
    assert 'v-if="!historyLoading && modelScorecard"' not in html


def _chart_data(records):
    script = r'''
        const fs = require('fs');
        const source = fs.readFileSync('api/static/js/chart-manager.js', 'utf8');
        const uri = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
        import(uri).then(m => console.log(JSON.stringify(m.buildVerifiedPriceChartData(JSON.parse(process.argv[1])))));
    '''
    return json.loads(subprocess.run(["node", "-e", script, json.dumps(records)],
        text=True, capture_output=True, check=True).stdout)


def test_chart_uses_target_date_instead_of_next_record_after_missed_runs():
    records = [
        {"timestamp": "2026-01-05T22:00:00Z", "forecast_session": "2026-01-05", "current_price": 100,
         "model_name": "TimesFM", "predicted_price": 101, "horizon": 1, "target_date": "2026-01-06",
         "verification_status": "pending"},
        {"timestamp": "2026-01-08T22:00:00Z", "forecast_session": "2026-01-08", "current_price": 120,
         "model_name": "TimesFM", "predicted_price": 121, "horizon": 1, "target_date": "2026-01-09",
         "verification_status": "pending"},
    ]
    data = _chart_data(records)
    assert data["dates"] == ["2026-01-05", "2026-01-06", "2026-01-08", "2026-01-09"]
    assert data["timesfm"] == [None, 101, None, 121]
    assert data["price"] == [100, None, 120, None]


def test_chart_legacy_forecast_without_explicit_target_has_no_guessed_point():
    records = [
        {"timestamp": "2026-01-05T22:00:00Z", "current_price": 100, "model_name": "TimesFM", "predicted_price": 101},
        {"timestamp": "2026-01-08T22:00:00Z", "current_price": 120, "model_name": "TimesFM", "predicted_price": 121},
    ]
    data = _chart_data(records)
    assert data["timesfm"] == [None, None]
    assert data["legacy_count"] == 2


def test_chart_moirai_shadow_uses_its_own_target():
    data = _chart_data([{"timestamp": "2026-01-05T22:00:00Z", "forecast_session": "2026-01-05",
        "current_price": 100, "model_name": "MOIRAI-2", "predicted_price": 103,
        "target_date": "2026-01-06", "verification_status": "pending", "role": "shadow"}])
    assert data["moirai"] == [None, 103]
    assert data["meta"]["moirai"][1]["status"] == "pending"
