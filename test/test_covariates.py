"""Contract tests for the independent cross-market covariate pipeline."""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from data.covariates import (
    COVARIATE_REGISTRY,
    CovariateBuilder,
    LiveSourceProvider,
    Observation,
    normalize_observations,
    get_ticker_industry_series,
)
from data.covariate_store import CovariateStore


def obs(series, day, value, published=None, available=None, revision="r1"):
    return Observation(
        series_id=series,
        observation_date=date.fromisoformat(day),
        value=value,
        source="fixture",
        unit="fixture-unit",
        published_at=datetime.fromisoformat(published).replace(tzinfo=timezone.utc) if published else None,
        available_at=datetime.fromisoformat(available).replace(tzinfo=timezone.utc) if available else None,
        revision=revision,
    )


def test_registry_supports_verified_industry_proxies_and_resolves_tickers():
    assert COVARIATE_REGISTRY["USD_TWD"].normalization == "local_currency_per_usd"
    assert COVARIATE_REGISTRY["USD_JPY"].normalization == "local_currency_per_usd"
    assert COVARIATE_REGISTRY["UST_10Y_2Y"].normalization == "percentage_points"
    assert COVARIATE_REGISTRY["TOPIX"].status in {"verified", "blocked", "partial"}
    assert "industry" in COVARIATE_REGISTRY
    assert COVARIATE_REGISTRY["industry"].status == "verified"
    assert all(COVARIATE_REGISTRY[f"INDUSTRY_{market}"].status == "verified" for market in ("TW", "US", "JP"))

    # 驗證 Sector ETF Proxy series 註冊完整
    for proxy_id in ("XLK", "XLF", "XLV", "XLY", "XLE", "0052.TW", "1625.T", "1622.T"):
        assert proxy_id in COVARIATE_REGISTRY
        assert COVARIATE_REGISTRY[proxy_id].status == "verified"
        assert COVARIATE_REGISTRY[proxy_id].proxy_of is not None

    # 驗證股票產業映射與極端輸入防禦
    assert get_ticker_industry_series("2330.TW") == "0052.TW"
    assert get_ticker_industry_series("2330.tw") == "0052.TW"
    assert get_ticker_industry_series("2454") == "0052.TW"
    assert get_ticker_industry_series("2881.TW") == "0055.TW"
    assert get_ticker_industry_series("9999.TW") == "0052.TW"  # 未知台股預設電子科技
    assert get_ticker_industry_series("NVDA") == "XLK"
    assert get_ticker_industry_series("AAPL") == "XLK"
    assert get_ticker_industry_series("JPM") == "XLF"
    assert get_ticker_industry_series("UNKNOWN_US") == "XLK"  # 未知美股預設科技
    assert get_ticker_industry_series("7203.T") == "1622.T"
    assert get_ticker_industry_series("7203.JP") == "1622.T"
    assert get_ticker_industry_series("8035.T") == "1625.T"
    assert get_ticker_industry_series("9984.T") == "1625.T"
    assert get_ticker_industry_series("UNKNOWN.T") == "1625.T"  # 未知日股預設電氣精密
    assert get_ticker_industry_series("") == "XLK"
    assert get_ticker_industry_series("   ") == "XLK"
    assert get_ticker_industry_series(None) == "XLK"


def test_fx_quotes_are_one_usd_in_local_currency_and_treasury_spread_uses_percentage_points():
    got = normalize_observations([
        obs("USD_TWD", "2026-10-30", 32.1),
        obs("USD_JPY", "2026-10-30", 150.0),
        obs("UST_2Y", "2026-10-30", 3.5),
        obs("UST_10Y", "2026-10-30", 4.1),
    ])
    assert got["USD_TWD"].value == pytest.approx(32.1)
    assert got["USD_JPY"].value == pytest.approx(150.0)
    assert got["UST_10Y_2Y"].value == pytest.approx(0.6)
    assert got["UST_10Y_2Y"].unit == "percentage_points"
    with pytest.raises(ValueError, match="USD_TWD"):
        normalize_observations([obs("USD_TWD", "2026-10-30", 1 / 32.1)])


def test_asof_builder_masks_future_publications_and_preserves_missing_values(tmp_path):
    store = CovariateStore(tmp_path / "covariates.duckdb")
    store.upsert_observations([
        obs("UST_2Y", "2026-10-30", 3.9, "2026-11-02T20:00:00", "2026-11-02T20:05:00"),
    ])
    builder = CovariateBuilder(store=store, provider=lambda *_: [], session_provider=lambda ticker, as_of, lookback: [date(2026, 10, 29), date(2026, 10, 30)])
    result = builder.build_market_features(
        ["2330.TW"], datetime(2026, 10, 30, 3, tzinfo=timezone.utc), lookback=2,
        profile="TW", series_ids=["UST_2Y"], persist_snapshot=True,
    )["2330.TW"]
    assert result.past_covariates.shape == (1, 2)
    assert np.isnan(result.past_covariates[0, 0])
    assert np.isnan(result.past_covariates[0, 1])
    assert result.mask.tolist() == [[False], [False]]
    assert result.status == "features_unavailable"
    assert result.metadata["as_of"] == "2026-10-30T03:00:00+00:00"
    assert result.metadata["snapshot_id"]


def test_upsert_is_idempotent_and_revisions_remain_queryable(tmp_path):
    store = CovariateStore(tmp_path / "covariates.duckdb")
    first = obs("SPY", "2026-10-30", 680.0, revision="vendor-a")
    revised = obs("SPY", "2026-10-30", 681.0, revision="vendor-b")
    assert store.upsert_observations([first]) == 1
    assert store.upsert_observations([first]) == 0
    assert store.upsert_observations([revised]) == 1
    rows = store.get_observations("SPY", include_revisions=True)
    assert {r.revision for r in rows} == {"vendor-a", "vendor-b"}


def test_sync_retries_failed_series_without_overwriting_good_rows(tmp_path):
    from data.covariates import CovariateSynchronizer

    store = CovariateStore(tmp_path / "covariates.duckdb")
    previous = obs("UST_2Y", "2026-10-30", 3.5)
    store.upsert_observations([previous])
    calls = {"bad": 0}

    def provider(series, start, end):
        if series == "UST_10Y":
            calls["bad"] += 1
            raise RuntimeError("upstream unavailable")
        return [obs(series, "2026-10-31", 3.6)]

    sync = CovariateSynchronizer(store=store, provider=provider, retries=2, sleep=lambda _: None)
    report = sync.sync(["UST_2Y", "UST_10Y"], start=date(2021, 1, 1), end=date(2026, 10, 31))
    assert report["inserted"] == 1
    assert report["failed"]["UST_10Y"] == "upstream unavailable"
    assert calls["bad"] == 3
    assert len(store.get_observations("UST_2Y")) == 2


def test_source_empty_is_reported_not_filled_with_zero(tmp_path):
    from data.covariates import CovariateSynchronizer

    sync = CovariateSynchronizer(store=CovariateStore(tmp_path / "covariates.duckdb"), provider=lambda *_: [], retries=0)
    report = sync.sync(["TOPIX"], start=date(2021, 1, 1), end=date(2026, 10, 31))
    assert report["failed"]["TOPIX"] == "empty source response"


def test_each_historical_session_uses_its_own_point_in_time_cutoff(tmp_path):
    store = CovariateStore(tmp_path / "covariates.duckdb")
    store.upsert_observations([
        obs("VIX", "2026-10-29", 18.0, "2026-10-29T20:00:00", "2026-10-29T20:01:00", "old"),
        # A same observation-date revision arrived after the next target session cutoff.
        obs("VIX", "2026-10-29", 99.0, "2026-10-30T21:00:00", "2026-10-30T21:01:00", "late-revision"),
        obs("VIX", "2026-10-30", 21.0, "2026-10-30T20:00:00", "2026-10-30T20:01:00", "next"),
    ])
    builder = CovariateBuilder(
        store=store,
        provider=lambda *_: [],
        session_provider=lambda ticker, as_of, lookback: [date(2026, 10, 29), date(2026, 10, 30)],
    )
    result = builder.build_market_features(
        ["AAPL"], datetime(2026, 10, 31, 12, tzinfo=timezone.utc),
        lookback=2, profile="US", series_ids=["VIX"], persist_snapshot=False,
    )["AAPL"]
    assert result.values[:, 0].tolist() == [18.0, 21.0]
    assert result.mask[:, 0].tolist() == [True, True]
    assert result.metadata["row_cutoffs"][0] < "2026-10-30T21:01:00+00:00"


def test_rejects_nonfinite_observations_and_snapshot_json(tmp_path):
    store = CovariateStore(tmp_path / "covariates.duckdb")
    with pytest.raises(ValueError, match="finite"):
        store.upsert_observations([obs("VIX", "2026-10-30", float("nan"))])
    with pytest.raises(ValueError, match="JSON"):
        store.save_snapshot("bad", "AAPL", datetime(2026, 10, 30, tzinfo=timezone.utc), "US-v1", {"x": float("inf")})


def test_cross_market_freshness_counts_source_sessions_and_masks_stale_value(tmp_path):
    store = CovariateStore(tmp_path / "covariates.duckdb")
    store.upsert_observations([
        obs("VIX", "2026-10-20", 19.0, "2026-10-20T20:00:00", "2026-10-20T20:01:00"),
    ])
    builder = CovariateBuilder(
        store=store, provider=lambda *_: [], max_source_sessions=5,
        session_provider=lambda ticker, as_of, lookback: [date(2026, 10, 29), date(2026, 10, 30)],
    )
    result = builder.build_market_features(
        ["2330.TW"], datetime(2026, 10, 31, 12, tzinfo=timezone.utc),
        lookback=2, profile="TW", series_ids=["VIX"], persist_snapshot=False,
    )["2330.TW"]
    assert result.mask[:, 0].tolist() == [False, False]
    assert np.isnan(result.values[:, 0]).all()
    statuses = result.metadata["source_lineage"]["VIX"]["target_session_status"]
    assert statuses["2026-10-29"]["status"] == "stale"
    assert statuses["2026-10-29"]["age_sessions"] > 5


def test_concurrent_same_database_writes_are_serialized_and_idempotent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = CovariateStore(tmp_path / "covariates.duckdb")
    row = obs("VIX", "2026-10-30", 20.0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(lambda _: store.upsert_observations([row]), range(4)))
    assert sum(counts) == 1
    assert len(store.get_observations("VIX")) == 1


def test_fred_fx_adapter_uses_exact_usd_local_currency_series(monkeypatch):
    class Response:
        text = "observation_date,DEXTAUS\n2026-10-29,32.5000\n"

        def raise_for_status(self):
            return None

    requests_seen = {}

    def fake_get(url, *, params, timeout):
        requests_seen.update(url=url, params=params, timeout=timeout)
        return Response()

    monkeypatch.setattr("data.covariates.requests.get", fake_get)
    rows = LiveSourceProvider()("USD_TWD", date(2026, 10, 29), date(2026, 10, 29))
    assert requests_seen["params"]["id"] == "DEXTAUS"
    assert len(rows) == 1
    assert rows[0].value == pytest.approx(32.5)
    assert rows[0].unit == "TWD_per_USD"
    assert rows[0].available_at > rows[0].published_at


def test_h10_exchange_rate_is_unavailable_until_weekly_federal_release(monkeypatch):
    class Response:
        text = "observation_date,DEXTAUS\n2026-09-04,32.5000\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("data.covariates.requests.get", lambda *args, **kwargs: Response())
    rows = LiveSourceProvider()("USD_TWD", date(2026, 9, 4), date(2026, 9, 4))
    # September 7 is US Labor Day; the H.10 release moves to Tuesday 16:15 ET.
    assert rows[0].published_at >= datetime(2026, 9, 8, 20, 15, tzinfo=timezone.utc)
    assert rows[0].available_at > rows[0].published_at


def test_fred_revision_without_vintage_is_not_visible_before_ingestion():
    row = obs("USD_TWD", "2025-01-02", 32.5, "2025-01-06T21:15:00", "2025-01-07T13:00:00")
    row.source = "FRED"
    row.quality = "estimated_availability"
    row.ingested_at = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
    store = type("Store", (), {"get_observations": lambda self, *a, **k: [row]})()
    builder = CovariateBuilder(store=store, provider=lambda *_: [],
        session_provider=lambda *args: [date(2025, 1, 8)])
    result = builder.build_market_features(["2330.TW"], datetime(2026, 11, 3, 12, tzinfo=timezone.utc),
        lookback=1, profile="TW", series_ids=["USD_TWD"], persist_snapshot=False)["2330.TW"]
    assert result.status == "features_unavailable"
    assert not result.mask[0, 0]


@pytest.mark.parametrize("series,ticker,hour", [("TWII", "^TWII", 13), ("N225", "^N225", 15), ("TOPIX", "^TOPX", 15)])
def test_yahoo_index_uses_its_exchange_timezone(monkeypatch, series, ticker, hour):
    frame = pd.DataFrame({"Close": [100.]}, index=pd.to_datetime(["2026-10-30"]))
    monkeypatch.setattr("yfinance.download", lambda *a, **k: frame)
    rows = LiveSourceProvider()._fetch_yahoo(ticker, series, date(2026, 10, 30), date(2026, 10, 30))
    assert rows[0].published_at.hour == hour
    assert str(rows[0].published_at.tzinfo) in ("Asia/Taipei", "Asia/Tokyo")


def test_fred_fx_source_store_and_feature_builder_keep_historical_vintages(tmp_path):
    store = CovariateStore(tmp_path / "covariates.duckdb")
    old = obs("USD_TWD", "2026-10-30", 31.0, "2026-10-30T12:00:00", "2026-10-30T12:01:00", "released")
    correction = obs("USD_TWD", "2026-10-30", 31.5, "2026-11-02T12:00:00", "2026-11-02T12:01:00", "later-correction")
    builder = CovariateBuilder(
        store=store,
        provider=lambda series, *_: [old, correction] if series == "USD_TWD" else [],
        session_provider=lambda ticker, as_of, lookback: [date(2026, 10, 30), date(2026, 11, 2)],
    )
    result = builder.build_market_features(
        ["AAPL"], datetime(2026, 11, 3, 12, tzinfo=timezone.utc),
        lookback=2, profile="US", series_ids=["USD_TWD"], persist_snapshot=False,
    )["AAPL"]
    assert result.values[:, 0].tolist() == [31.0, 31.5]
    assert result.mask[:, 0].tolist() == [True, True]
    assert result.metadata["source_lineage"]["USD_TWD"]["selected_by_target_session"]["2026-10-30"]["revision"] == "released"
    assert result.metadata["source_lineage"]["USD_TWD"]["selected_by_target_session"]["2026-11-02"]["revision"] == "later-correction"


def test_build_market_features_full_features_ready(tmp_path):
    store = CovariateStore(tmp_path / "covariates.duckdb")
    days = [date(2026, 10, 26) + timedelta(days=i) for i in range(5)]
    as_of = datetime(2026, 10, 30, 22, tzinfo=timezone.utc)

    # 建立一個返回有效觀察值的 provider，涵蓋所有需要的 series
    def mock_provider(sid, start, end):
        out = []
        cur = start
        target_sids = [sid]
        if sid.startswith("OHLCV:") and sid.count(":") == 1:
            target_sids = [f"{sid}:{f}" for f in ("open", "high", "low", "close", "volume", "adj_close")]
        while cur <= end:
            val = 100.0
            pub = datetime.combine(cur, datetime.min.time()).replace(tzinfo=timezone.utc)
            for s in target_sids:
                out.append(Observation(
                    series_id=s,
                    observation_date=cur,
                    value=val,
                    source="fixture",
                    unit="unit",
                    published_at=pub,
                    available_at=pub,
                    revision=f"rev-{cur.isoformat()}",
                    raw_value=val,
                    metadata={"vintage_verified": True},
                ))
            cur += timedelta(days=1)
        return out

    builder = CovariateBuilder(
        store=store,
        provider=mock_provider,
        session_provider=lambda ticker, asof, lookback: days,
    )

    # 測試美股 NVDA 在 profile="full_features" 下產出 full_features
    res_us = builder.build_market_features(["NVDA"], as_of, lookback=5, profile="full_features")["NVDA"]
    assert res_us.status == "full_features"
    assert res_us.missing_required == []
    assert res_us.past_covariates.shape[1] == 5
    assert np.isfinite(res_us.past_covariates).all()
    # 驗證 NVDA 產業 series 展開為 XLK
    assert "XLK" in res_us.features

    # 測試台股 2330.TW 在 profile="auto" 下產出 full_features
    res_tw = builder.build_market_features(["2330.TW"], as_of, lookback=5, profile="auto")["2330.TW"]
    assert res_tw.status == "full_features"
    assert res_tw.missing_required == []
    assert res_tw.past_covariates.shape[1] == 5
    assert np.isfinite(res_tw.past_covariates).all()
    # 驗證 2330.TW 產業 series 展開為 0052.TW
    assert "0052.TW" in res_tw.features

    # 測試日股 7203.T 在 profile="full_features" 下產出 full_features
    res_jp = builder.build_market_features(["7203.T"], as_of, lookback=5, profile="full_features")["7203.T"]
    assert res_jp.status == "full_features"
    assert res_jp.missing_required == []
    assert res_jp.past_covariates.shape[1] == 5
    assert np.isfinite(res_jp.past_covariates).all()
    # 驗證 7203.T 產業 series 展開為 1622.T
    assert "1622.T" in res_jp.features

    # 測試 profile="market_features"（大盤+總經+匯率+美債，不含產業/籌碼）
    res_market = builder.build_market_features(["NVDA"], as_of, lookback=5, profile="market_features")["NVDA"]
    assert res_market.status == "full_features"
    assert "XLK" not in res_market.features
    assert "SPY" in res_market.features

