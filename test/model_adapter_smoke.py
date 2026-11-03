"""Opt-in live checkpoint smoke; run explicitly, one model per process.

This is intentionally not named ``test_*.py`` so ordinary mock pytest runs do
not download model weights. See docs/model_adapters.md for the isolated setup.
"""
import argparse
import importlib.metadata
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

# Support both `python -m test.model_adapter_smoke` and direct file execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.moirai_model import MoiraiWrapper
from models.timesfm_model import TimesFMWrapper
from models.timestral_model import TimestralWrapper


def _peak_rss_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return peak / (1024 if sys.platform.startswith("linux") else 1024 * 1024)


def _series_fixture():
    rng = np.random.default_rng(20261103)
    tickers = ["SMOKE_A", "SMOKE_B", "SMOKE_C", "SMOKE_D"]
    series = {}
    covariates = {}
    for i, ticker in enumerate(tickers):
        returns = rng.normal(loc=0.0003 * (i + 1), scale=0.01, size=128)
        series[ticker] = 100.0 * np.exp(np.cumsum(returns)).astype(np.float32)
        trend = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
        covariates[ticker] = np.stack([trend, np.sin(np.arange(128) / 8.0)]).astype(np.float32)
    return series, covariates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("timesfm", "timestral", "moirai"), required=True)
    args = parser.parse_args()

    wrappers = {
        "timesfm": lambda: TimesFMWrapper(context_len=128, max_horizon=5, batch_size=2, device="cpu"),
        "timestral": lambda: TimestralWrapper(context_len=128, horizon=5, batch_size=2,
                                              device="cpu"),
        "moirai": lambda: MoiraiWrapper(context_len=128, batch_size=2, device="cpu"),
    }
    wrapper = wrappers[args.model]()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    load_started = time.perf_counter()
    if not wrapper.load_model():
        raise SystemExit(f"model load failed: {wrapper.last_status}")
    load_seconds = time.perf_counter() - load_started

    series, covariates = _series_fixture()
    inference_seconds = {}
    result_counts = {}
    for horizon in (1, 5):
        started = time.perf_counter()
        result = wrapper.forecast_batch(series, horizon=horizon, past_covariates=covariates)
        inference_seconds[str(horizon)] = time.perf_counter() - started
        result_counts[str(horizon)] = len(result)
        if set(result) != set(series):
            raise SystemExit(
                f"horizon {horizon} returned {sorted(result)}; expected {sorted(series)}; "
                f"status={wrapper.last_status}"
            )
        for ticker, forecast in result.items():
            if not np.isfinite(forecast["predicted_price"]):
                raise SystemExit(f"non-finite T+{horizon} output for {ticker}")
            if forecast.get("quantile_source") != "native":
                raise SystemExit(f"non-native quantile output for {ticker}")

    try:
        versions = {name: importlib.metadata.version(name) for name in
                    ("torch", "numpy", "timesfm", "timestral", "uni2ts")}
    except importlib.metadata.PackageNotFoundError:
        versions = {"note": "one adapter package does not expose installed metadata"}
    print(json.dumps({
        "model": args.model,
        "status": wrapper.last_status,
        "model_id": wrapper.model_id,
        "checkpoint_revision": wrapper.checkpoint_revision,
        "package_revision": wrapper.PACKAGE_REVISION,
        "versions": versions,
        "device": "cpu",
        "series_count": len(series),
        "covariate_channels": 2,
        "result_counts_by_horizon": result_counts,
        "load_wall_seconds": round(load_seconds, 3),
        "inference_wall_seconds_by_horizon": {
            horizon: round(seconds, 3) for horizon, seconds in inference_seconds.items()
        },
        "total_wall_seconds": round(time.perf_counter() - wall_started, 3),
        "process_cpu_seconds": round(time.process_time() - cpu_started, 3),
        "peak_rss_mib": round(_peak_rss_mib(), 1),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
