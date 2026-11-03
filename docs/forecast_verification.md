# Forecast verification store

`data.forecast_store.ForecastStore(db_path)` keeps forecast evidence in a DuckDB database independent of the display-oriented predictions table. It creates its schema when instantiated for writes; importing the module does not create a database. Use `ForecastStore(path, read_only=True)` for queries that must not mutate the database.

## Python API

- `record_run(mapping)` stores an immutable run envelope. Repeating identical content is idempotent; reusing its ID with changed content raises `ValueError`.
- `record_forecast(mapping)` requires run, ticker, market, model revision, issue time, last complete bar, target date, horizon, and predicted price. A stable identity includes run, ticker, model/config revision, target, horizon, target type, and feature profile. Revisions remain separate records.
- `list_forecasts(filters=None)` and `get_forecast(id)` return the original forecast mappings. `query_records(as_of=None)` joins forecast and actual evidence for downstream scoring.
- `settle_forecast(id, actual)` appends a versioned actual. A changed source revision creates another version and invalidates existing performance snapshots; identical retries are idempotent. `mark_pending` preserves a reason such as `target_session_not_closed` or `target_price_missing`.
- `record_snapshot`, `record_health_snapshot`, `list_invalidated_snapshots`, `list_actuals`, and `enqueue_supabase` / `list_outbox` expose audit and optional outbox operations. Outbox writes remain local; this package does not send remote requests.

Payloads are checked with strict JSON serialization, including rejection of NaN and infinity. Do not place credentials in payloads.

## Session verification

`backtest.verification.resolve_target_session(last_complete_bar, market, horizon)` advances by exchange sessions using `MarketCalendar` for TW, US, and JP calendars. Session close checks use Taipei, New York (including daylight saving time), or Tokyo time zones; the US close helper accounts for standard Thanksgiving Friday, July 3, and Christmas Eve early closes. `verify_forecast` accepts daily bars and considers only the recorded target date. A later bar cannot replace a missing target price. Until that target session closes, the forecast remains pending.

The caller must supply a vendor-confirmed bar set and an appropriate adjustment basis. Vendor correction history is represented by `source_revision`; split/dividend corporate-action adjustment consistency remains the caller's responsibility. Calendar holiday data follows `MarketCalendar`'s existing cache/fallback behavior.

## CLI

Verify a local bar fixture; use `--dry-run` to open the database read-only and print the decision without writing:

```sh
python scripts/verify_forecasts.py --db /tmp/forecasts.duckdb --forecast-id fc_123 --bars bars.json --dry-run
```

Classify a legacy JSON array without writing. Rows without checkpoint revision, input cutoff, and issue timestamp are retained as `legacy_unverified` and excluded from trust ranking. Rows with lineage remain `replay` unless the source explicitly proves `evidence_type: prospective`:

```sh
python scripts/verify_forecasts.py --db /tmp/forecasts.duckdb --legacy-dry-run legacy.json
```

To check all due forecasts with yfinance adjusted daily closes, use:

```sh
python scripts/verify_forecasts.py --db /path/to/stock_quant.duckdb --all-due
```

The command verifies only the recorded target session after its market close. A missing bar or upstream failure stays pending and is retried on the next run; existing settled rows are skipped. A source correction currently requires an explicit re-verification and source revision. The command does not send data to Supabase. Production scheduling and correction detection are not configured by this slice.
