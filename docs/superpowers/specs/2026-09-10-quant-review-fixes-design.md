# Quant Review Fixes Design

## Goal

修正最近直接推送到 `main` 的 Polymarket、TimesFM、版本資訊與 CI/CD 變更，讓 API、WebMCP、前端顯示、量化計算、快取失敗語意與自動更新流程使用一致且可驗證的契約。

## Scope

1. Polymarket：統一機率為 0–100 的百分比格式；統一市場欄位；把 `all` 視為不篩選；區分有效資料、空資料與上游失敗；只快取有效結果；縮小 DoH fallback 的副作用。
2. TimesFM：讓程式計算與 README/API/Agent Skill 的風險報酬定義一致，並以 horizon 預測作為 5 日策略命中判定。
3. API metadata：集中版本號，避免健康檢查、首頁與 manifest 顯示不同版本。
4. CI/CD：自動 yfinance 更新改為可驗證的 branch/PR 流程，不以 `[skip ci]` 繞過依賴更新後的測試與 image build；補上 workflow 的最小權限與 main 保護建議文件。
5. 文件與測試：同步更新 `README.md`、`CHANGELOG.md`、`docs/CHANGELOG.md`，並新增可重現前述回歸的測試。

## Data Contract

Polymarket response uses:

- `fed_real_money_odds`: category key to numeric percentage in `[0, 100]`.
- Each market has `probability` (top outcome percentage), `top_outcome`, `odds_percent`, `yes_prob`, `no_prob`, `volume_24h`, and `category`.
- `source` identifies the successful route (`2md_reader` or `doh_direct`).
- `success` is `false` when all upstream routes fail; the response includes an `error` and does not replace a last-known-good cache entry.
- `category=None` and `category="all"` both mean all categories.

## Error and Cache Behavior

Network and payload errors are isolated per upstream and per market. A valid non-empty result is cached for 15 minutes. An upstream failure returns an explicit failure response and may expose stale last-known-good data with `stale=true`; an empty normalized market list is treated as unavailable so it cannot overwrite a valid snapshot.

## TimesFM Contract

The configured horizon is the strategy horizon. Potential and P50/P10 risk-reward values are evaluated at the horizon endpoint. The P90 value remains available as an upside reference but is not used as the documented risk-reward numerator.

## Verification

- Focused unit tests cover Polymarket percentage rendering data, `all`, failure/stale behavior, and TimesFM horizon/P50 calculations.
- Full unittest discovery must pass locally.
- GitHub Actions must pass on the resulting commit before it is reported as complete.
