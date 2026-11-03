# Forecast model adapters

## Runtime identities

| Adapter | Package source revision | Checkpoint revision | Role |
|---|---|---|---|
| TimesFM | `google-research/timesfm@e51928e27119cb17bebc005be2696b75e0a9e688` (`timesfm` 3.0.2) | `google/timesfm-3.0-pytorch@43046b85ec22d584a13f8098c2ed39c889e129c2` | `research` |
| Timestral | `mistralai/timestral@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` | `mistralai/Timestral-209M@9874d252ab0aae16f7f7c999f98f98d6ad7d8365` | `active` |
| MOIRAI-2 | `SalesforceAIResearch/uni2ts@cfd46d4510ed8896f263116f32928eede05b0a75` | `Salesforce/moirai-2.0-R-small@30f43ff08c8494f4943ae1521e9d4e94a0fbb389` | `shadow` |

Package revisions and Hugging Face revisions are independent. Runtime results expose both identities. TimesFM 2.5 is historical-only and its checkpoint is rejected by the TimesFM adapter. TimesFM 3.0 weights are restricted to noncommercial, nonproduction use; MOIRAI-2 weights are CC BY-NC 4.0. Timestral weights use Mistral's Modified MIT license, whose model grant excludes companies or employers with preceding-month consolidated revenue above US$20 million unless they hold a separate commercial license. The default `DEPLOYMENT_MODE=production` disables TimesFM3 and MOIRAI-2; set `DEPLOYMENT_MODE=research` explicitly for research runs. Timestral retains its active production role and defaults to eligible; deployments whose company or employer meets the revenue restriction should set `TIMESTRAL_PRODUCTION_ELIGIBLE=false` unless they hold a separate commercial license. MOIRAI-2 remains a shadow candidate and has no selection role.

The package source APIs were checked against the upstream repositories on 2026-11-03. The local environment used during this implementation lacked `torch`, `timesfm3`, and `uni2ts`; live checkpoint loading and device measurements therefore require the isolated remote runner. Unit tests use mocks only for adapter contract behavior and do not represent live inference evidence.

The pinned Timestral source requires PyTorch >=2.4, while the pinned Uni2TS source constrains PyTorch to >=2.1,<2.5. The shared requirement is `torch>=2.4,<2.5`; TimesFM 3.0 accepts that range. Uni2TS also constrains NumPy to the 1.26.x series, which the dependency resolver must preserve.

## Live checkpoint smoke in an isolated Linux container

The three pinned checkpoint files total about 2.20 GB: TimesFM 3.0 is 1.32 GB, Timestral-209M is 837,271,692 bytes, and MOIRAI-2 R-small is 45.6 MB. Reserve at least 10 GB of free isolated disk for model files, wheels, temporary downloads, and caches, plus 16 GB RAM and 4 CPU cores for the sequential CPU smoke. The models are loaded in separate processes so each JSON record reports a per-model peak RSS. A 2-series T+1/T+5 run takes less memory; the supplied smoke uses four series to exercise MOIRAI batching. These are conservative container limits, not measured runtime results.

The current developer Mac has 5.0 GiB free disk, 8 CPU cores, and 16 GiB RAM. It is below the isolated smoke disk allowance; keep all downloads and execution in the remote isolated image.

Example setup for a disposable Linux x86_64 image with Python 3.11 (run from the repository root):

```sh
set -eu
export SMOKE_ROOT=/tmp/forecast-model-smoke
mkdir -p "$SMOKE_ROOT"/{venv,hf,xdg,pip-cache,torch}
df -h "$SMOKE_ROOT"
nproc
free -h
export HF_HOME="$SMOKE_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export XDG_CACHE_HOME="$SMOKE_ROOT/xdg"
export PIP_CACHE_DIR="$SMOKE_ROOT/pip-cache"
export TORCH_HOME="$SMOKE_ROOT/torch"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
python3.11 -m venv "$SMOKE_ROOT/venv"
. "$SMOKE_ROOT/venv/bin/activate"
python -m pip install --upgrade pip==24.3.1
python -m pip install --dry-run --report "$SMOKE_ROOT/model-dependency-resolution.json" \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu \
  'torch==2.4.1+cpu' 'numpy==1.26.4' 'pandas==2.2.3' 'pytest==8.3.4' \
  'timesfm @ git+https://github.com/google-research/timesfm.git@e51928e27119cb17bebc005be2696b75e0a9e688' \
  'timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff' \
  'uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git@cfd46d4510ed8896f263116f32928eede05b0a75'
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.4.1+cpu'
python -m pip install 'numpy==1.26.4' 'pandas==2.2.3' 'pytest==8.3.4' \
  'timesfm @ git+https://github.com/google-research/timesfm.git@e51928e27119cb17bebc005be2696b75e0a9e688' \
  'timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff' \
  'uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git@cfd46d4510ed8896f263116f32928eede05b0a75'
python -m pip check
python -m pip freeze > "$SMOKE_ROOT/resolved-requirements.txt"
```

Run the mock tests and live weights as separate checks. Each live invocation uses the pinned default checkpoint revision, isolated Hugging Face cache, CPU-only PyTorch, four synthetic 128-step series, two past-only feature channels, and T+1/T+5 outputs:

```sh
DEPLOYMENT_MODE=research python -m pytest test/test_model_adapters.py test/test_timesfm_strategy.py test/test_timestral_strategy.py test/test_config.py -q
for model in timesfm timestral moirai; do
  DEPLOYMENT_MODE=research python -m test.model_adapter_smoke --model "$model"
done
```

The script reports load and inference wall time, process CPU time, peak RSS, package versions, model/checkpoint/package identities, and per-horizon result counts. Preserve each JSON line with the container image digest and resolved requirements for reproducibility. The live run is successful only when all four series yield finite results at both horizons with native quantiles; an installation or checkpoint failure is reported as a failed smoke, not as mock-test evidence.

## Covariate input contract

`forecast_batch(series_by_ticker, horizon=5, past_covariates=None)` accepts an optional mapping from ticker to a finite `float32` NumPy array shaped `[channels, time]`. Each adapter validates alignment before inference. TimesFM3 receives this layout as `past_only_covariates`; Timestral receives it as `past_covariates`; the MOIRAI2 adapter transposes it to Uni2TS `[time, channels]` past features. MOIRAI runs price-only and covariate batches separately because Uni2TS predictors use a fixed channel schema.

Strategies read `past_covariates`, `feature_profile`, `feature_status`, and `feature_snapshot` from `StockContext.metadata`. A `full_features` row without a valid matrix returns `features_unavailable` and does not silently become a price-only forecast. Results preserve model/checkpoint/package identity, role, feature profile/status/snapshot, and whether interval quantiles are native.

## Isolated remote verification

Run adapter contract tests in research mode:

```sh
DEPLOYMENT_MODE=research .venv/bin/python -m pytest test/test_model_adapters.py test/test_timesfm_strategy.py test/test_timestral_strategy.py test/test_config.py -q
```

The real checkpoint smoke must instantiate each wrapper with its pinned checkpoint revision, forecast a small fixed batch with T+1 and T+5, and record elapsed time and peak memory by model/device. A mock-only run is not live-checkpoint evidence. Do not enable automatic GPU provisioning; use an already available CPU/device and report dependency incompatibilities or resource limits.

MOIRAI-2 inference splits input stocks into chunks no larger than `MOIRAI_BATCH_SIZE` (default 4); the remote smoke should include a batch larger than this setting to verify chunking and result identity. The pinned Uni2TS 2.0 numpy padding path accepts one past-feature channel, so the adapter explicitly averages multiple supplied covariate channels into one channel and records this projection in the model metadata; it does not silently claim that all channels were consumed.
