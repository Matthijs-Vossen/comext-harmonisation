# comext-harmonisation

LT-style harmonisation of Comext CN8 trade data across code-vintage revisions.

## Scope
- Method baseline: *Harmonizing the Harmonized System* (Section 3, Eq. (1), and chaining composition).
- Unit of harmonisation: CN8 (not HS6).
- This repository consumes preprocessed parquet inputs; upstream extraction/preprocessing is external.

## Data Inputs
- Annual input (default): `data/extracted_annual_no_confidential/products_like`
- Monthly input (default): `data/extracted_no_confidential/products_like`
- Schema and processed-data notes: `data/metadata/PROCESSED.txt`

## Pipeline Entrypoints
- CLI runner: `python3 scripts/run_pipeline.py --config configs/pipeline/example.yaml`
- Programmatic orchestrator:
  - `comext_harmonisation.pipeline_runner.run_pipeline_from_config_path`
  - `comext_harmonisation.pipeline_runner.run_pipeline_with_config`
- Config loader: `comext_harmonisation.load_pipeline_config`

Pipeline stage order:
1. Estimate adjacent weights
2. Chain weights to target vintage
3. Apply chained weights to annual/monthly data

## Public API Stability
Primary public functions preserved:
- Estimation: `run_weight_estimation_for_period`, `run_weight_estimation_for_period_multi`
- Chaining: `chain_weights_for_year`, `build_chained_weights_for_range`
- Apply: `apply_weights_to_annual_period`, `apply_chained_weights_wide_for_range`, `apply_chained_weights_wide_for_month_range`
- Config: `load_pipeline_config`

## Method Invariants (Intentional Behavior)
1. Estimation sample uses imports flow (`FLOW="1"`); estimated weights are then applicable to both flows.
2. Strict revised-link handling hard-fails unresolved revised links when strict/fail flags are enabled.
3. Intermediate chain compositions are not normalized; only finalization paths normalize.
4. Apply paths finalize effective weights before multiplication.
5. Chaining universe checks use observed annual code universes.

## Internal Architecture
See [docs/architecture.md](docs/architecture.md) for layer boundaries and extension rules.

## Development
Run tests:
- `python3 -m pytest -q`

Current suite baseline in this refactor wave: `97 passed`.
