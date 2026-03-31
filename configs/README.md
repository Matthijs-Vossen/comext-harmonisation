# Config Catalog

This repository keeps explicit YAML scenarios under `configs/` for reproducible runs.

## Pipeline Configs
- `configs/pipeline/example.yaml`: small reference run (estimate + chain + annual apply).
- `configs/pipeline/run_2000_2010_cn2007.yaml`: period-focused run targeting CN2007 with monthly apply enabled.
- `configs/pipeline/estimate_chain_all_years_backward.yaml`: full-range backward-to-1988 estimation/chaining.
- `configs/pipeline/estimate_chain_all_years_forward.yaml`: full-range forward-to-2024 estimation/chaining.

Canonical invocation:
```bash
python -m comext_harmonisation.cli.run_pipeline --config configs/pipeline/example.yaml
```

## Analysis Configs
- `configs/analysis/bilateral_persistence_cn2007_raw.yaml`: raw-data CN analogue of LT Table 3 around the `2006->2007` break, using zero-completed pair-code panels with a break-centered deterministic-all broad row, a filtered break-group broad row, and an adjusted row defined by the union of non-`1:1` break groups within that retained universe.
- `configs/analysis/chained_link_distribution.yaml`: thesis-facing LT Table 1 style chained CN link-distribution figure using fixed anchor universes and observed-universe implied identities.
- `configs/analysis/chain_length.yaml`: chain-length diagnostics and delta plotting.
- `configs/analysis/crm_revision_exposure_2023.yaml`: CN2023-anchored CRM application diagnostic showing cumulative revision exposure for unique CRM-related codes versus the full observed CN2023 code universe.
- `configs/analysis/link_distribution_adjacent.yaml`: LT Table 1 style adjacent-break CN link-distribution summary for revised codes only, reported for both focal directions.
- `configs/analysis/link_distribution_adjacent_observed_universe.yaml`: adjacent-break CN link-distribution summary using the observed annual code universe with implied unchanged `1:1` identities.
- `configs/analysis/sampling_robustness_cn2007.yaml`: leave-one-bin-out LT-style adjacent-weight stability for the focal `2007->2006` CN analogue.
- `configs/analysis/share_stability.yaml`: baseline share-stability analysis.
- `configs/analysis/share_stability_filtered.yaml`: share-stability with stability filter enabled.
- `configs/analysis/stress_test.yaml`: long-horizon stress-test analysis.
- `configs/analysis/synthetic_persistence.yaml`: qualitative evidence analysis for challenging pre-history/afterlife product codes, with optional `candidates.display_labels` for human-readable figure titles.
- `configs/analysis/synthetic_persistence_thesis_examples.yaml`: thesis-ready 6-example synthetic-persistence figure using a reduced pre-history/afterlife code set.

Canonical invocation:
```bash
python -m comext_harmonisation.cli.run_analysis --config configs/analysis/chain_length.yaml
```
