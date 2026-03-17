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
- `configs/analysis/chain_length.yaml`: chain-length diagnostics and delta plotting.
- `configs/analysis/share_stability.yaml`: baseline share-stability analysis.
- `configs/analysis/share_stability_filtered.yaml`: share-stability with stability filter enabled.
- `configs/analysis/stress_test.yaml`: long-horizon stress-test analysis.
- `configs/analysis/synthetic_persistence.yaml`: qualitative evidence analysis for challenging pre-history/afterlife product codes.

Canonical invocation:
```bash
python -m comext_harmonisation.cli.run_analysis --config configs/analysis/chain_length.yaml
```
