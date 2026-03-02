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
- `configs/analysis/chain_length.yaml`: chain-length diagnostics and delta plotting.
- `configs/analysis/share_stability.yaml`: baseline share-stability analysis.
- `configs/analysis/share_stability_filtered.yaml`: share-stability with stability filter enabled.
- `configs/analysis/stress_test.yaml`: long-horizon stress-test analysis.

Canonical invocation:
```bash
python -m comext_harmonisation.cli.run_analysis --config configs/analysis/chain_length.yaml
```
