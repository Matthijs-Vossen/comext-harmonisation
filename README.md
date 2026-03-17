# comext-harmonisation

Research pipeline for LT-style harmonisation of Comext CN8 trade data across product-code vintage revisions.

## Research Context and Objective
This repository supports thesis work on replicating and adapting the method in *"Harmonizing the Harmonized System"* to Comext CN8 data. The goal is to produce classification-consistent trade series over time, despite CN code revisions.

The pipeline is organized as:
1. Estimate adjacent-vintage conversion weights.
2. Chain adjacent weights to a chosen target vintage.
3. Apply chained weights to annual and/or monthly trade panels.

## Method Baseline and Adaptation to CN8

| LT baseline | This repository | Why |
| --- | --- | --- |
| LT Section 3 and Eq. (1) for conversion weights and chaining | Same estimation/chaining logic is implemented in Python modules and pipeline stages | Preserve method fidelity while making runs reproducible |
| HS-focused framing in paper context | Applied to Comext CN8 codes | Thesis scope and dataset are CN8-based |
| Method description abstracted from specific ETL system | Upstream extraction/preprocessing handled outside this repo (`comext-fetcher`) | Keep this repository focused on harmonisation logic |

## Data Dependencies and Upstream Boundary
This repository consumes prepared parquet inputs.

Default input paths:
- Annual: `data/extracted_annual_no_confidential/products_like`
- Monthly: `data/extracted_no_confidential/products_like`

Concordance and schema references:
- Concordance file: `data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls`
- Processed-data schema notes: `data/metadata/PROCESSED.txt`

Upstream boundary:
- Preparation of those parquet inputs is external to this repository (typically from `comext-fetcher`).

## Repository Layout
```text
.
├── src/comext_harmonisation/
│   ├── concordance/
│   ├── estimation/
│   ├── chaining/
│   ├── apply/
│   ├── pipeline/
│   ├── weights/
│   ├── analysis/
│   ├── cli/
│   └── core/
├── configs/
│   ├── pipeline/
│   └── analysis/
├── tests/
├── data/
├── outputs/
├── pyproject.toml
└── README.md
```

| Path | Purpose |
| --- | --- |
| `src/comext_harmonisation/concordance/` | Parse concordance tables, build groups, and derive deterministic mappings |
| `src/comext_harmonisation/estimation/` | Build LT share matrices and estimate adjacent conversion weights |
| `src/comext_harmonisation/chaining/` | Compose adjacent weights into target-vintage chains |
| `src/comext_harmonisation/apply/` | Apply adjacent/chained weights to annual and monthly trade data |
| `src/comext_harmonisation/pipeline/` | Typed config loading and end-to-end stage orchestration |
| `src/comext_harmonisation/weights/` | Weight schema, validation, I/O, and finalization |
| `src/comext_harmonisation/analysis/` | Post-harmonisation diagnostics and research analyses |
| `src/comext_harmonisation/cli/` | Module CLI entrypoints for pipeline, estimation, and analysis |
| `src/comext_harmonisation/core/` | Shared normalization, revised-link, and diagnostics helpers |

## Reproducible Setup
From repository root:

```bash
python3 -m pip install -e '.[dev]'
python3 -m pytest -q
```

Python requirement is defined in `pyproject.toml` (`>=3.9`).

## Run the Harmonisation Pipeline
Canonical module CLI:

```bash
python -m comext_harmonisation.cli.run_pipeline --config configs/pipeline/example.yaml
```

Full-range backward chaining run:

```bash
python -m comext_harmonisation.cli.run_pipeline --config configs/pipeline/estimate_chain_all_years_backward.yaml
```

Console-script equivalent (after editable install):

```bash
comext-run-pipeline --config configs/pipeline/example.yaml
```

## Run Estimation Only
Run adjacent-period estimation without full chain/apply stages:

```bash
python -m comext_harmonisation.cli.run_estimation --period 20102011 --direction a_to_b --measure BOTH
```

Console-script equivalent:

```bash
comext-run-estimation --period 20102011 --direction a_to_b --measure BOTH
```

## Run Analysis
Example chain-length analysis:

```bash
python -m comext_harmonisation.cli.run_analysis --config configs/analysis/chain_length.yaml
```

Synthetic-persistence qualitative evidence analysis:

```bash
python -m comext_harmonisation.cli.run_analysis --config configs/analysis/synthetic_persistence.yaml
```

Raw-data bilateral persistence analysis around the CN2007 break:

```bash
python -m comext_harmonisation.cli.run_analysis --config configs/analysis/bilateral_persistence_cn2007_raw.yaml
```

Console-script equivalent:

```bash
comext-run-analysis --config configs/analysis/chain_length.yaml
```

## Key Configuration Switches

| Key | Meaning | Default in loader |
| --- | --- | --- |
| `estimation.flow` | Flow used to build estimation shares | `"1"` |
| `chaining.strict_revised_link_validation` | Fail on unresolved revised-link coverage in chaining | `true` |
| `chaining.write_unresolved_details` | Persist unresolved revised-link detail rows during chaining | `true` |
| `apply.assume_identity_for_missing` | Inject identity rows for missing apply codes | `true` |
| `apply.fail_on_missing` | Raise if missing apply coverage remains | `true` |
| `apply.strict_revised_link_validation` | Validate unresolved revised-link coverage before apply | `true` |
| `chaining.finalize_weights` | Finalize chained weights at chain-output stage | `false` |
| `chaining.neg_tol`, `chaining.pos_tol`, `chaining.row_sum_tol` | Numerical tolerances for clamping/normalization checks | `1e-6`, `1e-10`, `1e-6` |

Note: checked-in example configs can override these defaults for specific runs.

## Outputs and Artifacts
Default artifacts written by the pipeline include:
- Adjacent estimation weights: `outputs/weights/adjacent/<period>/<direction>/<measure>/`
- Estimation diagnostics and summary: `outputs/weights/diagnostics/`, `outputs/weights/summary.csv`
- Per-run pipeline outputs: `outputs/runs/run_<timestamp>_CN<target>/`
- Chained weights and diagnostics inside a run: `<run_dir>/chain/`
- Applied annual/monthly wide outputs and summaries inside a run: `<run_dir>/apply/`

Strict-link diagnostics:
- Chaining unresolved details can be written to `.../chain/CN<target>/unresolved_details.csv`
- Apply unresolved details can be written to `.../apply/CN<target>/diagnostics/unresolved_details.csv`

Analysis artifacts:
- Bilateral-persistence outputs: `outputs/analysis/bilateral_persistence_cn2007_raw/{table.csv,table.tex,regression_details.csv,sample_diagnostics.csv}`
- Synthetic-persistence outputs: `outputs/analysis/synthetic_persistence_qualitative/{code_catalog.csv,candidate_series.csv,code_evidence.csv,qualitative_summary.png}`

The bilateral-persistence analysis is a CN-adapted LT Table 3 diagnostic built on raw reported data. It now reports three rows: a break-centered deterministic-all broad row (`All deterministically break-comparable CN codes`), a filtered `2006->2007` break-group broad row (`All break-group CN codes`), and an adjusted row defined as the ambiguous subset of that retained break universe (`Adjusted CN codes`). The new broad row uses the `2006` or `2007` break basis and includes both linked break-group concepts and deterministically comparable singleton concepts outside the break groups; the linked-group broad and adjusted rows remain restricted to the retained `2006->2007` break universe after nearby non-bijective revision filtering.

## Pragmatic Implementation Choices vs LT Baseline
1. Estimation sample is fixed to imports flow (`FLOW="1"`).
2. Deterministic concordance links are represented as fixed `weight=1.0`; only ambiguous links are estimated.
3. Chaining universe checks are based on observed annual code universes.
4. Strict revised-link validation can fail-fast when unresolved revised links are detected.
5. Unresolved revised-link details can be written as explicit diagnostics tables.
6. Intermediate chain compositions are intentionally left non-normalized.
7. Apply paths finalize effective weights before multiplication.
8. Apply can optionally use identity fallback for missing codes (`assume_identity_for_missing`), with strict fail mode available (`fail_on_missing`).

## Citation and Thesis Context (Template)
- LT paper citation: `TODO`
- Thesis citation: `TODO`
- Repository citation (version/commit): `TODO`
