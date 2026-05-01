# Comext CN Harmonisation

Research software for LT-style harmonisation of Eurostat Comext CN8 trade data
across product-code vintage revisions.

The package estimates adjacent-year conversion weights from observed trade
shares, chains those weights to a target CN vintage, and applies the resulting
weights to annual or monthly Comext panels. It was developed during an MSc
thesis internship with the Port of Rotterdam to study longitudinal trade-series
harmonisation under changing product classifications.

The method follows the conversion-weight estimation and chaining approach of
Lukaszuk and Torun and adapts it to Comext CN8 data.

## What This Repository Does

CN product codes change over time. That makes direct longitudinal comparison
difficult: a code observed in one year may split, merge, disappear, or map only
ambiguously to codes in another year.

This repository provides a reproducible Python implementation for:

- parsing official CN concordance tables;
- identifying deterministic and ambiguous revision groups;
- estimating ambiguous conversion weights with LT-style constrained least
  squares;
- chaining adjacent conversion weights to a target vintage;
- applying chained weights to annual or monthly trade panels;
- writing diagnostics for missing coverage, row-sum checks, and application
  totals.

Upstream Comext download and Parquet preparation are intentionally handled
outside this package, typically by
[`comext-fetcher`](https://github.com/Matthijs-Vossen/comext-fetcher).

## Quickstart

Install the package in editable mode:

```bash
python -m pip install -e '.[dev]'
```

Run the test suite:

```bash
python -m pytest -q
```

Run the small demo pipeline:

```bash
comext-run-pipeline --config examples/demo_cn_revision/configs/demo_pipeline.yaml
```

The demo uses real `2007->2008` CN concordance topology with synthetic annual
trade values. It estimates adjacent weights, chains them to CN2008, and applies
them to two tiny annual input files. Generated outputs are written under
`examples/demo_cn_revision/outputs/`, which is ignored by git.

## Core Commands

Run an end-to-end pipeline from a YAML config. This requires prepared Comext
Parquet inputs at the paths specified by the config:

```bash
comext-run-pipeline --config configs/pipeline/example.yaml
```

Run adjacent-period estimation directly:

```bash
comext-run-estimation --period 20102011 --direction a_to_b --measure BOTH
```

Run a configured research diagnostic:

```bash
comext-run-analysis --config configs/analysis/chain_length.yaml
```

The pipeline command is the main public entry point. The direct estimation and
analysis commands are useful for advanced or research-specific workflows.

## Data Inputs

This repository includes the CN concordance reference file used by the package:

```text
data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls
```

It does not include Comext trade data. Full runs expect prepared Parquet files,
usually produced by
[`comext-fetcher`](https://github.com/Matthijs-Vossen/comext-fetcher), in
directories such as:

```text
data/extracted_annual_no_confidential/products_like/
data/extracted_no_confidential/products_like/
```

The expected annual schema is:

```text
REPORTER, PARTNER, TRADE_TYPE, PRODUCT_NC, FLOW, PERIOD, VALUE_EUR, QUANTITY_KG
```

Monthly inputs use the same product and measure columns with monthly `PERIOD`
values.

### Using Full Comext Data

For real Comext runs, use `comext-fetcher` first to download, validate, and
convert Eurostat bulk archives to products-like Parquet. Its default
no-confidential outputs line up with this repository's default input paths:

```text
comext-fetcher output                                       comext-harmonisation input
data/non_confidential/extracted_annual/products_like/  ->   data/extracted_annual_no_confidential/products_like/
data/non_confidential/extracted/products_like/         ->   data/extracted_no_confidential/products_like/
```

In practice, copy or symlink the relevant `products_like/` directories from the
fetcher workspace into this repository before running a full pipeline, or adjust
`paths.annual_base_dir` and `paths.monthly_base_dir` in your pipeline config to
point directly at the fetcher outputs.

The small demo under `examples/demo_cn_revision/` does not require
`comext-fetcher`; it uses checked-in synthetic Parquet inputs.

## Configuration

Pipeline configs live in `configs/pipeline/`. Important settings include:

- `years.start`, `years.end`, `years.target`: processed year range and target CN
  vintage;
- `measures`: `VALUE_EUR`, `QUANTITY_KG`, or both;
- `stages`: enable estimation, chaining, annual apply, and monthly apply;
- `paths`: concordance, input data, weight cache, and run-output locations;
- `estimation.flow`: Comext flow used to estimate shares;
- `chaining.*` and `apply.*`: validation, finalisation, and missing-code
  behaviour.

The checked-in configs are examples and research workflows. For a new run, copy
one and adjust paths and years explicitly.

## Repository Layout

```text
.
├── src/comext_harmonisation/
│   ├── concordance/    # Parse CN concordances and build revision groups
│   ├── estimation/     # Build share matrices and estimate adjacent weights
│   ├── chaining/       # Compose adjacent weights into target-vintage chains
│   ├── apply/          # Apply chained weights to annual/monthly panels
│   ├── pipeline/       # Config loading and end-to-end orchestration
│   ├── weights/        # Weight schema, validation, finalisation, and I/O
│   ├── analysis/       # Optional research diagnostics
│   ├── cli/            # Console command entry points
│   └── core/           # Shared code normalisation and diagnostics helpers
├── configs/            # Pipeline and analysis configs
├── examples/           # Small runnable demo data and config
├── tests/              # Unit and integration tests
├── pyproject.toml      # Package metadata and tool configuration
└── README.md
```

## Research Diagnostics

The `analysis/` package contains diagnostics used during the thesis work:
chain-length sensitivity, link-distribution summaries, revision validation,
sampling robustness, synthetic persistence checks, and related plots/tables.

These modules are retained because they document and test downstream research
workflows, but they are not required for the core harmonisation pipeline.

## Method Reference

This implementation follows and adapts the method described in:

Lukaszuk, Piotr and Torun, David, *Harmonizing the Harmonized System*.
Available at SSRN: https://ssrn.com/abstract=4302540 or
http://dx.doi.org/10.2139/ssrn.4302540

## Development

Use Python 3.10 or newer.

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
python -m pytest -q
```

GitHub Actions runs the same checks on Python 3.10 and 3.12.

## Limitations

- Demo trade values are synthetic and are only intended to exercise the pipeline.
- Full-data runs require local Comext Parquet inputs, typically produced by
  `comext-fetcher`, that are not redistributed in this repository.
- Estimated weights depend on observed trade shares, the selected flow, and the
  quality and interpretation of the CN concordance.
- The implementation adapts an HS-focused method to CN8 data; CN-specific
  assumptions and validation choices matter.
- Identity fallback and strict revised-link validation settings affect coverage
  and should be chosen deliberately for substantive analysis.
- Full runs can be disk- and data-intensive.

## License

Code in this repository is released under the MIT License. See `LICENSE`.
