# Demo CN Revision Pipeline

This demo runs the core harmonisation pipeline on a tiny synthetic annual trade
panel. It uses real `2007->2008` CN concordance topology from the checked-in CN
concordance file, but the trade values are synthetic.

The input data includes:

- a real 1:1 mapping: `29163600 -> 29161950`;
- a real 3:1 merge: `90241091`, `90241093`, `90241099 -> 90241090`;
- a real ambiguous 2:2 group: `57032011`, `57032019 -> 57032012`, `57032018`.

Run from the repository root:

```bash
comext-run-pipeline --config examples/demo_cn_revision/configs/demo_pipeline.yaml
```

The command estimates adjacent weights, chains them to CN2008, and applies them
to the annual 2007 and 2008 input files. Generated files are written under
`examples/demo_cn_revision/outputs/`, which is ignored by git.

The demo is intentionally small enough to run in seconds. It is meant to show
the mechanics and data contracts of the pipeline, not to reproduce thesis-scale
results.
