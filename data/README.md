# Seed data

`arithmetic_operations.csv` is the math seed (`operation,solution`).

It ships with this project; no parent-folder copy is required.

## Reproducibility samples

From a real loop run (`output/autoresearch/run`, iteration 0):

- `train_data_sample.csv` — the train pool actually used for training:
  the seed's train split (held-out rows excluded) plus generated rows that
  passed programmatic + high-reasoning validation.
- `validated_generated_sample.csv` — just the model-generated rows that
  survived validation in that iteration.

Both use the same `operation,solution` schema as the seed.
