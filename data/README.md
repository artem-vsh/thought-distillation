# Loop seed data

`arithmetic_operations.csv` is a copy of the repo-root math dataset
`data/arithmetic_operations.csv` (`operation,solution`).

The autoresearch loop defaults to this path so the loop is self-contained.
Keep it in sync with `data/arithmetic_operations.csv` when the source set changes:

```bash
cp data/arithmetic_operations.csv data/arithmetic_operations.csv
```
