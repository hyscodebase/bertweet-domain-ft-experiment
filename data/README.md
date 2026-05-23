# Data

Place CSV files in this directory using the paths configured in `configs/experiment.yaml`.

Required columns for sequence classification:

- `text`
- `label`

Optional metadata columns:

- `domain`
- `demographic_group`
- `perturbation_type`
- `source_id`

The code can create tiny synthetic smoke-test CSVs when files are missing. Those files are marked by their content and are suitable only for checking that the pipeline runs.
