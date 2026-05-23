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

The default Hugging Face import command materializes:

- `nyu-mll/glue`, config `sst2`, as the baseline sentiment dataset
- `cardiffnlp/tweet_eval`, config `sentiment`, as the Twitter-domain sentiment dataset with neutral rows filtered out
- `stanfordnlp/imdb`, config `plain_text`, as the general-domain sentiment dataset

The generated `bias_eval.csv` and `robustness_eval.csv` are derived evaluation scaffolds from imported baseline examples. They are useful for exercising the metrics pipeline, but a serious fairness analysis should replace `bias_eval.csv` with a dataset that has real demographic annotations.
