# ClinVar Conflict Explorer

An independent bioinformatics ML project that asks whether population allele
frequencies and molecular annotations help rank variants with conflicting
classifications in a historical ClinVar snapshot.

## Run locally

Use Python 3.10 or newer. Download and unzip the
[Genetic Variant Classifications dataset](https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting)
by Kevin Arvai. The CSV used for the included results has SHA-256
`53061e207d437af397028760be5bc566d216f6932e21db5a10322b843c41689a`.

```sh
python -m pip install -r requirements.txt
python train.py /path/to/clinvar_conflicting.csv
```

Open `results/demo.html` in a browser. It works offline and includes a
reproducible sample of 1,200 held-out predictions with gene search and an
outcome filter. The training command also generates `results/metrics.json`
and `results/held_out_predictions.csv` with all eligible variants and their
assigned folds. The original dataset is available separately through Kaggle.

## Evaluation

The historical snapshot has 65,188 rows. Removing 16 variants without a
usable gene symbol leaves 65,172 variants across 2,328 genes. The script
checks that chromosome, position, reference, and alternate allele form a
unique key. It uses five stratified group folds, with each gene appearing
in exactly one evaluation fold. Imputation, scaling, and category encoding
are fitted inside each training fold.

The benchmark compares a prevalence-only baseline with logistic regression
on three population allele frequencies, then adds molecular consequence,
predicted impact, and variant type. The model inputs are explicitly
allowlisted. Gene symbols, coordinates, database identifiers, clinical
assertions, and review metadata are excluded from model inputs.

| Model | Mean average precision | Mean ROC AUC |
|---|---:|---:|
| Fold prevalence | 0.252 | 0.500 |
| Population frequencies | 0.299 | 0.574 |
| Frequencies + annotations | 0.319 | 0.601 |

Annotations add 0.019 average precision over the frequencies-only model.
The resulting ranking signal is modest. These results describe disagreement
recorded in this snapshot. They do not establish pathogenicity or show that
the model predicts disagreements that arise in later ClinVar releases.
The project is a research demonstration and the scores have no clinical use.

## Data credit

Kevin Arvai, [Genetic Variant Classifications](https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting),
Kaggle (CC0). Original resource: [NCBI ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/).
