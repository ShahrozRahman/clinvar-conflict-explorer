"""Benchmark models for historical ClinVar classification disagreement."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FREQUENCIES = ["AF_ESP", "AF_EXAC", "AF_TGP"]
ANNOTATIONS = ["Consequence", "IMPACT", "CLNVC"]
VARIANT = ["CHROM", "POS", "REF", "ALT"]
COLUMNS = VARIANT + ["SYMBOL", "CLASS"] + FREQUENCIES + ANNOTATIONS


def load_data(path):
    data = pd.read_csv(path, usecols=COLUMNS, dtype={"CHROM": str})
    data["SYMBOL"] = data["SYMBOL"].astype("string").str.strip().str.upper()
    data = data.loc[data["SYMBOL"].str.fullmatch(r"[A-Z0-9_.-]+", na=False)].copy()
    if set(data["CLASS"]) != {0, 1}:
        raise ValueError("CLASS must contain both binary labels without missing values.")
    if data[VARIANT].isna().any().any() or data.duplicated(VARIANT).any():
        raise ValueError("Variant keys must be complete and unique.")
    return data.reset_index(drop=True)


def logistic(categories):
    transformers = [
        ("frequency", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                    StandardScaler()), FREQUENCIES),
    ]
    if categories:
        transformers.append(
            ("annotation", OneHotEncoder(handle_unknown="ignore", min_frequency=10), categories)
        )
    return make_pipeline(ColumnTransformer(transformers), LogisticRegression(max_iter=1000))


def benchmark(data):
    inputs = data[FREQUENCIES + ANNOTATIONS].copy()
    inputs[FREQUENCIES] = np.log1p(inputs[FREQUENCIES] * 1_000_000)
    inputs[ANNOTATIONS] = inputs[ANNOTATIONS].fillna("missing")
    labels = data["CLASS"]
    genes = data["SYMBOL"]
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(inputs, labels, genes))
    folds = np.empty(len(data), dtype=int)
    for fold, (train, test) in enumerate(splits, 1):
        if set(genes.iloc[train]) & set(genes.iloc[test]):
            raise RuntimeError("Gene overlap detected.")
        folds[test] = fold

    models = {
        "Fold prevalence": DummyClassifier(strategy="prior"),
        "Population frequencies": logistic([]),
        "Frequencies + annotations": logistic(ANNOTATIONS),
    }
    results = {}
    scores = {}
    for name, model in models.items():
        scores[name] = cross_val_predict(model, inputs, labels, cv=splits, method="predict_proba")[:, 1]
        fold_results = [
            {"average_precision": average_precision_score(labels.iloc[test], scores[name][test]),
             "roc_auc": roc_auc_score(labels.iloc[test], scores[name][test])}
            for _, test in splits
        ]
        results[name] = {
            metric: float(np.mean([row[metric] for row in fold_results]))
            for metric in ("average_precision", "roc_auc")
        }
        results[name]["folds"] = fold_results
    return results, folds, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to the extracted Kaggle CSV.")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    data = load_data(args.csv)
    models, folds, scores = benchmark(data)
    metrics = {
        "variants": len(data),
        "genes": int(data["SYMBOL"].nunique()),
        "conflict_rate": float(data["CLASS"].mean()),
        "models": models,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions = data[VARIANT + ["SYMBOL", "CLASS"]].assign(
        fold=folds, frequency_score=scores["Population frequencies"],
        annotation_score=scores["Frequencies + annotations"],
    )
    predictions.to_csv(args.output / "held_out_predictions.csv", index=False)
    demo_rows = predictions.sample(min(1200, len(predictions)), random_state=42)
    demo_payload = json.dumps({"metrics": metrics, "rows": demo_rows.to_dict("records")})
    template = Path(__file__).with_name("demo_template.html").read_text(encoding="utf-8")
    (args.output / "demo.html").write_text(
        template.replace("/* PAYLOAD */", demo_payload.replace("<", "\\u003c")),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
