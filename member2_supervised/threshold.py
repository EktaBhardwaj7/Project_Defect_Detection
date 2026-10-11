"""Select a decision threshold using validation predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


def select_threshold(
    y_true,
    probabilities,
    strategy="recall",
    min_recall=0.95,
):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal nonzero length.")

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probabilities must be finite and in [0, 1].")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")

    if len(np.unique(y_true)) < 2:
        raise ValueError("Validation data must contain both classes.")

    if not 0 <= min_recall <= 1:
        raise ValueError("min_recall must be in [0, 1].")

    thresholds = np.unique(
        np.r_[0.0, probabilities, 0.5, 1.0]
    )

    rows = []

    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)

        rows.append({
            "threshold": float(threshold),
            "precision": float(
                precision_score(
                    y_true, predicted, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_true, predicted, zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_true, predicted, zero_division=0
                )
            ),
        })

    table = pd.DataFrame(rows)

    if strategy == "recall":
        eligible = table[table["recall"] >= min_recall]

        if eligible.empty:
            raise RuntimeError(
                "No threshold meets the requested minimum recall."
            )

        best = eligible.sort_values(
            ["threshold", "precision"],
            ascending=[False, False],
        ).iloc[0]

    elif strategy == "f1":
        best = table.sort_values(
            ["f1", "threshold"],
            ascending=[False, False],
        ).iloc[0]

    else:
        raise ValueError("strategy must be 'recall' or 'f1'.")

    return best.to_dict(), table


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="threshold.json")
    parser.add_argument(
        "--strategy",
        choices=["recall", "f1"],
        default="recall",
    )
    parser.add_argument("--min-recall", type=float, default=0.95)

    args = parser.parse_args()

    df = pd.read_csv(args.predictions)

    required = {"true_label", "defect_probability"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    best, table = select_threshold(
        df["true_label"],
        df["defect_probability"],
        strategy=args.strategy,
        min_recall=args.min_recall,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(best, indent=2),
        encoding="utf-8",
    )

    table.to_csv(
        output.with_name("threshold_search_results.csv"),
        index=False,
    )

    print("Selected threshold:")
    print(json.dumps(best, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()