"""Create supervised and cross-dataset splits."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .dataset import load_metadata
except ImportError:
    from dataset import load_metadata


def split_dataframe(
    df,
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15,
):
    if (
        not 0 < train_ratio < 1
        or not 0 < val_ratio < 1
        or train_ratio + val_ratio >= 1
    ):
        raise ValueError("Invalid train/validation ratios.")

    counts = df["label"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    if (counts < 3).any():
        raise ValueError(
            "Need at least 3 images per class for stratification. "
            f"Counts: {counts.to_dict()}"
        )

    train_df, remainder = train_test_split(
        df,
        test_size=1 - train_ratio,
        random_state=seed,
        stratify=df["label"],
    )

    relative_val = val_ratio / (1 - train_ratio)

    val_df, test_df = train_test_split(
        remainder,
        test_size=1 - relative_val,
        random_state=seed,
        stratify=remainder["label"],
    )

    return tuple(
        part.reset_index(drop=True)
        for part in (train_df, val_df, test_df)
    )


def save_split(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve paths again when loading, so don't save machine-specific
    # resolved paths in the portable CSV.
    df.drop(
        columns=["resolved_image_path"],
        errors="ignore",
    ).to_csv(path, index=False)

    counts = df["label"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    print(
        f"Saved {path}: {len(df)} images; "
        f"normal={counts[0]}, defective={counts[1]}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mvtec-csv", required=True)
    parser.add_argument("--visa-csv", required=True)

    parser.add_argument(
        "--mvtec-root",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--visa-root",
        action="append",
        default=[],
    )

    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    datasets = {
        "mvtec": load_metadata(
            args.mvtec_csv,
            args.mvtec_root,
            strict=True,
        ),
        "visa": load_metadata(
            args.visa_csv,
            args.visa_root,
            strict=True,
        ),
    }

    split_map = {}

    # Independent splits for each dataset.
    for name, df in datasets.items():
        parts = split_dataframe(
            df,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

        split_map[name] = dict(
            zip(("train", "val", "test"), parts)
        )

        for split_name, split_df in split_map[name].items():
            save_split(
                split_df,
                output / f"{name}_supervised_{split_name}.csv",
            )

    # Primary supervised experiment: combine datasets, then split.
    combined = pd.concat(
        [
            df.assign(dataset=name)
            for name, df in datasets.items()
        ],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["resolved_image_path"]
    ).reset_index(drop=True)

    for split_name, split_df in zip(
        ("train", "val", "test"),
        split_dataframe(
            combined,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        ),
    ):
        save_split(
            split_df,
            output / f"combined_supervised_{split_name}.csv",
        )

    # Cross-dataset experiments:
    # Source train/validation; target test only.
    for source, target in [
        ("mvtec", "visa"),
        ("visa", "mvtec"),
    ]:
        experiment_dir = output / f"{source}_to_{target}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        save_split(
            split_map[source]["train"],
            experiment_dir / "train.csv",
        )
        save_split(
            split_map[source]["val"],
            experiment_dir / "val.csv",
        )
        save_split(
            split_map[source]["test"],
            experiment_dir / "source_test.csv",
        )
        save_split(
            split_map[target]["test"],
            experiment_dir / "target_test.csv",
        )

        print(f"Prepared experiment: {experiment_dir}")

    # Category overlap audit, if category metadata exists.
    if all("category" in df.columns for df in datasets.values()):
        mvtec_categories = set(
            datasets["mvtec"]["category"].dropna().astype(str)
        )
        visa_categories = set(
            datasets["visa"]["category"].dropna().astype(str)
        )

        audit = {
            "mvtec_categories": sorted(mvtec_categories),
            "visa_categories": sorted(visa_categories),
            "exact_category_overlap": sorted(
                mvtec_categories & visa_categories
            ),
        }

        pd.Series(audit, dtype=object).to_json(
            output / "category_overlap_audit.json",
            indent=2,
        )

        print(
            "Exact category overlap:",
            sorted(mvtec_categories & visa_categories),
        )

    print(f"Splits saved under: {output.resolve()}")


if __name__ == "__main__":
    main()