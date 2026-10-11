
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

try:
    from .dataset import load_metadata
except ImportError:
    from dataset import load_metadata


def split_dataframe(df, seed=42, train_ratio=0.70, val_ratio=0.15):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios.")

    counts = df["label"].value_counts()
    if len(counts) != 2 or counts.min() < 3:
        raise ValueError(
            "Need at least 3 samples in each class for stratified splitting."
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

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec-csv", required=True)
    parser.add_argument("--visa-csv", required=True)
    parser.add_argument("--mvtec-root", action="append", default=[])
    parser.add_argument("--visa-root", action="append", default=[])
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, csv_path, roots in [
        ("mvtec", args.mvtec_csv, args.mvtec_root),
        ("visa", args.visa_csv, args.visa_root),
    ]:
        print(f"\nProcessing {name}: {csv_path}")
        df = load_metadata(csv_path, roots=roots, strict=True)
        print("Usable images:", len(df))
        print("Class counts:\n", df["label"].value_counts().sort_index())

        train_df, val_df, test_df = split_dataframe(df, seed=args.seed)

        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            path = output_dir / f"{name}_supervised_{split_name}.csv"
            split_df.to_csv(path, index=False)
            print(f"Saved {path}: {len(split_df)} rows")


if __name__ == "__main__":
    main()
