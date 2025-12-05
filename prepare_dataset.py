import os
import pandas as pd
from datasets import load_dataset

def prepare_imdb(output_path="data/reviews_imdb.csv", sample_size=None):
    """
    Download the IMDB dataset, combine train+test,
    map labels to 0/1, and save as a CSV.
    """
    print("Downloading IMDB dataset from HuggingFace...")
    dataset = load_dataset("imdb")

    # train and test are separate; we combine them
    train_data = dataset["train"]
    test_data = dataset["test"]

    def to_df(dsplit):
        return pd.DataFrame({
            "text": dsplit["text"],
            "label": dsplit["label"]  # 0 = negative, 1 = positive
        })

    train_df = to_df(train_data)
    test_df = to_df(test_data)

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # optional: shuffle
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # optional: take a subset for faster training
    if sample_size is not None:
        full_df = full_df.iloc[:sample_size]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_csv(output_path, index=False)

    print(f"Saved {len(full_df)} rows to {output_path}")


if __name__ == "__main__":
    # full dataset (~50k reviews) – good for final project
    prepare_imdb(output_path="data/reviews_imdb.csv", sample_size=None)

    # smaller subset for quick experiments (e.g., 5k reviews)
    prepare_imdb(output_path="data/reviews_imdb_5k.csv", sample_size=5000)
