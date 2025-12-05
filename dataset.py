import os
import pandas as pd

def load_review_data(csv_path: str) -> pd.DataFrame:
    """
    Load review data from a CSV file.
    Expects columns: 'text', 'label'
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    df = df.dropna(subset=["text", "label"])
    return df


if __name__ == "__main__":
    # We run this from the project root: C:\Users\reddy\OneDrive\reviewinsight
    csv_path = os.path.join("data", "reviews_small.csv")
    df = load_review_data(csv_path)
    print("Loaded rows:", len(df))
    print(df.head())
