"""
Data Preparation
Loads the raw dataset FROM the Hugging Face dataset repo, cleans it,
caps outliers, splits it into train/test, saves both locally, and
uploads them back to the same dataset repo.
"""

import os
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

HF_USERNAME = "VamiSai"
DATASET_REPO = f"{HF_USERNAME}/visit-with-us-tourism-data"

HF_TOKEN = os.environ.get("HF_TOKEN")
TARGET_COL = "ProdTaken"
DROP_COLS = ["Unnamed: 0", "CustomerID"]
OUTLIER_COLS = ["MonthlyIncome", "DurationOfPitch"]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_raw_from_hub() -> pd.DataFrame:
    print(f"Downloading raw data from {DATASET_REPO} ...")
    local_path = hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset", filename="raw/tourism.csv", token=HF_TOKEN
    )
    return pd.read_csv(local_path)


def cap_outliers_iqr(df: pd.DataFrame, cols, factor: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"  {col}: capped {n_capped} outlier(s) to [{lower:.1f}, {upper:.1f}]")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        print(f"Dropped {before - len(df)} duplicate rows")

    print("Capping outliers (IQR method):")
    df = cap_outliers_iqr(df, OUTLIER_COLS)

    return df


def main():
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    df = load_raw_from_hub()
    print(f"Raw shape: {df.shape}")

    df_clean = clean_data(df)
    print(f"Clean shape: {df_clean.shape}")

    train_df, test_df = train_test_split(
        df_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df_clean[TARGET_COL]
    )

    train_path = "tourism_project/data/train.csv"
    test_path = "tourism_project/data/test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train {train_df.shape} -> {train_path}")
    print(f"Saved test  {test_df.shape} -> {test_path}")

    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=train_path, path_in_repo="processed/train.csv",
        repo_id=DATASET_REPO, repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=test_path, path_in_repo="processed/test.csv",
        repo_id=DATASET_REPO, repo_type="dataset",
    )
    print(f"Uploaded processed train/test to https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    main()
