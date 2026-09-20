"""
Data Registration
Uploads the raw tourism.csv (already placed in tourism_project/data/) to a
Hugging Face dataset repo, so the rest of the pipeline reads from a
versioned, shared source instead of a local file.
"""

import os
from huggingface_hub import HfApi

HF_USERNAME = "VamiSai"
DATASET_REPO = f"{HF_USERNAME}/visit-with-us-tourism-data"

HF_TOKEN = os.environ.get("HF_TOKEN")
RAW_DATA_PATH = "tourism_project/data/tourism.csv"


def main():
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    api = HfApi(token=HF_TOKEN)
    print(f"Creating/reusing dataset repo: {DATASET_REPO}")
    api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)

    print(f"Uploading {RAW_DATA_PATH} ...")
    api.upload_file(
        path_or_fileobj=RAW_DATA_PATH,
        path_in_repo="raw/tourism.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Done: https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    main()
