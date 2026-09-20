"""
Hosting
Creates (or reuses) a Hugging Face Space configured for the Streamlit SDK,
pinned to Python 3.10 so pip installs pre-built wheels instead of
compiling scikit-learn/pandas from source (which caused build timeouts
under the Space's default Python 3.13).
"""

import os
from huggingface_hub import HfApi

HF_USERNAME = "VamiSai"
SPACE_REPO = f"{HF_USERNAME}/visit-with-us-tourism-app"

HF_TOKEN = os.environ.get("HF_TOKEN")
DEPLOY_DIR = "tourism_project/deployment"

SPACE_README = """---
title: Visit With Us Tourism Predictor
emoji: 🧳
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
python_version: "3.10"
app_file: app.py
pinned: false
---
"""


def main():
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    api = HfApi(token=HF_TOKEN)
    print(f"Creating/reusing Space: {SPACE_REPO}")
    api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="static", exist_ok=True)

    readme_path = os.path.join(DEPLOY_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(SPACE_README)

    for filename in ["app.py", "requirements.txt", "README.md"]:
        local_path = os.path.join(DEPLOY_DIR, filename)
        print(f"Uploading {filename} ...")
        api.upload_file(
            path_or_fileobj=local_path, path_in_repo=filename,
            repo_id=SPACE_REPO, repo_type="space",
        )

    print(f"Done. App will build at https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
