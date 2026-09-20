"""
Model Training and Registration with Experimentation Tracking
Loads train/test from the Hugging Face dataset repo, tunes six candidate
algorithms with MLflow experiment tracking, evaluates each on the held-out
test set, and registers the best model (by test F1-score) on the
Hugging Face model hub.
"""

import os
import json

import joblib
import mlflow
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
)
from xgboost import XGBClassifier

HF_USERNAME = "VamiSai"
DATASET_REPO = f"{HF_USERNAME}/visit-with-us-tourism-data"
MODEL_REPO = f"{HF_USERNAME}/visit-with-us-tourism-model"

HF_TOKEN = os.environ.get("HF_TOKEN")
TARGET_COL = "ProdTaken"
RANDOM_STATE = 42

mlflow.set_experiment("tourism-wellness-package")


def load_split(filename: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset", filename=f"processed/{filename}", token=HF_TOKEN
    )
    return pd.read_csv(path)


def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )


CANDIDATES = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"model__max_depth": [3, 5, 7, 10, None], "model__min_samples_split": [2, 5, 10],
         "model__min_samples_leaf": [1, 2, 4]},
    ),
    "Bagging": (
        BaggingClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [50, 100, 200], "model__max_samples": [0.6, 0.8, 1.0]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200, 300], "model__max_depth": [5, 10, 15, None],
         "model__min_samples_split": [2, 5, 10]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.01, 0.1, 0.5, 1.0]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.05, 0.1],
         "model__max_depth": [2, 3, 4]},
    ),
    "XGBoost": (
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.01, 0.05, 0.1],
         "model__max_depth": [3, 4, 5, 6], "model__subsample": [0.7, 0.85, 1.0]},
    ),
}


def main():
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    os.makedirs("tourism_project/model_building/models", exist_ok=True)

    train_df = load_split("train.csv")
    test_df = load_split("test.csv")
    X_train, y_train = train_df.drop(columns=TARGET_COL), train_df[TARGET_COL]
    X_test, y_test = test_df.drop(columns=TARGET_COL), test_df[TARGET_COL]

    preprocessor = build_preprocessor(X_train)

    results = []
    best_score, best_name, best_pipeline = -1, None, None

    for name, (estimator, param_grid) in CANDIDATES.items():
        print(f"\n=== Tuning {name} ===")
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
            search = RandomizedSearchCV(
                pipe, param_distributions=param_grid, n_iter=10, scoring="f1",
                cv=5, random_state=RANDOM_STATE, n_jobs=-1,
            )
            search.fit(X_train, y_train)

            best_est = search.best_estimator_
            y_pred = best_est.predict(X_test)
            y_proba = best_est.predict_proba(X_test)[:, 1]

            metrics = {
                "model": name,
                "best_params": json.dumps(search.best_params_),
                "cv_best_f1": round(search.best_score_, 4),
                "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
                "test_precision": round(precision_score(y_test, y_pred), 4),
                "test_recall": round(recall_score(y_test, y_pred), 4),
                "test_f1": round(f1_score(y_test, y_pred), 4),
                "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            }

            mlflow.log_params(search.best_params_)
            mlflow.log_metrics({
                "cv_best_f1": metrics["cv_best_f1"],
                "test_accuracy": metrics["test_accuracy"],
                "test_precision": metrics["test_precision"],
                "test_recall": metrics["test_recall"],
                "test_f1": metrics["test_f1"],
                "test_roc_auc": metrics["test_roc_auc"],
            })
            mlflow.sklearn.log_model(best_est, name, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

            results.append(metrics)
            print(metrics)

            if metrics["test_f1"] > best_score:
                best_score, best_name, best_pipeline = metrics["test_f1"], name, best_est

    results_df = pd.DataFrame(results).sort_values("test_f1", ascending=False)
    results_path = "tourism_project/model_building/models/experiment_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nBest model: {best_name} (test F1 = {best_score})")
    print(results_df[["model", "test_f1", "test_recall", "test_roc_auc"]])

    model_path = "tourism_project/model_building/models/best_model.joblib"
    joblib.dump(best_pipeline, model_path)

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=model_path, path_in_repo="best_model.joblib",
                     repo_id=MODEL_REPO, repo_type="model")
    api.upload_file(path_or_fileobj=results_path, path_in_repo="experiment_results.csv",
                     repo_id=MODEL_REPO, repo_type="model")
    print(f"Model registered at https://huggingface.co/{MODEL_REPO}")


if __name__ == "__main__":
    main()
