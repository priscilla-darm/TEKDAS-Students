"""Prepare student dropout/outcome data for the EduSight ML workflow.

Run:
    python data_prep.py

Creates model-ready data splits and a data-quality report from
student_doa_dataset.csv. No customer/e-commerce assumptions are used.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_doa_dataset.csv"
TARGET = "Target"
ID_COLUMN = "student_record_id"

# These are categorical codes even though they are stored as numbers.
CATEGORICAL_COLUMNS = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]
FEATURE_COLUMNS = []


def load_source_data(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    frame[TARGET] = frame[TARGET].astype(str).str.strip()
    return frame


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in {TARGET, ID_COLUMN}]


def validate_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset has no rows.")
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")
    expected = {"Dropout", "Enrolled", "Graduate"}
    actual = set(df[TARGET].unique())
    if not actual.issubset(expected):
        raise ValueError(f"Unexpected target labels: {sorted(actual - expected)}")
    if df[TARGET].nunique() < 2:
        raise ValueError("At least two target classes are required for classification.")


def build_data_quality_report(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    features = infer_feature_columns(df)
    counts = df[TARGET].value_counts().to_dict()
    return {
        "source_file": DATA_FILE.name,
        "rows": int(len(df)),
        "columns_including_target": int(df.shape[1]),
        "feature_count_before_encoding": int(len(features)),
        "target_column": TARGET,
        "target_classes": sorted(df[TARGET].unique().tolist()),
        "target_counts": {str(k): int(v) for k, v in counts.items()},
        "target_percentages": {str(k): round(float(v / len(df) * 100), 3) for k, v in counts.items()},
        "missing_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "columns_with_missing_values": {str(k): int(v) for k, v in df.isna().sum().items() if v > 0},
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split_random_state": 42,
        "test_size": 0.20,
        "categorical_code_columns": [c for c in CATEGORICAL_COLUMNS if c in features],
        "note": (
            "Target is a three-class student outcome. Academic results from both semesters "
            "are present; models using them are not suitable to claim early-enrollment prediction. "
            "Associations do not establish causation."
        ),
    }


def main() -> None:
    df = load_source_data()
    validate_data(df)
    # A stable row key is for traceability only and is never a model feature.
    df.insert(0, ID_COLUMN, [f"STU-{i + 1:05d}" for i in range(len(df))])
    features = infer_feature_columns(df)
    train, test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df[TARGET]
    )
    df.to_csv(BASE_DIR / "student_ml_dataset.csv", index=False)
    train.to_csv(BASE_DIR / "student_train.csv", index=False)
    test.to_csv(BASE_DIR / "student_test.csv", index=False)
    report = build_data_quality_report(df, train, test)
    (BASE_DIR / "data_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Student data preparation complete.")
    print(f"Rows / features: {len(df):,} / {len(features)}")
    print(f"Train / test: {len(train):,} / {len(test):,}")
    print("Target counts:", df[TARGET].value_counts().to_dict())
    print("Created: student_ml_dataset.csv, student_train.csv, student_test.csv")
    print("Created: data_quality_report.json")


if __name__ == "__main__":
    main()
