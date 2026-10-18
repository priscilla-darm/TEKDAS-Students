"""Train and evaluate a multiclass student-outcome model.

Run after data_prep.py:
    python ml_model.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import CATEGORICAL_COLUMNS, TARGET, ID_COLUMN

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "student_outcome_pipeline.pkl"
LABEL_ORDER = ["Dropout", "Enrolled", "Graduate"]


def build_pipeline(train: pd.DataFrame, features: list[str]) -> Pipeline:
    cat_cols = [
        c for c in features
        if c in CATEGORICAL_COLUMNS or train[c].dtype == "object" or train[c].nunique(dropna=True) <= 12
    ]
    num_cols = [c for c in features if c not in cat_cols]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    preprocessor = ColumnTransformer([
        ("cat", categorical_pipe, cat_cols),
        ("num", numeric_pipe, num_cols),
    ], remainder="drop")
    classifier = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def main() -> None:
    train_path, test_path = BASE_DIR / "student_train.csv", BASE_DIR / "student_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run `python data_prep.py` before `python ml_model.py`.")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    features = [c for c in train.columns if c not in {TARGET, ID_COLUMN}]
    X_train, y_train = train[features], train[TARGET].astype(str)
    X_test, y_test = test[features], test[TARGET].astype(str)

    print("Training Random Forest student-outcome model...")
    pipeline = build_pipeline(train, features)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)
    classes = list(pipeline.named_steps["model"].classes_)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    metrics = {
        "model": "RandomForestClassifier",
        "target": TARGET,
        "classes": classes,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "dropout_precision": float(precision_score(y_test, pred, labels=["Dropout"], average="macro", zero_division=0)),
        "dropout_recall": float(recall_score(y_test, pred, labels=["Dropout"], average="macro", zero_division=0)),
        "confusion_matrix_labels": classes,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=classes).tolist(),
        "classification_report": report,
        "baseline_majority_accuracy": float(accuracy_score(y_test, dummy_pred)),
        "test_rows": int(len(test)),
        "random_state": 42,
        "note": "Held-out split metrics; not a guarantee of future performance or proof of causality.",
    }
    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=3, zero_division=0))
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro-F1: {metrics['macro_f1']:.3f}")
    print(f"Dropout recall: {metrics['dropout_recall']:.3f}")
    joblib.dump({
        "model": pipeline,
        "features": features,
        "target": TARGET,
        "classes": classes,
    }, MODEL_FILE)

    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    fi.to_csv(BASE_DIR / "feature_importance.csv", index=False)

    predictions = test[[ID_COLUMN, TARGET]].copy()
    predictions["predicted_outcome"] = pred
    for i, class_name in enumerate(classes):
        predictions[f"probability_{class_name.lower()}"] = probs[:, i]
    predictions["correct_prediction"] = predictions[TARGET].eq(pred)
    predictions.to_csv(BASE_DIR / "student_predictions.csv", index=False)

    (BASE_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved: {MODEL_FILE.name}")
    print("Saved: feature_importance.csv, student_predictions.csv, model_metrics.json")


if __name__ == "__main__":
    main()
