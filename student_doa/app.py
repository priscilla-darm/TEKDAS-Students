from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("STUDENT_DOA_DATA", BASE_DIR / "student_doa_dataset.csv"))
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "student_outcome_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
TARGET = "Target"
LABELS = ["Dropout", "Enrolled", "Graduate"]
PALETTE = {"Dropout": "#D95757", "Enrolled": "#4F80D8", "Graduate": "#24865B"}

st.set_page_config(
    page_title="EduSight | Student Outcome Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #F5F7FB; }
  [data-testid="stSidebar"] { background: linear-gradient(180deg,#111B30 0%,#172641 100%); }
  [data-testid="stSidebar"] * { color: #EDF2FC !important; }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #C1CBE0 !important; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1500px; }
  h1, h2, h3 { font-family: 'Manrope', sans-serif; color: #16233A; letter-spacing: -0.025em; }
  .hero { background: linear-gradient(120deg,#111B30 0%,#203A68 100%); padding: 1.7rem 1.9rem; border-radius: 20px; color: white; margin-bottom: 1.15rem; box-shadow: 0 10px 28px rgba(17,27,48,.12); }
  .hero h1 { color: white !important; margin: 0; font-size: 2rem; }
  .hero p { color: #D1DCEF; margin: .45rem 0 0 0; font-size: .98rem; }
  .eyebrow { color: #9DBBFF; text-transform: uppercase; font-size: .72rem; letter-spacing: .13em; font-weight: 700; margin-bottom: .5rem; }
  .section-note { color: #64748B; font-size: .9rem; margin-top: -.35rem; }
  div[data-testid="stMetric"] { background: white; padding: 1rem 1.15rem; border: 1px solid #E6EAF2; border-radius: 15px; box-shadow: 0 3px 12px rgba(26,44,75,.035); }
  div[data-testid="stMetricLabel"] { color: #64748B; font-weight: 600; }
  div[data-testid="stMetricValue"] { color: #17243B; font-family: 'Manrope', sans-serif; }
  div[data-testid="stTabs"] button { border-radius: 10px 10px 0 0; font-weight: 600; }
  div[data-testid="stDataFrame"] { border: 1px solid #E5EAF2; border-radius: 12px; overflow: hidden; }
  div.stButton > button, div.stDownloadButton > button { border-radius: 10px; font-weight: 600; border: 1px solid #D5DDEA; }
  div.stButton > button[kind="primary"] { background: #315EDB; border-color: #315EDB; }
  .small-muted { color: #718096; font-size: .82rem; }
  .risk-card { background: white; padding: 1.1rem; border-radius: 15px; border: 1px solid #E5EAF2; }
  hr { border-color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data(path_str: str, modified_time: float) -> pd.DataFrame:
    frame = pd.read_csv(path_str)
    if TARGET not in frame.columns:
        raise ValueError(f"Dataset must contain a '{TARGET}' target column.")
    frame[TARGET] = frame[TARGET].astype(str).str.strip()
    frame = frame.dropna(subset=[TARGET]).copy()
    return frame


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            f"Dataset not found at `{DATA_PATH}`. Place `student_doa_dataset.csv` beside `app.py`, "
            "or set the `STUDENT_DOA_DATA` environment variable to its path."
        )
        st.stop()
    try:
        return load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()


def get_categorical_columns(df: pd.DataFrame, features: list[str]) -> list[str]:
    # These columns are codes/categories even though the CSV stores them as numbers.
    known_categorical = {
        "Marital status", "Application mode", "Application order", "Course",
        "Daytime/evening attendance", "Previous qualification", "Nacionality",
        "Mother's qualification", "Father's qualification", "Mother's occupation",
        "Father's occupation", "Displaced", "Educational special needs", "Debtor",
        "Tuition fees up to date", "Gender", "Scholarship holder", "International",
    }
    return [c for c in features if c in known_categorical or df[c].dtype == "object" or df[c].nunique(dropna=True) <= 12]


def make_pipeline(df: pd.DataFrame, features: list[str], model_name: str) -> Pipeline:
    cat_cols = get_categorical_columns(df, features)
    num_cols = [c for c in features if c not in cat_cols]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    preprocess = ColumnTransformer([
        ("categorical", categorical_pipe, cat_cols),
        ("numeric", numeric_pipe, num_cols),
    ], remainder="drop")
    if model_name == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1800, class_weight="balanced", random_state=42)
    else:
        classifier = RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        )
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


@st.cache_resource(show_spinner=False)
def train_models(cache_key: str, csv_path: str, modified_time: float, feature_columns: tuple[str, ...]):
    df = pd.read_csv(csv_path)
    df[TARGET] = df[TARGET].astype(str).str.strip()
    features = list(feature_columns)
    X = df[features].copy()
    y = df[TARGET].copy()
    # Normalize mixed CSV types before preprocessing.
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("Unknown").astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )
    results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for name in ["Random Forest", "Logistic Regression"]:
        pipe = make_pipeline(df, features, name)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)
        classes = list(pipe.named_steps["model"].classes_)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "dropout_precision": float(precision_score(y_test, pred, labels=["Dropout"], average="macro", zero_division=0)),
            "dropout_recall": float(recall_score(y_test, pred, labels=["Dropout"], average="macro", zero_division=0)),
            "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
            "confusion": confusion_matrix(y_test, pred, labels=classes).tolist(),
            "classes": classes,
            "y_test": y_test.tolist(),
            "predictions": pred.tolist(),
            "test_indices": X_test.index.tolist(),
        }
        fitted[name] = pipe
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    results["Baseline (majority class)"] = {
        "accuracy": float(accuracy_score(y_test, dummy_pred)),
        "macro_f1": float(f1_score(y_test, dummy_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, dummy_pred, average="weighted", zero_division=0)),
        "dropout_precision": float(precision_score(y_test, dummy_pred, labels=["Dropout"], average="macro", zero_division=0)),
        "dropout_recall": float(recall_score(y_test, dummy_pred, labels=["Dropout"], average="macro", zero_division=0)),
    }
    best_name = max(fitted, key=lambda n: results[n]["macro_f1"])
    best_model = fitted[best_name]
    # Fit selected model on the complete dataset for prediction use after evaluation.
    best_model.fit(X, y)
    return {"results": results, "models": fitted, "best_name": best_name, "best_model": best_model,
            "X_test": X_test, "y_test": y_test, "features": features, "labels": sorted(y.unique())}


def plot_theme(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=38, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#334155", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E5EAF2")
    fig.update_yaxes(gridcolor="#EDF1F7", zeroline=False)
    return fig


def show_hero(title: str, subtitle: str, section: str = "STUDENT OUTCOME INTELLIGENCE"):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{section}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def dataframe_download(df: pd.DataFrame, filename: str, label: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


df = load_dataset()
features_all = [c for c in df.columns if c != TARGET]
outcomes = df[TARGET].value_counts()
n_students = len(df)
dropout_rate = float((df[TARGET] == "Dropout").mean())
graduate_rate = float((df[TARGET] == "Graduate").mean())
enrolled_rate = float((df[TARGET] == "Enrolled").mean())

with st.sidebar:
    st.markdown("## 🎓 EduSight")
    st.markdown("<p style='font-size:.78rem'>STUDENT OUTCOME INTELLIGENCE</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Dataset Explorer", "Business Intelligence", "Machine Learning Lab", "Prediction Studio", "AI Analyst"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Dataset status**")
    st.caption(f"{n_students:,} records · {len(features_all)} features")
    st.caption(f"Target: `{TARGET}`")
    st.markdown("<p class='small-muted'>Built for transparent, evidence-based student success analysis.</p>", unsafe_allow_html=True)

if page == "Overview":
    show_hero("Student Success Overview", "A clear view of student outcomes, academic patterns, and predictive insights.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total students", f"{n_students:,}")
    k2.metric("Graduation rate", f"{graduate_rate:.1%}")
    k3.metric("Dropout rate", f"{dropout_rate:.1%}", help="Observed historical label, not a model prediction.")
    k4.metric("Enrolled", f"{int((df[TARGET] == 'Enrolled').sum()):,}")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Outcome distribution")
        counts = df[TARGET].value_counts().reindex(LABELS).dropna().rename_axis("Outcome").reset_index(name="Students")
        fig = px.bar(counts, x="Outcome", y="Students", color="Outcome", color_discrete_map=PALETTE, text="Students")
        fig.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(plot_theme(fig, 350), use_container_width=True)
    with right:
        st.subheader("Dataset health")
        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())
        health = pd.DataFrame({"Check": ["Missing cells", "Exact duplicate rows", "Target classes"], "Result": [f"{missing:,}", f"{duplicates:,}", f"{df[TARGET].nunique()}"]})
        st.dataframe(health, hide_index=True, use_container_width=True)
        st.info("Observed outcomes describe this dataset. They should not be interpreted as current institution-wide rates without checking representativeness.")
    st.subheader("What this workspace includes")
    cards = st.columns(3)
    cards[0].markdown("**01 · Explore**  \nInspect columns, data quality, and the original records.")
    cards[1].markdown("**02 · Understand**  \nCompare outcomes across enrollment, financial, and academic indicators.")
    cards[2].markdown("**03 · Predict**  \nEvaluate models and estimate outcome probabilities with explicit limitations.")

elif page == "Dataset Explorer":
    show_hero("Dataset Explorer", "Inspect the source data before drawing conclusions.", "DATA QUALITY & DISCOVERY")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    with st.expander("Column profile", expanded=True):
        profile = pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(df[c].dtype) for c in df.columns],
            "Non-null": [int(df[c].notna().sum()) for c in df.columns],
            "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
            "Missing": [int(df[c].isna().sum()) for c in df.columns],
        })
        st.dataframe(profile, hide_index=True, use_container_width=True)
    all_cols = st.checkbox("Show all columns", value=False)
    selected_cols = st.multiselect("Columns to display", options=list(df.columns), default=list(df.columns) if all_cols else list(df.columns[:8]))
    target_filter = st.multiselect("Filter by observed outcome", options=sorted(df[TARGET].unique()), default=sorted(df[TARGET].unique()))
    view = df.loc[df[TARGET].isin(target_filter), selected_cols] if selected_cols else df.loc[df[TARGET].isin(target_filter), [TARGET]]
    st.caption(f"Showing {len(view):,} matching rows · {len(view.columns)} selected columns")
    st.dataframe(view, use_container_width=True, height=440)
    dataframe_download(view, "student_doa_filtered.csv", "Download filtered CSV")
    st.markdown("**Target balance**")
    target_counts = df[TARGET].value_counts().rename_axis("Target").reset_index(name="Count")
    fig = px.pie(target_counts, names="Target", values="Count", hole=.62, color="Target", color_discrete_map=PALETTE)
    fig.update_traces(textposition="outside", textinfo="label+percent")
    st.plotly_chart(plot_theme(fig, 340), use_container_width=True)

elif page == "Business Intelligence":
    show_hero("Business Intelligence", "Explore patterns associated with student outcomes.", "DESCRIPTIVE ANALYTICS")
    st.caption("These are descriptive comparisons. They do not establish that a feature causes a student outcome.")
    filt1, filt2 = st.columns(2)
    with filt1:
        outcome_filter = st.multiselect("Outcomes", options=LABELS, default=LABELS)
    with filt2:
        color_by = st.selectbox("Break down by", ["Target", "Gender", "Scholarship holder", "Tuition fees up to date", "Debtor", "Daytime/evening attendance"])
    view = df[df[TARGET].isin(outcome_filter)].copy()
    a, b = st.columns(2)
    with a:
        st.subheader("Outcome mix")
        mix = view[TARGET].value_counts().reindex(LABELS).fillna(0).rename_axis("Outcome").reset_index(name="Students")
        fig = px.bar(mix, x="Outcome", y="Students", color="Outcome", color_discrete_map=PALETTE, text_auto=True)
        st.plotly_chart(plot_theme(fig), use_container_width=True)
    with b:
        st.subheader("Outcome by selected group")
        if color_by == "Target":
            group = view.groupby(TARGET, dropna=False).size().rename("Students").reset_index()
            fig = px.bar(group, x=TARGET, y="Students", color=TARGET, color_discrete_map=PALETTE)
        else:
            group = view.groupby([color_by, TARGET], dropna=False).size().rename("Students").reset_index()
            group[color_by] = group[color_by].astype(str)
            fig = px.bar(group, x=color_by, y="Students", color=TARGET, barmode="group", color_discrete_map=PALETTE)
            fig.update_xaxes(tickangle=-25)
        st.plotly_chart(plot_theme(fig), use_container_width=True)
    academic_cols = ["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)", "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]
    available_academic = [c for c in academic_cols if c in view.columns]
    if available_academic:
        st.subheader("Academic indicators by outcome")
        chosen_metric = st.selectbox("Academic indicator", available_academic)
        fig = px.box(view, x=TARGET, y=chosen_metric, color=TARGET, color_discrete_map=PALETTE, points="outliers")
        st.plotly_chart(plot_theme(fig, 390), use_container_width=True)
    st.subheader("Outcome rates across groups")
    group_col = st.selectbox("Group for rate comparison", ["Scholarship holder", "Tuition fees up to date", "Debtor", "Gender", "Age at enrollment", "Course"])
    rate = df.groupby(group_col, dropna=False).agg(
        Students=(TARGET, "size"),
        **{
            "Dropout rate": (TARGET, lambda s: (s == "Dropout").mean()),
            "Graduate rate": (TARGET, lambda s: (s == "Graduate").mean()),
        },
    ).reset_index()
    if group_col == "Age at enrollment":
        rate[group_col] = rate[group_col].astype(str)
    rate["Group"] = rate[group_col].astype(str)
    rate = rate.sort_values("Students", ascending=False).head(25)
    rate_long = rate.melt(id_vars=["Group", "Students"], value_vars=["Dropout rate", "Graduate rate"], var_name="Metric", value_name="Rate")
    fig = px.bar(rate_long, x="Group", y="Rate", color="Metric", barmode="group", hover_data=["Students"])
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickangle=-25)
    st.plotly_chart(plot_theme(fig, 400), use_container_width=True)

elif page == "Machine Learning Lab":
    show_hero("Machine Learning Lab", "Train, compare, and evaluate outcome classifiers with a reproducible holdout split.", "MODEL EVALUATION")
    st.warning("Prediction quality depends on when the prediction is made. This dataset contains second-semester results; a model trained with them is not an early-enrollment warning model.")
    with st.expander("Modeling setup", expanded=True):
        st.markdown("""
        - Target: `Target` with **Dropout**, **Enrolled**, and **Graduate** classes.
        - Evaluation: stratified 78/22 train/test split, fixed random seed (42).
        - Candidate models: Random Forest and Logistic Regression, compared using macro-F1.
        - Preprocessing: one-hot encoding for coded/categorical fields and scaling for continuous fields.
        - A majority-class baseline is reported for context. Test metrics are for a held-out split; they are not a guarantee of future performance.
        """)
    if st.button("Train and evaluate models", type="primary"):
        st.session_state["model_bundle"] = None
        with st.spinner("Training models and evaluating the held-out test set..."):
            try:
                bundle = train_models(str(DATA_PATH.stat().st_mtime), str(DATA_PATH), DATA_PATH.stat().st_mtime, tuple(features_all))
                st.session_state["model_bundle"] = bundle
                ARTIFACT_DIR.mkdir(exist_ok=True)
                joblib.dump({"model": bundle["best_model"], "features": features_all, "best_name": bundle["best_name"]}, MODEL_PATH)
                METRICS_PATH.write_text(json.dumps({k: {m: v for m, v in val.items() if m not in ["report", "confusion", "classes", "y_test", "predictions", "test_indices"]} for k, val in bundle["results"].items()}, indent=2), encoding="utf-8")
                st.success(f"Training complete. Best candidate by macro-F1: {bundle['best_name']}.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
    bundle = st.session_state.get("model_bundle")
    if bundle:
        result_rows = []
        for name, metrics in bundle["results"].items():
            result_rows.append({"Model": name, "Accuracy": metrics["accuracy"], "Macro F1": metrics["macro_f1"], "Weighted F1": metrics["weighted_f1"], "Dropout precision": metrics["dropout_precision"], "Dropout recall": metrics["dropout_recall"]})
        results_df = pd.DataFrame(result_rows)
        st.dataframe(results_df.style.format({c: "{:.3f}" for c in results_df.columns if c != "Model"}), hide_index=True, use_container_width=True)
        candidates = [n for n in bundle["results"] if n != "Baseline (majority class)"]
        metric_plot = results_df[results_df["Model"].isin(candidates)].melt(id_vars="Model", value_vars=["Accuracy", "Macro F1", "Dropout recall"], var_name="Metric", value_name="Score")
        fig = px.bar(metric_plot, x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1])
        st.plotly_chart(plot_theme(fig), use_container_width=True)
        selected_model = st.selectbox("Inspect confusion matrix", candidates, index=candidates.index(bundle["best_name"]))
        met = bundle["results"][selected_model]
        cm = np.array(met["confusion"])
        classes = met["classes"]
        fig = px.imshow(cm, x=classes, y=classes, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                        labels={"x": "Predicted label", "y": "Actual label", "color": "Count"})
        fig.update_layout(height=390)
        st.plotly_chart(plot_theme(fig, 390), use_container_width=True)
        report = pd.DataFrame(met["report"]).T
        st.subheader("Per-class classification report")
        st.dataframe(report.style.format("{:.3f}"), use_container_width=True)
        st.download_button("Download model metrics JSON", METRICS_PATH.read_bytes(), file_name="model_metrics.json", mime="application/json")
        st.caption(f"Selected model for the Prediction Studio: {bundle['best_name']} (chosen by macro-F1).")
    else:
        st.info("Select **Train and evaluate models** to generate metrics and enable prediction. Training runs locally in this app.")

elif page == "Prediction Studio":
    show_hero("Prediction Studio", "Estimate the three possible student outcomes and inspect the model's confidence.", "INDIVIDUAL-LEVEL ESTIMATION")
    bundle = st.session_state.get("model_bundle")
    if bundle is None:
        standalone_model_path = BASE_DIR / "student_outcome_pipeline.pkl"
        candidate_path = MODEL_PATH if MODEL_PATH.exists() else standalone_model_path
        if candidate_path.exists():
            try:
                loaded = joblib.load(candidate_path)
                # app.py stores a dict with a "model" key; ml_model.py stores the
                # equivalent standalone pipeline bundle at project root.
                st.session_state["saved_model"] = loaded
            except Exception:
                st.session_state.pop("saved_model", None)
        saved = st.session_state.get("saved_model")
        if saved is not None:
            model = saved["model"]
            features = saved["features"]
        else:
            model = None
            features = features_all
    else:
        model = bundle["best_model"]
        features = bundle["features"]
    if model is None:
        st.info("Train a model in **Machine Learning Lab** first. The selected model is then available here.")
    else:
        mode = st.radio("Choose student input", ["Select an existing record", "Enter feature values"], horizontal=True)
        if mode == "Select an existing record":
            idx = st.selectbox("Dataset row", options=list(df.index), format_func=lambda i: f"Record {i + 1} · observed outcome: {df.loc[i, TARGET]}")
            input_row = df.loc[[idx], features].copy()
            st.dataframe(df.loc[[idx], [TARGET]], hide_index=True, use_container_width=True)
        else:
            st.caption("Values are taken from the dataset's observed range. Categorical codes are numeric IDs, not descriptive labels.")
            input_values = {}
            with st.form("student_prediction_form"):
                cols = st.columns(3)
                for i, col in enumerate(features):
                    series = df[col]
                    with cols[i % 3]:
                        if series.dtype == "object":
                            choices = sorted(series.dropna().astype(str).unique().tolist())
                            input_values[col] = st.selectbox(col, choices, index=0, key=f"pred_{col}")
                        else:
                            median = float(series.median()) if series.notna().any() else 0.0
                            minimum = float(series.min()) if series.notna().any() else 0.0
                            maximum = float(series.max()) if series.notna().any() else 1.0
                            if series.nunique(dropna=True) <= 12:
                                options = sorted(series.dropna().unique().tolist())
                                input_values[col] = st.selectbox(col, options, index=options.index(median) if median in options else 0, key=f"pred_{col}")
                            else:
                                input_values[col] = st.number_input(col, min_value=minimum, max_value=maximum, value=min(max(median, minimum), maximum), key=f"pred_{col}")
                submitted = st.form_submit_button("Estimate outcome", type="primary")
            if not submitted:
                st.stop()
            input_row = pd.DataFrame([input_values], columns=features)
        try:
            pred = str(model.predict(input_row)[0])
            probabilities = model.predict_proba(input_row)[0]
            classes = list(model.named_steps["model"].classes_)
            probability_df = pd.DataFrame({"Outcome": classes, "Estimated probability": probabilities})
            probability_df["Color"] = probability_df["Outcome"].map(PALETTE)
            pcols = st.columns(3)
            for i, label in enumerate(LABELS):
                value = float(probabilities[classes.index(label)]) if label in classes else 0.0
                pcols[i].metric(f"{label} probability", f"{value:.1%}")
            st.subheader("Estimated outcome probabilities")
            fig = px.bar(probability_df, x="Outcome", y="Estimated probability", color="Outcome", color_discrete_map=PALETTE, range_y=[0, 1], text=probability_df["Estimated probability"].map(lambda x: f"{x:.1%}"))
            fig.update_traces(textposition="outside")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(plot_theme(fig, 350), use_container_width=True)
            st.success(f"Model's predicted class: **{pred}**")
            st.warning("This is a statistical estimate, not a certainty or a decision about a person. Validate performance on the intended prediction stage and use results only to guide supportive, human-reviewed outreach.")
            with st.expander("Input features used"):
                st.dataframe(input_row, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not generate prediction: {exc}")

elif page == "AI Analyst":
    show_hero("AI Analyst", "Create a grounded summary from the observed data and any trained model results.", "EVIDENCE-BASED INTERPRETATION")
    st.caption("Ollama is optional. The built-in summary works without an external LLM.")
    dropout_df = df[df[TARGET] == "Dropout"]
    graduate_df = df[df[TARGET] == "Graduate"]
    summaries = [
        f"The dataset contains {len(df):,} records and {len(features_all)} input features.",
        f"Observed outcomes: {dropout_rate:.1%} Dropout, {enrolled_rate:.1%} Enrolled, and {graduate_rate:.1%} Graduate.",
    ]
    for col in ["Scholarship holder", "Tuition fees up to date", "Debtor", "Gender"]:
        if col in df.columns:
            table = df.groupby(col)[TARGET].apply(lambda s: (s == "Dropout").mean()).sort_values(ascending=False)
            if len(table):
                summaries.append(f"Dropout share varies across `{col}` groups; the highest observed group in this dataset is code/value `{table.index[0]}` ({table.iloc[0]:.1%}). This is descriptive and does not imply causation.")
    bundle = st.session_state.get("model_bundle")
    if bundle:
        name = bundle["best_name"]
        met = bundle["results"][name]
        summaries.append(f"The current selected model is {name}, with held-out macro-F1 {met['macro_f1']:.3f}, accuracy {met['accuracy']:.3f}, and dropout recall {met['dropout_recall']:.3f}.")
    st.subheader("Evidence-based summary")
    for item in summaries:
        st.markdown(f"- {item}")
    st.download_button("Download summary as text", "\n".join(f"- {s}" for s in summaries), file_name="student_outcome_summary.txt", mime="text/plain")
    st.divider()
    st.subheader("Optional Ollama analysis")
    ollama_model = st.text_input("Ollama model name", value=os.environ.get("OLLAMA_MODEL", "llama3.2"))
    custom_question = st.text_area("Question for the analyst", value="Summarize the main student-outcome patterns and propose supportive, non-punitive interventions. Clearly distinguish observations from hypotheses.")
    if st.button("Ask local LLM", type="primary"):
        prompt = (
            "You are an education analytics assistant. Use only the evidence below. Do not claim causality, "
            "do not invent facts, avoid stigmatizing individual students, and suggest supportive interventions. "
            "State limitations and recommend human review.\n\n"
            f"Dataset evidence:\n" + "\n".join(f"- {s}" for s in summaries) +
            f"\n\nUser question: {custom_question}"
        )
        try:
            import ollama
            with st.spinner("Contacting local Ollama model..."):
                response = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": prompt}])
            st.markdown(response["message"]["content"])
        except ImportError:
            st.error("The Python package `ollama` is not installed. Install it with `pip install ollama`.")
        except Exception as exc:
            st.error(f"Could not contact Ollama. Ensure Ollama is running and the model is installed (`ollama pull {ollama_model}`). Details: {exc}")

st.divider()
st.markdown(
    "<p class='small-muted'>EduSight · Student Outcome Intelligence · Descriptive analytics and model predictions are decision support, not definitive judgments. "
    "Protect student data and validate models before operational use.</p>",
    unsafe_allow_html=True,
)
