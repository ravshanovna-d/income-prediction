"""Train the income (>50K) classifier, evaluate it, and export it for the web demo.

Usage:  python train.py
Outputs: web/model.onnx, web/meta.json, web/assets/*.png
"""
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from export_utils import check_onnx, examples, feature_schema, onnx_input_types, write_meta

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent
WEB, ASSETS = ROOT / "web", ROOT / "web" / "assets"
SEED = 42

NUM_COLS = ["age", "capital_gain", "capital_loss", "hours_per_week"]
CAT_COLS = ["workclass", "education", "marital_status", "occupation", "relationship",
            "race", "sex", "native_country"]
FEATURES = ["age", "workclass", "education", "marital_status", "occupation", "relationship",
            "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]

EDUCATION_ORDER = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
                   "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters",
                   "Prof-school", "Doctorate"]
READABLE = {
    "education": {"HS-grad": "High school graduate", "Some-college": "Some college",
                  "Assoc-voc": "Associate (vocational)", "Assoc-acdm": "Associate (academic)",
                  "Bachelors": "Bachelor's", "Masters": "Master's",
                  "Prof-school": "Professional school", "Doctorate": "Doctorate",
                  **{e: f"Grade {e.replace('th', '')}" for e in ["9th", "10th", "11th", "12th"]},
                  "1st-4th": "Grades 1-4", "5th-6th": "Grades 5-6", "7th-8th": "Grades 7-8",
                  "Preschool": "Preschool"},
    "workclass": {"Private": "Private company", "Self-emp-not-inc": "Self-employed",
                  "Self-emp-inc": "Self-employed (own company)", "Federal-gov": "Federal government",
                  "Local-gov": "Local government", "State-gov": "State government",
                  "Without-pay": "Unpaid", "Never-worked": "Never worked"},
    "marital_status": {"Married-civ-spouse": "Married", "Married-AF-spouse": "Married (spouse in armed forces)",
                       "Married-spouse-absent": "Married (spouse absent)", "Never-married": "Never married",
                       "Divorced": "Divorced", "Separated": "Separated", "Widowed": "Widowed"},
    "occupation": {"Tech-support": "Tech support", "Craft-repair": "Craft / repair",
                   "Other-service": "Other service", "Sales": "Sales",
                   "Exec-managerial": "Executive / managerial", "Prof-specialty": "Professional specialty",
                   "Handlers-cleaners": "Handlers / cleaners", "Machine-op-inspct": "Machine operator",
                   "Adm-clerical": "Administrative / clerical", "Farming-fishing": "Farming / fishing",
                   "Transport-moving": "Transport / moving", "Priv-house-serv": "Private household service",
                   "Protective-serv": "Protective service", "Armed-Forces": "Armed forces"},
    "relationship": {"Husband": "Spouse (male)", "Wife": "Spouse (female)", "Own-child": "Own child",
                     "Not-in-family": "Not in family", "Other-relative": "Other relative",
                     "Unmarried": "Unmarried"},
}
LABELS = {
    "age": "Age", "workclass": "Employer type", "education": "Education",
    "marital_status": "Marital status", "occupation": "Occupation", "relationship": "Role in household",
    "race": "Race", "sex": "Sex", "capital_gain": "Capital gains", "capital_loss": "Capital losses",
    "hours_per_week": "Hours worked per week", "native_country": "Country of origin",
}

# XGBoost / LightGBM converters for skl2onnx (from onnxmltools)
update_registered_converter(XGBClassifier, "XGBoostXGBClassifier", calculate_linear_classifier_output_shapes,
                            convert_xgboost, options={"nocl": [True, False], "zipmap": [True, False, "columns"]})
update_registered_converter(LGBMClassifier, "LightGbmLGBMClassifier", calculate_linear_classifier_output_shapes,
                            convert_lightgbm, options={"nocl": [True, False], "zipmap": [True, False, "columns"]})


def load_data():
    """Download (and cache in data/) the Adult census income data from OpenML."""
    d = fetch_openml("adult", version=2, as_frame=True, data_home=ROOT / "data")
    df = d.frame.copy()
    df.columns = [c.replace("-", "_") for c in df.columns]
    y = (df.pop("class").astype(str) == ">50K").astype(int)
    for c in CAT_COLS:
        s = df[c].astype(object).fillna("Unknown").astype(str)
        df[c] = s.map(READABLE.get(c, {})).fillna(s.str.replace("-", " ")) if c in READABLE else s.str.replace("-", " ")
    for c in NUM_COLS:
        df[c] = df[c].astype(float)
    # fnlwgt is a census sampling weight (not a property of the person) and
    # education_num duplicates education, so both are left out.
    return df[FEATURES], y


def build_models():
    def prep(scale=False):
        # Dense one-hot: tree boosters treat sparse zeros as "missing", ONNX treats them as 0.
        # Trees get raw numbers: scaling is useless for them, and float64 scaled values can
        # land on the other side of a split once rounded to float32 in ONNX.
        return ColumnTransformer([
            ("num", StandardScaler() if scale else "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ])
    return {
        "Logistic Regression": Pipeline([("prep", prep(scale=True)), ("model", LogisticRegression(max_iter=2000, C=1.0))]),
        "Random Forest": Pipeline([("prep", prep()), ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_features="sqrt", random_state=SEED, n_jobs=1))]),
        "Gradient Boosting": Pipeline([("prep", prep()), ("model", GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4, subsample=0.6, random_state=SEED))]),
        "XGBoost": Pipeline([("prep", prep()), ("model", XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1))]),
        "LightGBM": Pipeline([("prep", prep()), ("model", LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEED, n_jobs=1, verbose=-1))]),
        # CatBoost handles categories natively, so it gets the raw columns.
        "CatBoost": CatBoostClassifier(iterations=500, depth=6, learning_rate=0.08, random_seed=SEED,
                                       cat_features=CAT_COLS, verbose=False, allow_writing_files=False,
                                       thread_count=1),
    }


ONNX_READY = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]


def export_pipeline(pipeline, X, path):
    """Like export_utils.export_onnx, but pins the ai.onnx.ml opset that the
    XGBoost/LightGBM tree converters (and onnxruntime-web) support."""
    onx = convert_sklearn(pipeline, initial_types=onnx_input_types(X),
                          options={id(pipeline): {"zipmap": False}},
                          target_opset={"": 17, "ai.onnx.ml": 3})
    Path(path).write_bytes(onx.SerializeToString())


def cv_catboost(model, X, y, cv):
    """Manual CV loop: CatBoost's cat_features parameter does not survive sklearn's clone()."""
    out = {"test_ROC-AUC": [], "test_F1": [], "test_Accuracy": [], "fit_time": []}
    for tr, va in cv.split(X, y):
        m = CatBoostClassifier(**model.get_params())  # single thread, like the other models
        t0 = time.perf_counter()
        m.fit(X.iloc[tr], y.iloc[tr])
        out["fit_time"].append(time.perf_counter() - t0)
        p = m.predict_proba(X.iloc[va])[:, 1]
        out["test_ROC-AUC"].append(roc_auc_score(y.iloc[va], p))
        out["test_F1"].append(f1_score(y.iloc[va], p >= 0.5))
        out["test_Accuracy"].append(accuracy_score(y.iloc[va], p >= 0.5))
    return out


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 1) Compare six models with 5-fold stratified cross-validation (single-threaded, so fit times are comparable)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = build_models()
    rows = []
    for name, model in models.items():
        t0 = time.perf_counter()
        if name == "CatBoost":
            r = cv_catboost(model, X_train, y_train, cv)
        else:
            r = cross_validate(model, X_train, y_train, cv=cv, n_jobs=1, error_score="raise",
                               scoring={"ROC-AUC": "roc_auc", "F1": "f1", "Accuracy": "accuracy"})
        rows.append({"model": name, **{k: float(np.mean(r[f"test_{k}"])) for k in ("ROC-AUC", "F1", "Accuracy")},
                     "Fit time (s)": float(np.mean(r["fit_time"]))})
        print(f"{name:20s} ROC-AUC={rows[-1]['ROC-AUC']:.4f}  F1={rows[-1]['F1']:.4f}  "
              f"fit={rows[-1]['Fit time (s)']:.1f}s  (wall {time.perf_counter() - t0:.0f}s)")
    comparison = sorted(rows, key=lambda r: -r["ROC-AUC"])
    best_overall = comparison[0]["model"]
    best_name = next(r["model"] for r in comparison if r["model"] in ONNX_READY)
    print(f"Best overall: {best_overall}; best ONNX-exportable: {best_name}")
    best = models[best_name]
    if "model__n_jobs" in best.get_params():
        best.set_params(model__n_jobs=-1)
    best.fit(X_train, y_train)

    # 2) Final evaluation on the untouched test set
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }
    print("Test:", {k: round(v, 4) for k, v in test_metrics.items()})

    # 3) Figures
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["≤50K", ">50K"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, name=best_name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "roc_curve.png", dpi=130); plt.close(fig)

    # SHAP on the boosted trees (one-hot columns grouped back to the original features)
    prep, clf = best.named_steps["prep"], best.named_steps["model"]
    sample = X_test.sample(2000, random_state=SEED)
    Z = prep.transform(sample)
    sv = shap.TreeExplainer(clf).shap_values(Z)
    sv = sv[1] if isinstance(sv, list) else sv
    names = prep.get_feature_names_out()
    owner = [next(f for f in FEATURES if n.split("__", 1)[1] == f or n.split("__", 1)[1].startswith(f + "_"))
             for n in names]
    grouped = pd.DataFrame(sv, columns=names).T.groupby(owner).sum().T
    mean_abs = grouped.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    mean_abs.rename(LABELS).plot.barh(ax=ax, color="#4f46e5")
    ax.set_xlabel("Mean |SHAP value| (log-odds)"); ax.set_title(f"What drives the prediction ({best_name})")
    fig.tight_layout(); fig.savefig(ASSETS / "shap_importance.png", dpi=130); plt.close(fig)

    df = X.assign(high=y)
    edu = df.groupby("education")["high"].mean().reindex([READABLE["education"][e] for e in EDUCATION_ORDER])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    edu.plot.barh(ax=ax, color="#4f46e5"); ax.set_ylabel("")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share earning >50K"); ax.set_title("Income rises steadily with education")
    fig.tight_layout(); fig.savefig(ASSETS / "income_by_education.png", dpi=130); plt.close(fig)

    # 4) Export for the browser and verify it matches the Python pipeline
    export_pipeline(best, X_train, WEB / "model.onnx")
    err = check_onnx(WEB / "model.onnx", best, X_test, n=2000)
    print(f"ONNX check OK (max diff {err:.2e})")

    schema = feature_schema(X_train, LABELS, {"capital_gain": "USD per year, usually 0",
                                              "capital_loss": "USD per year, usually 0",
                                              "age": "years"})
    for f in schema:
        if f["name"] == "education":
            f["options"] = [READABLE["education"][e] for e in EDUCATION_ORDER]
        if f["name"] == "native_country":  # most common first
            f["options"] = X_train["native_country"].value_counts().index.tolist()
    pos_rate = float(y.mean())
    write_meta(WEB / "meta.json", {
        "title": "Income Prediction",
        "subtitle": ("Predicts whether a person earns more than $50K a year from census information "
                     "such as age, education, occupation and hours worked."),
        "tags": ["Classification", "Pipelines", "Cross-validation", best_name, "SHAP", "ONNX"],
        "task": "classification",
        "classes": [0, 1], "positive_class": 1,
        "output_label": "Probability of earning more than $50K / year",
        "verdicts": ["Likely ≤ $50K", "Uncertain", "Likely > $50K"],
        "thresholds": [0.35, 0.65],
        "positive_is_good": True,
        "result_note": "Try changing Education, Occupation or Hours worked per week. "
                       "Census data from 1994 — dollar amounts are not inflation-adjusted.",
        "features": schema,
        "examples": examples(X_test),
        "model_name": best_name,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "metric_formats": {k: "percent" for k in test_metrics if k != "ROC-AUC"},
        "metrics_note": f"Held-out test set: {len(X_test):,} people never used for training or model selection.",
        "figures": [
            {"src": "assets/shap_importance.png", "caption": "SHAP: marital status, age, capital gains and education matter most."},
            {"src": "assets/income_by_education.png", "caption": "The share of high earners climbs with every education level."},
            {"src": "assets/roc_curve.png", "caption": "ROC curve on the test set."},
            {"src": "assets/confusion_matrix.png", "caption": "Correct and wrong predictions on the test set."},
        ],
        "steps": [
            f"Loaded the UCI Adult census data ({len(X):,} people, {pos_rate:.0%} earn >50K) from OpenML.",
            "Renamed columns to simple names, marked missing answers as “Unknown”, dropped the census sampling weight.",
            "Built one scikit-learn Pipeline per model (one-hot encoding for categories, scaling for the linear model).",
            "Compared Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM and CatBoost "
            "with 5-fold stratified cross-validation (ROC-AUC, F1, training time).",
            f"Shipped {best_name} (best exportable model) and evaluated it once on a held-out 20% test set.",
            "Explained it with SHAP values and exported the whole pipeline to ONNX for the browser.",
        ],
        "dataset": {"name": "UCI Adult (Census Income), OpenML", "url": "https://www.openml.org/d/1590", "rows": len(X)},
        "repo_url": "https://github.com/ravshanovna-d/income-prediction",
        "notebook_url": "https://github.com/ravshanovna-d/income-prediction/blob/main/notebook.ipynb",
        "footer": "Built by <a href=\"https://github.com/ravshanovna-d\">ravshanovna-d</a>",
    })
    print("Wrote web/model.onnx and web/meta.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
