"""Research dashboard for the saved readmission model and experiment outputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# Streamlit may be launched with a working directory outside the repository.
# The saved sklearn pipeline references custom transformers under ``src``.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "models" / "best_model.pkl"
META_PATH = ARTIFACTS / "models" / "model_metadata.json"
REPORTS = ARTIFACTS / "reports"

st.set_page_config(
    page_title="Readmission Research Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#17324d; --muted:#61758a; --blue:#176b87; --mint:#e7f5f1; }
    .stApp { background: linear-gradient(135deg,#f4f8fb 0%,#f8fbfa 55%,#edf5f6 100%); }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] { background: #102d43; }
    [data-testid="stSidebar"] * { color: #eef6f9 !important; }
    .hero { padding: 1.6rem 1.8rem; border-radius: 20px; color:#f8fbff;
      background: linear-gradient(120deg,#12334b,#176b87 65%,#48a99b); margin: .3rem 0 1.2rem; }
    .hero h1 { color:white; margin:0; font-size:2rem; }
    .hero p { color:#e2f2f4; margin:.45rem 0 0; font-size:1rem; }
    .eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; opacity:.8; }
    .metric-card { background:white; border:1px solid #e3ebef; border-radius:16px;
      padding:1rem 1.15rem; box-shadow:0 5px 18px rgba(24,55,76,.05); min-height:105px; }
    .metric-label { color:#63778a; font-size:.82rem; margin-bottom:.45rem; }
    .metric-value { color:#17324d; font-size:1.55rem; font-weight:700; }
    .metric-note { color:#718497; font-size:.74rem; margin-top:.2rem; }
    .section-kicker { color:#176b87; text-transform:uppercase; letter-spacing:.12em;
      font-size:.72rem; font-weight:700; margin:1.1rem 0 .25rem; }
    .soft-note { background:#eaf4f6; border-left:4px solid #31899b; padding:.8rem 1rem;
      border-radius:8px; color:#27485d; }
    .small-muted { color:#657a8e; font-size:.85rem; }
    div[data-testid="stForm"] { background:white; padding:1.2rem; border:1px solid #e3ebef;
      border-radius:16px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading saved model…")
def load_model_and_metadata():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Saved model files are missing. Run `python -m src.train` from the project root first.")
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


@st.cache_data(show_spinner=False)
def read_csv_artifact(relative_path: str) -> pd.DataFrame | None:
    path = ARTIFACTS / relative_path
    return pd.read_csv(path) if path.exists() else None


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def show_image(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"This report has not been generated yet: `{path.relative_to(ROOT)}`")


try:
    model, metadata = load_model_and_metadata()
except Exception as error:
    st.error(str(error))
    st.stop()

test_metrics = metadata.get("held_out_test_metrics", {})
validation_metrics = metadata.get("validation_metrics", {})
threshold = float(metadata.get("threshold", 0.5))

with st.sidebar:
    st.markdown("## 🏥 Readmission Lab")
    st.caption("Explainable and fairness-aware ML research")
    page = st.radio(
        "Workspace",
        ["Overview", "Make a prediction", "Model comparison", "Explainability", "Fairness audit", "Data profile"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Selected model**")
    st.write(metadata.get("model_name", "Unknown").replace("_", " ").title())
    st.markdown("**Validation threshold**")
    st.write(f"{threshold:.2f}")
    st.caption(f"Seed {metadata.get('random_state', '—')} · scikit-learn {metadata.get('sklearn_version', '—')}")

st.markdown(
    '<div class="hero"><div class="eyebrow">Research prototype · Model results</div>'
    '<h1>Hospital readmission research lab</h1>'
    '<p>Explore saved experiment results, model explanations, subgroup metrics, and encounter-level estimates.</p></div>',
    unsafe_allow_html=True,
)
st.warning(
    "This system is an academic research prototype and is not intended for clinical diagnosis, treatment, or medical decision-making.",
    icon="⚠️",
)


if page == "Overview":
    st.markdown('<div class="section-kicker">Held-out test results</div>', unsafe_allow_html=True)
    cards = st.columns(5)
    for col, label, key in zip(cards, ["PR-AUC", "ROC-AUC", "Recall", "F1 score", "Specificity"], ["pr_auc", "roc_auc", "recall", "f1", "specificity"]):
        with col:
            value = test_metrics.get(key)
            metric_card(label, f"{float(value):.3f}" if value is not None else "—", "Held-out test set")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### Model comparison")
        show_image(REPORTS / "figures" / "model_comparison.png", "Validation PR-AUC across fitted candidates")
    with right:
        st.markdown("### Confusion matrix")
        show_image(REPORTS / "figures" / "confusion_matrix.png", "Held-out test set at the validation-selected threshold")

    st.markdown("### Experiment summary")
    comparison = read_csv_artifact("reports/model_comparison.csv")
    if comparison is not None:
        visible = [c for c in ["model", "threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy"] if c in comparison]
        st.dataframe(comparison[visible].sort_values("pr_auc", ascending=False), hide_index=True, use_container_width=True)
    st.markdown(
        '<div class="soft-note">The negative target class means <b>not readmitted within 30 days</b>. '
        'It does not mean never readmitted. This is a retrospective end-of-encounter model.</div>',
        unsafe_allow_html=True,
    )


elif page == "Make a prediction":
    st.markdown("### Estimate a research score")
    st.markdown('<p class="small-muted">Enter an encounter record. The form uses the same feature names and saved preprocessing pipeline used during training.</p>', unsafe_allow_html=True)
    feature_names = metadata["feature_list"]
    age_values = [f"[{n}-{n+10})" for n in range(0, 100, 10)]
    race_values = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"]
    gender_values = ["Female", "Male", "Unknown"]
    weight_values = ["Not recorded", "[0-25)", "[25-50)", "[50-75)", "[75-100)", "[100-125)", "[125-150)", "[150-175)", "[175-200)", "[200-250)", "[250-300)", ">300"]
    status_values = ["No", "Steady", "Up", "Down"]
    missing_values = ["Not recorded", "None", "Norm", ">200", ">300"]
    text_defaults = {
        "race": "Caucasian", "gender": "Female", "age": "[70-80)", "weight": "Not recorded",
        "payer_code": "Not recorded", "medical_specialty": "Not recorded", "diag_1": "250.00",
        "diag_2": "Not recorded", "diag_3": "Not recorded", "max_glu_serum": "None",
        "A1Cresult": "None", "change": "No", "diabetesMed": "Yes",
    }
    categorical_options = {
        "race": race_values, "gender": gender_values, "age": age_values, "weight": weight_values,
        "max_glu_serum": missing_values, "A1Cresult": ["Not recorded", "None", "Norm", ">7", ">8"],
        "change": ["No", "Ch"], "diabetesMed": ["Yes", "No"],
    }
    numeric_defaults = {
        "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
        "time_in_hospital": 4, "num_lab_procedures": 40, "num_procedures": 1,
        "num_medications": 12, "number_outpatient": 0, "number_emergency": 0,
        "number_inpatient": 0, "number_diagnoses": 6,
    }
    row: dict[str, object] = {}
    with st.form("encounter_input"):
        profile, encounter, history, medication = st.tabs(["Patient profile", "Encounter", "Prior utilization", "Medication indicators"])
        with profile:
            a, b, c = st.columns(3)
            for i, name in enumerate([n for n in ["race", "gender", "age", "weight", "payer_code", "medical_specialty"] if n in feature_names]):
                with [a, b, c][i % 3]:
                    options = categorical_options.get(name)
                    if options:
                        default = text_defaults.get(name, options[0])
                        row[name] = st.selectbox(name.replace("_", " ").title(), options, index=options.index(default) if default in options else 0)
                    else:
                        row[name] = st.text_input(name.replace("_", " ").title(), value=text_defaults.get(name, "Not recorded"))
        with encounter:
            a, b, c = st.columns(3)
            encounter_fields = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "time_in_hospital", "num_lab_procedures", "num_procedures", "number_diagnoses", "diag_1", "diag_2", "diag_3"]
            for i, name in enumerate([n for n in encounter_fields if n in feature_names]):
                with [a, b, c][i % 3]:
                    if name.startswith("diag_"):
                        row[name] = st.text_input(name.replace("_", " ").title(), value=text_defaults.get(name, "Not recorded"))
                    else:
                        row[name] = st.number_input(name.replace("_", " ").title(), min_value=0, max_value=1000, value=int(numeric_defaults.get(name, 0)), step=1)
        with history:
            a, b, c = st.columns(3)
            for i, name in enumerate(["number_inpatient", "number_emergency", "number_outpatient", "num_medications"]):
                if name in feature_names:
                    with [a, b, c][i % 3]:
                        row[name] = st.number_input(name.replace("_", " ").title(), min_value=0, max_value=500, value=int(numeric_defaults.get(name, 0)), step=1)
        with medication:
            medication_fields = [n for n in feature_names if n not in set(row) and n not in numeric_defaults and not n.startswith("diag_") and n not in {"race", "gender", "age", "weight", "payer_code", "medical_specialty", "max_glu_serum", "A1Cresult", "change", "diabetesMed"}]
            medication_fields += [n for n in ["max_glu_serum", "A1Cresult", "change", "diabetesMed"] if n in feature_names and n not in medication_fields]
            a, b, c = st.columns(3)
            for i, name in enumerate(medication_fields):
                with [a, b, c][i % 3]:
                    options = categorical_options.get(name, status_values if name not in {"max_glu_serum", "A1Cresult"} else missing_values)
                    default = text_defaults.get(name, "No")
                    row[name] = st.selectbox(name.replace("_", " ").title(), options, index=options.index(default) if default in options else 0)
        submitted = st.form_submit_button("Run prediction", type="primary", use_container_width=True)

    if submitted:
        # Every saved model input is supplied, even when optional fields use the explicit missing category.
        complete_row = {
            name: "__MISSING__" if row.get(name) == "Not recorded" else row.get(name, "__MISSING__")
            for name in feature_names
        }
        frame = pd.DataFrame([complete_row], columns=feature_names)
        score = float(model.predict_proba(frame)[0, 1])
        st.session_state["prediction_result"] = {"score": score, "row": complete_row}
        st.session_state.pop("counterfactual_result", None)

    result = st.session_state.get("prediction_result")
    if result:
        score = result["score"]
        predicted = score >= threshold
        st.markdown("### Model output")
        c1, c2, c3 = st.columns([1, 1, 1.4])
        with c1:
            metric_card("Estimated probability", f"{score:.1%}", "Model score for readmission within 30 days")
        with c2:
            metric_card("Thresholded model flag", "Positive" if predicted else "Negative", f"Validation threshold: {threshold:.2f}")
        with c3:
            st.progress(min(max(score, 0.0), 1.0), text=f"Probability score · {score:.3f}")
            st.caption("A negative output means not readmitted within 30 days according to this model's label.")

        from src.recommendations import generate_utilization_review
        review = generate_utilization_review(result["row"])
        st.markdown("**Potential areas for review**")
        if review:
            for item in review:
                st.write(f"• {item}")
        else:
            st.caption("No nonzero prior utilization counts were entered.")
        st.caption("These summaries describe entered data only; they are not care recommendations.")

        if st.button("Explore utilization what-if scenarios"):
            try:
                from src.counterfactual import counterfactual
                scenarios = counterfactual(pd.DataFrame([result["row"]], columns=feature_names))
                st.session_state["counterfactual_result"] = scenarios
            except Exception as error:
                st.error(f"Counterfactual analysis is unavailable: {error}")
        if "counterfactual_result" in st.session_state:
            scenarios = st.session_state["counterfactual_result"]
            if scenarios:
                st.dataframe(pd.DataFrame(scenarios), hide_index=True, use_container_width=True)
            else:
                st.info("No lower-score scenarios were found in the configured count-change search. This analysis is mathematical model sensitivity, not a causal intervention.")


elif page == "Model comparison":
    st.markdown("### Candidate models")
    comparison = read_csv_artifact("reports/model_comparison.csv")
    if comparison is None:
        st.info("Run `python -m src.train` to generate model comparison results.")
    else:
        shown = [c for c in ["model", "threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy"] if c in comparison]
        st.dataframe(comparison[shown].sort_values("pr_auc", ascending=False), hide_index=True, use_container_width=True)
        l, r = st.columns(2)
        with l:
            st.bar_chart(comparison.set_index("model")[["pr_auc", "roc_auc"]], height=330)
        with r:
            show_image(REPORTS / "figures" / "roc_curve.png", "Selected model ROC curve on held-out test records")
        l, r = st.columns(2)
        with l:
            show_image(REPORTS / "figures" / "precision_recall_curve.png", "Selected model precision-recall curve on held-out test records")
        with r:
            show_image(REPORTS / "figures" / "confusion_matrix.png", "Selected model at the validation-selected threshold")
        fallback = metadata.get("xgboost_fallback_reason")
        if fallback:
            st.info("XGBoost could not load in the training environment; the saved comparison used the documented HistGradientBoosting fallback.")


elif page == "Explainability":
    st.markdown("### Global model explanation")
    st.markdown('<div class="soft-note">SHAP values summarize how input features influenced model output across the sampled records. They describe the fitted model, not causal effects.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        show_image(ARTIFACTS / "explainability" / "shap_summary.png", "Global SHAP summary, computed on a 300-record sample")
    with right:
        shap_table = read_csv_artifact("explainability/global_shap_importance.csv")
        if shap_table is not None:
            st.markdown("#### Highest mean absolute contributions")
            st.dataframe(shap_table.head(15), hide_index=True, use_container_width=True)
        else:
            st.info("Run SHAP analysis with `python -c 'from src.explainability import run_shap; run_shap()'`.")


elif page == "Fairness audit":
    st.markdown("### Subgroup performance")
    st.markdown('<div class="soft-note">Descriptive subgroup measurements are shown for review. They do not establish fairness or determine whether a model should be used.</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Race", "Gender"])
    for tab, dimension in zip(tabs, ["race", "gender"]):
        with tab:
            table = read_csv_artifact(f"fairness/{dimension}_audit.csv")
            if table is None:
                st.info(f"No {dimension} audit found. Run the fairness audit to create it.")
                continue
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.bar_chart(table.set_index("group")[["selection_rate", "recall"]], height=320)
            if (table["n"] < 100).any():
                st.caption("Some groups contain fewer than 100 held-out records; estimates for these groups may be especially unstable.")


elif page == "Data profile":
    st.markdown("### Dataset profile")
    summary_path = REPORTS / "eda_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        a, b, c = st.columns(3)
        with a: metric_card("Encounter records", f"{summary.get('rows', 0):,}")
        with b: metric_card("Early readmission rate", f"{summary.get('target_positive_rate', 0):.1%}")
        with c: metric_card("Model input features", str(len(metadata.get("feature_list", []))))
    else:
        st.info("Run `python -m src.eda` to generate the data profile.")
    l, r = st.columns(2)
    with l: show_image(REPORTS / "figures" / "eda_target_distribution.png", "Readmission target counts")
    with r: show_image(REPORTS / "figures" / "missingness.png", "Most-missing source fields")
    missing = read_csv_artifact("reports/missing_value_report.csv")
    if missing is not None:
        st.markdown("#### Missing-value report")
        st.dataframe(missing.sort_values("missing_fraction", ascending=False).head(15), use_container_width=True)


st.divider()
st.caption(
    f"Model: {metadata.get('model_name', '—').replace('_', ' ')} · "
    f"Threshold selected on validation data: {threshold:.2f} · "
    f"Training record: {metadata.get('training_date_utc', '—')}"
)
