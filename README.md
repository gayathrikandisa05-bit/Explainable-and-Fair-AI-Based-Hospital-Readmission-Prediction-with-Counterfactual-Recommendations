# Explainable and Fair AI-Based Hospital Readmission Prediction with Counterfactual Recommendations

Academic research prototype for estimating whether a diabetic patient encounter is followed by a hospital readmission within 30 days, with interpretable model analysis, subgroup evaluation, and constrained what-if scenarios.

> **Medical safety:** This system is an academic research prototype and is not intended for clinical diagnosis, treatment, or medical decision-making.

## Problem and motivation

Early readmission is a challenging prediction target in historical encounter data. This project presents an end-to-end, reproducible workflow that examines predictive performance alongside explanations and subgroup metrics. It is not clinically validated and makes no claim that its hypothetical scenarios cause better outcomes.

## Dataset and target

Use the UCI Diabetes 130-US Hospitals dataset (`diabetic_data.csv`, `IDS_mapping.csv`). Place both files in `data/raw/`. The project does not download them. The label `readmitted_30` maps `<30` to 1 and `>30` and `NO` to 0. Therefore class 0 means **not readmitted within 30 days**, never “never readmitted.” Dataset source: [UCI Machine Learning Repository, Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008). Follow the dataset's citation and terms.

## Method

Identifiers and the original target are removed. `?` becomes missing. Median numeric imputation with missingness indicators and an explicit categorical missing value, scaling, and one-hot encoding are fit only inside the training pipeline. Engineered variables include previous utilization, medication and diagnosis burden, encounter intensity, and training-quantile flags. Logistic regression, random forest, and XGBoost are compared with class weighting. Random forest and XGBoost receive bounded randomized searches. Patient-disjoint stratified group folds approximate a 71% train, 14% validation, and 14% test split, avoiding the same patient's encounters crossing partitions. Validation PR-AUC selects the model; validation F1 (recall tie-break) selects a threshold. The test partition is held out until this selection is fixed. If XGBoost cannot load because the host is missing its OpenMP runtime, the run records and uses a dense histogram gradient boosting fallback.

Some available fields (stay duration, procedures, laboratory count, and discharge context) are only known during/after an encounter. This is currently a **retrospective end-of-encounter estimate**. For admission-time prediction, remove such fields and retrain. See [methodology](docs/methodology.md) and [limitations](docs/limitations.md).

## Explainability, fairness, and counterfactual analysis

SHAP summaries are optional and computed on a sample. The subgroup audit reports selection rate and predictive metrics by race (descriptive only; no parity criterion is asserted). Counterfactual analysis varies prior utilization counts mathematically and does not establish causal or actionable interventions. Dashboard language is restricted to “potential areas for review.” No diagnosis, treatment plan, medication dosage, or emergency instruction is generated.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy the two source files into `data/raw/`. Without `data/raw/diabetic_data.csv`, commands give a direct placement error.

## Run

```bash
python -m src.eda
python -m src.train
python -m src.evaluate
python -c 'from src.fairness import audit; print(audit())'
python -c 'from src.explainability import run_shap; print(run_shap())'
streamlit run app/streamlit_app.py
pytest
```

Run EDA, training, and evaluation sequentially. SHAP requires its optional package dependencies. Counterfactual API: load an encounter-shaped pandas DataFrame and call `src.counterfactual.counterfactual(record)`. Inspect returned hypothetical changes critically.

## Project structure

`src/` contains modular data, feature, training, evaluation, explanation, fairness, counterfactual, and recommendation code. `app/` contains Streamlit. `notebooks/` contains nine phase notebooks. `docs/` contains research gap, methods, architecture, experiments, limitations, and 55 viva Q&As. `artifacts/` receives generated reports and models; raw data and model binaries are excluded from git.

Expected outputs include missingness report, EDA figures and summary, `model_comparison.csv`, `experiment_summary.csv`, `best_parameters.json`, `best_model.pkl`, `model_metadata.json`, held-out metrics and plots, race audit, and (when installed) SHAP importance. Results must be generated from an actual run; this repository does not encode fabricated metrics.

## Reproducibility and privacy

Random seed is 42. The training metadata captures Python and scikit-learn versions, selected features, threshold, split, and measured metrics. Raw healthcare data and patient-level data must not be committed. Suggested commits: `feat: add leakage-aware training pipeline`, `feat: add evaluation and fairness reports`, `docs: document research protocol`.

## Limitations and future work

Historical single-context data may not transport; missingness and label definitions may encode local practice. Subgroup results need uncertainty intervals and stakeholder interpretation. Follow-up work should define a prospective decision point, conduct external validation, evaluate calibration and decision utility, audit uncertainty, and obtain appropriate clinical, ethical, and governance review. Full details are in `docs/limitations.md`.
