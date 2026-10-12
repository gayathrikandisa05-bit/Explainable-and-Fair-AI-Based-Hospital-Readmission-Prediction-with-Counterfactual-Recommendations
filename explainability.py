"""Optional SHAP summaries with graceful dependency handling."""
import json
import joblib
import pandas as pd
from src.config import MODELS, REPORTS, ARTIFACTS, TARGET
from src.data_loader import load_data

def run_shap(sample_size=300):
    import matplotlib
    matplotlib.use("Agg")
    try: import shap
    except ImportError as exc: raise RuntimeError("SHAP is unavailable. Install requirements.txt to run explainability analysis.") from exc
    df=load_data(); df.pop(TARGET); df.pop("_patient_group"); X=df.sample(min(sample_size,len(df)),random_state=42)
    model=joblib.load(MODELS/"best_model.pkl"); pre=model.named_steps["preprocess"].transform(X); estimator=model.named_steps["model"]
    explainer=shap.TreeExplainer(estimator) if hasattr(estimator,"feature_importances_") else shap.Explainer(estimator,pre)
    values=explainer(pre)
    names=model.named_steps["preprocess"].named_steps["transform"].get_feature_names_out()
    import numpy as np
    import matplotlib.pyplot as plt
    vals=values.values
    if vals.ndim==3: vals=vals[:,:,1]
    importance=pd.DataFrame({"feature":names,"mean_abs_shap":np.abs(vals).mean(axis=0)}).sort_values("mean_abs_shap",ascending=False)
    importance.to_csv(ARTIFACTS/"explainability"/"global_shap_importance.csv",index=False)
    shap.summary_plot(vals,pre,feature_names=names,show=False,max_display=20)
    plt.tight_layout(); plt.savefig(ARTIFACTS/"explainability"/"shap_summary.png",dpi=150,bbox_inches="tight"); plt.close()
    return importance.head(20)
