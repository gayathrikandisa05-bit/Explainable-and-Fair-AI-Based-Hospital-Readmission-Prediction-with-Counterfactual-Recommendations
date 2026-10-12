"""Constrained, descriptive what-if analysis for mutable utilization counts."""
import joblib, json
from src.config import MODELS

def counterfactual(record, max_steps=5):
    """Search modest count reductions; output is a mathematical scenario, not advice."""
    model=joblib.load(MODELS/"best_model.pkl"); threshold=json.loads((MODELS/"model_metadata.json").read_text())["threshold"]
    baseline=float(model.predict_proba(record)[0,1]); candidates=[]
    mutable=[c for c in ("number_outpatient","number_emergency","number_inpatient") if c in record]
    for column in mutable:
        for reduction in range(1,max_steps+1):
            changed=record.copy(); changed.loc[:,column]=(changed[column].astype(float)-reduction).clip(lower=0)
            score=float(model.predict_proba(changed)[0,1])
            if score < baseline: candidates.append({"feature":column,"hypothetical_change":-reduction,"baseline_probability":baseline,"scenario_probability":score,"crosses_threshold":bool(baseline>=threshold and score<threshold)})
    return sorted(candidates,key=lambda row:row["scenario_probability"]-row["baseline_probability"])
