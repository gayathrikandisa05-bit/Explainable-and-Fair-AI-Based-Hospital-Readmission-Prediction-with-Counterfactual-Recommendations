"""Descriptive subgroup performance and selection-rate audit."""
import joblib, json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.config import MODELS, ARTIFACTS, RANDOM_STATE, TARGET
from src.data_loader import load_data
from src.splitting import patient_disjoint_split

def audit(attribute="race"):
    df=load_data(); y=df.pop(TARGET).astype(int); patient_groups=df.pop("_patient_group"); groups=df[attribute].fillna("Unknown").astype(str)
    _,_,test_idx=patient_disjoint_split(df,y,patient_groups,RANDOM_STATE); xt=df.iloc[test_idx]; yt=y.iloc[test_idx]
    g=groups.iloc[test_idx]; model=joblib.load(MODELS/"best_model.pkl"); threshold=json.loads((MODELS/"model_metadata.json").read_text())["threshold"]; p=model.predict_proba(xt)[:,1]; pred=(p>=threshold).astype(int)
    rows=[]
    for value in sorted(g.unique()):
        m=(g==value).to_numpy(); actual=yt.to_numpy()[m]; decision=pred[m]
        rows.append({"attribute":attribute,"group":value,"n":int(m.sum()),"positive_rate":float(actual.mean()),"selection_rate":float(decision.mean()),"recall":float(recall_score(actual,decision,zero_division=0)),"precision":float(precision_score(actual,decision,zero_division=0)),"accuracy":float(accuracy_score(actual,decision))})
    out=pd.DataFrame(rows); out.to_csv(ARTIFACTS/"fairness"/f"{attribute}_audit.csv",index=False); return out
