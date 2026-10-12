"""Held-out evaluation plots and report generation."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve)
from src.config import MODELS, REPORTS, FIGURES, TARGET
from src.data_loader import load_data
from src.config import RANDOM_STATE
from src.splitting import patient_disjoint_split

def evaluate():
    df=load_data(); y=df.pop(TARGET).astype(int); groups=df.pop("_patient_group"); _,_,test_idx=patient_disjoint_split(df,y,groups,RANDOM_STATE); Xtest=df.iloc[test_idx]; yt=y.iloc[test_idx]
    model=joblib.load(MODELS/"best_model.pkl"); meta=json.loads((MODELS/"model_metadata.json").read_text()); p=model.predict_proba(Xtest)[:,1]; pred=(p>=meta["threshold"]).astype(int)
    fpr,tpr,_=roc_curve(yt,p); prec,rec,_=precision_recall_curve(yt,p)
    for name,x,yv,xlabel,ylabel in [("roc_curve.png",fpr,tpr,"False positive rate","True positive rate"),("precision_recall_curve.png",rec,prec,"Recall","Precision")]:
        plt.figure(); plt.plot(x,yv); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(FIGURES/name,dpi=140); plt.close()
    cm=confusion_matrix(yt,pred); plt.figure(); plt.imshow(cm,cmap="Blues"); plt.xticks([0,1],["Not within 30d","Within 30d"]); plt.yticks([0,1],["Not within 30d","Within 30d"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.colorbar(); plt.tight_layout(); plt.savefig(FIGURES/"confusion_matrix.png",dpi=140); plt.close()
    comparison_path=REPORTS/"model_comparison.csv"
    if comparison_path.exists():
        comparison=pd.read_csv(comparison_path); plt.figure(figsize=(8,4)); plt.bar(comparison["model"],comparison["pr_auc"]); plt.ylabel("Validation PR-AUC"); plt.xticks(rotation=25,ha="right"); plt.tight_layout(); plt.savefig(FIGURES/"model_comparison.png",dpi=140); plt.close()
    result={"accuracy":accuracy_score(yt,pred),"precision":precision_score(yt,pred,zero_division=0),"recall":recall_score(yt,pred,zero_division=0),"f1":f1_score(yt,pred,zero_division=0),"roc_auc":roc_auc_score(yt,p),"pr_auc":average_precision_score(yt,p),"specificity":cm[0,0]/cm[0].sum(),"confusion_matrix":cm.tolist()}
    pd.DataFrame([result]).to_csv(REPORTS/"held_out_evaluation.csv",index=False); return result

if __name__ == "__main__": print(evaluate())
