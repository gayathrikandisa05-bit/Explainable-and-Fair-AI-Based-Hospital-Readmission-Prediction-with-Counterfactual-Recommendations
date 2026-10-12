"""Reproducible exploratory summaries and plots."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.config import FIGURES, REPORTS, RAW_DATA, TARGET
from src.data_loader import load_data

def run_eda():
    raw=pd.read_csv(RAW_DATA,na_values=["?"],low_memory=False); df=load_data(); df=df.drop(columns=["_patient_group"]); data=df.copy()
    data[TARGET]=df[TARGET]
    charts={"target_distribution":"readmitted_30","age":"age","gender":"gender","race":"race","time_in_hospital":"time_in_hospital","num_medications":"num_medications","number_inpatient":"number_inpatient","number_emergency":"number_emergency","number_diagnoses":"number_diagnoses"}
    summary={"rows":len(df),"columns":len(df.columns),"target_positive_rate":float(df[TARGET].mean()),"missingness_top":raw.isna().mean().sort_values(ascending=False).head(20).to_dict()}
    for label,col in charts.items():
        if col not in data: continue
        plt.figure(figsize=(9,5))
        if label=="target_distribution": sns.countplot(data=data,x=col)
        else:
            if col in {"time_in_hospital","num_medications","number_inpatient","number_emergency","number_diagnoses"}:
                data[col+"_group"]=pd.cut(data[col],bins=[-1,0,1,3,5,10,25,1000],include_lowest=True).astype(str)
                plot_col=col+"_group"
            else: plot_col=col
            sns.barplot(data=data,x=plot_col,y=TARGET,errorbar=None)
        plt.xticks(rotation=40,ha="right"); plt.tight_layout(); plt.savefig(FIGURES/f"eda_{label}.png",dpi=130); plt.close()
    numeric=raw.drop(columns=["encounter_id","patient_nbr"],errors="ignore").select_dtypes(include="number"); plt.figure(figsize=(12,9)); sns.heatmap(numeric.corr(),cmap="coolwarm",center=0); plt.tight_layout(); plt.savefig(FIGURES/"correlation.png",dpi=130); plt.close()
    raw.isna().mean().sort_values(ascending=False).head(30).plot(kind="bar",figsize=(12,5)); plt.tight_layout(); plt.savefig(FIGURES/"missingness.png",dpi=130); plt.close()
    (REPORTS/"eda_summary.json").write_text(json.dumps(summary,indent=2)); pd.DataFrame([summary]).to_csv(REPORTS/"eda_summary.csv",index=False); return summary

if __name__ == "__main__": print(run_eda())
