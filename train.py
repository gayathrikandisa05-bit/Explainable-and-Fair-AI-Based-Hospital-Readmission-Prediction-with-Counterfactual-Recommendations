"""Train, tune, select threshold, and save a complete prediction pipeline."""
import json, logging, platform, sys
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from src.config import MODELS, PREPROCESSORS, REPORTS, RANDOM_STATE, TARGET
from src.data_loader import load_data
from src.preprocessing import build_preprocessor, build_dense_preprocessor
from src.splitting import patient_disjoint_split

def _metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {"threshold": float(threshold), "precision": float(precision_score(y,pred,zero_division=0)), "recall": float(recall_score(y,pred,zero_division=0)), "f1": float(f1_score(y,pred,zero_division=0)), "roc_auc": float(roc_auc_score(y,p)), "pr_auc": float(average_precision_score(y,p))}

def train(path=None):
    df = load_data(path) if path else load_data()
    y = df.pop(TARGET).astype(int); groups=df.pop("_patient_group"); X = df
    train_idx,valid_idx,test_idx=patient_disjoint_split(X,y,groups,RANDOM_STATE)
    X_train,X_valid,X_test=X.iloc[train_idx],X.iloc[valid_idx],X.iloc[test_idx]
    y_train,y_valid,y_test=y.iloc[train_idx],y.iloc[valid_idx],y.iloc[test_idx]
    group_train=groups.iloc[train_idx]
    candidates = {
      "logistic_regression": (LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE), {}),
      "random_forest": (RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE,n_jobs=-1), {"model__n_estimators":[150,250],"model__max_depth":[None,16,28],"model__min_samples_leaf":[2,8]}),
    }
    try:
        from xgboost import XGBClassifier
        # Import may fail when the host lacks macOS libomp; retain a runnable dense fallback.
        candidates["xgboost"]=(XGBClassifier(n_estimators=180,max_depth=5,learning_rate=.08,subsample=.85,colsample_bytree=.85,reg_lambda=1.0,scale_pos_weight=float((y_train==0).sum()/(y_train==1).sum()),eval_metric="logloss",tree_method="hist",n_jobs=4,random_state=RANDOM_STATE),{"model__n_estimators":[120,200],"model__max_depth":[4,6],"model__learning_rate":[.05,.1]})
        factories={name:build_preprocessor for name in candidates}
    except Exception as exc:
        candidates["hist_gradient_boosting_fallback"]=(HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=20,class_weight="balanced",random_state=RANDOM_STATE),{})
        factories={name:build_preprocessor for name in candidates}
        factories["hist_gradient_boosting_fallback"]=build_dense_preprocessor
        xgb_fallback_reason=f"XGBoost unavailable at runtime: {exc}"
    rows=[]; best=None; best_score=-1; params_report={}
    for name,(estimator,space) in candidates.items():
        pipe=Pipeline([("preprocess",factories[name](X_train)),("model",estimator)])
        if space:
            cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=RANDOM_STATE)
            search=RandomizedSearchCV(pipe,space,n_iter=5,scoring="average_precision",cv=cv,random_state=RANDOM_STATE,n_jobs=1,refit=True)
            search.fit(X_train,y_train,groups=group_train); fitted=search.best_estimator_; params_report[name]={"best_parameters":search.best_params_,"cv_pr_auc":float(search.best_score_),"selected_metric":"average_precision","cv_grouped_by_patient":True}
        else:
            fitted=pipe.fit(X_train,y_train); params_report[name]={"best_parameters":{},"selected_metric":"default class-weighted baseline"}
        valid_p=fitted.predict_proba(X_valid)[:,1]
        thresholds=np.linspace(.10,.90,81)
        scored=[(_metrics(y_valid,valid_p,t),t) for t in thresholds]
        valid_metrics, threshold=max(scored,key=lambda item:(item[0]["f1"],item[0]["recall"]))
        rows.append({"experiment_id":name,"model":name,"preprocessing_version":"v1","features":"|".join(X.columns),"feature_count":int(X.shape[1]),"class_imbalance_method":"class_weight","threshold":threshold,**valid_metrics,"accuracy":float(((valid_p>=threshold)==y_valid).mean())})
        params_report[name]["validation_metrics"] = valid_metrics
        if valid_metrics["pr_auc"] > best_score:
            best_score=valid_metrics["pr_auc"]; best=(name,fitted,float(threshold),valid_metrics)
    name,model,threshold,valid_metrics=best
    test_p=model.predict_proba(X_test)[:,1]
    test_metrics=_metrics(y_test,test_p,threshold)
    test_metrics["accuracy"]=float(((test_p>=threshold)==y_test).mean())
    pred=(test_p>=threshold).astype(int)
    tn,fp,fn,tp=__import__("sklearn.metrics",fromlist=["confusion_matrix"]).confusion_matrix(y_test,pred,labels=[0,1]).ravel()
    test_metrics["specificity"]=float(tn/(tn+fp)) if tn+fp else 0.
    test_metrics["confusion_matrix"]=[[int(tn),int(fp)],[int(fn),int(tp)]]
    MODELS.mkdir(parents=True,exist_ok=True); PREPROCESSORS.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,MODELS/"best_model.pkl"); joblib.dump(model.named_steps["preprocess"],PREPROCESSORS/"preprocessor.pkl")
    with open(REPORTS/"best_parameters.json","w") as f: json.dump(params_report,f,indent=2)
    comparison=pd.DataFrame(rows); comparison.to_csv(REPORTS/"model_comparison.csv",index=False); comparison.to_csv(REPORTS/"experiment_summary.csv",index=False)
    metadata={"model_name":name,"threshold":threshold,"feature_list":list(X.columns),"training_date_utc":datetime.now(timezone.utc).isoformat(),"validation_metrics":valid_metrics,"held_out_test_metrics":test_metrics,"random_state":RANDOM_STATE,"python_version":sys.version,"sklearn_version":sklearn.__version__,"class_weighting":True,"split":"patient-disjoint stratified group folds: approximately 71% train / 14% validation / 14% held-out test","test_touched_once_after_selection":True}
    if "xgb_fallback_reason" in locals(): metadata["xgboost_fallback_reason"]=xgb_fallback_reason
    with open(MODELS/"model_metadata.json","w") as f: json.dump(metadata,f,indent=2)
    return metadata

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO); print(json.dumps(train(),indent=2))
