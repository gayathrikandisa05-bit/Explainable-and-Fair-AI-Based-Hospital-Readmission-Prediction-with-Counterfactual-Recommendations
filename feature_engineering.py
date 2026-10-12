"""Train-fitted engineered features for encounter-level readmission modeling."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add utilization and complexity features; learn cut points on training data."""
    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.emergency_cutoff_ = float(X["number_emergency"].fillna(0).quantile(.75)) if "number_emergency" in X else 0.
        self.stay_cutoff_ = float(X["time_in_hospital"].fillna(0).quantile(.75)) if "time_in_hospital" in X else 0.
        self.medication_cutoff_ = float(X["num_medications"].fillna(0).median()) if "num_medications" in X else 0.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in ("number_inpatient", "number_emergency", "number_outpatient", "num_medications", "number_diagnoses", "time_in_hospital"):
            if col in out:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        def col(name):
            return out[name].fillna(0) if name in out else pd.Series(0, index=out.index, dtype=float)
        out["previous_utilization"] = col("number_inpatient") + col("number_emergency") + col("number_outpatient")
        out["medication_burden"] = col("num_medications")
        out["diagnosis_burden"] = col("number_diagnoses")
        # Available only after the encounter: interpret as encounter-complexity association, not prospective triage.
        out["hospitalization_intensity"] = col("time_in_hospital") + col("num_lab_procedures") + col("num_procedures")
        out["high_emergency_use"] = (col("number_emergency") > self.emergency_cutoff_).astype("int8")
        out["previous_admission_flag"] = (col("number_inpatient") > 0).astype("int8")
        out["long_stay_flag"] = (col("time_in_hospital") > self.stay_cutoff_).astype("int8")
        out["high_medication_burden"] = (col("num_medications") > self.medication_cutoff_).astype("int8")
        return out

    def get_feature_names_out(self, input_features=None):
        names = list(input_features if input_features is not None else self.feature_names_in_)
        return np.asarray(names + ["previous_utilization", "medication_burden", "diagnosis_burden", "hospitalization_intensity", "high_emergency_use", "previous_admission_flag", "long_stay_flag", "high_medication_burden"], dtype=object)
