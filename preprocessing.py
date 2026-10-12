"""Leakage-safe sklearn preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from src.feature_engineering import FeatureEngineer

def build_preprocessor(X: pd.DataFrame) -> Pipeline:
    """Return a pipeline whose imputers, encoders, scaling, and cutoffs fit on training data."""
    engineered_numeric = ["previous_utilization", "medication_burden", "diagnosis_burden", "hospitalization_intensity", "high_emergency_use", "previous_admission_flag", "long_stay_flag", "high_medication_burden"]
    numeric = X.select_dtypes(include=["number"]).columns.tolist() + engineered_numeric
    categorical = [c for c in X.columns if c not in X.select_dtypes(include=["number"]).columns]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant",fill_value="__MISSING__")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10))])
    transformer = ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)], remainder="drop")
    return Pipeline([("features", FeatureEngineer()), ("transform", transformer)])

def build_dense_preprocessor(X: pd.DataFrame) -> Pipeline:
    """Compact ordinal encoded pipeline for estimators requiring dense input."""
    numeric=X.select_dtypes(include=["number"]).columns.tolist()+["previous_utilization","medication_burden","diagnosis_burden","hospitalization_intensity","high_emergency_use","previous_admission_flag","long_stay_flag","high_medication_burden"]
    categorical=[c for c in X.columns if c not in X.select_dtypes(include=["number"]).columns]
    transform=ColumnTransformer([("numeric",SimpleImputer(strategy="median",add_indicator=True),numeric),("categorical",Pipeline([("imputer",SimpleImputer(strategy="constant",fill_value="__MISSING__")),("encode",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]),categorical)],sparse_threshold=0)
    return Pipeline([("features",FeatureEngineer()),("transform",transform)])
