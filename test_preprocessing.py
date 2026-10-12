import pandas as pd
from src.data_loader import load_data
from src.preprocessing import build_preprocessor

def test_load_creates_binary_target_and_drops_identifiers(tmp_path):
    path=tmp_path/"tiny.csv"
    pd.DataFrame({"encounter_id":[1,2,3],"patient_nbr":[8,9,10],"readmitted":["<30",">30","NO"],"x":["?","a","b"]}).to_csv(path,index=False)
    df=load_data(path,tmp_path)
    assert df["readmitted_30"].tolist()==[1,0,0]
    assert "encounter_id" not in df and "patient_nbr" not in df
    assert pd.isna(df.loc[0,"x"])

def test_preprocessor_fits_small_mixed_data():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from src.feature_engineering import FeatureEngineer
    from sklearn.impute import SimpleImputer
    X=pd.DataFrame({"number_inpatient":[0,1,2,0],"gender":["F","M","F","M"]})
    transformed=FeatureEngineer().fit_transform(X)
    assert "previous_utilization" in transformed
    pipe=build_preprocessor(X)
    result=pipe.fit_transform(X)
    assert result.shape[0]==4
