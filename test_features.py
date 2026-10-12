import pandas as pd
from src.feature_engineering import FeatureEngineer

def test_engineered_utilization_and_flags():
    x=pd.DataFrame({"number_inpatient":[2,0],"number_emergency":[1,0],"number_outpatient":[3,0],"time_in_hospital":[8,2],"num_medications":[10,2],"number_diagnoses":[7,1]})
    result=FeatureEngineer().fit(x).transform(x)
    assert result["previous_utilization"].tolist()==[6,0]
    assert result["previous_admission_flag"].tolist()==[1,0]
    assert result["long_stay_flag"].tolist()==[1,0]
