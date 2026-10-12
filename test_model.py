import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor

def test_pipeline_prediction_probability_range():
    x=pd.DataFrame({"number_inpatient":[0,1,0,2,1,0],"race":["A","B","A","B","A","B"]})
    y=[0,1,0,1,0,1]
    pipe=Pipeline([("preprocess",build_preprocessor(x)),("model",LogisticRegression())]).fit(x,y)
    p=pipe.predict_proba(x)[:,1]
    assert np.all((p>=0)&(p<=1))
