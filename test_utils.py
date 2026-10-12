from src.recommendations import generate_recommendations, generate_utilization_review

def test_recommendations_are_review_only():
    result=generate_recommendations({"number_inpatient":.4,"race":.2,"number_emergency":-.1})
    assert len(result)==1
    assert result[0].startswith("Potential area for review:")

def test_utilization_review_does_not_prescribe():
    result=generate_utilization_review({"number_inpatient":2,"number_emergency":0})
    assert len(result)==1
    assert "prior inpatient utilization" in result[0]
