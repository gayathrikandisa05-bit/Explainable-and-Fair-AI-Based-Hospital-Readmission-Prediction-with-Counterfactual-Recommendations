"""Non-clinical, review-oriented summaries of model drivers."""
def generate_recommendations(contributions: dict, top_n: int = 3) -> list[str]:
    """Return potential areas for review; never prescribe care."""
    labels = {
        "number_inpatient": "prior inpatient utilization", "number_emergency": "prior emergency utilization",
        "number_outpatient": "prior outpatient utilization", "time_in_hospital": "length of the recorded stay",
        "num_medications": "medication count", "number_diagnoses": "recorded diagnosis count",
    }
    protected_or_contextual = {"race", "gender", "age"}
    positive = sorted(((k, float(v)) for k, v in contributions.items() if float(v) > 0 and k not in protected_or_contextual), key=lambda pair: pair[1], reverse=True)[:top_n]
    return [f"Potential area for review: {labels.get(name, name.replace('_', ' '))} contributed to the model's higher estimated risk." for name, _ in positive]

def generate_utilization_review(record: dict) -> list[str]:
    """Describe recorded utilization fields for human review without prescribing action."""
    labels={"number_inpatient":"prior inpatient utilization","number_emergency":"prior emergency utilization","number_outpatient":"prior outpatient utilization"}
    return [f"Potential area for review: recorded {label} is {record[name]}." for name,label in labels.items() if name in record and float(record[name])>0]
