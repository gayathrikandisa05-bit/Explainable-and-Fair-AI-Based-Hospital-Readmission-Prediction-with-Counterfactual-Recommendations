"""Dataset loading, target construction, and descriptive missingness report."""
import logging
from pathlib import Path
import pandas as pd
from src.config import RAW_DATA, REPORTS, ID_COLUMNS, TARGET

LOGGER = logging.getLogger(__name__)

def load_data(path: Path = RAW_DATA, report_dir: Path = REPORTS) -> pd.DataFrame:
    """Load source encounters, normalize missing markers, and create binary target."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Place diabetic_data.csv under data/raw/.")
    frame = pd.read_csv(path, na_values=["?"], keep_default_na=True, low_memory=False)
    required = {"readmitted", *ID_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    report_dir.mkdir(parents=True, exist_ok=True)
    missingness = frame.isna().agg(["sum", "mean"]).T.rename(columns={"sum": "missing_count", "mean": "missing_fraction"})
    missingness.to_csv(report_dir / "missing_value_report.csv")
    invalid = set(frame["readmitted"].dropna().unique()) - {"<30", ">30", "NO"}
    if invalid:
        raise ValueError(f"Unexpected readmitted values: {sorted(invalid)}")
    frame[TARGET] = frame["readmitted"].eq("<30").astype("int8")
    # Keep a private split key so encounters from one patient cannot cross partitions.
    groups = frame["patient_nbr"].copy()
    result = frame.drop(columns=["readmitted", *ID_COLUMNS])
    result["_patient_group"] = groups.to_numpy()
    return result
