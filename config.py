"""Central project paths and reproducibility settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
PREPROCESSORS = ARTIFACTS / "preprocessors"
REPORTS = ARTIFACTS / "reports"
FIGURES = REPORTS / "figures"
RANDOM_STATE = 42
TARGET = "readmitted_30"
ID_COLUMNS = ["encounter_id", "patient_nbr"]
RAW_DATA = DATA_RAW / "diabetic_data.csv"
MAPPING_DATA = DATA_RAW / "IDS_mapping.csv"

for _path in (DATA_PROCESSED, MODELS, PREPROCESSORS, ARTIFACTS / "explainability", ARTIFACTS / "fairness", FIGURES):
    _path.mkdir(parents=True, exist_ok=True)
