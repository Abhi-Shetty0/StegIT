from pathlib import Path

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"

OUTPUT_DIR = ROOT_DIR / "output"

MODEL_DIR = ROOT_DIR / "models"

# ---------------------------------------------------
# HuggingFace Model
# ---------------------------------------------------

# Clinical language model (for embeddings/classification later)
CLINICAL_BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

# Token Classification Model for Medical NER
NER_MODEL = "Clinical-AI-Apollo/Medical-NER"

# ---------------------------------------------------
# SciSpaCy
# ---------------------------------------------------

SCISPACY_MODEL = "en_core_sci_sm"

# ---------------------------------------------------
# NLP Settings
# ---------------------------------------------------

MAX_LENGTH = 512

DEVICE = "cpu"

# ---------------------------------------------------
# JSON Output
# ---------------------------------------------------

OUTPUT_JSON = OUTPUT_DIR / "patient_information.json"