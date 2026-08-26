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
# JSON Output & Directories
# ---------------------------------------------------

OUTPUT_JSON = OUTPUT_DIR / "patient_information.json"
STEGO_OUTPUT_DIR = OUTPUT_DIR / "stego_images"
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"
EXTRACTED_PAYLOADS_DIR = OUTPUT_DIR / "extracted_payloads"
IMAGES_DIR = ROOT_DIR / "images"

# Default Cover Image
DEFAULT_COVER_IMAGE = IMAGES_DIR / "xray.png"

# ---------------------------------------------------
# Non-ROI (Region of Non-Interest) Detection Settings
# ---------------------------------------------------

BLOCK_SIZE = 48
VARIANCE_THRESHOLD = 120
DARK_MEAN_THRESHOLD = 30
DARK_VAR_THRESHOLD = 50
DENSE_WHITE_MEAN_THRESHOLD = 180
DENSE_WHITE_VAR_THRESHOLD = 300

# ---------------------------------------------------
# Cryptography & Steganography Settings
# ---------------------------------------------------

DEFAULT_CIPHER = "AES-256-GCM"  # Options: "AES-256-GCM", "ChaCha20-Poly1305"
PBKDF2_ITERATIONS = 100_000
MAGIC_HEADER_BYTES = b"STEG"
LSB_BITS_PER_CHANNEL = 1  # Standard 1-bit LSB embedding per color channel (or grayscale)