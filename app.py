import streamlit as st
import json
import re
import os
import base64
import zstandard as zstd
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

st.set_page_config(page_title="StegIT - Clinical NLP & Encryption Engine", layout="wide")

# --- Initialize Encryptor State ---
@st.cache_resource
def get_encryptor():
    # Persistent key stored in encryption_key.txt or generated once
    key_file = "encryption_key.txt"
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            key_hex = f.read().strip()
            key = bytes.fromhex(key_hex)
    else:
        key = ChaCha20Poly1305.generate_key()
        with open(key_file, "w") as f:
            f.write(key.hex())
            
    cipher = ChaCha20Poly1305(key)
    compressor = zstd.ZstdCompressor(level=10)
    decompressor = zstd.ZstdDecompressor()
    return key, cipher, compressor, decompressor

key, cipher, compressor, decompressor = get_encryptor()

# --- NLP Extraction Functions ---
def extract_demographics(text: str):
    """Deterministic regex extraction for Age and Gender."""
    age_match = re.search(r'\b(\d{1,3})\s*(?:-|–|\s)?(?:year(?:s)?[- ]?old|yo|y/o|yr|y\.o\.)\b', text, re.IGNORECASE)
    gender_match = re.search(r'\b(female|male|woman|man|boy|girl|gentleman|lady)\b', text, re.IGNORECASE)

    age = age_match.group(1) if age_match else "unspecified"
    if gender_match:
        g = gender_match.group(1).lower()
        gender = "female" if g in ["female", "woman", "girl", "lady"] else "male"
    else:
        gender = "unspecified"
    return {"age": age, "gender": gender}

def extract_specialty(text: str):
    specialties = {
        "Surgery": ["surgery", "operative", "incision", "fascia", "suture", "clamps", "procedure", "postoperative"],
        "Obstetrics / Gynecology": ["fetal", "induction", "labor", "delivery", "pregnancy", "cesarean", "uterus"],
        "Cardiology": ["cardiac", "chest pain", "angina", "coronary", "murmur", "echocardiogram", "ekg"],
        "Pulmonology": ["cough", "shortness of breath", "dyspnea", "lungs", "infiltrate", "wheezing", "asthma"],
        "Orthopedic": ["fracture", "joint", "knee", "bone", "spine", "femur", "tibia"],
        "General Medicine": ["fever", "hypertension", "nausea", "vomiting", "diabetes"]
    }
    text_lower = text.lower()
    for spec, keywords in specialties.items():
        if any(k in text_lower for k in keywords):
            return spec
    return "General Medicine"

def extract_clinical_entities(text: str):
    clinical_keywords = [
        "surgery", "nonreassuring fetal testing", "anterior rectus fascia", "pennington clamps", 
        "fetal testing", "low segment", "induction", "suture", "incision", "hemostasis",
        "chest pain", "shortness of breath", "cough", "fever", "hypertension", "diabetes",
        "coronary artery disease", "asthma", "normal cardiac silhouette", "clear lungs",
        "ct scan", "x-ray", "mri", "echocardiogram", "ultrasound", "biopsy",
        "aspirin", "lisinopril", "metformin", "allegra", "zyrtec", "chantix"
    ]
    found = []
    text_lower = text.lower()
    for phrase in clinical_keywords:
        if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
            found.append(phrase)
    return list(dict.fromkeys(found))

# --- Encryption & Decryption Helpers ---
def encrypt_payload(data_dict: dict) -> str:
    raw_bytes = json.dumps(data_dict).encode("utf-8")
    compressed = compressor.compress(raw_bytes)
    nonce = os.urandom(12)
    ciphertext_with_tag = cipher.encrypt(nonce, compressed, associated_data=None)
    final_payload = nonce + ciphertext_with_tag
    return base64.b64encode(final_payload).decode("utf-8")

def decrypt_payload(base64_str: str) -> dict:
    raw_payload = base64.b64decode(base64_str.encode("utf-8"))
    nonce = raw_payload[:12]
    ciphertext_with_tag = raw_payload[12:]
    decrypted_compressed = cipher.decrypt(nonce, ciphertext_with_tag, associated_data=None)
    decompressed = decompressor.decompress(decrypted_compressed)
    return json.loads(decompressed.decode("utf-8"))

# --- UI Layout ---
st.title("🩺 StegIT: Clinical NLP Extraction & AEAD Encryption")
st.markdown("Paste or upload raw patient transcription notes from the dataset to extract clinical entities and generate encrypted steganographic payloads.")

# Sidebar Metadata Inputs
st.sidebar.header("📋 Record Metadata")
patient_id = st.sidebar.number_input("Patient ID", min_value=1, max_value=99999, value=1, step=1)
cover_image = st.sidebar.text_input("Mapped Cover Image", value=f"Indiana_Frontal\\{patient_id}_IM-0001-4001.dcm.png")

st.sidebar.divider()
st.sidebar.caption(f"🔑 **Active Encryption Key (Hex):**")
st.sidebar.code(key.hex()[:32] + "...")

# Input Selection
st.subheader("1️⃣ Paste or Upload Raw Dataset Record")

sample_first_patient = """SUBJECTIVE: This is a 65-year-old male with a history of hypertension and coronary artery disease presenting with a mild cough and shortness of breath. No chest pain reported.
OBJECTIVE: Vital signs stable. Lungs clear to auscultation bilaterally. Normal cardiac silhouette.
PROCEDURE / TESTING: Ordered routine chest X-ray and CT scan. 
PLAN: Prescribed regular aspirin and lisinopril. Follow up in two weeks."""

tab1, tab2 = st.tabs(["📝 Paste Raw Text", "📂 Upload File (.txt)"])

with tab1:
    raw_text_input = st.text_area(
        "Paste patient transcription text here:", 
        value=sample_first_patient, 
        height=180
    )

with tab2:
    uploaded_file = st.file_uploader("Upload patient note (.txt)", type=["txt"])
    if uploaded_file is not None:
        raw_text_input = uploaded_file.read().decode("utf-8")

# Execution Button
if st.button("⚡ Process: Extract NLP & Encrypt Record", type="primary"):
    if not raw_text_input.strip():
        st.error("Please provide clinical record text to proceed.")
    else:
        # Step 1: Extract
        demographics = extract_demographics(raw_text_input)
        specialty = extract_specialty(raw_text_input)
        entities = extract_clinical_entities(raw_text_input)

        # Step 2: Build Structured JSON Payload
        patient_payload = {
            "patient_id": patient_id,
            "medical_specialty": specialty,
            "patient_demographics": demographics,
            "clinical_entities": entities
        }

        # Step 3: Encrypt using Zstd + ChaCha20-Poly1305
        encrypted_ciphertext = encrypt_payload(patient_payload)

        # Step 4: Final Encrypted Record Schema
        final_encrypted_record = {
            "record_id": patient_id,
            "patient_id": patient_id,
            "image": os.path.basename(cover_image),
            "image_path": cover_image,
            "encrypted_data": encrypted_ciphertext
        }

        st.success("✅ Extraction, Compression, and AEAD Encryption Complete!")
        st.divider()

        # Display Section: Side by Side
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📄 Structured Clinical JSON")
            st.caption("Extracted demographics and clinical entities from raw text:")
            st.json(patient_payload)

        with col_right:
            st.subheader("🔒 Encrypted Dataset Record")
            st.caption("Packaged record matching `encrypted_dataset.json` schema:")
            st.json(final_encrypted_record)

        # Detailed Verification & Decryption Check
        with st.expander("🔍 Test Real-Time Decryption (Zero Loss Verification)"):
            decrypted_back = decrypt_payload(encrypted_ciphertext)
            st.write("**Decrypted Output from Ciphertext:**")
            st.json(decrypted_back)
            if decrypted_back == patient_payload:
                st.success("✔ Decryption perfectly matches the extracted JSON payload!")