import streamlit as st
import json
import os
import random

st.set_page_config(page_title="StegIT - Clinical NLP Explorer", layout="wide")

PAYLOADS_DIR = "extracted_payloads"

st.title("🩺 StegIT: Clinical NLP Extraction Dashboard")
st.markdown("Inspect extracted clinical payloads generated from `mtsamples.csv` for steganographic embedding.")

# Sidebar Controls
st.sidebar.header("🔍 Patient Selector")

if "patient_id" not in st.session_state:
    st.session_state.patient_id = 50

# Random Button
if st.sidebar.button("🎲 Pick Random Patient"):
    st.session_state.patient_id = random.randint(1, 4966)

# Number input / slider
patient_id = st.sidebar.number_input(
    "Select Patient ID (1 - 4966)", 
    min_value=1, 
    max_value=4966, 
    value=st.session_state.patient_id,
    step=1
)
st.session_state.patient_id = patient_id

filepath = os.path.join(PAYLOADS_DIR, f"patient_{patient_id}_payload.json")

if not os.path.exists(filepath):
    st.error(f"Payload file `{filepath}` not found. Please verify the `extracted_payloads/` folder.")
else:
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Top Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    demo = payload.get("patient_demographics", {})
    
    col1.metric("Patient ID", f"#{patient_id}")
    col2.metric("Age", demo.get("age", "Not specified"))
    col3.metric("Gender", demo.get("gender", "Not specified"))
    col4.metric("Specialty", demo.get("medical_specialty", "General")[:20])

    st.divider()

    # Split View: Left = Categorized Entities, Right = Raw JSON
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("🏷️ Extracted Clinical Entities")
        st.caption(f"Cover Image Mapped: `{payload.get('mapped_cover_image', 'N/A')}`")
        
        breakdown = payload.get("category_breakdown", {})
        
        for category, terms in breakdown.items():
            with st.expander(f"**{category}** ({len(terms)} items)", expanded=True):
                if terms:
                    st.write(", ".join([f"`{t}`" for t in terms]))
                else:
                    st.write("_None detected_")

    with right_col:
        st.subheader("📄 Standardized JSON Payload")
        st.caption("Payload ready for AES / ChaCha20 encryption & non-ROI embedding")
        st.json(payload)