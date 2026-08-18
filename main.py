import os
import json
from pathlib import Path
from src.preprocessing import TextPreprocessor
from src.section_parser import SectionParser
from src.clinical_rules import ClinicalRules
from src.ner import MedicalNER
from src.utils import read_text
from config import OUTPUT_DIR, OUTPUT_JSON

# --------------------------------------------------
# 1. Read Raw Medical Note
# --------------------------------------------------
note_path = "data/sample_note.txt"
note = read_text(note_path)

# --------------------------------------------------
# 2. Text Preprocessing & Section Parsing
# --------------------------------------------------
processor = TextPreprocessor()
clean = processor.clean_text(note)

parser = SectionParser()
sections = parser.parse(clean)

# --------------------------------------------------
# 3. Clinical Rules & Demographics Extraction
# --------------------------------------------------
rules = ClinicalRules()
patient = rules.patient_information(sections)

# --------------------------------------------------
# 4. Medical Named Entity Recognition (NER)
# --------------------------------------------------
ner = MedicalNER()
entities = ner.extract_entities(clean)
ner.print_entities(entities)

grouped = ner.extract(clean)

print("\n" + "=" * 60)
print("GROUPED ENTITIES EXTRACTED")
print("=" * 60)
print(json.dumps(grouped, indent=4))

# --------------------------------------------------
# 5. Populate Patient Demographics from NER (Fallback)
# --------------------------------------------------
if not patient.get("age") and "AGE" in grouped and grouped["AGE"]:
    patient["age"] = grouped["AGE"][0]

if not patient.get("gender") and "SEX" in grouped and grouped["SEX"]:
    patient["gender"] = grouped["SEX"][0]

# --------------------------------------------------
# 6. Build Final Structured Payload & Map Cover Image
# --------------------------------------------------
cover_image_path = "images/cover_image.png"

# Collect unique clinical symptoms/diagnoses for stego embedding
clinical_findings = []
for category in ["SIGN_SYMPTOM", "HISTORY", "DIAGNOSTIC_PROCEDURE", "BIOLOGICAL_STRUCTURE"]:
    if category in grouped:
        clinical_findings.extend(grouped[category])

# Remove duplicates while keeping order
clinical_findings = list(dict.fromkeys(clinical_findings))

payload = {
    "patient_demographics": {
        "age": patient.get("age") or "Unspecified",
        "gender": patient.get("gender") or "Unspecified"
    },
    "clinical_entities": clinical_findings[:10],
    "category_breakdown": grouped,
    "mapped_cover_image": cover_image_path
}

# --------------------------------------------------
# 7. Save Structured Payload to output/patient_information.json
# --------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_JSON, "w") as f:
    json.dump(payload, f, indent=4)

print("\n" + "=" * 60)
print("FINAL STRUCTURED PAYLOAD (Saved to JSON for Phase 2/3)")
print("=" * 60)
print(json.dumps(payload, indent=4))
print(f"\n[SUCCESS] Output saved to: {OUTPUT_JSON}")