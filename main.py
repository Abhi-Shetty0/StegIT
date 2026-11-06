from src.preprocessing import TextPreprocessor
from src.section_parser import SectionParser
from src.clinical_rules import ClinicalRules
from src.ner import MedicalNER
from src.utils import read_text
from src.encryption import encrypt_json, decrypt_json

import json
import os
from getpass import getpass

# --------------------------------------------------

note = read_text("data/sample_note.txt")

# --------------------------------------------------

processor = TextPreprocessor()
clean = processor.clean_text(note)

# --------------------------------------------------

parser = SectionParser()
sections = parser.parse(clean)

# --------------------------------------------------

rules = ClinicalRules()

patient = rules.patient_information(sections)

print("\nPATIENT INFORMATION")
print(json.dumps(patient, indent=4))

# --------------------------------------------------

ner = MedicalNER()

entities = ner.extract_entities(clean)

ner.print_entities(entities)

grouped = ner.extract(clean)

print("\nGROUPED ENTITIES")
print(json.dumps(grouped, indent=4))

# --------------------------------------------------
# ENCRYPTION

result = {
    "patient": patient,
    "entities": grouped,
}

password = os.environ.get("STEGIT_PASSWORD") or getpass("\nEncryption password: ")

blob = encrypt_json(result, password)

os.makedirs("output", exist_ok=True)
with open("output/encrypted_report.bin", "wb") as f:
    f.write(blob)

print(f"\nENCRYPTED PAYLOAD SAVED ({len(blob)} bytes)")

# sanity check: decrypting must give back the original
assert decrypt_json(blob, password) == json.loads(json.dumps(result))
print("Decryption check passed")