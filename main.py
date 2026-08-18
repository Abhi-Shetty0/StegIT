from src.preprocessing import TextPreprocessor
from src.section_parser import SectionParser
from src.clinical_rules import ClinicalRules
from src.ner import MedicalNER
from src.utils import read_text

import json

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