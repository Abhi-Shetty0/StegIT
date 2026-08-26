import os
import json
import re
import pandas as pd

def categorize_entity(entity_text):
    text_lower = entity_text.lower()
    if any(k in text_lower for k in ["female", "male", "woman", "man", "gentleman"]):
        return "SEX"
    elif re.search(r'\b\d{1,2}[- ]?(year|yr)[- ]?old\b', text_lower):
        return "AGE"
    elif any(k in text_lower for k in ["x-ray", "echocardiogram", "doppler", "m-mode", "mri", "ct", "endoscopy", "scan", "bypass", "study", "ultrasound", "biopsy"]):
        return "DIAGNOSTIC_PROCEDURE"
    elif any(k in text_lower for k in ["chest", "heart", "atrium", "ventricle", "valve", "abdomen", "flank", "breast", "lung", "throat", "nasal", "knee", "spine", "mucosa", "joint", "kidney", "stomach", "brain", "colon"]):
        return "BIOLOGICAL_STRUCTURE"
    elif any(k in text_lower for k in ["mild", "severe", "moderate", "trace", "significant", "low-grade", "acute", "chronic", "mildly"]):
        return "SEVERITY"
    elif any(k in text_lower for k in ["persistent", "difficulty", "chronic", "acute", "long-standing", "overweight", "recurrent"]):
        return "DETAILED_DESCRIPTION"
    elif any(k in text_lower for k in ["asthma", "rhinitis", "diabetes", "obesity", "hypertension", "gout", "reflux", "copd", "cad", "osteoarthritis", "allergies", "gerd", "cancer", "infection"]):
        return "HISTORY"
    elif any(k in text_lower for k in ["allegra", "zyrtec", "claritin", "nasonex", "ortho", "penicillin", "diovan", "crestor", "tricor", "chantix", "aspirin", "heparin"]):
        return "MEDICATION"
    else:
        return "SIGN_SYMPTOM"

def clean_term(text):
    text = re.sub(r'[^\w\s-]', '', text)
    return text.strip()

def create_separate_json_for_every_patient(csv_path="mtsamples.csv", output_dir="extracted_payloads"):
    print("[1/2] Loading dataset...")
    df = pd.read_csv(csv_path).dropna(subset=['transcription']).reset_index(drop=True)
    total_records = len(df)
    print(f"[✓] Found {total_records} patients.")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[2/2] Generating {total_records} separate individual JSON files in '{output_dir}/'...")

    for idx in range(total_records):
        row = df.iloc[idx]
        text = str(row['transcription'])
        keywords_raw = str(row.get('keywords', ''))
        specialty = str(row.get('medical_specialty', 'General Medicine')).strip()
        
        # 1. Demographics
        age_match = re.search(r'\b(\d{1,2}\s*[- ]?(?:year|yr)[- ]?old)\b', text, re.IGNORECASE)
        if not age_match:
            age_match = re.search(r'\b(\d{1,2})\s*years?\s*old\b', text, re.IGNORECASE)
            age = f"{age_match.group(1)}-year-old" if age_match else "unspecified"
        else:
            age = age_match.group(1).lower()

        gender_match = re.search(r'\b(female|male|woman|man|gentleman)\b', text, re.IGNORECASE)
        gender = "female" if gender_match and ("fem" in gender_match.group(1).lower() or gender_match.group(1).lower() == "woman") else ("male" if gender_match else "unspecified")
        
        # 2. Entities & Categories
        clinical_entities = []
        category_breakdown = {
            "AGE": [age] if age != "unspecified" else [],
            "SEX": [gender] if gender != "unspecified" else [],
            "DETAILED_DESCRIPTION": [],
            "SIGN_SYMPTOM": [],
            "SEVERITY": [],
            "BIOLOGICAL_STRUCTURE": [],
            "HISTORY": [],
            "DIAGNOSTIC_PROCEDURE": [],
            "MEDICATION": []
        }
        
        kw_list = [k.strip() for k in keywords_raw.split(',') if k.strip()]
        for kw in kw_list:
            cleaned = clean_term(kw)
            if len(cleaned) > 2 and cleaned.lower() not in ["sample", "pounds", "months", "bariatrics"]:
                clinical_entities.append(cleaned)
                cat = categorize_entity(cleaned)
                if cleaned not in category_breakdown[cat]:
                    category_breakdown[cat].append(cleaned)

        # 3. Individual Payload
        payload = {
            "patient_id": idx + 1,
            "medical_specialty": specialty,
            "patient_demographics": {
                "age": age,
                "gender": gender
            },
            "clinical_entities": list(dict.fromkeys(clinical_entities))[:8],
            "category_breakdown": {k: v for k, v in category_breakdown.items() if len(v) > 0},
            "mapped_cover_image": f"images/patient_{idx+1}_scan.png"
        }
        
        # Save separate file for each patient
        out_file = os.path.join(output_dir, f"patient_{idx+1}_payload.json")
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=4)
        
        # Print progress update every 500 records
        if (idx + 1) % 500 == 0 or (idx + 1) == total_records:
            print(f"  [✓] Processed and saved {idx + 1}/{total_records} JSON files...")

    print(f"\n[SUCCESS] All {total_records} separate patient JSON files have been created in '{output_dir}/'!")

if __name__ == "__main__":
    create_separate_json_for_every_patient()