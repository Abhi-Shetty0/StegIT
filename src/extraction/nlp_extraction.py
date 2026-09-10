import pandas as pd
import json
import re
import os


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = "data/raw/final_multimodal_dataset.csv"
OUTPUT_FILE = "outputs/extracted/extracted_patient_data.json"


# ---------------------------------------------------------
# Extract sections from medical transcription
# ---------------------------------------------------------

def extract_sections(transcription):
    """
    Extract standard sections from the medical transcription.

    The dataset commonly contains sections such as:
    SUBJECTIVE, MEDICATIONS, ALLERGIES, OBJECTIVE,
    ASSESSMENT, PLAN, etc.
    """

    if pd.isna(transcription):
        return {}

    text = str(transcription).strip()

    # Known section headings in the dataset
    section_names = [
        "SUBJECTIVE",
        "MEDICATIONS",
        "ALLERGIES",
        "OBJECTIVE",
        "PHYSICAL EXAMINATION",
        "ASSESSMENT",
        "PLAN",
        "PAST MEDICAL HISTORY",
        "PAST SURGICAL HISTORY",
        "SOCIAL HISTORY",
        "FAMILY HISTORY",
        "CURRENT MEDICATIONS",
        "REVIEW OF SYSTEMS",
        "HISTORY OF PRESENT ILLNESS",
        "IMPRESSION",
        "RECOMMENDATIONS"
    ]

    # Create regex for all headings
    headings = "|".join(
        re.escape(section) for section in section_names
    )

    pattern = rf"(?:^|,)\s*({headings})\s*:\s*"

    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    sections = {}

    for i, match in enumerate(matches):

        section_name = match.group(1).strip().lower()

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()

        # Remove unnecessary commas/spaces
        content = content.strip(" ,")

        sections[section_name] = content

    return sections


# ---------------------------------------------------------
# Extract basic patient information
# ---------------------------------------------------------

def extract_patient_details(text):
    """
    Extract basic patient details when explicitly present
    in the report.

    Currently extracts:
    - Age
    - Gender

    If information is not found, None is returned.
    """

    if not text:
        return {
            "age": None,
            "gender": None
        }

    text = str(text)

    # Example:
    # "23-year-old white female"
    age_match = re.search(
        r"\b(\d{1,3})[- ]year[- ]old\b",
        text,
        re.IGNORECASE
    )

    age = age_match.group(1) if age_match else None

    gender = None

    if re.search(r"\bfemale\b", text, re.IGNORECASE):
        gender = "female"

    elif re.search(r"\bmale\b", text, re.IGNORECASE):
        gender = "male"

    return {
        "age": age,
        "gender": gender
    }


# ---------------------------------------------------------
# Process one medical record
# ---------------------------------------------------------

def process_record(row):
    """
    Convert one CSV record into structured patient data.
    """

    transcription = row.get("transcription", "")

    sections = extract_sections(transcription)

    # Use description + transcription for basic details
    combined_text = (
        str(row.get("description", "")) + " " +
        str(transcription)
    )

    patient_details = extract_patient_details(combined_text)

    record = {
        "record_id": row.get("record_id"),
        "indiana_uid": row.get("indiana_uid"),

        "image": row.get("image"),
        "image_path": row.get("image_path"),

        "medical_specialty": row.get("medical_specialty"),
        "sample_name": row.get("sample_name"),

        "patient_data": patient_details,

        "medical_sections": sections
    }

    return record


# ---------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("PATIENT DATA EXTRACTION")
    print("=" * 60)

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset loaded successfully.")
    print(f"Number of records: {len(df)}")

    # Verify required column
    if "transcription" not in df.columns:
        raise ValueError(
            "The CSV does not contain the 'transcription' column."
        )

    print("\nExtracting medical information...")

    extracted_records = []

    for index, row in df.iterrows():

        record = process_record(row)

        extracted_records.append(record)

        # Progress display
        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/{len(df)} records..."
            )

    # Create output directory
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    # Save JSON
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            extracted_records,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETED")
    print("=" * 60)

    print(f"\nRecords processed: {len(extracted_records)}")
    print(f"Output file: {OUTPUT_FILE}")


# ---------------------------------------------------------
# Program entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()