import os
import json
import random

PAYLOADS_DIR = "extracted_payloads"

def show_patient(patient_id):
    filename = f"patient_{patient_id}_payload.json"
    filepath = os.path.join(PAYLOADS_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"❌ Error: {filename} not found in '{PAYLOADS_DIR}'. Make sure patient ID is between 1 and 4966.")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "="*60)
    print(f"   PATIENT RECORD: ID {patient_id} ({filename})")
    print("="*60)
    
    # 1. Demographics
    demo = data.get("patient_demographics", {})
    print(f"📌 Demographics:")
    print(f"   • Age    : {demo.get('age', 'N/A')}")
    print(f"   • Gender : {demo.get('gender', 'N/A')}")
    print(f"   • Specialty: {demo.get('medical_specialty', 'N/A')}")
    print(f"   • Cover Scan Mapped: {data.get('mapped_cover_image', 'N/A')}")

    # 2. Extracted Entities Breakdown
    print(f"\n🏷️  Clinical Category Breakdown:")
    breakdown = data.get("category_breakdown", {})
    for category, terms in breakdown.items():
        if terms:
            print(f"   • {category:<22}: {', '.join(terms)}")

    # 3. Formatted JSON Payload
    print("\n📄 Raw Standardized JSON Payload:")
    print(json.dumps(data, indent=2))
    print("="*60 + "\n")

if __name__ == "__main__":
    user_input = input("Enter Patient ID (1-4966) or 'r' for random patient: ").strip()
    
    if user_input.lower() in ['r', 'random', '']:
        p_id = random.randint(1, 4966)
        print(f"🎲 Selected Random Patient ID: {p_id}")
    else:
        try:
            p_id = int(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            exit(1)
            
    show_patient(p_id)