import os
import pdfplumber
import medspacy
from medspacy.visualization import visualize_ent

def extract_text_from_pdf(pdf_path):
    """
    Step 1: Extract clean, structured text from a digital PDF.
    Handles tables and maintains layout flow.
    """
    print(f"[Step 1] Extracting text from: {pdf_path}...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} could not be found.")
        
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True)  # layout=True preserves multi-column alignments
            if text:
                full_text.append(text)
    
    # Combine all pages with spacing
    return "\n\n--- Page Break ---\n\n".join(full_text)


def process_clinical_text(clinical_text):
    """
    Step 2 & 3: Initialize MedSpacy NLP model and process text.
    Extracts medical concepts and determines context (e.g., Negation).
    """
    print("[Step 2] Initializing MedSpacy Clinical Pipeline...")
    # Loads clinical components: tokenizer, sentence splitter, target matcher, and context flagger
    nlp = medspacy.load()
    
    print("[Step 3] Running NLP model over extracted text...")
    doc = nlp(clinical_text)
    return doc


def display_results(doc):
    """
    Step 4: Parse and structure the extracted entities.
    Prints medical insights and their context rules.
    """
    print("\n" + "="*50)
    print(" EXTRACTED MEDICAL DATA & CONTEXT")
    print("="*50)
    
    # Format results into a readable printout
    print(f"{'Medical Entity/Concept':<35} | {'Category/Label':<15} | {'Status/Context'}")
    print("-" * 75)
    
    for ent in doc.ents:
        # Check context attributes (MedSpacy adds rules like is_negated, is_historical)
        status = "Present"
        if ent._.is_negated:
            status = "Negated / Absent"
        elif ent._.is_historical:
            status = "Historical / Past History"
        elif ent._.is_hypothetical:
            status = "Hypothetical / Possible"
            
        print(f"{ent.text:<35} | {ent.label_:<15} | {status}")
    print("="*50)


# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # 1. Define your path
    PDF_FILE_PATH = "UMNwriteup (2).pdf" 
    
    try:
        # Step 1: PDF to Text
        raw_text = extract_text_from_pdf(PDF_FILE_PATH)
        
        # Step 2 & 3: Run through Clinical Model
        processed_doc = process_clinical_text(raw_text)
        
        # Step 4: Output structured data
        display_results(processed_doc)
        
        # Optional Step 5: Visualise in your browser/notebook (Uncomment to use)
        # visualize_ent(processed_doc)
        
    except Exception as e:
        print(f"\n[Error] Pipeline failed: {e}")
