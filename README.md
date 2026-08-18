# StegIT

StegIT is a clinical text processing pipeline. It extracts medical entities, parses clinical notes, and interprets medical context from digital documents and text notes.

## Project Structure

- `main.py`: Processes raw text notes (`data/sample_note.txt`) to extract sections, patient information, and medical entities using a HuggingFace NER model.
- `clinical_pipeline.py`: A specialized pipeline to extract text from digital PDFs (e.g., `UMNwriteup (2).pdf`) using `pdfplumber`, and process it using `medspacy` to identify medical concepts and their context (e.g., Negated, Historical, Hypothetical).
- `src/`: Contains the core logic (Text Preprocessing, Section Parsing, Clinical Rules, and Medical NER).

## Setup & Installation

### 1. Set up a Virtual Environment
We recommend using a virtual environment (e.g., `medical_env`) to keep dependencies isolated:

```bash
# Create the virtual environment
python -m venv medical_env

# Activate the virtual environment
# On Windows:
medical_env\Scripts\activate
# On macOS/Linux:
source medical_env/bin/activate
```

### 2. Install Dependencies
Once the virtual environment is activated, install all the required Python packages:

```bash
pip install -r requirements.txt
```

> **Note on SciSpacy**: Although `scispacy` is included in the requirements, depending on future changes, you might optionally need to install its statistical models (like `en_core_sci_sm`). Currently, the pipeline relies on MedSpacy's default models and HuggingFace pipelines.

## How to Run

### Option 1: Running the Text Note Pipeline (`main.py`)
This script processes plain text medical notes.

1. Ensure you have a text file at `data/sample_note.txt`. (Create the `data` directory and add a clinical note text file if it does not exist, since this directory is ignored in Git).
2. Run the pipeline:
   ```bash
   python main.py
   ```
   This will output parsed patient information, grouped medical entities, and the medical NER output.

### Option 2: Running the PDF Clinical Pipeline (`clinical_pipeline.py`)
This script uses MedSpacy and `pdfplumber` to extract structured information from a digital PDF report.

1. Ensure you have the target PDF file (by default, `UMNwriteup (2).pdf` in the root directory). 
2. Run the pipeline:
   ```bash
   python clinical_pipeline.py
   ```
   This will extract text, pass it through MedSpacy, and print the medical entities along with their inferred context (e.g., "Negated / Absent").
