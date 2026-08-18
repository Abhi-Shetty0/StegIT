import re


class TextPreprocessor:
    """
    Cleans raw medical transcription.
    """

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Clean OCR/transcription artifacts.
        """

        # Remove repeated commas
        text = re.sub(r",+", ",", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        # Normalize line endings
        text = text.replace("\r", "\n")

        return text.strip()

    def lowercase_copy(self, text: str):
        return text.lower()