"""
Medical Named Entity Recognition

Uses a Hugging Face token-classification model.
"""

from transformers import pipeline
from config import NER_MODEL, DEVICE


class MedicalNER:
    def __init__(self):
        print("=" * 60)
        print("Loading Medical NER Model...")
        print("=" * 60)

        self.pipeline = pipeline(
            task="token-classification",
            model=NER_MODEL,
            tokenizer=NER_MODEL,
            aggregation_strategy="simple",
            device=DEVICE
        )

        print("Model Loaded Successfully!\n")

    ############################################################

    def extract_entities(self, text):

        predictions = self.pipeline(text)

        entities = []

        for item in predictions:

            entity = {
                "text": item.get("word", ""),
                "label": item.get("entity_group", item.get("entity", "UNKNOWN")),
                "score": round(float(item.get("score", 0)), 4),
                "start": item.get("start"),
                "end": item.get("end")
            }

            entities.append(entity)

        return entities

    ############################################################

    def group_entities(self, entities):

        grouped = {}

        for entity in entities:

            label = entity["label"]

            grouped.setdefault(label, [])

            grouped[label].append(entity["text"])

        # Remove duplicates
        for key in grouped:

            grouped[key] = sorted(list(set(grouped[key])))

        return grouped

    ############################################################

    def print_entities(self, entities):

        print("=" * 80)
        print("MEDICAL NER OUTPUT")
        print("=" * 80)

        if not entities:
            print("No entities detected.")
            return

        for entity in entities:

            print(
                f"{entity['text']:<35}"
                f"{entity['label']:<20}"
                f"{entity['score']:.3f}"
            )

    ############################################################

    def extract(self, text):

        entities = self.extract_entities(text)

        grouped = self.group_entities(entities)

        return grouped