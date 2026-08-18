import re


class SectionParser:

    def __init__(self):

        self.headers = [
            "SUBJECTIVE",
            "OBJECTIVE",
            "ASSESSMENT",
            "PLAN",
            "MEDICATIONS",
            "ALLERGIES",
            "HISTORY",
            "FAMILY HISTORY",
            "SOCIAL HISTORY",
            "PHYSICAL EXAM",
            "REVIEW OF SYSTEMS",
            "IMPRESSION",
            "DIAGNOSIS"
        ]

    def parse(self, text):

        sections = {}

        pattern = "|".join(
            [re.escape(x) + r"\s*:" for x in self.headers]
        )

        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

        if not matches:
            return {"FULL_TEXT": text}

        for i, match in enumerate(matches):

            start = match.end()

            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )

            title = match.group().replace(":", "").strip().upper()

            body = text[start:end].strip()

            sections[title] = body

        return sections