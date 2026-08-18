import re


class ClinicalRules:
    """
    Rule-based extraction for structured clinical information.
    """

    def __init__(self):
        pass

    ############################################################
    # DEMOGRAPHICS
    ############################################################

    def age(self, text):

        patterns = [

            r'(\d+)-year-old',

            r'Age:\s*(\d+)'
        ]

        for p in patterns:

            m = re.search(p, text, re.IGNORECASE)

            if m:

                return int(m.group(1))

        return None

    def gender(self, text):

        genders = [

            "female",

            "male"

        ]

        lower = text.lower()

        for g in genders:

            if g in lower:

                return g.capitalize()

        return None

    def race(self, text):

        races = [

            "white",

            "black",

            "asian",

            "hispanic",

            "native american",

            "african american"

        ]

        lower = text.lower()

        for r in races:

            if r in lower:

                return r.title()

        return None

    ############################################################
    # VITALS
    ############################################################

    def blood_pressure(self, text):

        m = re.search(

            r'Blood Pressure:\s*([\d]+/[\d]+)',

            text,

            re.IGNORECASE

        )

        return m.group(1) if m else None

    def temperature(self, text):

        m = re.search(

            r'Temperature:\s*([\d\.]+)',

            text,

            re.IGNORECASE

        )

        return m.group(1) if m else None

    def pulse(self, text):

        m = re.search(

            r'Heart Rate:\s*(\d+)',

            text,

            re.IGNORECASE

        )

        return m.group(1) if m else None

    def respiratory_rate(self, text):

        m = re.search(

            r'Respiratory Rate:\s*(\d+)',

            text,

            re.IGNORECASE

        )

        return m.group(1) if m else None

    def spo2(self, text):

        m = re.search(

            r'Oxygen Saturation:\s*([\d]+%)',

            text,

            re.IGNORECASE

        )

        return m.group(1) if m else None

    def weight(self, text):

        m = re.search(

            r'Weight:\s*([\d\.]+\s*(?:kg|lbs?|pounds))',

            text,

            re.IGNORECASE

        )

        return m.group(1) if m else None

    ############################################################
    # PATIENT
    ############################################################

    def patient_information(self, sections):

        subject = sections.get("SUBJECTIVE", "")

        objective = sections.get("OBJECTIVE", "")

        return {

            "age": self.age(subject),

            "gender": self.gender(subject),

            "race": self.race(subject),

            "blood_pressure": self.blood_pressure(objective),

            "temperature": self.temperature(objective),

            "heart_rate": self.pulse(objective),

            "respiratory_rate": self.respiratory_rate(objective),

            "spo2": self.spo2(objective),

            "weight": self.weight(objective)

        }