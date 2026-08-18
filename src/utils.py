import json


def save_json(data, path):

    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=4
        )


def read_text(path):

    with open(path) as f:

        return f.read()