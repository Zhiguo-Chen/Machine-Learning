import json


def get_config(path):
    with open(path) as f:
        raw_config = json.load(f)
    return raw_config
