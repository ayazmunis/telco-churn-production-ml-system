"""
config.py

Loads project configuration from YAML.
"""

import yaml


def load_config():

    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)


config = load_config()

if __name__ == "__main__":

    print(config)