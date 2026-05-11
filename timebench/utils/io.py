"""Shared I/O and identifier utilities used across multiple pipeline scripts."""

import json
from typing import Any


def get_base_scenario_id(result: dict) -> str:
    """Return a unique identifier for a scenario, independent of the run seed."""
    return json.dumps(result.get("test_messages", []), sort_keys=True)


def get_scenario_identifier(result: dict) -> tuple:
    """Return a unique identifier for a single test instance (scenario + seed)."""
    return (get_base_scenario_id(result), result.get("seed"))


def update_json_file(file_path: str, data: Any, key: str) -> None:
    """Read *file_path*, set ``existing[key] = data``, and write it back.

    Creates the file from scratch if it does not exist or contains invalid JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}

    existing[key] = data

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)

    print(f"Updated '{file_path}' with key '{key}'.")
