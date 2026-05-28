import json
import pathlib

from vultrition.models.results import AnalysisResults

import json
from dataclasses import fields, is_dataclass
from typing import get_args, get_origin


def load_analysis_results_from_json(path: pathlib.Path) -> AnalysisResults:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return _from_dict(AnalysisResults, data)


def _from_dict(cls, data):
    """
    Recursively construct dataclass instances from dictionaries.
    Also handles tuples such as tuple[str, str].
    """
    if data is None:
        return None

    origin = get_origin(cls)

    # Handle tuple[str, str], tuple[int, ...], etc.
    if origin is tuple:
        return tuple(data)

    # Handle list[...] if needed later
    if origin is list:
        item_type = get_args(cls)[0]
        return [_from_dict(item_type, item) for item in data]

    # Handle dataclasses
    if is_dataclass(cls):
        kwargs = {}

        for field in fields(cls):
            field_name = field.name
            field_type = field.type

            if field_name not in data:
                raise KeyError(f"Missing field '{field_name}' for {cls.__name__}")

            kwargs[field_name] = _from_dict(field_type, data[field_name])

        return cls(**kwargs)

    # Primitive types: str, float, bool, etc.
    return data