from __future__ import annotations

import csv
import json
import pathlib
import sys
import typing as t

from ..models.config import DatasetConfig
from ..models.dataset import Dataset, Sample

FieldMapping = dict[str, str]


def _increase_csv_field_size_limit() -> None:
    max_size = sys.maxsize

    while True:
        try:
            csv.field_size_limit(max_size)
            break
        except OverflowError:
            max_size = int(max_size / 10)

def _normalize_path(path: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(path).expanduser().resolve()


def _read_csv_records(path: pathlib.Path) -> list[dict[str, t.Any]]:
    _increase_csv_field_size_limit()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _read_json_records(path: pathlib.Path) -> list[dict[str, t.Any]]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    if isinstance(document, list):
        return document

    if isinstance(document, dict):
        for candidate in ("data", "records", "samples"):
            if isinstance(document.get(candidate), list):
                return document[candidate]

        raise ValueError(
            f"JSON file {path} must contain a top-level list or one of 'data', 'records', 'samples' keys."
        )

    raise ValueError(f"Unsupported JSON root object in {path}: {type(document).__name__}")


def _read_jsonl_records(path: pathlib.Path) -> list[dict[str, t.Any]]:
    records: list[dict[str, t.Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL document at {path}:{lineno}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL entry at {path}:{lineno} must be an object, got {type(record).__name__}."
                )

            records.append(record)

    return records


def _normalize_label(raw_label: t.Any, vuln_label_value: t.Any) -> int:
    if raw_label == vuln_label_value:
        return 1

    if raw_label is None or vuln_label_value is None:
        return 0

    raw = str(raw_label).strip().lower()
    target = str(vuln_label_value).strip().lower()
    return 1 if raw == target else 0


def _normalize_cwe(raw_cwe: t.Any) -> list[str]:
    if raw_cwe is None:
        return []
    if isinstance(raw_cwe, list):
        return [str(item).strip() for item in raw_cwe if str(item).strip()]

    raw = str(raw_cwe).strip()
    if not raw:
        return []

    # Remove outer quotes if the field is wrapped like '"[cwe-32]"' or "'cwe-32'".
    while len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1].strip()
        if not raw:
            return []

    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass

        separators = [",", ";", "|"]
        for sep in separators:
            if sep in inner:
                return [item.strip().strip('"\'') for item in inner.split(sep) if item.strip()]

        return [inner.strip().strip('"\'')]

    separators = [",", ";", "|"]
    for sep in separators:
        if sep in raw:
            return [item.strip() for item in raw.split(sep) if item.strip()]

    return [raw]


def _record_to_sample(
    record: dict[str, t.Any],
    field_map: FieldMapping,
    vuln_label_value: t.Any,
) -> Sample:
    function_key = field_map.get("function")
    label_key = field_map.get("label")
    cve_key = field_map.get("cve")
    cwe_key = field_map.get("cwe")
    project_key = field_map.get("project")

    if function_key is None or label_key is None or cve_key is None or cwe_key is None or project_key is None:
        raise ValueError(
            "Config field mapping must include 'function', 'label', 'cve', 'cwe', and 'project' keys."
        )

    function = str(record.get(function_key, "") or "")
    raw_label = record.get(label_key)
    cve = str(record.get(cve_key, "") or "")
    cwe = _normalize_cwe(record.get(cwe_key))
    project = str(record.get(project_key, "") or "")

    return Sample(
        function=function,
        label=_normalize_label(raw_label, vuln_label_value),
        cve=cve,
        cwe=cwe,
        project=project,
    )


def _load_records_from_path(path: pathlib.Path) -> list[dict[str, t.Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file path, got {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_records(path)
    if suffix == ".json":
        return _read_json_records(path)
    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl_records(path)

    raise ValueError(
        f"Unsupported dataset file format for {path}. Supported extensions: .csv, .json, .jsonl, .ndjson"
    )


def _load_split(
    path: str | pathlib.Path | None,
    field_mapping: dict[str, t.Any],
) -> list[Sample]:
    if path is None:
        return []

    path_obj = _normalize_path(path)
    vuln_label_value = field_mapping.get("vuln_label_value")
    field_map: FieldMapping = {
        "function": str(field_mapping.get("function", "")),
        "label": str(field_mapping.get("label", "")),
        "cve": str(field_mapping.get("cve", "")),
        "cwe": str(field_mapping.get("cwe", "")),
        "project": str(field_mapping.get("project", "")),
    }

    raw_records = _load_records_from_path(path_obj)
    return [_record_to_sample(record, field_map, vuln_label_value) for record in raw_records]


def load_dataset_from_config(config: DatasetConfig) -> Dataset:
    if not isinstance(config, DatasetConfig):
        raise ValueError("Expected DatasetConfig object for load_dataset_from_config.")

    fields = {
        "function": config.fields.function,
        "label": config.fields.label,
        "vuln_label_value": config.fields.vuln_label_value,
        "cve": config.fields.cve,
        "cwe": config.fields.cwe,
        "project": config.fields.project,
    }

    train_path = config.files.train
    test_path = config.files.test
    valid_path = config.files.valid
    data_path = config.files.data

    if data_path is not None:
        # If data file is present, load it into data, ignore splits
        return Dataset(
            name=config.name,
            description=config.description,
            version=config.version,
            license=config.license,
            data=_load_split(data_path, fields),
            train=[],
            test=[],
            validation=[],
        )
    else:
        # Otherwise, load the split files
        return Dataset(
            name=config.name,
            description=config.description,
            version=config.version,
            license=config.license,
            train=_load_split(train_path, fields) if train_path else [],
            test=_load_split(test_path, fields) if test_path else [],
            validation=_load_split(valid_path, fields) if valid_path else [],
            data=[],
        )

__all__ = ["load_dataset_from_config"]
