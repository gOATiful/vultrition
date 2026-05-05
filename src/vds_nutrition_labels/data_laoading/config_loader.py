from __future__ import annotations

import pathlib
import typing as t

from ..models.config import (
    DatasetAnalysis,
    DatasetConfig,
    DatasetFiles,
    DatasetFields,
    QualityMetrics,
    StructuralMetrics,
)


def _load_toml_module() -> t.Any:
    try:
        import tomllib as toml_module
    except ModuleNotFoundError:
        try:
            import tomli as toml_module  # type: ignore[import]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TOML support requires Python 3.11+ or the tomli package. "
                "Install tomli with `pip install tomli`."
            ) from exc
    return toml_module


def _normalize_path(path: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(path).expanduser().resolve()


def _ensure_dict(value: t.Any, name: str) -> dict[str, t.Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected section '{name}' to be a table in the config.")
    return value


def _cast_str(value: t.Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"Missing required config value '{field_name}'.")
    return str(value)


def _cast_bool(value: t.Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValueError(f"Expected boolean for '{field_name}', got {value!r}.")


def _load_dataset_files(files_section: dict[str, t.Any]) -> DatasetFiles:
    return DatasetFiles(
        train=_cast_str(files_section.get("train"), "dataset.files.train"),
        test=_cast_str(files_section.get("test"), "dataset.files.test"),
        valid=_cast_str(files_section.get("valid"), "dataset.files.valid"),
    )


def _load_dataset_fields(fields_section: dict[str, t.Any]) -> DatasetFields:
    return DatasetFields(
        function=_cast_str(fields_section.get("function"), "dataset.fields.function"),
        label=_cast_str(fields_section.get("label"), "dataset.fields.label"),
        vuln_label_value=fields_section.get("vuln_label_value"),
        cve=_cast_str(fields_section.get("cve"), "dataset.fields.cve"),
        cwe=_cast_str(fields_section.get("cwe"), "dataset.fields.cwe"),
        project=_cast_str(fields_section.get("project"), "dataset.fields.project"),
    )


def _load_quality_metrics(analysis_section: dict[str, t.Any]) -> QualityMetrics:
    quality_section = analysis_section.get("quality_metrics")
    if quality_section is None:
        quality_section = analysis_section.get("quiality_metrics")
    quality_section = _ensure_dict(quality_section, "dataset.analysis.quality_metrics")

    return QualityMetrics(
        completeness=_cast_bool(quality_section.get("completeness"), "dataset.analysis.quality_metrics.completeness"),
        diversity=_cast_bool(quality_section.get("diversity"), "dataset.analysis.quality_metrics.diversity"),
        balance=_cast_bool(quality_section.get("balance"), "dataset.analysis.quality_metrics.balance"),
        timespan=_cast_bool(quality_section.get("timespan"), "dataset.analysis.quality_metrics.timespan"),
        uniqueness=_cast_bool(quality_section.get("uniqueness"), "dataset.analysis.quality_metrics.uniqueness"),
        cross_contamination=_cast_bool(quality_section.get("cross_contamination"), "dataset.analysis.quality_metrics.cross_contamination"),
    )


def _load_structural_metrics(analysis_section: dict[str, t.Any]) -> StructuralMetrics:
    structural_section = _ensure_dict(analysis_section.get("structural_metrics"), "dataset.analysis.structural_metrics")

    return StructuralMetrics(
        loc=_cast_bool(structural_section.get("loc"), "dataset.analysis.structural_metrics.loc"),
        tokens=_cast_bool(structural_section.get("tokens"), "dataset.analysis.structural_metrics.tokens"),
        cyclomatic_complexity=_cast_bool(
            structural_section.get("cyclomatic_complexity"),
            "dataset.analysis.structural_metrics.cyclomatic_complexity",
        ),
    )


def _load_dataset_analysis(analysis_section: dict[str, t.Any]) -> DatasetAnalysis:
    analysis_section = _ensure_dict(analysis_section, "dataset.analysis")
    return DatasetAnalysis(
        quality_metrics=_load_quality_metrics(analysis_section),
        structural_metrics=_load_structural_metrics(analysis_section),
    )


def load_dataset_config(path: str | pathlib.Path) -> DatasetConfig:
    config_path = _normalize_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Expected a file for config path, got: {config_path}")

    toml_module = _load_toml_module()
    with config_path.open("rb") as handle:
        config_data = toml_module.load(handle)

    dataset_section = _ensure_dict(config_data.get("dataset"), "dataset")

    return DatasetConfig(
        name=_cast_str(dataset_section.get("name"), "dataset.name"),
        description=_cast_str(dataset_section.get("description"), "dataset.description"),
        version=_cast_str(dataset_section.get("version"), "dataset.version"),
        license=_cast_str(dataset_section.get("license"), "dataset.license"),
        languages=_cast_str(dataset_section.get("languages"), "dataset.languages"),
        has_runable_code_or_test_cases=_cast_bool(dataset_section.get("has_runable_code_or_test_cases"), "dataset.has_runable_code_or_test_cases"),
        files=_load_dataset_files(_ensure_dict(dataset_section.get("files"), "dataset.files")),
        fields=_load_dataset_fields(_ensure_dict(dataset_section.get("fields"), "dataset.fields")),
        analysis=_load_dataset_analysis(dataset_section.get("analysis")),
    )


__all__ = ["load_dataset_config"]
