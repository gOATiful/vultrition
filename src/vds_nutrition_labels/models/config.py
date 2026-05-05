from dataclasses import dataclass
import typing as t



@dataclass
class DatasetFiles:
    train: str
    test: str
    valid: str

    def _to_lines(self, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        return [
            f"{prefix}files:",
            f"{prefix}  train: {self.train}",
            f"{prefix}  test: {self.test}",
            f"{prefix}  valid: {self.valid}",
        ]


@dataclass
class DatasetFields:
    function: str
    label: str
    vuln_label_value: t.Any
    cve: str
    cwe: str
    project: str

    def _format_value(self, value: t.Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _to_lines(self, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        return [
            f"{prefix}fields:",
            f"{prefix}  function: {self.function}",
            f"{prefix}  label: {self.label}",
            f"{prefix}  vuln_label_value: {self._format_value(self.vuln_label_value)}",
            f"{prefix}  cve: {self.cve}",
            f"{prefix}  cwe: {self.cwe}",
            f"{prefix}  project: {self.project}",
        ]


@dataclass
class QualityMetrics:
    completeness: bool
    diversity: bool
    balance: bool
    timespan: bool
    uniqueness: bool
    cross_contamination: bool

    def _format_value(self, value: t.Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _to_lines(self, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        return [
            f"{prefix}quality_metrics:",
            f"{prefix}  completeness: {self._format_value(self.completeness)}",
            f"{prefix}  diversity: {self._format_value(self.diversity)}",
            f"{prefix}  balance: {self._format_value(self.balance)}",
            f"{prefix}  timespan: {self._format_value(self.timespan)}",
            f"{prefix}  uniqueness: {self._format_value(self.uniqueness)}",
            f"{prefix}  cross_contamination: {self._format_value(self.cross_contamination)}",
        ]


@dataclass
class StructuralMetrics:
    loc: bool
    tokens: bool
    cyclomatic_complexity: bool

    def _format_value(self, value: t.Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _to_lines(self, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        return [
            f"{prefix}structural_metrics:",
            f"{prefix}  loc: {self._format_value(self.loc)}",
            f"{prefix}  tokens: {self._format_value(self.tokens)}",
            f"{prefix}  cyclomatic_complexity: {self._format_value(self.cyclomatic_complexity)}",
        ]


@dataclass
class DatasetAnalysis:
    quality_metrics: QualityMetrics
    structural_metrics: StructuralMetrics

    def _format_value(self, value: t.Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _to_lines(self, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        lines: list[str] = [f"{prefix}analysis:"]
        lines.extend(self.quality_metrics._to_lines(indent + 1))
        lines.extend(self.structural_metrics._to_lines(indent + 1))
        return lines

    def __str__(self) -> str:
        return "\n".join(self._to_lines())


@dataclass
class DatasetConfig:
    name: str
    description: str
    version: str
    license: str
    has_runable_code_or_test_cases: bool
    languages: str
    files: DatasetFiles
    fields: DatasetFields
    analysis: DatasetAnalysis

    def _format_value(self, value: t.Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _to_lines(self, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        lines: list[str] = [f"{prefix}dataset:"]
        lines.append(f"{prefix}  name: {self.name}")
        lines.append(f"{prefix}  description: {self.description}")
        lines.append(f"{prefix}  version: {self.version}")
        lines.append(f"{prefix}  license: {self.license}")
        lines.append(f"{prefix}  languages: {self.languages}")
        lines.append(
            f"{prefix}  has_runable_code_or_test_cases: {self._format_value(self.has_runable_code_or_test_cases)}"
        )
        lines.extend(self.files._to_lines(indent + 1))
        lines.extend(self.fields._to_lines(indent + 1))
        lines.extend(self.analysis._to_lines(indent + 1))
        return lines

    def __str__(self) -> str:
        return "\n".join(self._to_lines())
