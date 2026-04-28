from dataclasses import dataclass


@dataclass
class SplitNumbericalMetricsResults:
    train: int
    test: int
    validation: int
    overall: int


@dataclass
class DiversityResults:
    unique_cwes: SplitNumbericalMetricsResults
    unique_projects: SplitNumbericalMetricsResults


@dataclass
class TimeSpanResults:
    train: tuple[str, str]
    test: tuple[str, str]
    validation: tuple[str, str]
    overall: tuple[str, str]


@dataclass
class CrossContaminationResults:
    train_test: float
    train_valid: float
    test_valid: float


@dataclass
class CompletenessResults:
    train: float
    test: float
    valid: float
    overall: float


@dataclass
class QualityMetricsResults:
    completeness: CompletenessResults
    diversity: DiversityResults
    balance: float
    timespan: TimeSpanResults
    uniqueness: float
    cross_contamination: CrossContaminationResults


@dataclass
class CountingResult:
    min: float
    max: float
    mean: float
    std: float


@dataclass
class SplitStatisticalMetricsResults:
    train: CountingResult
    test: CountingResult
    validation: CountingResult
    overall: CountingResult


@dataclass
class StructuralMetricsResults:
    loc: SplitStatisticalMetricsResults
    tokens: SplitStatisticalMetricsResults
    cyclomatic_complexity: SplitStatisticalMetricsResults


@dataclass
class AnalysisResults:
    name: str
    version: str
    description: str
    license: str
    languages: str
    has_runable_code_or_test_cases: bool
    quality_metrics: QualityMetricsResults
    structural_metrics: StructuralMetricsResults
