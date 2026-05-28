from dataclasses import dataclass


@dataclass
class SplitNumericalMetricsResults:
    train: float
    test: float
    validation: float
    overall: float
    
    


@dataclass
class DiversityResults:
    unique_cwes: SplitNumericalMetricsResults
    unique_projects: SplitNumericalMetricsResults


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
    samples: SplitNumericalMetricsResults
    completeness: CompletenessResults
    diversity: DiversityResults
    balance: SplitNumericalMetricsResults
    timespan: TimeSpanResults
    similarity_top1: SplitNumericalMetricsResults
    similar_functions_top1: SplitNumericalMetricsResults
    similarity_top3: SplitNumericalMetricsResults
    similar_functions_top3: SplitNumericalMetricsResults
    cross_contamination: CrossContaminationResults
    cross_contamination_a_b_above_threshold: CrossContaminationResults
    cross_contamination_b_a_above_threshold: CrossContaminationResults


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
    preprocessor_directives: SplitNumericalMetricsResults


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
