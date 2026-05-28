

from vultrition.models import config
from vultrition.models.results import AnalysisResults


def run_meta_analysis(config: config) -> None:
    return AnalysisResults(
        name=config.name,
        version=config.version,
        description=config.description,
        license=config.license,
        languages=config.languages,
        has_runable_code_or_test_cases=config.has_runable_code_or_test_cases,
        quality_metrics=None,
        structural_metrics=None,
    )
#     @dataclass
# class AnalysisResults:
#     name: str
#     version: str
#     description: str
#     license: str
#     languages: str
#     has_runable_code_or_test_cases: bool
#     quality_metrics: QualityMetricsResults
#     structural_metrics: StructuralMetricsResults
#     quality_results = run_quality_analysis(config, dataset)
#     structural_results = run_structural_analysis(config, dataset)
