from vultrition.analysis.quality_analysis import analyze_quality_metrics
from vultrition.models.config import DatasetConfig
from vultrition.models.dataset import Dataset, Sample
from vultrition.models.results import StructuralMetricsResults


def run_quality_analysis(config: DatasetConfig, dataset: Dataset) -> StructuralMetricsResults:
    print("Running quality analysis...")
    results = analyze_quality_metrics(config, dataset)
    
    return results