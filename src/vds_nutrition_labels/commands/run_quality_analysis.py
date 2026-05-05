from vds_nutrition_labels.analysis.quality_analysis import analyze_quality_metrics
from vds_nutrition_labels.models.config import DatasetConfig
from vds_nutrition_labels.models.dataset import Dataset, Sample
from vds_nutrition_labels.models.results import StructuralMetricsResults


def run_quality_analysis(config: DatasetConfig, dataset: Dataset) -> StructuralMetricsResults:
    print("Running quality analysis...")
    results = analyze_quality_metrics(config, dataset)
    
    return results