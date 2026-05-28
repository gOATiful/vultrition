

from vultrition.analysis.structural_analysis import analyze_structural_metrics
from vultrition.models.config import DatasetConfig
from vultrition.models.dataset import Dataset


def run_structural_analysis(config: DatasetConfig, dataset: Dataset) -> None:
    structural_metrics = analyze_structural_metrics(config, dataset)
    return structural_metrics