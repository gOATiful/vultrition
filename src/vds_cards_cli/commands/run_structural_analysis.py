

from vds_cards_cli.analysis.structural_analysis import analyze_structural_metrics
from vds_cards_cli.models.config import DatasetConfig
from vds_cards_cli.models.dataset import Dataset


def run_structural_analysis(config: DatasetConfig, dataset: Dataset) -> None:
    structural_metrics = analyze_structural_metrics(config, dataset)
    print("Structural Metrics Results:")
    print(structural_metrics)