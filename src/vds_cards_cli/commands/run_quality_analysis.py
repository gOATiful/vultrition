from vds_cards_cli.analysis.quality_analysis import analyze_quality_metrics
from vds_cards_cli.models.config import DatasetConfig
from vds_cards_cli.models.dataset import Dataset, Sample


def run_quality_analysis(config: DatasetConfig, dataset: Dataset) -> None:
    print(config.analysis.quality_metrics)
    print("Running quality analysis...")
    results = analyze_quality_metrics(config, dataset)
    print(results)