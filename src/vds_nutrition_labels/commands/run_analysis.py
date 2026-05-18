from vds_nutrition_labels.commands.run_meta_analysis import run_meta_analysis
from vds_nutrition_labels.commands.run_quality_analysis import run_quality_analysis
from vds_nutrition_labels.commands.run_structural_analysis import run_structural_analysis
from vds_nutrition_labels.data_laoading.dataset_loader import load_dataset_from_config
from vds_nutrition_labels.models import config
from vds_nutrition_labels.models.dataset import Dataset


def print_samples(dataset: Dataset):
    print("Sample records from the dataset:")
    NUM_SAMPLES_TO_PRINT = 2
    samples = dataset.train[:NUM_SAMPLES_TO_PRINT] if dataset.train else dataset.data[:NUM_SAMPLES_TO_PRINT]

    for i, sample in enumerate(samples):
        print(f"Record {i+1}:")
        print(f"  Function: {sample.function}")
        print(f"  Label: {sample.label}")
        print(f"  CVE: {sample.cve}")
        print(f"  CWE: {sample.cwe}")
        print(f"  Project: {sample.project}")


def run_full_analysis(config: config) -> None:

    dataset = load_dataset_from_config(config)
    print(dataset.summary())
    print_samples(dataset)

    meta_results = run_meta_analysis(config)

    quality_results = run_quality_analysis(config, dataset)
    meta_results.quality_metrics = quality_results
    structural_results = run_structural_analysis(config, dataset)
    meta_results.structural_metrics = structural_results
    return meta_results
