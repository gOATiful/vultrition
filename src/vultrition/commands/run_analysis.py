from vultrition.commands.run_meta_analysis import run_meta_analysis
from vultrition.commands.run_quality_analysis import run_quality_analysis
from vultrition.commands.run_structural_analysis import run_structural_analysis
from vultrition.data_laoading.dataset_loader import load_dataset_from_config
from vultrition.models import config
from vultrition.models.dataset import Dataset
from vultrition.models.results import AnalysisResults


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


def run_full_analysis(config: config, verbose = False) -> AnalysisResults:

    dataset = load_dataset_from_config(config)
    
    print(dataset.summary())
    
    if verbose:
        print_samples(dataset)

    results = run_meta_analysis(config)

    quality_results = run_quality_analysis(config, dataset)
    results.quality_metrics = quality_results
    structural_results = run_structural_analysis(config, dataset)
    results.structural_metrics = structural_results
    return results
