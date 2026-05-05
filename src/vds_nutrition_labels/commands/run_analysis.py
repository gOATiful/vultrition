from vds_nutrition_labels.commands.run_meta_analysis import run_meta_analysis
from vds_nutrition_labels.commands.run_quality_analysis import run_quality_analysis
from vds_nutrition_labels.commands.run_structural_analysis import run_structural_analysis
from vds_nutrition_labels.data_laoading.dataset_loader import load_dataset_from_config
from vds_nutrition_labels.models import config


def run_full_analysis(config: config) -> None:
    
    dataset = load_dataset_from_config(config)
    print(dataset.summary())  
    
    meta_results = run_meta_analysis(config)
    
    quality_results = run_quality_analysis(config, dataset)
    structural_results = run_structural_analysis(config, dataset)
    meta_results.quality_metrics = quality_results
    meta_results.structural_metrics = structural_results
    return meta_results
