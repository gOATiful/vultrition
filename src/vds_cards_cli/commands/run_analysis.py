from vds_cards_cli.commands.run_quality_analysis import run_quality_analysis
from vds_cards_cli.commands.run_structural_analysis import run_structural_analysis
from vds_cards_cli.data_laoading.dataset_loader import load_dataset_from_config
from vds_cards_cli.models import config


def run_full_analysis(config: config) -> None:
    
    dataset = load_dataset_from_config(config)
    print(f"Loaded dataset")
    print(dataset.summary())  
    
    run_quality_analysis(config, dataset)
    run_structural_analysis(config, dataset)
