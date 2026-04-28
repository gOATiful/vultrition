import pathlib
import typing as t
from vds_cards_cli.data_laoading.config_loader import load_dataset_config
from vds_cards_cli.models.config import DatasetConfig


def load_config(path: pathlib.Path) -> DatasetConfig:
    return load_dataset_config(path)



def print_config(config: DatasetConfig) -> None:
    print(config)