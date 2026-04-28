from __future__ import annotations
import argparse
import pathlib


PROG = "dataset_cards_cli"
VERSION = "0.1.0"


def get_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Dataset cards CLI",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=pathlib.Path,
        help="Path to a dataset card config TOML file.",
    )
    parser.add_argument(
        "--create_config_template",
        "--create-config-template",
        type=pathlib.Path,
        help="Create a template configuration file for the dataset card.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + VERSION,
        help="Show the version of the dataset cards CLI.",
    )

    parser.add_argument("-o", "--output", type=pathlib.Path,
                        help="Path to save the generated dataset card.", default=pathlib.Path("vds_card_analysis.json"))

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output during processing.")

    parser.add_argument("--run_analysis", action="store_true",
                        help="Run the full analysis pipeline (quality and structural).")

    return parser
