from __future__ import annotations
import argparse
import pathlib


PROG = "vds-nutrition-labels"
VERSION = "0.1.0"


def get_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Vulnerability Dataset Nutrition Labels CLI",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=pathlib.Path,
        help="Path to a vulnerability dataset nutrition label config TOML file.",
    )
    parser.add_argument(
        "--create_config_template",
        "--create-config-template",
        type=pathlib.Path,
        help="Create a template configuration file for the vulnerability dataset nutrition label.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + VERSION,
        help="Show the version of the Vulnerability Dataset Nutrition Labels CLI.",
    )

    parser.add_argument("-o", "--output", type=pathlib.Path,
                        help="Path to save the generated vulnerability dataset nutrition label data.", default=pathlib.Path("vds_nutrition_label_data.json"))

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output during processing.")

    parser.add_argument("--run_analysis", action="store_true",
                        help="Run the full analysis pipeline (quality and structural).")
    parser.add_argument("--create_vultrition_label", type=pathlib.Path,
                        help="Create vulnerability dataset nutrition label from results file.")
    parser.add_argument("--svg_output", type=pathlib.Path,
                        help="Path to save the generated vulnerability dataset nutrition label SVG file.", default=pathlib.Path("vultrition_label.svg"))
    return parser
