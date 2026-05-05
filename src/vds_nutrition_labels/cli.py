from __future__ import annotations
from dataclasses import asdict
import json
import sys


from vds_nutrition_labels.commands.load_config import load_config, print_config
from vds_nutrition_labels.commands.create_config import create_config_template
from vds_nutrition_labels.commands.parser import get_parser
from vds_nutrition_labels.commands.run_analysis import run_full_analysis


def main() -> int:
    parser = get_parser()

    args = parser.parse_args()

    if args.create_config_template:
        try:
            created_path = create_config_template(args.create_config_template)
        except FileExistsError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Created config template at {created_path}")
        return 0

    if args.config:
        if not args.output:
            print("Error: --output is required when using --config", file=sys.stderr)
            return 1
        try:
            config = load_config(args.config)
        except Exception as exc:
            print(f"Failed to parse config: {exc}", file=sys.stderr)
            return 1
        if args.verbose:
            print(f"Parsed config from {args.config}")
            print_config(config)

        if args.run_analysis:
            results = run_full_analysis(config)
            print(results)
            with open(args.output, "w") as f:
                json.dump(asdict(results), f, indent=2)
            
        else:
            print(
                "No analysis option specified. Use --run_analysis to run the full analysis pipeline.")
        return 0

    parser.print_help()

    return 0
