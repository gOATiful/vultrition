from __future__ import annotations
from dataclasses import asdict
import json
import logging
import sys


from vultrition.commands.create_label import run_create_vultrition_label
from vultrition.commands.load_config import load_config, print_config
from vultrition.commands.create_config import create_config_template
from vultrition.commands.parser import get_parser
from vultrition.commands.run_analysis import run_full_analysis


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
    if args.create_vultrition_label and args.svg_output:
        print("Creating vulnerability dataset nutrition label from results file...")
        return run_create_vultrition_label(args.create_vultrition_label, args.svg_output)
    if args.create_vultrition_label:
        print("Creating vulnerability dataset nutrition label from results file...")
        return run_create_vultrition_label(args.create_vultrition_label)
        
        
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
            results = run_full_analysis(config, args.verbose)
            with open(args.output, "w") as f:
                json.dump(asdict(results), f, indent=2)
            logging.info(f"Analysis results saved to {args.output}")
        else:
            print(
                "No analysis option specified. Use --run_analysis to run the full analysis pipeline.")
        return 0

    parser.print_help()

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())