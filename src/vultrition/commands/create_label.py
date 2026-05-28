

import pathlib

from vultrition.data_laoading.results_loader import load_analysis_results_from_json
from vultrition.visualization.painter import draw_label


def run_create_vultrition_label(path_to_results_file: pathlib.Path, output_svg_path: pathlib.Path = pathlib.Path("vultrition_label.svg")) -> int:
    print(f"Creating vulnerability dataset nutrition label from results file: {path_to_results_file}")
    if not path_to_results_file.is_file():
        raise FileNotFoundError(f"Results file not found: {path_to_results_file}")
    loaded_result = load_analysis_results_from_json(path_to_results_file)
    svg_str = draw_label(loaded_result)
    with open(output_svg_path, "w") as f:
        f.write(svg_str)
    print(f"Vulnerability dataset nutrition label saved to: {output_svg_path}")
    return 0