# Dataset Cards CLI

A command-line tool for building dataset cards from vulnerability dataset configuration files.

## Features

- Generate dataset cards for vulnerability datasets using a TOML config file
- Built-in config template creation
- Runs both quality and structural dataset analysis
- Supports train/test/validation split metadata and field mappings

## Installation

Install locally using the included Python environment or pip:

```bash
python -m pip install -e .
```

If you are running on Python 3.9 or 3.10, install `tomli` for TOML support:

```bash
python -m pip install tomli
```

## Usage

Create a default configuration template:

```bash
vds-cards-cli --create-config-template
```

Or create it at a specific location:

```bash
vds-cards-cli --create-config-template path/to/vds-config.toml
```

Run the full analysis pipeline using a config file:

```bash
vds-cards-cli --config vds-config.toml --output vds_card_analysis.json --run_analysis
```

Enable verbose output:

```bash
vds-cards-cli --config vds-config.toml --output vds_card_analysis.json --run_analysis --verbose
```

## Configuration

The CLI expects a TOML file with a top-level `[dataset]` section. The built-in template includes:

- `name`, `description`, `version`, `license`
- `has_runable_code_or_test_cases`
- `languages`
- `files` section for `train`, `test`, and `valid` dataset paths
- `fields` section for field names such as `function`, `label`, `cve`, `cwe`, and `project`
- `analysis.quality_metrics` including `completeness`, `diversity`, `balance`, `timespan`, `uniqueness`, and `cross_contamination`
- `analysis.structural_metrics` including `loc`, `tokens`, `cyclomatic_complexity`, and `node_diversity`

Example:

```toml
[dataset]
name = "Example Dataset"
description = "Dataset of annotated images"
version = "1.0.0"
license = "MIT"
has_runable_code_or_test_cases = false
languages = "c,c++,python"

[dataset.files]
train = "path/to/trainfile"
test = "path/to/testfile"
valid = "path/to/validfile"

[dataset.fields]
function = "func"
label = "label"
vuln_label_value = 1
cve = "cve"
cwe = "cwe"
project = "project_url"

[dataset.analysis.quality_metrics]
completeness = true
diversity = true
balance = true
timespan = true
uniqueness = true
cross_contamination = true

[dataset.analysis.structural_metrics]
loc = true
tokens = true
cyclomatic_complexity = true
node_diversity = true
```

## Project

- Package name: `vds-cards-cli`
- Entry point: `vds_cards_cli.cli:main`
- License: MIT

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request on the GitHub repository.
