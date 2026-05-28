# Vultrition: Vulnerability Dataset Nutrition Labels

`vultrition` is a Python command-line tool for creating nutrition-label style summaries of software vulnerability datasets. It reads a TOML dataset configuration, loads vulnerability dataset records, runs quality and structural analyses, exports the results as JSON, and can render the results as an SVG label.

## Features

* Create a TOML configuration template for a dataset
* Load datasets from CSV, JSON, JSONL, or NDJSON files
* Support either train/test/validation splits or a single dataset file
* Map arbitrary dataset column names to the fields used by Vultrition
* Normalize vulnerability labels using a configurable vulnerable-label value
* Compute dataset quality metrics, including completeness, balance, diversity, CVE timespan, uniqueness, and split contamination
* Compute structural code metrics, including lines of code, token counts, cyclomatic complexity, and C/C++ preprocessor directive counts
* Export analysis results as JSON
* Generate an SVG “Dataset Vultrition Label” from an analysis JSON file

## Installation


Install via PyPi.
```bash
pip install vultrition
```


Install the package from source:

```bash
python3 -m pip install -e .
```

Python 3.11 or newer is recommended because Vultrition can use the standard-library `tomllib` TOML parser. On older Python versions, install `tomli` as well.

The analysis pipeline uses several runtime dependencies. If they are not already declared in your project metadata, install them with:

```bash
python3 -m pip install tomli svgwrite lizard tiktoken pygments tqdm numpy torch transformers faiss-cpu
```

For GPU-enabled FAISS, install a FAISS build that matches your CUDA environment instead of `faiss-cpu`.

## Quick start

Create a configuration file:

```bash
vultrition --create-config-template vds-config.toml
```

Edit `vds-config.toml` so that the file paths and field names match your dataset.

Run the full analysis pipeline:

```bash
vultrition --config vds-config.toml --run_analysis --output vds_nutrition_label_data.json
```

Create an SVG label from the JSON results:

```bash
vultrition --create_vultrition_label vds_nutrition_label_data.json --svg_output vultrition_label.svg
```

## CLI usage

Show help:

```bash
vultrition --help
```

Show the installed CLI version:

```bash
vultrition --version
```

Create a config template at a specific file path:

```bash
vultrition --create-config-template path/to/vds-config.toml
```

Create a config template inside an existing directory:

```bash
vultrition --create-config-template path/to/config-directory/
```

When the target is a directory, Vultrition writes `vds-config.toml` inside that directory. Existing files are not overwritten.

Run analysis with the default output path, `vds_nutrition_label_data.json`:

```bash
vultrition --config vds-config.toml --run_analysis
```

Run analysis with an explicit output path:

```bash
vultrition --config vds-config.toml --run_analysis --output results.json
```

Enable verbose output, including parsed config information and sample records:

```bash
vultrition --config vds-config.toml --run_analysis --output results.json --verbose
```

Generate an SVG label with the default SVG output path, `vultrition_label.svg`:

```bash
vultrition --create_vultrition_label results.json
```

Generate an SVG label with an explicit output path:

```bash
vultrition --create_vultrition_label results.json --svg_output labels/vultrition_label.svg
```

## CLI options

| Option                                                 | Description                                                                           |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `-c`, `--config`                                       | Path to a vulnerability dataset TOML config file.                                     |
| `--create-config-template`, `--create_config_template` | Create a template config file. The current parser expects a target path or directory. |
| `-o`, `--output`                                       | JSON output path for analysis results. Defaults to `vds_nutrition_label_data.json`.   |
| `--run_analysis`                                       | Run the full analysis pipeline: metadata, quality metrics, and structural metrics.    |
| `--create_vultrition_label`                            | Create an SVG Vultrition label from an existing JSON analysis result file.            |
| `--svg_output`                                         | SVG output path. Defaults to `vultrition_label.svg`.                                  |
| `-v`, `--verbose`                                      | Print parsed config details and sample records during analysis.                       |
| `--version`                                            | Print the CLI version.                                                                |

## Configuration

Vultrition expects a TOML file with a top-level `[dataset]` section.

Use `[dataset.files]` in one of two ways:

1. Provide `train`, `test`, and optionally `valid` paths for split datasets.
2. Provide `data` for a single-file dataset.

If `data` is set, Vultrition loads that file as the full dataset and ignores split paths.

Supported dataset file extensions are:

* `.csv`
* `.json`
* `.jsonl`
* `.ndjson`

JSON files may contain either a top-level list of records or a top-level object with a `data`, `records`, or `samples` list.

### Example config with train/test/validation splits

```toml
[dataset]
name = "Example Vulnerability Dataset"
description = "Dataset of vulnerable and non-vulnerable code functions"
version = "1.0.0"
license = "MIT"
has_runable_code_or_test_cases = false
languages = "c,c++,python"

[dataset.files]
train = "data/train.jsonl"
test = "data/test.jsonl"
valid = "data/valid.jsonl"

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
```

### Example config with a single dataset file

```toml
[dataset]
name = "Example Vulnerability Dataset"
description = "Single-file vulnerability dataset"
version = "1.0.0"
license = "MIT"
has_runable_code_or_test_cases = false
languages = "c,c++"

[dataset.files]
data = "data/all_samples.csv"

[dataset.fields]
function = "function_source"
label = "is_vulnerable"
vuln_label_value = true
cve = "cve_id"
cwe = "cwe_id"
project = "repository"

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
```

## Dataset field mapping

The `[dataset.fields]` section tells Vultrition how to read each record in your dataset.

| Config key         | Meaning                                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `function`         | Field containing the source code snippet or function body.                                                                |
| `label`            | Field containing the original vulnerability label.                                                                        |
| `vuln_label_value` | Value that should be interpreted as vulnerable. Matching records are normalized to `1`; all others are normalized to `0`. |
| `cve`              | Field containing a CVE identifier, such as `CVE-2023-12345`. Used for metadata completeness and timespan analysis.        |
| `cwe`              | Field containing one or more CWE identifiers. Strings, delimited strings, and lists are supported.                        |
| `project`          | Field containing the project, repository, or source identifier.                                                           |

## Metrics

### Quality metrics

Vultrition reports quality metrics per split and overall where applicable:

* `samples`: number of records
* `completeness`: share of records with all required fields populated
* `balance`: ratio of vulnerable to non-vulnerable records
* `diversity.unique_cwes`: number of unique CWE identifiers
* `diversity.unique_projects`: number of unique projects
* `timespan`: minimum and maximum CVE year found in CVE identifiers
* `similarity_top1` and `similarity_top3`: nearest-neighbor code similarity metrics based on code embeddings
* `similar_functions_top1` and `similar_functions_top3`: share of entries with near duplicates above the similarity threshold
* `cross_contamination`: split-to-split similarity scores for train/test, train/validation, and test/validation

For uniqueness and cross-contamination, Vultrition creates code embeddings with `jinaai/jina-code-embeddings-1.5b` and uses FAISS similarity search.

### Structural metrics

Vultrition reports structural metrics as `min`, `max`, `mean`, and `std` per split and overall:

* `loc`: source lines of code per entry
* `tokens`: token count per entry using `tiktoken`
* `cyclomatic_complexity`: cyclomatic complexity per entry using `lizard`
* `preprocessor_directives`: count of entries skipped for C/C++ cyclomatic-complexity reporting because preprocessor directives were detected

## Outputs

### JSON analysis output

Running analysis writes a JSON file containing:

* dataset metadata: `name`, `version`, `description`, `license`, `languages`, and `has_runable_code_or_test_cases`
* `quality_metrics`
* `structural_metrics`

Example:

```bash
vultrition --config vds-config.toml --run_analysis --output results.json
```

### SVG label output

Create a nutrition-label style SVG from an analysis JSON file:

```bash
vultrition --create_vultrition_label results.json --svg_output vultrition_label.svg
```

The generated SVG includes dataset metadata, quality facts, split contamination metrics when available, and structural facts.

## Notes and troubleshooting

* The config key is spelled `has_runable_code_or_test_cases` to match the current implementation.
* `--create-config-template` currently expects a path argument, for example `vds-config.toml` or `configs/`.
* The loader accepts both `[dataset.analysis.quality_metrics]` and the legacy misspelled `[dataset.analysis.quiality_metrics]`, but new configs should use `quality_metrics`.
* The first analysis run may download the `jinaai/jina-code-embeddings-1.5b` model through Hugging Face Transformers.
* If you see `ModuleNotFoundError` for `transformers`, `torch`, `faiss`, `lizard`, `tiktoken`, `pygments`, `svgwrite`, or `tqdm`, install the missing runtime dependency.
* If a dataset file fails to load, check that it has one of the supported extensions: `.csv`, `.json`, `.jsonl`, or `.ndjson`.
* For JSON datasets, use a top-level list or an object containing a `data`, `records`, or `samples` list.
* Existing config files are not overwritten by the template command.

## Project

* Package name: `vultrition`
* CLI entry point: `vultrition.cli:main`
* Internal parser program name: `vds-nutrition-labels`
* Version: `0.1.0`
* License: MIT

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request on the GitHub repository.
