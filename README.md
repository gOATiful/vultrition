# VULTRITION: Vulnerability Dataset Nutrition Labels

`vultrition` is a lightweight Python command-line tool for generating nutrition-label style summaries for function-level vulnerability datasets. It turns a dataset and a TOML configuration file into machine-readable JSON facts and an SVG label that can be included in dataset READMEs, benchmark reports, papers, or dataset catalogs.

VULTRITION does **not** rank datasets or certify their quality. Instead, it exposes facts that are often hidden, incomplete, or reported inconsistently: dataset size, metadata completeness, vulnerability-type coverage, class imbalance, duplicate or near-duplicate functions, train/test contamination, and structural characteristics of the code.

## Core contributions

- **Compact dataset nutrition labels:** Produce standardized numerical summaries for function-level vulnerability datasets instead of long, hard-to-compare documentation.
- **Reproducible fact extraction:** Use a TOML config to describe dataset metadata, file locations, split information, and field mappings.
- **Quality and structure in one view:** Report facts about completeness, balance, CWE/project coverage, CVE timespan, uniqueness, split contamination, lines of code, token counts, and cyclomatic complexity.
- **Benchmark transparency:** Help researchers choose datasets, document dataset versions, and interpret vulnerability-detection results beyond headline model scores.

## Installation

Install from the repository root:

```bash
python3 -m pip install -e .
```

Python 3.11 or newer is recommended. On older Python versions, install `tomli` for TOML support.

If runtime dependencies are not installed automatically, install them with:

```bash
python3 -m pip install tomli svgwrite lizard tiktoken pygments tqdm numpy torch transformers faiss-cpu
```

For GPU-enabled FAISS, install a FAISS build that matches your CUDA environment instead of `faiss-cpu`.

## Quick start

Create a configuration template:

```bash
vultrition --create-config-template vds-config.toml
```

Edit the generated TOML file so that the dataset metadata, file paths, and field names match your dataset.

Run the analysis pipeline:

```bash
vultrition --config vds-config.toml --run_analysis --output vds_nutrition_label_data.json
```

Generate the SVG nutrition label:

```bash
vultrition --create_vultrition_label vds_nutrition_label_data.json --svg_output vultrition_label.svg
```

## Input data

VULTRITION supports vulnerability datasets stored as:

- `.csv`
- `.json`
- `.jsonl`
- `.ndjson`

A dataset can be provided either as predefined `train`, `test`, and optional `valid` splits, or as a single `data` file. JSON files may contain a top-level list or an object with a `data`, `records`, or `samples` list.

## Configuration

VULTRITION expects a TOML file with a top-level `[dataset]` section. The fields section maps the column names in your dataset to the canonical fields used by the analysis.

```toml
[dataset]
name = "Example Vulnerability Dataset"
description = "Function-level vulnerability dataset"
version = "1.0.0"
license = "MIT"
has_runable_code_or_test_cases = false
languages = "c,c++"

[dataset.files]
train = "data/train.jsonl"
test = "data/test.jsonl"
valid = "data/valid.jsonl"
# For a single-file dataset, use this instead of the split paths:
# data = "data/all.jsonl"

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

### Field mapping

| Config key | Meaning |
| --- | --- |
| `function` | Source code snippet or function body. |
| `label` | Original vulnerability label. |
| `vuln_label_value` | Label value that should be interpreted as vulnerable. Matching records are normalized to `1`; all others to `0`. |
| `cve` | CVE identifier field. Used for completeness and timespan analysis. |
| `cwe` | CWE identifier field. Strings, delimited strings, and lists are supported. |
| `project` | Project, repository, or source identifier. |

## Reported facts

### Quality facts

VULTRITION reports facts that help characterize dataset composition and reliability:

- number of functions
- metadata completeness
- vulnerable/non-vulnerable balance
- unique projects and CWEs
- CVE-year timespan
- nearest-neighbor similarity and near-duplicate rate
- cross-contamination between train, validation, and test splits when splits are available

Similarity-based facts are computed with code embeddings and FAISS nearest-neighbor search. The default near-duplicate threshold is `0.95`.

### Structural facts

VULTRITION also reports structural properties of the analyzed functions:

- lines of code
- token counts
- cyclomatic complexity

These facts are useful when comparing datasets for modern code models, where long or complex functions may affect training and evaluation.

## CLI reference

| Option | Description |
| --- | --- |
| `-c`, `--config` | Path to a VULTRITION TOML config file. |
| `--create-config-template`, `--create_config_template` | Create a template config file at the provided path or inside the provided directory. |
| `-o`, `--output` | JSON output path for analysis results. Defaults to `vds_nutrition_label_data.json`. |
| `--run_analysis` | Run the full analysis pipeline. |
| `--create_vultrition_label` | Create an SVG label from an existing JSON result file. |
| `--svg_output` | SVG output path. Defaults to `vultrition_label.svg`. |
| `-v`, `--verbose` | Print parsed config details and sample records during analysis. |
| `--version` | Print the installed CLI version. |

## Outputs

Running analysis produces a JSON file with dataset metadata, quality facts, and structural facts:

```bash
vultrition --config vds-config.toml --run_analysis --output results.json
```

The JSON file can be rendered as an SVG label:

```bash
vultrition --create_vultrition_label results.json --svg_output vultrition_label.svg
```

## Paper and demo

VULTRITION is described in the paper **“VULTRITION: Nutrition Label Generation for Function Vulnerability Datasets.”** The paper demonstrates the tool by generating nutrition labels for popular function-level vulnerability datasets across C, C++, Java, and Python. (Link avaiable if paper gets accepted)

Video demo: https://youtu.be/wEaGWKP5Szo

## Notes

- VULTRITION is intended to support dataset selection, documentation, and benchmark interpretation, not to produce a single quality score.
- The config key is currently spelled `has_runable_code_or_test_cases` to match the implementation.
- The loader accepts both `[dataset.analysis.quality_metrics]` and the legacy misspelled `[dataset.analysis.quiality_metrics]`, but new configs should use `quality_metrics`.
- The first analysis run may download the code-embedding model used for similarity analysis.
- Existing config files are not overwritten by the template command.

## Project

- Package name: `vultrition`
- CLI entry point: `vultrition.cli:main`
- Internal parser program name: `vds-nutrition-labels`
- Version: `0.1.0`
- License: MIT

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request on the GitHub repository.
