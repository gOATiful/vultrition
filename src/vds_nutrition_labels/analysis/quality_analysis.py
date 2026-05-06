
import itertools
import json
import os
import shutil
from typing import Tuple

from vds_nutrition_labels.models import config
from vds_nutrition_labels.models.dataset import Dataset, Sample
from vds_nutrition_labels.models.results import CrossContaminationResults, DiversityResults, QualityMetricsResults, SplitNumbericalMetricsResults, StructuralMetricsResults, CompletenessResults, TimeSpanResults
from vds_nutrition_labels.analysis.uniqueness_analysis import run_uniqueness_detection, UniquenessConfig


def _create_source_files(samples: list[Sample], split_name: str, file_name: str, extension: str):
    os.makedirs(f"uniqueness_source/{split_name}", exist_ok=True)
    for i, sample in enumerate(samples):
        with open(f"uniqueness_source/{split_name}/{file_name}_{i}.{extension}", "w", encoding="utf-8") as f:
            if sample.function and isinstance(sample.function, str):
                f.write(sample.function + "\n")


def _get_file_extension(config: config) -> str:
    file_extension_mapping = {
        "cpp": "cpp",
        "c": "c",
        "java": "java",
        "python": "py",
    }

    extension = ""
    for lang, ext in file_extension_mapping.items():
        if lang in config.languages.lower():
            extension = ext
            break
    return extension


def _eval_uniqueness(config: config, dataset: Dataset) -> float:
    splits = {
        "train": dataset.train or [],
        "test": dataset.test or [],
        "validation": dataset.validation or []
    }

    extension = _get_file_extension(config)

    if dataset.has_splits():
        for split_name, samples in splits.items():
            _create_source_files(samples, split_name, split_name, extension)

        scores = {"train": 0.0, "test": 0.0, "validation": 0.0}

        for split_name in splits.keys():
            print("Running uniqueness detection for split:", split_name)
            run_uniqueness_detection(
                f"uniqueness_source/{split_name}",
                UniquenessConfig(
                    output_dir="uniqueness_results",
                ),
            )

            with open(f"uniqueness_results/uniqueness_metrics.json", "r", encoding="utf-8") as f:
                stats = json.load(f)
            scores[split_name] = stats['true_uniqueness_score']
            shutil.rmtree(f"uniqueness_source/{split_name}")
            shutil.rmtree("uniqueness_results")
        return scores

    else:
        _create_source_files(dataset.data or [],
                             "overall", "overall", extension)
        print("Running uniqueness detection for overall dataset...")
        run_uniqueness_detection(
            f"uniqueness_source/overall",
            UniquenessConfig(
                output_dir="uniqueness_results",
            ),
        )

        with open(f"uniqueness_results/uniqueness_metrics.json", "r", encoding="utf-8") as f:
            stats = json.load(f)
        score = stats['true_uniqueness_score']
        shutil.rmtree(f"uniqueness_source/overall")
        shutil.rmtree("uniqueness_results")
        return {
            "overall": score
        }


def _eval_cross_contamination(config: config, dataset: Dataset) -> CrossContaminationResults:
    splits = {
        "train": dataset.train or [],
        "test": dataset.test or [],
        "validation": dataset.validation or []
    }

    extension = _get_file_extension(config)
    combos = list(itertools.combinations(splits.keys(), 2))
    cross_splits = [f"{s1}_{s2}" for s1, s2 in combos]

    scores = {k: 0.0 for k in cross_splits}

    for split1, split2 in combos:
        os.makedirs(f"uniqueness_source/{split1}_{split2}", exist_ok=True)
        for i, sample in enumerate(splits[split1]):
            with open(f"uniqueness_source/{split1}_{split2}/{split1}_{i}.{extension}", "w", encoding="utf-8") as f:
                if sample.function and isinstance(sample.function, str):
                    f.write(sample.function + "\n")

        for j, sample2 in enumerate(splits[split2]):
            with open(f"uniqueness_source/{split1}_{split2}/{split2}_{j}.{extension}", "w", encoding="utf-8") as f:
                if sample2.function and isinstance(sample2.function, str):
                    f.write(sample2.function + "\n")

    for split_name in cross_splits:
        print("Running uniqueness detection for split:", split_name)
        result = run_uniqueness_detection(
            f"uniqueness_source/{split_name}",
            UniquenessConfig(
                output_dir="uniqueness_results",
            ),
        )

        cnt = 0
        with open(f"uniqueness_results/duplicate_pairs.csv", "r") as fin:
            content = fin.readlines()
            if len(content) == 1:  # no duplicates found
                scores[split_name] = 1
                continue
            content = content[1:]
            for l in content:
                file1, file2 = l.split(",")[:2]
                # only count cross-split duplicates
                if file1.split("_")[0] != file2.split("_")[0]:
                    cnt += 1

        scores[split_name] = 1 - \
            (cnt / len(os.listdir(f"uniqueness_source/{split_name}")))
        shutil.rmtree(f"uniqueness_source/{split_name}")
        shutil.rmtree("uniqueness_results")

    return scores


def _eval_balance(samples: list[Sample]) -> float:
    if not samples:
        return 0.0
    vuln = 0
    for sample in samples:
        if sample.label is not None:
            if sample.label:
                vuln += 1
    total_samples = len(samples)
    balance_score = vuln/total_samples if total_samples > 0 else 0.0
    return balance_score


def _eval_diversity(samples: list[Sample]) -> Tuple[float, float]:
    projects = set()
    cwes = set()
    for sample in samples:
        if sample.project and isinstance(sample.project, str):
            projects.add(sample.project)
        if sample.cwe and isinstance(sample.cwe, list):
            cwes.update(sample.cwe)

    return len(cwes) / len(samples) if samples else 0.0, len(projects) / len(samples) if samples else 0.0


def _is_sample_complete(sample: Sample) -> bool:
    if not sample.function or not isinstance(sample.function, str) or not sample.function.strip():
        return False
    if sample.label is None:
        return False
    if not sample.cve or not isinstance(sample.cve, str) or not sample.cve.strip():
        return False
    if not sample.cwe or not isinstance(sample.cwe, list) or len(sample.cwe) == 0:
        return False
    if not sample.project or not isinstance(sample.project, str) or not sample.project.strip():
        return False
    return True


def _eval_completeness(samples: list[Sample]) -> float:
    if not samples:
        return []
    return [1 if not _is_sample_complete(sample) else 0 for sample in samples]


def _eval_timespan(samples: list[Sample]) -> Tuple[str, str]:
    cve_years = []
    for sample in samples:
        if sample.cve and isinstance(sample.cve, str):
            parts = sample.cve.split("-")
            if len(parts) >= 3 and parts[1].isdigit():
                cve_years.append(int(parts[1]))
    return str(min(cve_years)) if cve_years else "-", str(max(cve_years)) if cve_years else "-"


def analyze_quality_metrics(config: config, dataset: Dataset) -> StructuralMetricsResults:
    if config.analysis.quality_metrics.completeness:
        print("Evaluating completeness metrics...")
        if dataset.has_splits():
            completeness_samples_train = _eval_completeness(
                dataset.train or [])
            completeness_samples_test = _eval_completeness(dataset.test or [])
            completeness_samples_valid = _eval_completeness(
                dataset.validation or [])
            completeness_samples_overall = [
                *completeness_samples_train, *completeness_samples_test, *completeness_samples_valid]

            completeness_train = sum(completeness_samples_train) / len(
                completeness_samples_train) if completeness_samples_train else 0.0
            completeness_test = sum(completeness_samples_test) / len(
                completeness_samples_test) if completeness_samples_test else 0.0
            completeness_valid = sum(completeness_samples_valid) / len(
                completeness_samples_valid) if completeness_samples_valid else 0.0
            completeness_overall = sum(completeness_samples_overall) / len(
                completeness_samples_overall) if completeness_samples_overall else 0.0
        else:
            completeness_samples_overall = _eval_completeness(
                dataset.data or [])
            completeness_overall = sum(completeness_samples_overall) / len(
                completeness_samples_overall) if completeness_samples_overall else 0.0
            completeness_train = completeness_test = completeness_valid = None

        completeness_results = CompletenessResults(
            train=completeness_train,
            test=completeness_test,
            valid=completeness_valid,
            overall=completeness_overall,
        )
    else:
        completeness_results = None

    timespan_results = None
    if config.analysis.quality_metrics.timespan:
        print("Evaluating timespan metrics...")
        if dataset.has_splits():
            timespan_train = _eval_timespan(dataset.train or [])
            timespan_test = _eval_timespan(dataset.test or [])
            timespan_valid = _eval_timespan(dataset.validation or [])
            timespan_overall = _eval_timespan(
                [*dataset.train, *dataset.test, *dataset.validation])
        else:
            timespan_overall = _eval_timespan(dataset.data or [])
            timespan_train = timespan_test = timespan_valid = None
        timespan_results = TimeSpanResults(
            train=timespan_train,
            test=timespan_test,
            validation=timespan_valid,
            overall=timespan_overall,
        )
    else:
        timespan_train = timespan_test = timespan_valid = timespan_overall = None

    print("Evaluating diversity metrics...")
    unique_cwes_train, unique_projects_train = _eval_diversity(
        dataset.train or [])
    unique_cwes_test, unique_projects_test = _eval_diversity(
        dataset.test or [])
    unique_cwes_valid, unique_projects_valid = _eval_diversity(
        dataset.validation or [])
    unique_cwes_overall, unique_projects_overall = _eval_diversity(
        [*dataset.train, *dataset.test, *dataset.validation])

    print("Evaluating balance metrics...")
    balance_train = _eval_balance(dataset.train or [])
    balance_test = _eval_balance(dataset.test or [])
    balance_valid = _eval_balance(dataset.validation or [])
    balance_overall = _eval_balance(
        [*dataset.train, *dataset.test, *dataset.validation])

    print("Evaluating uniqueness and cross-contamination metrics...")
    print("creating source folders...")
    if dataset.has_splits():
        uniqueness = _eval_uniqueness(config, dataset)
        print(f"uniqueness: {uniqueness}")
        cross_contamination = _eval_cross_contamination(config, dataset)
        print(f"cross_contamination: {cross_contamination}")
        shutil.rmtree("uniqueness_source")
        uniqueness_results = SplitNumbericalMetricsResults(
            train=uniqueness["train"],
            test=uniqueness["test"],
            validation=uniqueness["validation"],
            overall=-1,
        )
    else:
        uniqueness_score = _eval_uniqueness(config, dataset)
        print(f"uniqueness: {uniqueness_score}")
        uniqueness_results = SplitNumbericalMetricsResults(
            train=-1,
            test=-1,
            validation=-1,
            overall=uniqueness_score["overall"],
        )
        cross_contamination = {
            "train_test": -1,
            "train_valid": -1,
            "test_valid": -1,
        }

    return QualityMetricsResults(
        completeness=completeness_results,
        diversity=DiversityResults(
            unique_cwes=SplitNumbericalMetricsResults(
                train=unique_cwes_train,
                test=unique_cwes_test,
                validation=unique_cwes_valid,
                overall=unique_cwes_overall
            ),
            unique_projects=SplitNumbericalMetricsResults(
                train=unique_projects_train,
                test=unique_projects_test,
                validation=unique_projects_valid,
                overall=unique_projects_overall
            )
        ),
        balance=SplitNumbericalMetricsResults(
            train=balance_train,
            test=balance_test,
            validation=balance_valid,
            overall=balance_overall
        ),
        timespan=timespan_results,
        uniqueness=uniqueness_results,
        cross_contamination=CrossContaminationResults(
            train_test=cross_contamination["train_test"],
            train_valid=cross_contamination["train_validation"],
            test_valid=cross_contamination["test_validation"],
        )
    )
