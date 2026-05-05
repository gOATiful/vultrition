
from typing import Tuple

from vds_nutrition_labels.models import config
from vds_nutrition_labels.models.dataset import Dataset, Sample
from vds_nutrition_labels.models.results import CrossContaminationResults, DiversityResults, QualityMetricsResults, SplitNumbericalMetricsResults, StructuralMetricsResults, CompletenessResults, TimeSpanResults


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
        uniqueness=SplitNumbericalMetricsResults(
            train=0,
            test=0,
            validation=0,
            overall=0
        ),
        cross_contamination=CrossContaminationResults(
            train_test=0,
            train_valid=0,
            test_valid=0,
            )
    )
