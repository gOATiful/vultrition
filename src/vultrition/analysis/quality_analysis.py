import itertools
from typing import Tuple

from vultrition.analysis.clustering_analysis import compute_embedding_cluster_diversity, print_cluster_diversity_summary
from vultrition.analysis.cross_contamination_analysis import embedding_dataset_similarity
from vultrition.analysis.similarity_analysis import create_code_embeddings
from vultrition.models import config
from vultrition.models import results
from vultrition.models.dataset import Dataset, Sample
from vultrition.models.results import CrossContaminationResults, DiversityResults, QualityMetricsResults, SplitNumbericalMetricsResults, StructuralMetricsResults, CompletenessResults, TimeSpanResults


REQUIRED_FIELDS = {
    "function",
    "label",
    "cve",
    "cwe",
    "project",
}


def _eval_balance(samples: list[Sample]) -> float:
    if not samples:
        return 0.0
    vuln = 0
    non_vuln = 0
    for sample in samples:
        if sample.label is not None:
            if sample.label:
                vuln += 1
            else:
                non_vuln += 1

    balance_score = vuln/non_vuln
    return balance_score


def _eval_diversity(samples: list[Sample]) -> Tuple[float, float]:
    projects = set()
    cwes = set()
    for sample in samples:
        if sample.project and isinstance(sample.project, str):
            projects.add(sample.project)
        if sample.cwe and isinstance(sample.cwe, list):
            cwes.update(sample.cwe)

    return len(cwes)if samples else 0.0, len(projects) if samples else 0.0


def has_all_required_fields(sample: dict) -> bool:
    for field in REQUIRED_FIELDS:
        if field not in sample:
            return False

        value = sample[field]

        if value is None:
            return False

        if isinstance(value, str) and not value.strip():
            return False

        if isinstance(value, list) and len(value) == 0:
            return False

    return True


def _eval_completeness(samples: list[Sample]) -> float:
    if not samples:
        return []
    return [1 if has_all_required_fields(sample.__dict__) else 0 for sample in samples]


def _eval_timespan(samples: list[Sample]) -> Tuple[str, str]:
    cve_years = []
    for sample in samples:
        if sample.cve and isinstance(sample.cve, str):
            parts = sample.cve.split("-")
            if len(parts) >= 3 and parts[1].isdigit():
                cve_years.append(int(parts[1]))
    return str(min(cve_years)) if cve_years else "-", str(max(cve_years)) if cve_years else "-"


def _eval_uniqueness(dataset_embeddings: dict) -> dict:

    scores = {"train": -1, "test": -1, "validation": -1, "overall": -1}
    for split_name, embeddings in dataset_embeddings.items():
        print(f"Computing uniqueness for split: {split_name}")
        r = compute_embedding_cluster_diversity(
            embeddings,
            min_cluster_size=10,
            min_samples=1,
        )
        print_cluster_diversity_summary(r)
        scores[split_name] = r.diversity.diversity_score

    return scores

    # scores = {"train": -1, "test": -1, "validation": -1, "overall": -1}

    # if dataset.has_splits():
    #     splits = {
    #         "validation": dataset.validation or [],
    #         "train": dataset.train or [],
    #         "test": dataset.test or [],
    #     }

    #     for split_name, samples in splits.items():
    #         print("create embeddings for split:", split_name)
    #         embeddings = create_code_embeddings(samples)
    #         print("compute cluster diversity for split:", split_name)
    #         r = compute_embedding_cluster_diversity(
    #             embeddings,
    #             min_cluster_size=10,
    #             min_samples=1,
    #         )
    #         print(f"Cluster diversity results for {split_name}: {r}")
    #         scores[split_name] = r.diversity.diversity_score

    # data = dataset.data if not dataset.has_splits(
    # ) else [*dataset.train, *dataset.test, *dataset.validation]
    # print("create embeddings for split:", "overall")
    # embeddings = create_code_embeddings(data)
    # print("compute cluster diversity for split:", "overall")
    # r = compute_embedding_cluster_diversity(
    #     embeddings,
    #     min_cluster_size=10,
    #     min_samples=1,
    # )
    # print_cluster_diversity_summary(r)
    # print(f"Cluster diversity results for overall: {r}")
    # scores["overall"] = r.diversity.diversity_score
    # return scores


def _eval_cross_contamination(split_embeddings: dict) -> dict:
    splits = ["train", "test", "validation"]
    combos = list(itertools.combinations(splits, 2))
    cross_splits = [f"{s1}_{s2}" for s1, s2 in combos]
    scores = {k: 0.0 for k in cross_splits}
    for s1, s2 in combos:
        A = split_embeddings.get(s1, [])
        B = split_embeddings.get(s2, [])
        if len(A) == 0 or len(B) == 0:
            print(f"Skipping cross-contamination analysis for {s1} and {s2} due to empty embeddings.")
            continue
        summary = embedding_dataset_similarity(A, B)
        scores[f"{s1}_{s2}"] = summary["symmetric_similarity_score"]

    return scores


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
    if dataset.has_splits():
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
    else:
        unique_cwes_overall, unique_projects_overall = _eval_diversity(
            dataset.data or [])
        balance_overall = _eval_balance(dataset.data or [])
        unique_cwes_train = unique_cwes_test = unique_cwes_valid = None
        unique_projects_train = unique_projects_test = unique_projects_valid = None
        balance_train = balance_test = balance_valid = None

    cross_contamination = {
        "train_test": -1,
        "train_validation": -1,
        "test_validation": -1,
    }

    uniqueness = {
        "train": -1,
        "test": -1,
        "validation": -1,
        "overall": -1,
    }

    if config.analysis.quality_metrics.cross_contamination or config.analysis.quality_metrics.uniqueness:
        print("Creating code embeddings for uniqueness and cross-contamination analysis...")
        dataset_embeddings = {}
        if dataset.has_splits():
            print("Creating code embeddings for train split...")
            dataset_embeddings["train"] = create_code_embeddings(
                dataset.train or [])
            print("Creating code embeddings for test split...")
            dataset_embeddings["test"] = create_code_embeddings(
                dataset.test or [])
            print("Creating code embeddings for validation split...")
            dataset_embeddings["validation"] = create_code_embeddings(
                dataset.validation or [])
        print("Creating code embeddings for overall dataset...")
        dataset_embeddings["overall"] = create_code_embeddings(dataset.data if not dataset.has_splits() else [
                                                               *dataset.train, 
                                                               *dataset.test, 
                                                               *dataset.validation])

        if config.analysis.quality_metrics.uniqueness:
            print("Evaluating uniqueness metrics...")
            uniqueness = _eval_uniqueness(dataset_embeddings)
            print(f"uniqueness: {uniqueness}")
        if config.analysis.quality_metrics.cross_contamination:
            print("Evaluating cross-contamination...")
            cross_contamination = _eval_cross_contamination(dataset_embeddings)
            print(f"cross_contamination: {cross_contamination}")

        uniqueness_results = SplitNumbericalMetricsResults(
            train=uniqueness["train"] or -1,
            test=uniqueness["test"] or -1,
            validation=uniqueness["validation"] or -1,
            overall=uniqueness["overall"] or -1,
        )

    overall_samples_cnt = len(dataset.train) + len(dataset.test) + len(
        dataset.validation) if dataset.has_splits() else len(dataset.data)

    return QualityMetricsResults(
        samples=SplitNumbericalMetricsResults(
            train=len(dataset.train) if dataset.train else -1,
            test=len(dataset.test) if dataset.test else -1,
            validation=len(dataset.validation) if dataset.validation else -1,
            overall=overall_samples_cnt,
        ),
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
            train_test=cross_contamination["train_test"] or -1,
            train_valid=cross_contamination["train_validation"] or -1,
            test_valid=cross_contamination["test_validation"] or -1,
        )
    )
