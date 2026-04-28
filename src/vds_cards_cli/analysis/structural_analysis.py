import lizard
import tiktoken
from tqdm import tqdm

from vds_cards_cli.models import config
from vds_cards_cli.models.dataset import Dataset, Sample
from vds_cards_cli.models.results import CountingResult, SplitStatisticalMetricsResults, StructuralMetricsResults


def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def _get_metrics(samples: list[Sample], label: str, count_tokens: bool = False, count_structural: bool = False) -> list[int]:
    if not samples:
        return []

    loc_values = []
    cyclomatic_complexity_values = []
    token_counts = []
    # TODO: !!!! remove limit
    for sample in tqdm(samples[:100], desc=f"Analyzing Structural Metrics for {label}"):
        try:
            if count_structural:
                analysis = lizard.analyze_file.analyze_source_code(
                    "f", sample.function)
                if analysis and len(analysis.function_list) > 0:
                    loc_values.append(analysis.function_list[0].nloc)
                    cyclomatic_complexity_values.append(
                        analysis.function_list[0].cyclomatic_complexity)

            if count_tokens:
                token_counts.append(_count_tokens(sample.function))
        except Exception as exc:
            print(
                f"Warning: Failed to analyze function for Structural Metrics: {exc}")

    return loc_values, cyclomatic_complexity_values, token_counts


def analyze_structural_metrics(config: config, dataset: Dataset) -> StructuralMetricsResults:
    print("Analyzing structural metrics...")

    if config.analysis.structural_metrics.loc or config.analysis.structural_metrics.cyclomatic_complexity or config.analysis.structural_metrics.tokens:
        count_tokens = config.analysis.structural_metrics.tokens
        count_structural = config.analysis.structural_metrics.loc or config.analysis.structural_metrics.cyclomatic_complexity
        if dataset.has_splits():
            train_nlocs, train_cyclomatic_complexity, train_token_counts = _get_metrics(
                dataset.train or [], "Train", count_tokens=count_tokens, count_structural=count_structural)
            test_nlocs, test_cyclomatic_complexity, test_token_counts = _get_metrics(
                dataset.test or [], "Test", count_tokens=count_tokens, count_structural=count_structural)
            valid_nlocs, valid_cyclomatic_complexity, valid_token_counts = _get_metrics(
                dataset.validation or [], "Validation", count_tokens=count_tokens, count_structural=count_structural)
            overall_nlocs, overall_cyclomatic_complexity, overall_token_counts = [*train_nlocs, *test_nlocs, *valid_nlocs], [
                *train_cyclomatic_complexity, *test_cyclomatic_complexity, *valid_cyclomatic_complexity], [*train_token_counts, *test_token_counts, *valid_token_counts]

            nloc_train_results = CountingResult(
                mean=sum(train_nlocs) /
                len(train_nlocs) if train_nlocs else -1,
                min=min(train_nlocs) if train_nlocs else -1,
                max=max(train_nlocs) if train_nlocs else -1,
                std=(sum((x - (sum(train_nlocs) / len(train_nlocs))) **
                     2 for x in train_nlocs) / len(train_nlocs)) ** 0.5 if train_nlocs else -1,
            )
            nloc_train_results = CountingResult(
                mean=sum(test_nlocs) /
                len(test_nlocs) if test_nlocs else -1,
                min=min(test_nlocs) if test_nlocs else -1,
                max=max(test_nlocs) if test_nlocs else -1,
                std=(sum((x - (sum(test_nlocs) / len(test_nlocs))) **
                     2 for x in test_nlocs) / len(test_nlocs)) ** 0.5 if test_nlocs else -1,
            )

            nloc_test_results = CountingResult(
                mean=sum(test_nlocs) /
                len(test_nlocs) if test_nlocs else -1,
                min=min(test_nlocs) if test_nlocs else -1,
                max=max(test_nlocs) if test_nlocs else -1,
                std=(sum((x - (sum(test_nlocs) / len(test_nlocs))) **
                     2 for x in test_nlocs) / len(test_nlocs)) ** 0.5 if test_nlocs else -1,
            )

            nloc_valid_results = CountingResult(
                mean=sum(valid_nlocs) /
                len(valid_nlocs) if valid_nlocs else -1,
                min=min(valid_nlocs) if valid_nlocs else -1,
                max=max(valid_nlocs) if valid_nlocs else -1,
                std=(sum((x - (sum(valid_nlocs) / len(valid_nlocs))) **
                     2 for x in valid_nlocs) / len(valid_nlocs)) ** 0.5 if valid_nlocs else -1,
            )
            nloc_overall_results = CountingResult(
                mean=sum(overall_nlocs) /
                len(overall_nlocs) if overall_nlocs else -1,
                min=min(overall_nlocs) if overall_nlocs else -1,
                max=max(overall_nlocs) if overall_nlocs else -1,
                std=(sum((x - (sum(overall_nlocs) / len(overall_nlocs))) **
                     2 for x in overall_nlocs) / len(overall_nlocs)) ** 0.5 if overall_nlocs else -1,
            )
            # Complexity
            cyclomatic_train_results = CountingResult(
                mean=sum(train_cyclomatic_complexity) /
                len(train_cyclomatic_complexity) if train_cyclomatic_complexity else -1,
                min=min(
                    train_cyclomatic_complexity) if train_cyclomatic_complexity else -1,
                max=max(
                    train_cyclomatic_complexity) if train_cyclomatic_complexity else -1,
                std=(sum((x - (sum(train_cyclomatic_complexity) / len(train_cyclomatic_complexity))) **
                     2 for x in train_cyclomatic_complexity) / len(train_cyclomatic_complexity)) ** 0.5 if train_cyclomatic_complexity else -1,
            )
            cyclomatic_test_results = CountingResult(
                mean=sum(test_cyclomatic_complexity) /
                len(test_cyclomatic_complexity) if test_cyclomatic_complexity else -1,
                min=min(
                    test_cyclomatic_complexity) if test_cyclomatic_complexity else -1,
                max=max(
                    test_cyclomatic_complexity) if test_cyclomatic_complexity else -1,
                std=(sum((x - (sum(test_cyclomatic_complexity) / len(test_cyclomatic_complexity))) **
                     2 for x in test_cyclomatic_complexity) / len(test_cyclomatic_complexity)) ** 0.5 if test_cyclomatic_complexity else -1,
            )
            cyclomatic_valid_results = CountingResult(
                mean=sum(valid_cyclomatic_complexity) /
                len(valid_cyclomatic_complexity) if valid_cyclomatic_complexity else -1,
                min=min(
                    valid_cyclomatic_complexity) if valid_cyclomatic_complexity else -1,
                max=max(
                    valid_cyclomatic_complexity) if valid_cyclomatic_complexity else -1,
                std=(sum((x - (sum(valid_cyclomatic_complexity) / len(valid_cyclomatic_complexity))) **
                     2 for x in valid_cyclomatic_complexity) / len(valid_cyclomatic_complexity)) ** 0.5 if valid_cyclomatic_complexity else -1,
            )

            cyclomatic_overall_results = CountingResult(
                mean=sum(overall_cyclomatic_complexity) /
                len(overall_cyclomatic_complexity) if overall_cyclomatic_complexity else -1,
                min=min(
                    overall_cyclomatic_complexity) if overall_cyclomatic_complexity else -1,
                max=max(
                    overall_cyclomatic_complexity) if overall_cyclomatic_complexity else -1,
                std=(sum((x - (sum(overall_cyclomatic_complexity) / len(overall_cyclomatic_complexity))) **
                     2 for x in overall_cyclomatic_complexity) / len(overall_cyclomatic_complexity)) ** 0.5 if overall_cyclomatic_complexity else -1,
            )


            # Token counts
            token_train_results = CountingResult(
                    mean=sum(train_token_counts) /
                    len(train_token_counts) if train_token_counts else -1,
                    min=min(train_token_counts) if train_token_counts else -1,
                    max=max(train_token_counts) if train_token_counts else -1,
                    std=(sum((x - (sum(train_token_counts) / len(train_token_counts))) **
                        2 for x in train_token_counts) / len(train_token_counts)) ** 0.5 if train_token_counts else -1,
            )
            token_test_results = CountingResult(
                    mean=sum(test_token_counts) /
                    len(test_token_counts) if test_token_counts else -1,
                    min=min(test_token_counts) if test_token_counts else -1,
                    max=max(test_token_counts) if test_token_counts else -1,
                    std=(sum((x - (sum(test_token_counts) / len(test_token_counts))) **
                        2 for x in test_token_counts) / len(test_token_counts)) ** 0.5 if test_token_counts else -1,
            )
            token_valid_results = CountingResult(
                    mean=sum(valid_token_counts) /
                    len(valid_token_counts) if valid_token_counts else -1,
                    min=min(valid_token_counts) if valid_token_counts else -1,
                    max=max(valid_token_counts) if valid_token_counts else -1,
                    std=(sum((x - (sum(valid_token_counts) / len(valid_token_counts))) **
                        2 for x in valid_token_counts) / len(valid_token_counts)) ** 0.5 if valid_token_counts else -1,
            )
            token_overall_results = CountingResult(
                    mean=sum(overall_token_counts) /
                    len(overall_token_counts) if overall_token_counts else -1,
                    min=min(overall_token_counts) if overall_token_counts else -1,
                    max=max(overall_token_counts) if overall_token_counts else -1,
                    std=(sum((x - (sum(overall_token_counts) / len(overall_token_counts))) **
                        2 for x in overall_token_counts) / len(overall_token_counts)) ** 0.5 if overall_token_counts else -1,
            )
            


        else:
            nloc_overall, cyclomatic_complexity_overall = _get_metrics(
                dataset.data or [], "Overall")
            nloc_overall_results = CountingResult(
                mean=sum(nloc_overall) /
                len(nloc_overall) if nloc_overall else -1,
                min=min(nloc_overall) if nloc_overall else -1,
                max=max(nloc_overall) if nloc_overall else -1,
                std=(sum((x - (sum(nloc_overall) / len(nloc_overall))) **
                     2 for x in nloc_overall) / len(nloc_overall)) ** 0.5 if nloc_overall else -1,
            )
            cyclomatic_overall_results = CountingResult(
                mean=sum(cyclomatic_complexity_overall) /
                len(cyclomatic_complexity_overall) if cyclomatic_complexity_overall else -1,
                min=min(
                    cyclomatic_complexity_overall) if cyclomatic_complexity_overall else -1,
                max=max(
                    cyclomatic_complexity_overall) if cyclomatic_complexity_overall else -1,
                std=(sum((x - (sum(cyclomatic_complexity_overall) / len(cyclomatic_complexity_overall))) **
                     2 for x in cyclomatic_complexity_overall) / len(cyclomatic_complexity_overall)) ** 0.5 if cyclomatic_complexity_overall else -1,
            )

        nloc_split_results = SplitStatisticalMetricsResults(
            train=nloc_train_results if dataset.has_splits() else None,
            test=nloc_test_results if dataset.has_splits() else None,
            validation=nloc_valid_results if dataset.has_splits() else None,
            overall=nloc_overall_results,
        )

        complexity_split_results = SplitStatisticalMetricsResults(
            train=cyclomatic_train_results if dataset.has_splits() else None,
            test=cyclomatic_test_results if dataset.has_splits() else None,
            validation=cyclomatic_valid_results if dataset.has_splits() else None,
            overall=cyclomatic_overall_results,
        )

    if config.analysis.structural_metrics.tokens:
        token_split_results = SplitStatisticalMetricsResults(
            train=token_train_results if dataset.has_splits() else None,
            test=token_test_results if dataset.has_splits() else None,
            validation=token_valid_results if dataset.has_splits() else None,
            overall=token_overall_results,
        )

    return StructuralMetricsResults(
        loc=nloc_split_results if config.analysis.structural_metrics.loc else None,
        cyclomatic_complexity=complexity_split_results if config.analysis.structural_metrics.cyclomatic_complexity else None,
        tokens=token_split_results if config.analysis.structural_metrics.tokens else None,
    )
