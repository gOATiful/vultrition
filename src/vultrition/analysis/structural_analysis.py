import lizard
import tiktoken
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor

from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

from vultrition.models import config
from vultrition.models.dataset import Dataset, Sample
from vultrition.models.results import CountingResult, SplitNumbericalMetricsResults, SplitStatisticalMetricsResults, StructuralMetricsResults


LANGUAGE_TO_FILENAME = {
    "python": "sample.py",
    "java": "Sample.java",
    "c": "sample.c",
    "cpp": "sample.cpp",
}

C_CPP_PREPROCESSOR_DIRECTIVES = {
    "define",
    "undef",
    "include",
    "if",
    "ifdef",
    "ifndef",
    "elif",
    "elifdef",
    "elifndef",
    "else",
    "endif",
    "line",
    "error",
    "warning",
    "pragma",
    "embed",
}


def has_preprocessor_directives(source: str) -> bool:
    lines = source.splitlines()
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("#"):
            directive = stripped_line[1:].split()[0] if len(
                stripped_line) > 1 else ""
            if directive in C_CPP_PREPROCESSOR_DIRECTIVES:
                return True
    return False


def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def _calc_std(values: list[int], mean: float) -> float:
    if not values:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def detect_language_pygments(source: str) -> str:
    try:
        lexer = guess_lexer(source)
    except ClassNotFound:
        return "unknown"

    aliases = set(lexer.aliases)
    name = lexer.name.lower()

    if "python" in aliases:
        return "python"
    if "java" in aliases:
        return "java"
    if "c++" in aliases or "cpp" in aliases:
        return "cpp"
    if "c" in aliases and "c++" not in name:
        return "c"

    return "unknown"


def _get_metrics(samples: list[Sample], label: str, count_tokens: bool = True, count_structural: bool = True) -> tuple[list[int], list[int], list[int]]:
    if not samples:
        return [], [], [], 0

    loc_values = []
    cyclomatic_complexity_values = []
    token_counts = []
    cnt_preprocessor_directives = 0
    for sample in tqdm(samples, desc=f"Analyzing Structural Metrics for {label}"):
        try:
            if count_structural:
                source = sample.function
                language = detect_language_pygments(source)

                if language == "unknown":
                    language = "cpp"  # Default to C++ if language detection fails

                filename = LANGUAGE_TO_FILENAME.get(language, "sample.cpp")

                analysis = lizard.analyze_file.analyze_source_code(
                    filename, source)
                if analysis and len(analysis.function_list) > 0:
                    # Be careful: function_list[0] may not be the function you expect
                    func = analysis.function_list[0]
                    loc_values.append(func.nloc)
                    if language in {"c", "cpp"} and not has_preprocessor_directives(source):
                        cyclomatic_complexity_values.append(
                            func.cyclomatic_complexity)
                    else:
                        cnt_preprocessor_directives += 1
            if count_tokens:
                token_counts.append(_count_tokens(sample.function))
        except Exception as exc:
            print(
                f"Warning: Failed to analyze function for Structural Metrics: {exc}")

    return loc_values, cyclomatic_complexity_values, token_counts, cnt_preprocessor_directives





def _analyze_sample_metrics(args):
    source, count_tokens, count_structural = args

    loc = None
    cyclomatic_complexity = None
    token_count = None
    cnt_preprocessor_directives = 0
    warning = None

    try:
        if count_structural:
            language = detect_language_pygments(source)

            if language == "unknown":
                language = "cpp"

            filename = LANGUAGE_TO_FILENAME.get(language, "sample.cpp")

            analysis = lizard.analyze_file.analyze_source_code(filename, source)

            if analysis and len(analysis.function_list) > 0:
                func = analysis.function_list[0]
                loc = func.nloc

                if language in {"c", "cpp"} and not has_preprocessor_directives(source):
                    cyclomatic_complexity = func.cyclomatic_complexity
                else:
                    cnt_preprocessor_directives = 1

        if count_tokens:
            token_count = _count_tokens(source)

    except Exception as exc:
        warning = str(exc)

    return loc, cyclomatic_complexity, token_count, cnt_preprocessor_directives, warning


def _get_metrics_mpu(
    samples: list[Sample],
    label: str,
    count_tokens: bool = True,
    count_structural: bool = True,
) -> tuple[list[int], list[int], list[int], int]:
    if not samples:
        return [], [], [], 0

    loc_values = []
    cyclomatic_complexity_values = []
    token_counts = []
    cnt_preprocessor_directives = 0

    sources = [
        (sample.function, count_tokens, count_structural)
        for sample in samples
    ]

    max_workers = os.cpu_count() or 1
    chunksize = max(1, len(sources) // (max_workers * 4))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            _analyze_sample_metrics,
            sources,
            chunksize=chunksize,
        )

        for loc, complexity, tokens, preproc_count, warning in tqdm(
            results,
            total=len(sources),
            desc=f"Analyzing Structural Metrics for {label}",
        ):
            if warning is not None:
                print(
                    f"Warning: Failed to analyze function for Structural Metrics: {warning}"
                )
                continue

            if loc is not None:
                loc_values.append(loc)

            if complexity is not None:
                cyclomatic_complexity_values.append(complexity)

            if tokens is not None:
                token_counts.append(tokens)

            cnt_preprocessor_directives += preproc_count

    return (
        loc_values,
        cyclomatic_complexity_values,
        token_counts,
        cnt_preprocessor_directives,
    )



def analyze_structural_metrics(config: config, dataset: Dataset) -> StructuralMetricsResults:
    print("Analyzing structural metrics...")

    if config.analysis.structural_metrics.loc or config.analysis.structural_metrics.cyclomatic_complexity or config.analysis.structural_metrics.tokens:
        count_tokens = config.analysis.structural_metrics.tokens
        count_structural = config.analysis.structural_metrics.loc or config.analysis.structural_metrics.cyclomatic_complexity
        if dataset.has_splits():
            train_nlocs, train_cyclomatic_complexity, train_token_counts, train_cnt_preprocessor_directives = _get_metrics_mpu(
                dataset.train or [], "Train", count_tokens=count_tokens, count_structural=count_structural)
            test_nlocs, test_cyclomatic_complexity, test_token_counts, test_cnt_preprocessor_directives = _get_metrics_mpu(
                dataset.test or [], "Test", count_tokens=count_tokens, count_structural=count_structural)
            valid_nlocs, valid_cyclomatic_complexity, valid_token_counts, valid_cnt_preprocessor_directives = _get_metrics_mpu(
                dataset.validation or [], "Validation", count_tokens=count_tokens, count_structural=count_structural)
            overall_nlocs, overall_cyclomatic_complexity, overall_token_counts = [*train_nlocs, *test_nlocs, *valid_nlocs], [
                *train_cyclomatic_complexity, *test_cyclomatic_complexity, *valid_cyclomatic_complexity], [*train_token_counts, *test_token_counts, *valid_token_counts]

            nloc_train_results = CountingResult(
                mean=sum(train_nlocs) /
                len(train_nlocs) if train_nlocs else -1,
                min=min(train_nlocs) if train_nlocs else -1,
                max=max(train_nlocs) if train_nlocs else -1,
                std=_calc_std(train_nlocs, sum(train_nlocs) /
                              len(train_nlocs) if train_nlocs else 0)
            )
            nloc_train_results = CountingResult(
                mean=sum(test_nlocs) /
                len(test_nlocs) if test_nlocs else -1,
                min=min(test_nlocs) if test_nlocs else -1,
                max=max(test_nlocs) if test_nlocs else -1,
                std=_calc_std(test_nlocs, sum(test_nlocs) /
                              len(test_nlocs) if test_nlocs else 0)
            )

            nloc_test_results = CountingResult(
                mean=sum(test_nlocs) /
                len(test_nlocs) if test_nlocs else -1,
                min=min(test_nlocs) if test_nlocs else -1,
                max=max(test_nlocs) if test_nlocs else -1,
                std=_calc_std(test_nlocs, sum(test_nlocs) /
                              len(test_nlocs) if test_nlocs else 0)
            )

            nloc_valid_results = CountingResult(
                mean=sum(valid_nlocs) /
                len(valid_nlocs) if valid_nlocs else -1,
                min=min(valid_nlocs) if valid_nlocs else -1,
                max=max(valid_nlocs) if valid_nlocs else -1,
                std=_calc_std(valid_nlocs, sum(valid_nlocs) /
                              len(valid_nlocs) if valid_nlocs else 0)
            )
            nloc_overall_results = CountingResult(
                mean=sum(overall_nlocs) /
                len(overall_nlocs) if overall_nlocs else -1,
                min=min(overall_nlocs) if overall_nlocs else -1,
                max=max(overall_nlocs) if overall_nlocs else -1,
                std=_calc_std(overall_nlocs, sum(overall_nlocs) /
                              len(overall_nlocs) if overall_nlocs else 0)
            )
            print("Evaluated LOC metrics. train_nloc_results: ")
            # Complexity
            cyclomatic_train_results = CountingResult(
                mean=sum(train_cyclomatic_complexity) /
                len(train_cyclomatic_complexity) if train_cyclomatic_complexity else -1,
                min=min(
                    train_cyclomatic_complexity) if train_cyclomatic_complexity else -1,
                max=max(
                    train_cyclomatic_complexity) if train_cyclomatic_complexity else -1,
                std=_calc_std(train_cyclomatic_complexity, sum(train_cyclomatic_complexity) / len(
                    train_cyclomatic_complexity) if train_cyclomatic_complexity else 0)
            )
            print("Evaluated Cyclomatic Complexity metrics. train_cyclomatic_results: ")
            cyclomatic_test_results = CountingResult(
                mean=sum(test_cyclomatic_complexity) /
                len(test_cyclomatic_complexity) if test_cyclomatic_complexity else -1,
                min=min(
                    test_cyclomatic_complexity) if test_cyclomatic_complexity else -1,
                max=max(
                    test_cyclomatic_complexity) if test_cyclomatic_complexity else -1,
                std=_calc_std(test_cyclomatic_complexity, sum(test_cyclomatic_complexity) / len(
                    test_cyclomatic_complexity) if test_cyclomatic_complexity else 0)
            )
            print("Evaluated Cyclomatic Complexity metrics. test_cyclomatic_results: ")
            cyclomatic_valid_results = CountingResult(
                mean=sum(valid_cyclomatic_complexity) /
                len(valid_cyclomatic_complexity) if valid_cyclomatic_complexity else -1,
                min=min(
                    valid_cyclomatic_complexity) if valid_cyclomatic_complexity else -1,
                max=max(
                    valid_cyclomatic_complexity) if valid_cyclomatic_complexity else -1,
                std=_calc_std(valid_cyclomatic_complexity, sum(valid_cyclomatic_complexity) / len(
                    valid_cyclomatic_complexity) if valid_cyclomatic_complexity else 0)
            )
            print("Evaluated Cyclomatic Complexity metrics. overall_cyclomatic_results: ")
            cyclomatic_overall_results = CountingResult(
                mean=sum(overall_cyclomatic_complexity) /
                len(overall_cyclomatic_complexity) if overall_cyclomatic_complexity else -1,
                min=min(
                    overall_cyclomatic_complexity) if overall_cyclomatic_complexity else -1,
                max=max(
                    overall_cyclomatic_complexity) if overall_cyclomatic_complexity else -1,
                std=_calc_std(overall_cyclomatic_complexity, sum(overall_cyclomatic_complexity) / len(
                    overall_cyclomatic_complexity) if overall_cyclomatic_complexity else 0)
            )

            # Token counts
            print(
                "Evaluated token count metrics. Calculating token count results for train...")
            token_train_results = CountingResult(
                mean=sum(train_token_counts) /
                len(train_token_counts) if train_token_counts else -1,
                min=min(train_token_counts) if train_token_counts else -1,
                max=max(train_token_counts) if train_token_counts else -1,
                std=_calc_std(train_token_counts, sum(
                    train_token_counts) / len(train_token_counts) if train_token_counts else 0)
            )
            print(
                "Evaluated token count metrics. Calculating token count results for test...")
            token_test_results = CountingResult(
                mean=sum(test_token_counts) /
                len(test_token_counts) if test_token_counts else -1,
                min=min(test_token_counts) if test_token_counts else -1,
                max=max(test_token_counts) if test_token_counts else -1,
                std=_calc_std(test_token_counts, sum(
                    test_token_counts) / len(test_token_counts) if test_token_counts else 0)
            )
            print(
                "Evaluated token count metrics. Calculating token count results for validation...")
            token_valid_results = CountingResult(
                mean=sum(valid_token_counts) /
                len(valid_token_counts) if valid_token_counts else -1,
                min=min(valid_token_counts) if valid_token_counts else -1,
                max=max(valid_token_counts) if valid_token_counts else -1,
                std=_calc_std(valid_token_counts, sum(
                    valid_token_counts) / len(valid_token_counts) if valid_token_counts else 0),
            )
            token_overall_results = CountingResult(
                mean=sum(overall_token_counts) /
                len(overall_token_counts) if overall_token_counts else -1,
                min=min(overall_token_counts) if overall_token_counts else -1,
                max=max(overall_token_counts) if overall_token_counts else -1,
                std=_calc_std(overall_token_counts, sum(
                    overall_token_counts) / len(overall_token_counts) if overall_token_counts else 0),
            )

            preprocessor_directive_overall_results = SplitNumbericalMetricsResults(
                train=train_cnt_preprocessor_directives if dataset.has_splits() else -1,
                test=test_cnt_preprocessor_directives if dataset.has_splits() else -1,
                validation=valid_cnt_preprocessor_directives if dataset.has_splits() else -1,
                overall=train_cnt_preprocessor_directives + test_cnt_preprocessor_directives +
                valid_cnt_preprocessor_directives if dataset.has_splits() else -1
            )

        else:
            nloc_overall, cyclomatic_complexity_overall, token_overall, cnt_preprocessor_directives = _get_metrics_mpu(
                dataset.data, "Overall")
            nloc_overall_results = CountingResult(
                mean=sum(nloc_overall) /
                len(nloc_overall) if nloc_overall else -1,
                min=min(nloc_overall) if nloc_overall else -1,
                max=max(nloc_overall) if nloc_overall else -1,
                std=_calc_std(nloc_overall, sum(nloc_overall) /
                              len(nloc_overall) if nloc_overall else 0)
            )
            cyclomatic_overall_results = CountingResult(
                mean=sum(cyclomatic_complexity_overall) /
                len(cyclomatic_complexity_overall) if cyclomatic_complexity_overall else -1,
                min=min(
                    cyclomatic_complexity_overall) if cyclomatic_complexity_overall else -1,
                max=max(
                    cyclomatic_complexity_overall) if cyclomatic_complexity_overall else -1,
                std=_calc_std(cyclomatic_complexity_overall, sum(cyclomatic_complexity_overall) / len(
                    cyclomatic_complexity_overall) if cyclomatic_complexity_overall else 0)
            )
            token_overall_results = CountingResult(
                mean=sum(token_overall) /
                len(token_overall) if token_overall else -1,
                min=min(token_overall) if token_overall else -1,
                max=max(token_overall) if token_overall else -1,
                std=_calc_std(token_overall, sum(token_overall) /
                              len(token_overall) if token_overall else 0),
            )
            preprocessor_directive_overall_results = SplitNumbericalMetricsResults(
                train=-1,
                test=-1,
                validation=-1,
                overall=cnt_preprocessor_directives
            )

        print("Evaluated LOC and Cyclomatic Complexity metrics.")
        nloc_split_results = SplitStatisticalMetricsResults(
            train=nloc_train_results if dataset.has_splits() else None,
            test=nloc_test_results if dataset.has_splits() else None,
            validation=nloc_valid_results if dataset.has_splits() else None,
            overall=nloc_overall_results,
        )
        print("Evaluated LOC metrics.")
        complexity_split_results = SplitStatisticalMetricsResults(
            train=cyclomatic_train_results if dataset.has_splits() else None,
            test=cyclomatic_test_results if dataset.has_splits() else None,
            validation=cyclomatic_valid_results if dataset.has_splits() else None,
            overall=cyclomatic_overall_results,
        )

    if config.analysis.structural_metrics.tokens:
        print("Evaluating token count metrics...")
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
        preprocessor_directives=preprocessor_directive_overall_results
    )
