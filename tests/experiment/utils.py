import statistics
import subprocess
from dataclasses import dataclass

from langfuse.experiment import ExperimentItemResult, ExperimentResult
from scipy import stats  # type:ignore[reportMissingTypeStubs]


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:  # noqa: BLE001
        return "unknown"


def get_git_dirty() -> bool:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)  # noqa: S607
        return bool(status.strip())
    except Exception:  # noqa: BLE001
        return True


def get_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def extract_item_result_scores_by_name(
    item_results: list[ExperimentItemResult], name: str
) -> list[float]:
    scores: list[float] = [
        float(evaluation.value)
        for item_result in item_results
        for evaluation in item_result.evaluations
        if evaluation.name == name
        if evaluation.value is not None
    ]
    return scores


def extract_run_evaluator_scores_by_name(results: list[ExperimentResult], name: str) -> list[float]:
    return [
        float(run_evaluation.value)
        for result in results
        for run_evaluation in result.run_evaluations
        if run_evaluation.name == name
        if run_evaluation.value is not None
    ]


@dataclass
class MeanAndTCI:
    mean: float
    ci_low: float
    ci_high: float
    n: int


def mean_and_t_ci(values: list[float], confidence: float = 0.95) -> MeanAndTCI:
    """Mean and a t-distribution based confidence interval."""
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        # Can't estimate spread from a single run.
        return MeanAndTCI(mean=mean, ci_low=mean, ci_high=mean, n=n)

    stdev = statistics.stdev(values)
    std_err_mean = stdev / (n**0.5)
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)  # type: ignore[reportUnknownMemberType]
    margin = t_crit * std_err_mean
    return MeanAndTCI(mean=mean, ci_low=mean - margin, ci_high=mean + margin, n=n)
