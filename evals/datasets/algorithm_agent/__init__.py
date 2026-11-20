"""Algorithm-agent eval dataset, assembled from per-tier case modules.

The dataset is split across sibling modules in this package (`existing`, `refusals`, `easy`,
`medium`, `hard`) with shared setup in `common`. The module-level `dataset` is the public entry.
"""

from collections.abc import Callable

from llm_agent.agents.algorithm_agent.algorithm.base import BaseFunctionCall
from llm_agent.evaluation.algorithm_agent import AlgorithmAgentEvalInput
from llm_agent.evaluation.base_eval import EvalDataset, EvalInput

from .easy import CASES as EASY_CASES
from .hard import CASES as HARD_CASES
from .medium import CASES as MEDIUM_CASES
from .old import CASES as OLD_CASES
from .refusals import CASES as REFUSAL_CASES

_CASES: list[Callable[[], AlgorithmAgentEvalInput[BaseFunctionCall]]] = [
    *REFUSAL_CASES,
    *OLD_CASES,
    *EASY_CASES,
    *MEDIUM_CASES,
    *HARD_CASES,
]

# A loose bounding-box slice before a polygon filter is a valid optimization, so allow one extra
# spatial slice per polygon filter op in a case's reference solution.
_ALLOWED_EXTRA_OP_PER_POLYGON_FILTER = "SliceDataArray_LatLonTime"


def _build_item(case: Callable[[], AlgorithmAgentEvalInput[BaseFunctionCall]]) -> EvalInput:
    inp = case()
    num_polygon_filters = max(
        (
            sum(
                isinstance(s.value, BaseFunctionCall)
                and s.value.func.func_name == "FilterDataArrayByPolygon"
                for s in sol.algorithm.statements
            )
            for sol in inp.reference_solutions
        ),
        default=0,
    )
    if num_polygon_filters:
        inp.allowed_extra_ops = [_ALLOWED_EXTRA_OP_PER_POLYGON_FILTER] * num_polygon_filters
    return inp.to_eval_input()


dataset = EvalDataset(
    id="default",
    version="6",
    items=[_build_item(case) for case in _CASES],
)
