from collections.abc import Iterable
from itertools import chain

from llm_agent.agents.algorithm_agent.algorithm.base import Function


class FunctionSet(dict[str, type[Function]]):
    """Base class for function sets.

    Stores Functions as a dict mapping function names to function models. This
    gets us deduplication (by name) for free. Deduplicating by name is
    important for Functions defined at runtime that are identical in all
    respects other than being different Python objects.
    """

    def __init__(self, functions: Iterable[type[Function]] | None = None) -> None:
        """Constructor."""
        functions = functions or []
        super().__init__((f.__name__, f) for f in functions)

    @staticmethod
    def merge(sets: "Iterable[FunctionSet]") -> "FunctionSet":
        """Merge multiple function sets into one."""
        return FunctionSet(chain.from_iterable(fs.values() for fs in sets))

    @property
    def functions(self) -> list[type[Function]]:
        """List of constituent functions."""
        return list(self.values())
