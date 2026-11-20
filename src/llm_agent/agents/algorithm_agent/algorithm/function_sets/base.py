import operator
from collections.abc import Iterable
from functools import reduce
from itertools import chain, groupby
from typing import Annotated, Any, cast, get_args, get_origin

from pydantic import Field

from llm_agent.agents.algorithm_agent.algorithm.base import Function


class FunctionSet(dict[str, type[Function]]):
    """Base class for function sets.

    Stores Functions as a dict mapping function names to function models. This
    gets us deduplication (by name) for free. Deduplicating by name is
    important for Functions defined at runtime that are identical in all
    respects other than being different Python objects.
    """

    # See docs for _consolidate_generic_args() for details on how this is used.
    func_name_to_discriminators: dict[str, tuple[str, ...]]

    def __init__(
        self,
        functions: Iterable[type[Function]] | None = None,
        func_name_to_discriminators: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Constructor."""
        self.func_name_to_discriminators = func_name_to_discriminators or {}
        super().__init__((f.__name__, f) for f in functions or [])
        self._consolidate_generic_args()

    @staticmethod
    def merge(sets: "Iterable[FunctionSet]") -> "FunctionSet":
        """Merge multiple function sets into one."""
        sets = list(sets)
        functions = chain.from_iterable(fs.values() for fs in sets)
        func_name_to_discriminators = reduce(
            lambda x, y: x | y, [fs.func_name_to_discriminators for fs in sets]
        )
        return FunctionSet(functions, func_name_to_discriminators)

    @property
    def functions(self) -> list[type[Function]]:
        """List of constituent functions."""
        return list(self.values())

    def _consolidate_generic_args(self) -> None:
        """Consolidate Func[S1, T1] and Func[S2, T2] into Func[S1 | S2, T1 | T2] etc.

        This is motivated by MakePlot and MakeAnimation which both take type params that are
        unions of supported PlotSpec types. So, e.g., you might have differently parameterized
        versions of MakePlot from different function sets like so:
        - MakePlot[LinePlotSpec | BarPlotSpec | RasterPlotSpec]
        - MakePlot[SSTPlotSpec | SSTAnomalyPlotSpec]
        - MakePlot[WindSpeedPlotSpec]
        and so on.

        This method will essentially consolidate this into:
        MakePlot[
            LinePlotSpec |
            BarPlotSpec |
            RasterPlotSpec |
            SSTPlotSpec |
            SSTAnomalyPlotSpec |
            WindSpeedPlotSpec
        ]

        An added detail is that, if type params are Pydantic models, we want this union to be a
        Pydantic discriminated union for easier deserialization, so we in fact create:
        MakePlot[Annotate[LinePlotSpec | ..., Field(discriminator="spec_name")]]
        Rather than try to automatically figure out the right discriminator, this class simply
        allows users to specify them via func_name_to_discriminators.

        Lastly, while MakePlot and MakeAnimation accept single type params, this method can in fact
        handle Functions with multiple type params like MyFunc[T1, T2, T3, ...]. The types in each
        position are treated the same way.
        """
        normal_funcs = {k: v for k, v in self.items() if "[" not in k}
        generic_funcs = {k: v for k, v in self.items() if "[" in k}

        # Mapping from e.g. the name "MakePlot" to the MakePlot root (unparameterized) class
        func_name_to_root_class = {
            k.split("[")[0]: cast("type[Function]", v.__pydantic_generic_metadata__["origin"])
            for k, v in self.items()
            if "[" in k
        }
        # Mapping from e.g. the name "MakePlot" to [MakePlot[T1 | T2], MakePlot[T3], ...]
        func_name_to_parameterized_classes = {
            k: [_v for _, _v in g]
            for k, g in groupby(sorted(generic_funcs.items()), key=lambda kv: kv[0].split("[")[0])
        }

        consolidated_funcs = {
            func_name: _consolidate_single_function(
                func_root_class=func_name_to_root_class[func_name],
                func_parameterized_classes=func_parameterized_classes,
                discriminators=self.func_name_to_discriminators.get(func_name),
            )
            for func_name, func_parameterized_classes in func_name_to_parameterized_classes.items()
        }

        self.clear()
        self.update(normal_funcs)
        self.update(consolidated_funcs)


def _consolidate_single_function(
    func_root_class: type[Function],
    func_parameterized_classes: list[type[Function]],
    discriminators: tuple[str, ...] | None = None,
) -> type[Function]:
    # If func_parameterized_classes == [MyFunc[T1 | T2, S1], MyFunc[T3 | T4, S2]],
    # then type_arg_tuples_for_each_class == [(T1 | T2, S1), (T3 | T4, S2)]
    type_arg_tuples_for_each_class = [
        t.__pydantic_generic_metadata__["args"] for t in func_parameterized_classes
    ]
    # => type_args_for_each_position_across_classes == [[T1 | T2, T3], [S1, S2]]
    type_args_for_each_position_across_classes = list(
        zip(*type_arg_tuples_for_each_class, strict=True)
    )
    # => merged_args_for_each_position == [T1 | T2 | T3, S1 | S2]
    merged_args_for_each_position = [
        _merge_type_args(args) for args in type_args_for_each_position_across_classes
    ]
    if discriminators:
        # Convert to a Pydantic discriminated union
        merged_args_for_each_position = [
            Annotated[args, Field(discriminator=discriminator)]
            for args, discriminator in zip(
                merged_args_for_each_position, discriminators, strict=True
            )
        ]
    # => MyFunc[T1 | T2 | T3, S1 | S2]
    new_func = func_root_class[*merged_args_for_each_position]
    return new_func


def _merge_type_args(type_args: tuple[type[Any], ...]) -> type[Any]:
    def _unwrap_if_annotated(arg: type[Any]) -> type[Any]:
        return get_args(arg)[0] if get_origin(arg) == Annotated else arg

    type_args_unwrapped = [_unwrap_if_annotated(arg) for arg in type_args]
    return reduce(operator.or_, type_args_unwrapped)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
