from contextlib import nullcontext as does_not_raise

import pytest
from pydantic import ValidationError

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    ReduceDataArray,
    SliceDataArray,
)


class TestDynamicModels:
    def test_slice_data_array(self):
        cls = SliceDataArray.with_constrained_dims(["abc", "xyz"])

        assert issubclass(cls, SliceDataArray)
        assert cls.__name__ == "SliceDataArray_AbcXyz"
        assert cls.model_json_schema()["title"] == "SliceDataArray_AbcXyz"

        with does_not_raise():
            _ = cls(
                func_name="SliceDataArray_AbcXyz",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                dim_slices={"abc": (0, 0), "xyz": (0, 0)},
            )

        # incorrect func_name
        with pytest.raises(ValidationError):
            _ = cls(
                func_name="SliceDataArray",
                data_array=VariableReference(name="da"),
                dim_slices={"abc": (0, 0), "xyz": (0, 0)},
            )

        # unknown dim
        with pytest.raises(ValidationError):
            _ = cls(
                func_name="SliceDataArray_AbcXyz",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                dim_slices={"abc": (0, 0), "xyz": (0, 0), "z": (0, 0)},
            )

    def test_reduce_data_array(self):
        cls = ReduceDataArray.with_constrained_dims(["x", "y"])

        assert issubclass(cls, ReduceDataArray)
        assert cls.__name__ == "ReduceDataArray_XY"
        assert cls.model_json_schema()["title"] == "ReduceDataArray_XY"

        with does_not_raise():
            _ = cls(
                func_name="ReduceDataArray_XY",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                op="max",
                dim=None,
            )
            _ = cls(
                func_name="ReduceDataArray_XY",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                op="max",
                dim="x",
            )
            _ = cls(
                func_name="ReduceDataArray_XY",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                op="max",
                dim=("x", "y"),
            )

        # incorrect func_name
        with pytest.raises(ValidationError):
            _ = cls(
                func_name="ReduceDataArray",
                data_array=VariableReference(name="da"),
                op="max",
                dim="x",
            )

        # unknown dim - str
        with pytest.raises(ValidationError):
            _ = cls(
                func_name="ReduceDataArray_XY",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                op="max",
                dim="a",
            )

        # unknown dim - tuple
        with pytest.raises(ValidationError):
            _ = cls(
                func_name="ReduceDataArray_XY",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                op="max",
                dim=("x", "a"),
            )
