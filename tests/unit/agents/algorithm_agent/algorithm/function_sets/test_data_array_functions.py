from contextlib import nullcontext as does_not_raise

import pytest
from pydantic import ValidationError

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    CorrelateDataArrays,
    CumulativeReduce,
    ReduceDataArray,
    RollingReduce,
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

    def test_reduce_data_array_quantile_validation(self):
        # median and quantile are valid ops
        with does_not_raise():
            _ = ReduceDataArray(
                func_name="ReduceDataArray",
                data_array=VariableReference(name="da"),
                op="median",
                dim="x",
            )
            _ = ReduceDataArray(
                func_name="ReduceDataArray",
                data_array=VariableReference(name="da"),
                op="quantile",
                dim="x",
                q=0.9,
            )
            # q boundaries are inclusive
            _ = ReduceDataArray(
                func_name="ReduceDataArray",
                data_array=VariableReference(name="da"),
                op="quantile",
                dim="x",
                q=0.0,
            )
            _ = ReduceDataArray(
                func_name="ReduceDataArray",
                data_array=VariableReference(name="da"),
                op="quantile",
                dim="x",
                q=1.0,
            )

        # quantile without q is invalid
        with pytest.raises(ValidationError):
            _ = ReduceDataArray(
                func_name="ReduceDataArray",
                data_array=VariableReference(name="da"),
                op="quantile",
                dim="x",
            )

        # q out of [0, 1] is invalid
        with pytest.raises(ValidationError):
            _ = ReduceDataArray(
                func_name="ReduceDataArray",
                data_array=VariableReference(name="da"),
                op="quantile",
                dim="x",
                q=1.5,
            )

        # q set for a non-quantile op is invalid
        with pytest.raises(ValidationError):
            _ = ReduceDataArray(
                func_name="ReduceDataArray",
                data_array=VariableReference(name="da"),
                op="mean",
                dim="x",
                q=0.5,
            )

    def test_rolling_reduce(self):
        cls = RollingReduce.with_constrained_dims(["x", "y"])

        assert issubclass(cls, RollingReduce)
        assert cls.__name__ == "RollingReduce_XY"
        assert cls.model_json_schema()["title"] == "RollingReduce_XY"

        with does_not_raise():
            _ = cls(
                func_name="RollingReduce_XY",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                dim="x",
                window=3,
                op="mean",
            )

        # unknown dim
        with pytest.raises(ValidationError):
            _ = cls(
                func_name="RollingReduce_XY",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                dim="a",
                window=3,
                op="mean",
            )

        # window must be >= 1
        with pytest.raises(ValidationError):
            _ = cls(
                func_name="RollingReduce_XY",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                dim="x",
                window=0,
                op="mean",
            )

    def test_cumulative_reduce(self):
        cls = CumulativeReduce.with_constrained_dims(["x", "y"])

        assert issubclass(cls, CumulativeReduce)
        assert cls.__name__ == "CumulativeReduce_XY"
        assert cls.model_json_schema()["title"] == "CumulativeReduce_XY"

        with does_not_raise():
            _ = cls(
                func_name="CumulativeReduce_XY",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                dim="x",
                op="sum",
            )

        # unknown dim
        with pytest.raises(ValidationError):
            _ = cls(
                func_name="CumulativeReduce_XY",  # type: ignore[reportArgumentType]
                data_array=VariableReference(name="da"),
                dim="a",
                op="sum",
            )

    def test_correlate_data_arrays(self):
        cls = CorrelateDataArrays.with_constrained_dims(["x", "y"])

        assert issubclass(cls, CorrelateDataArrays)
        assert cls.__name__ == "CorrelateDataArrays_XY"
        assert cls.model_json_schema()["title"] == "CorrelateDataArrays_XY"

        with does_not_raise():
            _ = cls(
                func_name="CorrelateDataArrays_XY",  # type: ignore[reportArgumentType]
                a=VariableReference(name="a"),
                b=VariableReference(name="b"),
                dim="x",
            )

        # unknown dim
        with pytest.raises(ValidationError):
            _ = cls(
                func_name="CorrelateDataArrays_XY",  # type: ignore[reportArgumentType]
                a=VariableReference(name="a"),
                b=VariableReference(name="b"),
                dim="z",
            )
