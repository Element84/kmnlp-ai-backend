from llm_agent.agents.algorithm_agent.algorithm.function_sets.base import (
    FunctionSet,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.data_array_functions import (
    GeneralDataArrayFunctionSet,
    ReduceDataArray,
    SliceDataArray,
    SpatioTemporalDataArrayFunctionSet,
    TemporalDataArrayFunctionSet,
)


class TestFunctionSet:
    def test_merge(self):
        fs = FunctionSet.merge(
            [
                GeneralDataArrayFunctionSet(["lat", "lon"]),
                TemporalDataArrayFunctionSet(["time", "wavelength"]),
                SpatioTemporalDataArrayFunctionSet(),
            ]
        )
        slice_funcs = {k for k, v in fs.items() if issubclass(v, SliceDataArray)}
        reduce_funcs = {k for k, v in fs.items() if issubclass(v, ReduceDataArray)}
        assert slice_funcs == {
            "SliceDataArray_LatLon",
            "SliceDataArray_TimeWavelength",
            "SliceDataArray_LatLonTime",
        }
        assert reduce_funcs == {
            "ReduceDataArray_LatLon",
            "ReduceDataArray_TimeWavelength",
            "ReduceDataArray_LatLonTime",
        }

    def test_deduplication(self):
        fs = FunctionSet(
            [
                SliceDataArray.with_constrained_dims(["time", "wavelength"]),
                SliceDataArray.with_constrained_dims(["time", "wavelength"]),
                SliceDataArray.with_constrained_dims(["time", "xyz"]),
            ]
        )
        slice_funcs = {k for k, v in fs.items() if issubclass(v, SliceDataArray)}
        assert slice_funcs == {
            "SliceDataArray_TimeWavelength",
            "SliceDataArray_TimeXyz",
        }
