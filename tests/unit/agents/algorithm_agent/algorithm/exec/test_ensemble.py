import numpy as np
import pytest
import xarray as xr

from llm_agent.agents.algorithm_agent.algorithm.base import VariableReference
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    ExecutionConfig,
    ExecutionContext,
    eval_expression,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import DataArrayShapeError
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    AnnotationSpec,
    MakePlot,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.ensemble import (
    EnsemblePlumePlotSpec,
)
from llm_agent.context import ToolCallContext
from llm_agent.tools.data_analysis_tool.types import ImageRef
from tests.utils.mocks.context import MockChatContext

default_exec_config = ExecutionConfig()


@pytest.fixture
def context(chat_context: MockChatContext) -> ToolCallContext:
    return ToolCallContext(tool_call_id="test", chat_context=chat_context)


def _make_ensemble(nmembers: int = 15, nlead: int = 12, seed: int = 42) -> xr.DataArray:
    """2D ensemble forecast: (ensemble_member, lead_time)."""
    rng = np.random.default_rng(seed)
    trend = np.linspace(0, 5, nlead)
    spread = rng.normal(0, 1, size=(nmembers, nlead))
    return xr.DataArray(
        trend[None, :] + spread,
        dims=["ensemble_member", "lead_time"],
        coords={
            "ensemble_member": np.arange(nmembers),
            "lead_time": np.arange(nlead),
        },
    )


@pytest.mark.parametrize("band_type", ["none", "minmax", "iqr", "p10_p90"])
async def test_make_ensemble_plume_plot(context: ToolCallContext, band_type: str) -> None:
    da = _make_ensemble()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=EnsemblePlumePlotSpec(
            data_array=VariableReference(name="da"),
            band_type=band_type,  # pyright: ignore[reportArgumentType]
            annotation=AnnotationSpec(title="Ensemble", xlabel="lead", ylabel="value"),
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_make_ensemble_plume_plot_timedelta_lead_time(context: ToolCallContext) -> None:
    # Forecast lead_time is a timedelta64 coordinate; the plume must handle it (matplotlib cannot
    # mix timedelta64 with float data inside fill_between).
    lead = np.arange(0, 12).astype("timedelta64[h]").astype("timedelta64[ns]")
    da = _make_ensemble().assign_coords(lead_time=lead)  # pyright: ignore[reportUnknownMemberType]
    assert np.issubdtype(da["lead_time"].dtype, np.timedelta64)
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=EnsemblePlumePlotSpec(
            data_array=VariableReference(name="da"),
            band_type="p10_p90",
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_make_ensemble_plume_plot_no_members(context: ToolCallContext) -> None:
    da = _make_ensemble()
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=EnsemblePlumePlotSpec(
            data_array=VariableReference(name="da"),
            show_members=False,
            show_mean=True,
        ),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_make_ensemble_plume_plot_ignores_nans(context: ToolCallContext) -> None:
    da = _make_ensemble()
    da = da.where(da > -0.5)  # introduce NaNs across members
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=EnsemblePlumePlotSpec(data_array=VariableReference(name="da")),
    )
    result = await eval_expression(expr, exec_ctx)
    assert isinstance(result, ImageRef)


async def test_make_ensemble_plume_plot_missing_ensemble_dim_raises(
    context: ToolCallContext,
) -> None:
    da = xr.DataArray(np.arange(12.0), dims=["lead_time"], coords={"lead_time": np.arange(12)})
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=EnsemblePlumePlotSpec(data_array=VariableReference(name="da")),
    )
    with pytest.raises(DataArrayShapeError, match="ensemble"):
        await eval_expression(expr, exec_ctx)


async def test_make_ensemble_plume_plot_too_many_dims_raises(context: ToolCallContext) -> None:
    da = xr.DataArray(
        np.zeros((3, 4, 5)),
        dims=["ensemble_member", "lead_time", "lat"],
        coords={
            "ensemble_member": np.arange(3),
            "lead_time": np.arange(4),
            "lat": np.arange(5),
        },
    )
    exec_ctx = ExecutionContext(context, input_vars={"da": da}, config=default_exec_config)
    expr = MakePlot(
        func_name="MakePlot",
        plot_spec=EnsemblePlumePlotSpec(data_array=VariableReference(name="da")),
    )
    with pytest.raises(DataArrayShapeError):
        await eval_expression(expr, exec_ctx)
