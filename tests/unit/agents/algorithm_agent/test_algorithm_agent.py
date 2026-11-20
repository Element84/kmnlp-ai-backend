from typing import Any
from unittest.mock import patch

import pytest
from pydantic_ai.models.test import TestModel
from shapely.geometry import box

from llm_agent.agents.algorithm_agent.algorithm.base import make_function_call_model
from llm_agent.agents.algorithm_agent.algorithm.exec.base import ExecutionConfig, execute_algorithm
from llm_agent.agents.algorithm_agent.algorithm_agent import AlgorithmAgent
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmProblemSpec,
    GeometryVariable,
    input_vars_to_function_set,
)
from llm_agent.context import ToolCallContext
from tests.unit.agents.algorithm_agent.utils import make_test_data_array_spatiotemporal, test_params
from tests.utils.mocks.context import MockChatContext


@pytest.mark.parametrize(
    ("problem", "dummy_solution"),
    [(params["problem"], params["dummy_solution"]) for params in test_params.values()],
    ids=test_params.keys(),
)
@pytest.mark.asyncio
async def test_solution(
    chat_context: MockChatContext,
    problem: AlgorithmProblemSpec,
    dummy_solution: dict[str, Any],
) -> None:
    """Test whether the solution runs without errors."""
    function_set = input_vars_to_function_set(problem.inputs_provided)
    function_call_model = make_function_call_model(function_set.functions)

    model = TestModel(custom_output_args=dummy_solution)
    agent = AlgorithmAgent(model, problem=problem, function_call_type=function_call_model)
    solution, _ = await agent.run(chat_context)

    sst = make_test_data_array_spatiotemporal(start_date="2024-01-01", num_time_steps=365)

    tool_call_context = ToolCallContext(tool_call_id="", chat_context=chat_context)

    # Geometries are resolved upstream and passed to this agent as inputs. Provide a
    # fixed geometry for each geometry input the problem declares. Patch cartopy's
    # feature rendering so plot generation doesn't reach out to the network.
    input_vars: dict[str, Any] = {"sst": sst}
    for input_var in problem.inputs_provided:
        if isinstance(input_var, GeometryVariable):
            input_vars[input_var.name] = box(-10, -10, 10, 10)

    with (
        patch("cartopy.mpl.geoaxes.GeoAxes.add_feature"),
        patch("cartopy.mpl.geoaxes.GeoAxes.coastlines"),
        patch("cartopy.mpl.geoaxes.GeoAxes.stock_img"),
    ):
        await execute_algorithm(
            tool_call_context,
            solution.algorithm,
            input_vars=input_vars,
            exec_config=ExecutionConfig(),
        )
