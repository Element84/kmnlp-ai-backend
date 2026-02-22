from typing import Any

import xarray as xr


def transform_execute_algorithm_input_vars(input_vars: dict[str, Any]) -> dict[str, Any]:
    """Customize captured input for Langfuse.

    `LLMAgentContext` and `xr.DataArray` serialize to very large things.
    """

    def _transform_value(value: Any) -> Any:  # noqa: ANN401
        if isinstance(value, xr.DataArray):
            return {
                "type": "xr.DataArray",
                "name": value.name,
                "dims": value.dims,
            }
        return value

    return {k: _transform_value(v) for k, v in input_vars.items()}
