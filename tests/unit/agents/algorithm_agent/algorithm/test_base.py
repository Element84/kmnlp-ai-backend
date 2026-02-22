from contextlib import nullcontext as does_not_raise
from typing import Literal

import pytest
from pydantic import ValidationError

from llm_agent.agents.algorithm_agent.algorithm.base import (
    Function,
    FunctionCallWithoutFunctionsError,
    make_function_call_model,
)


class TestMakeFunctionCallModel:
    def test_empty(self):
        with pytest.raises(FunctionCallWithoutFunctionsError):
            make_function_call_model([])

    def test_basic(self):
        class Func1(Function, frozen=True):
            func_name: Literal["Func1"]
            param: str

        class Func2(Function, frozen=True):
            func_name: Literal["Func2"]
            param: int

        function_call = make_function_call_model([Func1, Func2])
        assert function_call.model_fields["func"].annotation == Func1 | Func2
        assert function_call.model_fields["func"].discriminator == "func_name"

        with does_not_raise():
            _ = function_call.model_validate({"func": {"func_name": "Func1", "param": "abc"}})

        with does_not_raise():
            _ = function_call.model_validate({"func": {"func_name": "Func2", "param": 1}})

        with pytest.raises(ValidationError):
            _ = function_call.model_validate({"func": {"func_name": "Func1"}})
