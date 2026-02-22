import logging
from abc import ABC
from typing import Any

import json_repair
from pydantic import BaseModel, model_validator

log = logging.getLogger(__name__)


class LLMToolInput(BaseModel, ABC):
    """Type for LLM Tool Input types to extend.

    Currently, this provides a before validator that tries to parse any JSON
    strings to dictionaries before parsing input.
    """

    @model_validator(mode="before")
    @classmethod
    def parse_json_strings(cls, data: Any) -> Any:  # noqa: ANN401
        """Before validation, handle JSON strings by parsing them into dicts.

        Sometimes, the LLM's tool use input args are a JSON string that would
        decode to a JSON object rather than a JSON object. If the data class
        passed to this input type is a string, attempt to parse it into a dict
        before having Pydantic try to parse it.
        """
        if isinstance(data, str):
            # `json_repair.loads` handles cases like extra `}`.
            #
            # It always returns something rather than raising an error if
            # the data is completely unreasonable. In that case, let
            # Pydantic handle the validation error.
            log.warning(
                "Model validator input was string. Attempting to parse as JSON. data: %s", data
            )
            dict_data = json_repair.loads(data)

            return dict_data

        return data
