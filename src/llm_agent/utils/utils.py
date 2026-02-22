import datetime
import logging
from collections.abc import Sequence
from os.path import dirname
from typing import Any, Literal

import fsspec  # type: ignore[reportMissingTypeStubs]
from e84_geoai_common.llm.core import (
    LLMDataContentType,
    LLMToolResultContent,
)
from pydantic import BaseModel, PrivateAttr, ValidationError

log = logging.getLogger(__name__)


def get_current_date(fmt: str = "%A, %B %d, %Y") -> str:
    """Get current date formatted by default as 'Monday, March 17, 2025'."""
    return datetime.datetime.now(tz=datetime.UTC).strftime(fmt)


def maximum_length_json_string_from_string(string: str) -> str:
    """Get the maximum-length candidate JSON object from a string.

    This function is intended to be used to remove preamble and post-amble text
    from around a JSON object returned by an LLM.

    The runtime complexity is O(n), where n is the length of the string. The
    implementation favors speed over correctness in all cases. If the input
    string contains unbalanced `{}`s, or if the characters between the
    outermost `{}` are not valid JSON, this method will return a string that
    does not parse as valid JSON.

    Args:
        string (str): The string from which to extract the longest-possible
            candidate string representing a JSON object.

    Return:
        The slice of the string from the first `{` to the last `}`

    Raises:
        ValueError if the provided string is missing either

    """
    first_open_curly_bracket = string.index("{")
    last_close_curly_bracket = string.rindex("}")
    return string[first_open_curly_bracket : last_close_curly_bracket + 1]


def model_to_file(uri: str, obj: BaseModel, indent: int | None = 2) -> None:
    """Serialize Pydantic model to a (possibly remote) JSON file."""
    log.debug("Serializing %s to file %s", type(obj), uri)
    fs, fs_path = fsspec.url_to_fs(uri)  # type: ignore[reportUnknownMemberType]
    fs.makedirs(dirname(fs_path), exist_ok=True)  # pyright: ignore[reportUnknownArgumentType]
    with fs.open(fs_path, "w") as f:
        f.write(obj.model_dump_json(indent=indent))


def model_from_file[T: BaseModel](uri: str, model: type[T]) -> T:
    """Deserialize Pydantic model from a (possibly remote) JSON file."""
    log.debug("Deserializing %s from file %s", model, uri)
    fs, fs_path = fsspec.url_to_fs(uri)  # type: ignore[reportUnknownMemberType]
    with fs.open(fs_path, "r") as f:
        return model.model_validate_json(f.read())


class EnrichedValidationError(Exception):
    """Pydantic ValidationError bundled with the input that caused it.

    Attributes:
        error (ValidationError): The original Pydantic ValidationError.
        data (Union[str, dict[str, Any]]): The input data that failed
            validation.
    """

    def __init__(self, error: ValidationError, data: str | dict[str, Any]) -> None:
        """Constructor.

        Args:
            error (ValidationError): The original Pydantic ValidationError.
            data (Union[str, dict[str, Any]]): The input data that failed
                validation.
        """
        self.error = error
        self.data = data


class EnrichedLLMToolResultContent[DataT: BaseModel](LLMToolResultContent):
    """LLMToolResultContent but with unserialized data attached."""

    _data: DataT = PrivateAttr()

    def __init__(
        self,
        *,
        id: str,  # noqa: A002
        content: Sequence[LLMDataContentType],
        status: Literal["success", "error"] | None = None,
        data: DataT,
    ) -> None:
        """Constructor."""
        super().__init__(id=id, content=content, status=status)
        self._data = data

    @property
    def data(self) -> DataT:
        """Access data."""
        return self._data
