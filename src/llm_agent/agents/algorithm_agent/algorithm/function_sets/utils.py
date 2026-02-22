from collections.abc import Iterable


def concat_camel_case(strings: Iterable[str]) -> str:
    """Concatenate strings in to a single CamelCase string.

    Removes underscores within strings.
    """
    return "".join(s.capitalize() for s in strings).replace("_", "")
