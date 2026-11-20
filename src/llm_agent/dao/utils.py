import os
from importlib.resources import files
from importlib.resources.abc import Traversable

import llm_agent.data

# Why `Traversable`?
#
# According to the setuptool docs [1], `importlib.resources` is preferred over
# `os` because it is compatible with PEP-302-based import hooks. It is
# recommended to include data files in a package in this way.
#
# `Traversable` is an ABC, and the relatively more common `Path` implements
# it. In the event a future implementor finds it easier to build a `Path`, it
# can be provided anywhere that a `Traversable` is required.

# One advantage of the `Traversable` ABC is that `as_file(traversable)`
# returns a `ContextManager` for use with the `with` statement, making it
# easy to remember to close the file once it is no longer being used.
#
# [1] https://setuptools.pypa.io/en/latest/userguide/datafiles.html#accessing-data-files-at-runtime
root = files(llm_agent.data)


def get_kg_ttl_paths() -> list[Traversable]:
    """Get the list of knowledge graph TTL file paths.

    Returns the appropriate TTL files based on the USE_ICECHUNK_STORES
    environment variable. When set to 'true', '1', or 'yes', returns the
    icechunk version of the CDR dataset. Otherwise, returns the previous
    version.
    """
    # Use icechunk versions of stores if env variable set. Otherwise, use previous
    # ones.
    should_use_icechunk_stores = os.getenv("USE_ICECHUNK_STORES", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    return (
        [
            root.joinpath("dataset-kg-crw-generated.ttl"),
            root.joinpath("dataset-kg-cdr-generated-icechunk.ttl"),
            root.joinpath("dataset-kg-dynamical-org-generated.ttl"),
        ]
        if should_use_icechunk_stores
        else [
            root.joinpath("dataset-kg-crw-generated.ttl"),
            root.joinpath("dataset-kg-cdr-generated.ttl"),
            root.joinpath("dataset-kg-dynamical-org-generated.ttl"),
        ]
    )


# Backwards compatibility: provide the constant for existing code
KG_TTL_PATHS = get_kg_ttl_paths()
