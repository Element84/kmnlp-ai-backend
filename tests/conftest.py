import pytest


# function executed right after test items collected but before test run
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mechanism to only run non-integration (or integration) tests."""
    if not config.getoption("-m"):
        skip_me = pytest.mark.skip(
            reason="use `-m integration` to run this test"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_me)
