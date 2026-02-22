import os

import pytest
from langfuse import get_client


@pytest.mark.skipif(
    os.getenv("SKIP_LANGFUSE_TESTS") == "true", reason="Langfuse environment variables must be set."
)
class LangfuseTest:
    """Provide Langfuse setup and teardown for integration tests.

    Tests that extend `LangfuseTest` are likely to only ever be integration
    tests. They should make direct use of the Langfuse SDK.
    """

    @classmethod
    def setup_class(cls) -> None:
        """Setup Langfuse client."""
        # Ensure we have the variables set to make this work for testing.
        assert os.getenv("LANGFUSE_HOST") is not None, "Langfuse environment variables must be set."
        assert os.getenv("LANGFUSE_PUBLIC_KEY") is not None, (
            "Langfuse environment variables must be set."
        )
        assert os.getenv("LANGFUSE_SECRET_KEY") is not None, (
            "Langfuse environment variables must be set."
        )
        cls.langfuse = get_client()

    @classmethod
    def teardown_class(cls) -> None:
        cls.langfuse.shutdown()
