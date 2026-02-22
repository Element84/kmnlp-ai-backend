import logging
from typing import cast

import pytest

from tests.experiment.langfuse_test import LangfuseTest

log = logging.getLogger(__name__)


class TestCompilePrompt(LangfuseTest):
    @pytest.mark.integration
    def test_basic_prompt(self):
        prompt = self.langfuse.get_prompt(
            "orchestration_agent/data_analysis_tool_no_bounding_box", label="latest"
        )
        compiled_prompt = prompt.compile(current_date="Thursday, October 16, 2025")
        assert len(compiled_prompt) == 4
        log.debug("compiled_prompt type: %s", type(compiled_prompt[0]))
        assert (
            "You are a helpful assistant" in cast("dict[str, str]", compiled_prompt[0])["content"]
        )
        assert "Thursday, October 16, 2025" in cast("dict[str, str]", compiled_prompt[0])["content"]
