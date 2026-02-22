import pytest
from e84_geoai_common.llm.core import Base64ImageContent
from pydantic import ValidationError

from llm_agent.tools.data_analysis_tool.data_analysis_tool import DataAnalysisToolOutput


class TestDataAnalysisToolOutput:
    def test_init(self) -> None:
        # Raise will fail test
        _ = DataAnalysisToolOutput(
            data_outputs=None,
            image_outputs=["image1"],
            image_data=[Base64ImageContent(media_type="image/png", data="")],
        )

        with pytest.raises(ValidationError):
            _ = DataAnalysisToolOutput(data_outputs=None)

        with pytest.raises(ValueError, match=r"len\(image_outputs\) != len\(images\)"):
            _ = DataAnalysisToolOutput(data_outputs=None, image_outputs=["image1"], image_data=None)

        with pytest.raises(ValueError, match=r"len\(image_outputs\) != len\(images\)"):
            _ = DataAnalysisToolOutput(data_outputs=None, image_outputs=["image1"], image_data=[])

        with pytest.raises(ValueError, match=r"len\(image_outputs\) != len\(images\)"):
            _ = DataAnalysisToolOutput(
                data_outputs=None,
                image_outputs=["abc"],
                image_data=[
                    Base64ImageContent(media_type="image/png", data=""),
                    Base64ImageContent(media_type="image/png", data=""),
                ],
            )
