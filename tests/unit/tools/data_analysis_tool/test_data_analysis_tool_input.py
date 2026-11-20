from textwrap import dedent

import pytest
from pydantic import ValidationError

from llm_agent.tools.data_analysis_tool.data_analysis_tool import DataAnalysisToolInput


class TestDataAnalysisToolInput:
    def test_expected_dict_parses(self):
        input_dict = {
            "problem_statement": (
                "Extract sea surface temperature data for the Gulf of Mexico on June 1, 2023"
            ),
            "input_data": ["sea_surface_temperature_crw_5km_sst_daily"],
            "outputs_requested": [
                {
                    "name": "average_sst",
                    "variable_type": "scalar_float",
                    "description": (
                        "Average sea surface temperature in the Gulf of Mexico on June 1, 2023 in "
                        "degrees Celsius"
                    ),
                },
                {
                    "name": "sst_map",
                    "variable_type": "raster_plot",
                    "description": (
                        "Map showing sea surface temperature distribution in the Gulf of Mexico on "
                        "June 1, 2023"
                    ),
                },
            ],
        }

        parsed_input = DataAnalysisToolInput.model_validate(input_dict)

        assert isinstance(parsed_input, DataAnalysisToolInput)
        assert parsed_input.problem_statement == (
            "Extract sea surface temperature data for the Gulf of Mexico on June 1, 2023"
        )

    def test_missing_key_fails_to_parse(self):
        input_dict = {
            "input_data": ["sea_surface_temperature_crw_5km_sst_daily"],
            "outputs_requested": [
                {
                    "name": "average_sst",
                    "variable_type": "scalar_float",
                    "description": (
                        "Average sea surface temperature in the Gulf of Mexico on June 1, 2023 in "
                        "degrees Celsius"
                    ),
                },
                {
                    "name": "sst_map",
                    "variable_type": "raster_plot",
                    "description": (
                        "Map showing sea surface temperature distribution in the Gulf of Mexico on "
                        "June 1, 2023"
                    ),
                },
            ],
        }

        with pytest.raises(ValidationError):
            DataAnalysisToolInput.model_validate(input_dict)

    def test_over_escaped_input_succeeds(self):
        input_str = dedent("""{
            "problem_statement": "Extract sea surface temperature data for the Gulf of Mexico on June 1, 2023",
            "input_data": ["sea_surface_temperature_crw_5km_sst_daily"],
            "outputs_requested": [
                {
                    "name": "average_sst",
                    "variable_type": "scalar_float",
                    "description": "Average sea surface temperature in the Gulf of Mexico on June 1, 2023 in degrees Celsius"
                },
                {
                    "name": "sst_map",
                    "variable_type": "raster_plot",
                    "description": "Map showing sea surface temperature distribution in the Gulf of Mexico on June 1, 2023"
                }
            ]
        }""")  # noqa: E501

        parsed_input = DataAnalysisToolInput.model_validate(input_str)

        assert isinstance(parsed_input, DataAnalysisToolInput)
        assert parsed_input.problem_statement == (
            "Extract sea surface temperature data for the Gulf of Mexico on June 1, 2023"
        )

    def test_actual_string(self):
        # This is an actual string returned by the LLM.
        # It is unexpected because:
        #   * It was returned as a JSON String rather than JSON Object.
        #   * It has an extra `}`.
        input_str = """{"problem_statement": "Extract sea surface temperature data for the Gulf of Mexico on June 1, 2023", "input_data": ["sea_surface_temperature_crw_5km_sst_daily"], "outputs_requested": [{"name": "average_sst", "variable_type": "scalar_float", "description": "Average sea surface temperature in the Gulf of Mexico on June 1, 2023"}, {"name": "min_sst", "variable_type": "scalar_float", "description": "Minimum sea surface temperature in the Gulf of Mexico on June 1, 2023"}, {"name": "max_sst", "variable_type": "scalar_float", "description": "Maximum sea surface temperature in the Gulf of Mexico on June 1, 2023"}, {"name": "sst_map", "variable_type": "raster_plot", "description": "Spatial map of sea surface temperature in the Gulf of Mexico on June 1, 2023"}]}}"""  # noqa: E501

        parsed_input = DataAnalysisToolInput.model_validate(input_str)

        assert isinstance(parsed_input, DataAnalysisToolInput)
        assert parsed_input.problem_statement == (
            "Extract sea surface temperature data for the Gulf of Mexico on June 1, 2023"
        )

    def test_fails_with_pure_string(self):
        input_str = "yo"

        with pytest.raises(ValidationError):
            DataAnalysisToolInput.model_validate(input_str)
