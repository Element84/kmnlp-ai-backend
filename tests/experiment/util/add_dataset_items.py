# ruff: noqa: E501

from langfuse import get_client

# NOTE: This file was executed to create the dataset in Langfuse. It's commited
# for 2 reasons:
#   * To make it easy to populate other datasets in the future.
#   * To serve as a record for what we meant to put in that dataset.


def main():
    langfuse = get_client()
    expected_output = {
        "role": "assistant",
        "content": [
            {
                "id": "toolu_bdrk_01UrPTqX6HTH8NqHaDohNyzN",
                "name": "AgentPlanTool",
                "input": {
                    "plan_or_update": {
                        "id": "gulf_mexico_sst_plan",
                        "content_type": "plan_update",
                        "step_updates": [
                            {
                                "id": "step_1",
                                "status": "success",
                                "outcome_summary": "Found NOAA Coral Reef Watch Daily Global 5km SST dataset covering 2020-2024",
                            },
                            {"id": "step_2", "status": "in_progress"},
                        ],
                    }
                },
            },
            {
                "id": "toolu_bdrk_016XTfV4NXD8FRXHURgk4hP3",
                "name": "DataAnalysisToolWithTracking",
                "input": {
                    "tracking": {"plan_id": "gulf_mexico_sst_plan", "step_id": "step_2"},
                    "tool_input": {
                        "variable_ids": ["sea_surface_temperature_crw_daily_5km"],
                        "outputs_requested": [
                            {
                                "name": "gulf_mexico_sst_june_1_2023",
                                "description": "Average sea surface temperature in the Gulf of Mexico on June 1, 2023 in degrees Celsius",
                                "variable_type": "scalar_float",
                            }
                        ],
                        "problem_statement": "Extract the sea surface temperature for the Gulf of Mexico region on June 1, 2023. The Gulf of Mexico polygon should be determined automatically by the analysis tool rather than using predefined coordinates.",
                    },
                },
            },
        ],
        "metadata": {"stop_reason": "tool_use", "input_tokens": 7141, "output_tokens": 361},
    }

    generated_queries = [
        "Surface temp Gulf of Mexico, 2023-06-01",
        "What was the SST near the Mississippi River delta on June 1, 2023?",
        "Can you show me water temperature readings off Galveston, Texas for 06/01/2023?",
        "Sea temperature within 10 nautical miles of Pensacola, Florida on June first, 2023?",
        "What were the SST values along the Louisiana shelf on 1 June 2023?",
        "Temperature gradient across the Loop Current on June 1st, 2023?",
        "Warmest SST recorded in the Gulf on 2023-06-01?",
        "Coldest sea surface reading in the Gulf of Mexico, June 1, 2023?",
        "Average water temp in the Bay of Campeche on 01/06/2023?",
        "SST distribution across the Florida Keys on June 1, 2023?",
        "What was the temperature difference between shallow and deep Gulf waters on June first, 2023?",
        "Sea surface temps along the Texas coastline, 2023-06-01?",
        "Water temperature at 25°N, 90°W on June 1, 2023?",
        "SST anomaly in the western Gulf on 06/01/23?",
        "Temperature profile from Corpus Christi to Veracruz on June 1, 2023?",
        "What were surface water temps near the Flower Garden Banks on 2023-06-01?",
        "June 1, 2023 sea temps within the Yucatan Channel?",
        "SST readings off Tampa Bay on 01 June 2023?",
        "Water temperature comparison between eastern and western Gulf on June first, 2023?",
        "Mean SST across the entire Gulf of Mexico basin on 6/1/2023?",
        "Temperature at the mouth of Mobile Bay, June 1st, 2023?",
        "SST within 50 miles of Brownsville, Texas on 2023-06-01?",
        "What was the thermal structure near the DeSoto Canyon on June 1, 2023?",
        "Sea surface temperature off Apalachicola on 06/01/2023?",
        "June first, 2023 water temps along the continental shelf break?",
        "SST variability in the northern Gulf on 01-06-2023?",
        "Temperature readings near oil rig platforms on June 1, 2023?",
        "What were sea temps off Biloxi, Mississippi on 2023-06-01?",
        "SST within Charlotte Harbor on June 1st, 2023?",
        "Water temperature along 28°N latitude on 6/1/23?",
        "June 1, 2023 thermal front locations in the Gulf?",
        "Sea surface temp off South Padre Island on 01/06/2023?",
        "SST patterns near the Sigsbee Deep on June first, 2023?",
        "Temperature within Galveston Bay versus open Gulf waters on 2023-06-01?",
        "What was the SST at the shelf edge off Louisiana on June 1, 2023?",
        "Water temps in the Straits of Florida on 06/01/2023?",
        "Sea surface temperature within 20 km of Cancún on June 1st, 2023?",
        "June first, 2023 SST along a transect from New Orleans to Cuba?",
        "Temperature near the Mississippi Canyon on 2023-06-01?",
        "What were surface water conditions off Clearwater, Florida on 6/1/2023?",
        "SST in shallow versus offshore waters near Matagorda Bay on June 1, 2023?",
        "June 1st, 2023 sea temp at coordinates 27.5°N, 92.5°W?",
        "Water temperature across the Florida Escarpment on 01 June 2023?",
        "SST variability within 100 km of the Texas-Louisiana border on 2023-06-01?",
        "What was the thermal gradient from coast to Loop Current on June first, 2023?",
        "Sea surface temps in Vermilion Bay on 06/01/23?",
        "June 1, 2023 water temperature off Port Aransas?",
        "SST readings within the Mississippi Bight on 2023-06-01?",
        "Temperature distribution in Atchafalaya Bay versus adjacent shelf waters on June 1st, 2023?",
        "What were the SST conditions near Dry Tortugas National Park on 01/06/2023?",
    ]

    for q in generated_queries:
        print(f"adding query: {q}")  # noqa: T201
        langfuse.create_dataset_item(
            dataset_name="Orchestration Agent Single Turn",
            input={"user_query": q},
            expected_output=expected_output,
        )


if __name__ == "__main__":
    main()
