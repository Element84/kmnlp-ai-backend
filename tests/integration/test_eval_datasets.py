# pyright: reportUnusedImport=false
import importlib
from contextlib import nullcontext as does_not_raise

import pytest


# Mark this test as an integratio test because it downloads the model if it
# isn't cached locally. In CI, it's never cached. It would be possible to hack
# around this by putting the model on the image. This is left an an exercise to
# the developer.
@pytest.mark.integration
class TestEvalDatasets:
    @pytest.mark.parametrize(
        "eval_dataset_module_name",
        [
            "evals.datasets.algorithm_agent",
            "evals.datasets.dataset_picking_agent",
        ],
    )
    def test_no_runtime_errors_when_building_eval_datasets(self, eval_dataset_module_name: str):
        with does_not_raise():
            importlib.import_module(eval_dataset_module_name)
