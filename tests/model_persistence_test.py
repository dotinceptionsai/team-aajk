from typing import Iterable, Any

import numpy as np

from dataload.dataloading import DataFilesRegistry
from pipelines import persistence
from pipelines.filtering import FilterPipeline, FilteredSentence
from pipelines.filtering import FilterTrainFiles


#
# Unit test for the generic loading and exporting of filter pipelines
#


class DummyFilterPipeline(FilterPipeline):
    name: str = "DummyFilterPipeline"
    weights: dict[str, Any]

    def filter_sentences(self, text: str) -> Iterable[FilteredSentence]:
        return []

    def fit(self, registry) -> None:
        self.weights = {
            "x": np.array([1, 2, 3]),
            "y": {"a": 1, "b": np.array([4, 5, 6])},
        }

    def export_weights(self) -> dict[str, Any]:
        return self.weights

    def load_weights(self, weights: dict[str, Any]) -> None:
        self.weights = weights


def test_should_convert_back_and_forth(tmp_path):
    data_registry = DataFilesRegistry(tmp_path)

    train_files = FilterTrainFiles(
        train_id=["train_id.txt"],
        train_ood=["train_ood.txt"],
        validation_id="validation_id.txt",
        validation_ood="validation_ood.txt",
    )
    run_params = {"alpha": 55.79}

    pipeline = DummyFilterPipeline(run_params, train_files)
    pipeline.fit(data_registry)

    persistence.save_pipeline(pipeline, tmp_path)

    loaded_model = persistence.load_typed_pipeline(tmp_path, DummyFilterPipeline)

    np.testing.assert_array_equal(np.array([1, 2, 3]), loaded_model.weights["x"])
    assert loaded_model.weights["y"]["a"] == 1
    np.testing.assert_array_equal(np.array([4, 5, 6]), loaded_model.weights["y"]["b"])
    assert loaded_model.run_params["alpha"] == 55.79
    assert loaded_model.datasets.train_id == ["train_id.txt"]
    assert loaded_model.datasets.train_ood == ["train_ood.txt"]
    assert loaded_model.datasets.validation_id == "validation_id.txt"
    assert loaded_model.datasets.validation_ood == "validation_ood.txt"

    assert loaded_model.name == "DummyFilterPipeline"
