import logging
import typing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Iterable

import h5py
import numpy as np
import yaml

from pipelines.filtering import FilterPipeline, FilterTrainFiles

PLACEHOLDER_NDARRAY = "__ndarray__"

PROP_EXPORTED_AT = "exported_at"
PROP_ENGINE_NAME = "name"
PROP_DATASETS = "datasets"
PROP_RUN_PARAMS = "run_params"
PROP_TRAIN_OOD_DATASETS = "train_ood"
PROP_TRAIN_ID_DATASETS = "train_id"
PROP_VALID_ID_DATASET = "validation_id"
PROP_VALID_OOD_DATASET = "validation_ood"

FILENAME_PIPELINE = "pipeline.yml"
FILENAME_WEIGHTS = "weights.yml"
FILENAME_ND_ARRAYS = "ndarrays.h5"

FriendlyPath = Path | str

P = typing.TypeVar("P", bound=FilterPipeline)


#
# Provides a simple and standard way to save and load pipelines, that either trained or not.
# A pipeline folder is a folder that contains a pipeline.yml file, and optionally a weights.yml file
# and a ndarrays.h5 file.
#
# The pipeline.yml file contains the all the properties to train the pipeline:
# - name: the name of the pipeline
# - run_params: the parameters used to train the pipeline
# - datasets: the datasets (train and validation) to use to train the pipeline
#
# If the folder contains a weights.yml file, it is considered a trained pipeline. The weights.yml file obtained during
# the training will be loaded and the filter is ready to be used (without training). If weights are np.arrays, they will
# be saved in a ndarrays.h5 file that is referenced by the weights.yml file.
#

logger = logging.getLogger(__name__)


@dataclass
class PipelineInfoCard:
    """All the parameters needed to train the pipeline"""

    name: str
    run_params: dict[str, Any]
    datasets: FilterTrainFiles


def get_pipeline_info(model_file: FriendlyPath) -> PipelineInfoCard:
    """Returns the info card of the pipeline, without loading the model"""
    model_file = _ensure_is_path(model_file)
    return _read_info_card(model_file / FILENAME_PIPELINE)


def list_pipeline_folders(root_folder: FriendlyPath) -> Iterable[Path]:
    """Returns all the folders that contain a pipeline.yml file"""
    root_folder = _ensure_is_path(root_folder)
    for folder in root_folder.iterdir():
        if folder.is_dir() and (folder / FILENAME_PIPELINE).exists():
            yield folder


def load_typed_pipeline(model_folder: FriendlyPath, clazz: typing.Type[P]) -> P:
    pipe = load_pipeline(model_folder)
    if not isinstance(pipe, clazz):
        raise ValueError(
            f"Expected a pipeline of type {clazz}, but got {type(pipe)} instead"
        )
    return typing.cast(P, pipe)


def load_pipeline(model_folder: FriendlyPath) -> FilterPipeline:
    """Loads a pipeline from a folder that contains a pipeline.yml file."""
    logger.info(f"Loading pipeline from {model_folder}")
    model_folder = _ensure_is_path(model_folder)
    model_dict = _read_info_card(model_folder / FILENAME_PIPELINE)
    weights = _read_model_weights(model_folder)
    return _create_model(model_dict, weights)


def save_pipeline(
    filter_model: FilterPipeline, dst_folder: Optional[FriendlyPath] = None
):
    dst_folder = _ensure_is_path("." if dst_folder is None else dst_folder)
    model_dict = _create_model_dict(filter_model)
    weights, nd_arrays = _extract_weights(filter_model.export_weights())

    dst_folder.mkdir(parents=True, exist_ok=True)
    _write_yml(dst_folder / FILENAME_PIPELINE, model_dict)
    _write_yml(dst_folder / FILENAME_WEIGHTS, weights)
    _write_h5(dst_folder / FILENAME_ND_ARRAYS, nd_arrays)


def _ensure_is_path(path: FriendlyPath) -> Path:
    return Path(path) if isinstance(path, str) else path


def _read_info_card(model_file: Path) -> PipelineInfoCard:
    model_dict = _read_yml(model_file, mandatory=True)
    return PipelineInfoCard(
        name=model_dict[PROP_ENGINE_NAME],
        run_params=model_dict[PROP_RUN_PARAMS],
        datasets=FilterTrainFiles(
            train_id=model_dict[PROP_DATASETS].get(PROP_TRAIN_ID_DATASETS),
            train_ood=model_dict[PROP_DATASETS].get(PROP_TRAIN_OOD_DATASETS),
            validation_id=model_dict[PROP_DATASETS].get(PROP_VALID_ID_DATASET),
            validation_ood=model_dict[PROP_DATASETS].get(PROP_VALID_OOD_DATASET),
        ),
    )


def _read_model_weights(model_folder: Path) -> dict[str, Any]:
    weights = _read_yml(model_folder / FILENAME_WEIGHTS)
    nd_arrays = _read_h5(model_folder / FILENAME_ND_ARRAYS)
    if nd_arrays is not None and len(nd_arrays) > 0:
        _resolve_ndarrays(weights, nd_arrays)
    return weights


def _create_model(
    model_card: PipelineInfoCard, weights: dict[str, Any]
) -> FilterPipeline:
    return FilterPipeline.create(
        model_card.name, model_card.run_params, model_card.datasets, weights
    )


def _read_yml(file: Path, mandatory: bool = False) -> dict[str, Any]:
    if mandatory and not file.exists():
        raise FileNotFoundError(f"File {file} is mandatory")
    if not file.exists():
        return {}
    with open(file, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _write_yml(file: Path, data: dict[str, Any]):
    if data is not None and len(data) > 0:
        with open(file, "w") as f:
            yaml.dump(
                data,
                f,
                sort_keys=False,
            )


def _read_h5(file: Path) -> dict[str, np.ndarray]:
    if not file.exists():
        return {}
    with h5py.File(file, "r") as f:
        return {k: v[:] for k, v in f.items()}


def _write_h5(file: Path, nd_arrays: dict[str, np.ndarray]):
    if nd_arrays is not None and len(nd_arrays) > 0:
        with h5py.File(file, "w") as f:
            for k, v in nd_arrays.items():
                f.create_dataset(k, data=v)


def _create_model_dict(engine: FilterPipeline) -> dict[str, str]:
    datasets = {
        PROP_TRAIN_ID_DATASETS: sorted(str(p) for p in engine.datasets.train_id),
        PROP_TRAIN_OOD_DATASETS: sorted(str(p) for p in engine.datasets.train_ood),
        PROP_VALID_ID_DATASET: engine.datasets.validation_id,
        PROP_VALID_OOD_DATASET: engine.datasets.validation_ood,
    }
    return {
        PROP_ENGINE_NAME: ".".join(
            [engine.__class__.__module__, engine.__class__.__name__]
        ),
        PROP_DATASETS: datasets,
        PROP_RUN_PARAMS: engine.run_params,
    }


def _extract_weights(
    raw_weights: dict[str, Any]
) -> (dict[str, Any], dict[str, np.ndarray]):
    weights: dict[str, Any] = {
        PROP_EXPORTED_AT: datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    nd_arrays: dict[str, np.ndarray] = {}
    _extract_ndarrays(raw_weights, weights, nd_arrays)
    return weights, nd_arrays


def _extract_ndarrays(
    raw_weights: dict[str, Any],
    weights: dict[str, Any],
    nd_arrays: dict[str, np.ndarray],
    path: Optional[list[str]] = None,
):
    if path is None:
        path = []
    for k, v in raw_weights.items():
        sub_path = path + [k]
        if isinstance(v, np.ndarray):
            nd_arrays[".".join(sub_path)] = v
            weights[k] = PLACEHOLDER_NDARRAY
        elif isinstance(v, dict):
            weights[k] = {}
            _extract_ndarrays(v, weights[k], nd_arrays, sub_path)
        else:
            weights[k] = v


def _resolve_ndarrays(
    weights: dict[str, Any],
    nd_arrays: dict[str, np.ndarray],
    path: Optional[list[str]] = None,
):
    if path is None:
        path = []
    for k, v in weights.items():
        sub_path = path + [k]
        if v == PLACEHOLDER_NDARRAY:
            weights[k] = nd_arrays[".".join(sub_path)]
        elif isinstance(v, dict):
            _resolve_ndarrays(v, nd_arrays, sub_path)
