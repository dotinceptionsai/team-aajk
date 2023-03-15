import abc
import typing
from dataclasses import dataclass, field
from typing import Iterable, Collection, Any

from dataload.dataloading import DataFilesRegistry

#
# Defines generic interface of any implementation of an ID (In-Domain) Filter pipeline
#

FP = typing.TypeVar("FP", bound="FilterPipeline")

__all__ = ["FilteredSentence", "FilterTrainFiles", "FilterPipeline", "FP"]


@dataclass(frozen=True, eq=True)
class FilteredSentence:
    """The result of the filtering of a sentence:
    the sentence plus a probability that it is out-of-domain (OOD)
    """

    sentence: str
    ood_proba: float
    similar_sents: Collection[str] = field(default_factory=list)


@dataclass
class FilterTrainFiles:
    """The files used for training a filter pipeline. Encompasses training data and validation data."""

    validation_id: str
    validation_ood: str
    train_id: Collection[str] = field(default_factory=list)
    train_ood: Collection[str] = field(default_factory=list)


class FilterPipeline(abc.ABC):
    """All filter pipeline implementations must subclass this class in order to be auto-loadable and exportable"""

    def __init__(self, run_params: dict[str, Any], datasets: FilterTrainFiles):
        self.datasets = datasets
        self.run_params = _normalize_run_params(run_params)
        self._post_init(**self.run_params)

    def _post_init(self, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def filter_sentences(self, text: str) -> Iterable[FilteredSentence]:
        """Analyzes a text and tags each sentences with a flag telling if it is relevant or not to the domain"""
        ...

    @abc.abstractmethod
    def fit(self, registry: DataFilesRegistry) -> None:
        """Trains the filter pipeline"""
        ...

    @abc.abstractmethod
    def export_weights(self) -> dict[str, Any]:
        """Exports the weights computed during training of the filter pipeline as a dictionary"""
        ...

    @abc.abstractmethod
    def load_weights(self, weights: dict[str, Any]) -> None:
        """Loads the weights computed during training.
        Should match the format of the dictionary returned by export_weights()"""
        ...

    @classmethod
    def create(
        cls,
        full_class_name: str,
        run_params: dict[str, Any],
        files: FilterTrainFiles,
        weights: dict[str, Any],
    ) -> "FilterPipeline":
        """Instantiates a filter pipeline of the given name. If weights are provided, they are loaded into the pipeline.
        If not provided, the pipeline will need to be trained (call method train)."""

        module_name, _, class_name = full_class_name.rpartition(".")
        from importlib import __import__

        loaded_module = __import__(module_name, fromlist=[class_name])
        engine_class = getattr(loaded_module, class_name)
        engine = engine_class(run_params, files)
        if weights is not None and len(weights) > 0:
            engine.load_weights(weights)
        return engine


def _normalize_run_params(value: Any) -> Any:
    """Normalizes the run parameters to a format that can be used to instantiate the pipeline"""
    if isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, typing.Sequence):
        return [_normalize_run_params(v) for v in value]
    elif isinstance(value, typing.Mapping):
        return {k: _normalize_run_params(v) for k, v in value.items()}
    else:
        raise ValueError(f"Unsupported type for run parameter {value}: {type(value)}")
