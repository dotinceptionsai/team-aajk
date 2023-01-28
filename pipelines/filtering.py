import abc
import typing
from dataclasses import dataclass, field
from typing import Iterable, Collection, Any

from dataload.dataloading import DataFilesRegistry

#
# Defines generic interface of any implementation of an ID (In-Domain) Filter pipeline
#

T = typing.TypeVar("T", bound="FilterPipeline")

__all__ = ["FilteredSentence", "FilterTrainFiles", "FilterPipeline", "T"]


@dataclass(frozen=True, eq=True)
class FilteredSentence:
    """The result of the filtering of a sentence:
    the sentence, a flag indicating if it is relevant to the domain and a confidence score."""

    sentence: str
    relevant: bool
    score: float


@dataclass
class FilterTrainFiles:
    """The files used for training a filter pipeline. Encompasses training data and validation data."""

    train_id: Collection[str] = field(default_factory=list)
    train_ood: Collection[str] = field(default_factory=list)
    validation_id: typing.Optional[str] = None
    validation_ood: typing.Optional[str] = None


class FilterPipeline(abc.ABC):
    """All filter pipeline implementations must subclass this class in order to be auto-loadable and exportable"""

    name: str
    _engines: dict[str, typing.Type[T]] = {}

    def __init__(
        self, run_params: dict[str, Any] = None, train_files: FilterTrainFiles = None
    ):
        self.files = train_files
        self.run_params = run_params
        self._post_init(**run_params)

    def _post_init(self, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def filter_sentences(self, text: str) -> Iterable[FilteredSentence]:
        """Analyzes a text and tags each sentences with a flag telling if it is relevant or not to the domain"""
        ...

    @abc.abstractmethod
    def train(self, registry: DataFilesRegistry) -> None:
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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        FilterPipeline._engines[cls.name] = cls

    @staticmethod
    def create(
        name: str,
        run_params: dict[str, Any],
        files: FilterTrainFiles,
        weights: dict[str, Any],
    ) -> T:
        """Instantiates a filter pipeline of the given name. If weights are provided, they are loaded into the pipeline.
        If not provided, the pipeline will need to be trained (call method train)."""
        class_name: typing.Type[T] = FilterPipeline._engines[name]
        engine: T = class_name(run_params, files)
        if weights is not None and len(weights) > 0:
            engine.load_weights(weights)
        return engine
