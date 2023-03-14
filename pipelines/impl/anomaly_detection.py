import logging
from enum import Enum
from typing import Any, Collection, Iterable, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline

import pipelines.impl.preprocessing as tk
from dataload.dataloading import DataFilesRegistry
from pipelines.filtering import FilteredSentence, FilterPipeline, FilterTrainFiles
from pipelines.impl.paragraph import ParagraphTransform

logger = logging.getLogger(__name__)
INVALID_SENTENCE_MSG = "Some inputs in your dataset are either less than 2 words, span multiple sentences or are duplicate"


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embedder_name: str = "all-MiniLM-L6-v2"):
        self.embedder = None
        self.embedder_name = embedder_name

    def __do_init(self):
        if not self.embedder:
            self.embedder = SentenceTransformer(self.embedder_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.__do_init()
        return self.embedder.encode(X, show_progress_bar=False)


class GaussianTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, robust_covariance: bool = False, support_fraction: float = 0.85):
        self.gaussian = None
        self.robust_covariance: bool = robust_covariance
        self.support_fraction: float = support_fraction

    def __do_init(self):
        if not self.gaussian:
            self.gaussian: EmpiricalCovariance = (
                MinCovDet(support_fraction=self.support_fraction)
                if self.robust_covariance
                else EmpiricalCovariance()
            )

    def fit(self, X, y=None):
        """X is an array-like of shape (n_samples, embeddings_dim)"""
        self.__do_init()
        self.gaussian.fit(X)
        return self

    def transform(self, X, y=None):
        """X is an array-like of shape (n_samples, embeddings_dim) or (embeddings_dim,)"""
        self._check_if_fitted()
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        return self.gaussian.mahalanobis(X)

    def transform_one(self, X) -> float:
        return float(self.transform(X).squeeze())

    def set_dist_params(self, location, covariance):
        self.__do_init()
        self.gaussian.location_ = location
        self.gaussian._set_covariance(covariance)

    def get_dist_params(self):
        self._check_if_fitted()
        return self.gaussian.location_, self.gaussian.covariance_

    def _check_if_fitted(self):
        if self.gaussian is None:
            raise ValueError("Must fit the transformer first")


class CutoffCalibrator(BaseEstimator, ClassifierMixin):
    """Takes in a list of id and ood scores to compute a cutoff threshold that maximizes f1-score"""

    def __init__(
        self,
        adjusted: bool = False,
        score_column: str = "score",
        label_column: str = "label",
    ):
        self.r_ood_ = None
        self.r_id_ = None
        self.cutoff_ = None
        self.f1_score_ = None
        self.score_column = score_column
        self.label_column = label_column
        self.adjusted = adjusted

    def fit(self, X, y=None):
        """Expects an array-like of shape (n_samples, 2) with the first column being the score and the second the label"""
        if isinstance(X, pd.DataFrame):
            y_true = X[self.label_column]
            y_score = X[self.score_column]
        else:
            y_true = X[:, 1]
            y_score = X[:, 0]

        validation_id_scores = y_score[y_true == 0]
        validation_ood_scores = y_score[y_true == 1]

        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = (2 * precision * recall) / (precision + recall)
        best_f1_idx = np.argmax(f1_scores)
        self.f1_score_ = float(f1_scores[best_f1_idx])
        self.cutoff_ = float(pr_thresholds[best_f1_idx])

        if self.adjusted:
            non_zero_id_scores = validation_id_scores[
                validation_id_scores < self.cutoff_
            ]
            non_zero_ood_scores = validation_ood_scores[
                validation_ood_scores > self.cutoff_
            ]
            nearest_id = np.argmin(np.abs(non_zero_id_scores - self.cutoff_))
            nearest_ood = np.argmin(np.abs(non_zero_ood_scores - self.cutoff_))
            self.cutoff_ = (
                non_zero_id_scores[nearest_id] + non_zero_ood_scores[nearest_ood]
            ) / 2

        # Make score user friendly:
        # idea from https://medium.com/balabit-unsupervised/calibrating-anomaly-scores-5e60b7e47553
        self.r_id_ = float(np.percentile(validation_id_scores, 50))
        self.r_ood_ = float(np.percentile(validation_ood_scores, 50))

    def predict(self, X):
        return X > self.cutoff_

    def predict_proba(self, X):
        distances = X - self.cutoff_
        full_distances = np.where(
            distances < 0, self.cutoff_ - self.r_id_, self.r_ood_ - self.cutoff_
        )
        proba = 0.5 + 0.5 * distances / full_distances
        return np.clip(proba, 0, 1)


class BinaryDictTransformer(BaseEstimator, TransformerMixin):
    """Given a dict with two keys "id" and "ood" that are lists
    returns a dataframe with two columns "score" and "label" where oods are mapped to 1 and ids to 0
    """

    def __init__(self, zero_mapped_key: str = "id", one_mapped_key: str = "ood"):
        self.zero_mapped_key = zero_mapped_key
        self.one_mapped_key = one_mapped_key

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        zero_scores = np.squeeze(X[self.zero_mapped_key])
        one_scores = np.squeeze(X[self.one_mapped_key])
        y_score = np.hstack([zero_scores, one_scores])
        y_true = np.hstack([np.zeros(len(zero_scores)), np.ones(len(one_scores))])

        return pd.DataFrame({"score": y_score, "label": y_true})


# Create an enum  class named OnInvalidSentence with values FAIL or WARN
class OnInvalidSentence(Enum):
    FAIL = 1
    WARN = 2


# noinspection PyAttributeOutsideInit
class GaussianEmbeddingsAnomalyDetector(FilterPipeline):
    """
    The model first computes the embeddings of the sentences. Then, it fits a Gaussian distribution to those.
    The embeddings are from HuggingFace's Sentence Transformers library. Only embeddings that have explicitly been
    trained to support Euclidian-distance comparisons are supported. The list of supported models is available in
    field `available_embedders`.

    The fitting of the Gaussian distribution is done using either sklearn.covariance.EmpiricalCovariance class for
    a quick and fast fit, or sklearn.covariance.MinCovDet for a more robust fit. The robust fit is slower, but can deal
    with a certain amount of sentences that are OOD but in the training ID dataset.
    """

    available_embedders = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "all-distilroberta-v1",
    ]
    available_text_normalizers = tk.get_available_normalizers()
    name: str = "GaussianEmbeddingsAnomalyDetector"

    def __init__(self, run_params: dict[str, Any], datasets: FilterTrainFiles):
        super().__init__(run_params, datasets)
        self.all_embeddings = None

    def _post_init(
        self,
        embedder_name: str = "all-MiniLM-L6-v2",
        robust_covariance: bool = False,
        support_fraction: float = 0.85,
        text_normalizer_keys: Sequence[str] | None = None,
    ):
        # Text preprocessing
        self.sentence_splitter = ParagraphTransform(
            [
                tk.make_text_normalizer(text_normalizer_keys),
                tk.split_into_sentences,
                tk.normalize_sent_min_words,
            ]
        )

        # Embeddings setup
        if embedder_name not in self.available_embedders:
            raise ValueError(
                f"Embedder {embedder_name} not available. Choose from {self.available_embedders}"
            )

        self.embedder_name = self.run_params.get("embedder_name", "all-MiniLM-L6-v2")
        self.embedder = EmbeddingTransformer(self.embedder_name)

        # Gaussian fit setup
        self.distribution = GaussianTransformer(robust_covariance, support_fraction)
        self.train_pipe = make_pipeline(
            self.sentence_splitter, self.embedder, self.distribution
        )

        # Calibration init
        self.binary_transformer = BinaryDictTransformer()
        self.calibrator = CutoffCalibrator()
        self.calibration_pipe = make_pipeline(self.binary_transformer, self.calibrator)

    def filter_sentences(self, text: str) -> Iterable[FilteredSentence]:
        if not self.calibrator.r_id_:
            raise ValueError("Calibrator has not been fitted yet.")

        for sentence in self.sentence_splitter.transform_one(text):
            embed = self.embedder.transform(sentence)
            raw_score = self.distribution.transform_one(embed)
            proba = self.calibrator.predict_proba(raw_score)
            yield FilteredSentence(sentence, proba)

    def fit(self, registry: DataFilesRegistry) -> None:
        train_paragraphs = _load_paragraphs(self.datasets.train_id, registry)
        self.train_pipe.fit(train_paragraphs)
        self.calibrate_from_val_files(registry)

    def predict(self, X, y=None, on_invalid_sentence=OnInvalidSentence.FAIL):
        return self._do_predict(X, on_invalid_sentence, proba=False)

    def predict_proba(self, X, y=None, on_invalid_sentence=OnInvalidSentence.FAIL):
        return self._do_predict(X, on_invalid_sentence, proba=True)

    def _do_predict(self, X, on_invalid_sentence, proba: bool = False):
        verified_sents = self._verified_sentences(X, on_invalid_sentence)
        raw_scores = self.train_pipe.transform(verified_sents)
        if proba:
            predictions = self.calibrator.predict_proba(raw_scores)
        else:
            predictions = self.calibrator.predict(raw_scores)
        return predictions, verified_sents

    def calibrate_from_val_files(self, registry: DataFilesRegistry) -> None:
        validation_id_paragraphs = _load_paragraphs(
            self.datasets.validation_id, registry
        )
        validation_ood_paragraphs = _load_paragraphs(
            self.datasets.validation_ood, registry
        )
        self._calibrate(
            validation_id_paragraphs, validation_ood_paragraphs, OnInvalidSentence.WARN
        )

    def recalibrate(
        self,
        id_sentences: Sequence[str],
        ood_sentences: Sequence[str],
        registry: DataFilesRegistry | None = None,
        on_invalid_sentence: OnInvalidSentence = OnInvalidSentence.FAIL,
    ):
        id_sents = list(id_sentences)
        ood_sents = list(ood_sentences)
        if registry:
            id_sents.extend(_load_paragraphs(self.datasets.validation_id, registry))
            ood_sents.extend(_load_paragraphs(self.datasets.validation_ood, registry))
        self._calibrate(id_sents, ood_sents, on_invalid_sentence)

    def _calibrate(
        self,
        id_sentences: Sequence[str],
        ood_sentences: Sequence[str],
        on_invalid_sentence: OnInvalidSentence,
    ) -> None:
        id_sentences = self._verified_sentences(id_sentences, on_invalid_sentence)
        ood_sentences = self._verified_sentences(ood_sentences, on_invalid_sentence)
        raw_val_id_scores = self.train_pipe.transform(id_sentences)
        raw_val_ood_scores = self.train_pipe.transform(ood_sentences)
        self.calibration_pipe.fit({"id": raw_val_id_scores, "ood": raw_val_ood_scores})

    def _verified_sentences(
        self, sentences: Sequence[str], on_invalid_sentence: OnInvalidSentence
    ) -> Sequence[str]:
        verified_sentences = self.sentence_splitter.transform(sentences)
        if len(verified_sentences) != len(sentences):
            self._report_invalid_sentences(sentences)

            if on_invalid_sentence == OnInvalidSentence.FAIL:
                raise ValueError(INVALID_SENTENCE_MSG)
            else:
                logger.warning(INVALID_SENTENCE_MSG)
        return verified_sentences

    def _report_invalid_sentences(self, sentences):
        found_sentences = set()
        for s in sentences:
            ts = list(self.sentence_splitter.transform_one(s))
            if len(ts) != 1:
                logger.warning(f"Invalid sentence: {s}")
            if s in found_sentences:
                logger.warning(f"Duplicate sentence: {s}")
            found_sentences.add(s)

    def export_weights(self) -> dict[str, Any]:
        location, covariance = self.distribution.get_dist_params()
        return {
            "cutoff_distance": self.calibrator.cutoff_,
            "r_id": self.calibrator.r_id_,
            "r_ood": self.calibrator.r_ood_,
            "mean": location,
            "covariance": covariance,
        }

    def load_weights(self, weights: dict[str, Any]) -> None:
        self.calibrator.cutoff_ = weights["cutoff_distance"]
        self.calibrator.r_id_ = weights["r_id"]
        self.calibrator.r_ood_ = weights["r_ood"]
        self.distribution.set_dist_params(weights["mean"], weights["covariance"])


def _load_paragraphs(
    file_ids: str | Collection[str], registry: DataFilesRegistry
) -> list[str]:
    if isinstance(file_ids, str):
        return registry.load_items(file_ids)
    else:
        paragraphs = []
        for file_id in file_ids:
            paragraphs.extend(registry.load_items(file_id))
        return paragraphs
