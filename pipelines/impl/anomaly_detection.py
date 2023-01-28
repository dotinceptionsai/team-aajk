import logging
from typing import Any, Iterable

import nltk
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.metrics import precision_recall_curve

import pipelines.impl.tokenizing as tk
from dataload.dataloading import DataFilesRegistry
from pipelines.filtering import FilterPipeline, FilteredSentence


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
    name: str = "GaussianEmbeddingsAnomalyDetector"
    sentence_parser: tk.TextTransformer = tk.pipelined(tk.to_sentences, tk.chunker(25))
    robust_covariance: bool = False
    distribution: EmpiricalCovariance = None
    embedder: SentenceTransformer = None
    embedder_name: str = None
    cutoff_distance: float = None
    r_id: float = None
    r_ood: float = None

    def _post_init(
        self, embedder_name: str = "all-MiniLM-L6-v2", robust_covariance: bool = False
    ):
        if embedder_name not in self.available_embedders:
            raise ValueError(
                f"Embedder {embedder_name} not available. Choose from {self.available_embedders}"
            )
        self.embedder_name = self.run_params.get("embedder_name", "all-MiniLM-L6-v2")
        self.robust_covariance = self.run_params.get("robust_covariance", False)
        self.embedder = SentenceTransformer(embedder_name)
        self.distribution = GaussianEmbeddingsAnomalyDetector._distribution_estimator(
            self.robust_covariance
        )

    def filter_sentences(self, text: str) -> Iterable[FilteredSentence]:
        if not self.cutoff_distance:
            raise ValueError("Anomaly detector has not been trained yet")

        for sentence in GaussianEmbeddingsAnomalyDetector.sentence_parser(text):
            score = self.mahalanobis(sentence)
            yield FilteredSentence(
                sentence, score < self.cutoff_distance, self._calibrated_score(score)
            )

    def mahalanobis(self, sentence: str):
        embedding = self.embedding(sentence)
        return self.distribution.mahalanobis(embedding.reshape(1, -1)).squeeze()

    def embedding(self, sentence):
        return self.embedder.encode(_standardize_sentence(sentence))

    def train(self, registry: DataFilesRegistry) -> None:
        # Fit distribution to embeddings of in-domain training data
        sentence_embeddings = []
        for file_id in self.files.train_id:
            sentence_embeddings.extend(self._embed_file(file_id, registry))
        self.distribution = GaussianEmbeddingsAnomalyDetector._distribution_estimator(
            self.robust_covariance
        )
        self.distribution.fit(np.vstack(sentence_embeddings))

        # Fit cutoff based on validation data
        self.cutoff_distance, id_scores, ood_scores = self._compute_cutoff(registry)

        # Make score user friendly:
        # idea from https://medium.com/balabit-unsupervised/calibrating-anomaly-scores-5e60b7e47553
        self.r_id = float(np.percentile(id_scores, 50))
        self.r_ood = float(np.percentile(ood_scores, 50))

    @staticmethod
    def _distribution_estimator(robust: bool) -> EmpiricalCovariance:
        return MinCovDet(support_fraction=0.85) if robust else EmpiricalCovariance()

    def _compute_cutoff(self, registry: DataFilesRegistry) -> (float, NDArray, NDArray):
        validation_id_embeddings = list(
            self._embed_file(self.files.validation_id, registry)
        )
        validation_ood_embeddings = list(
            self._embed_file(self.files.validation_ood, registry)
        )
        validation_id_scores = self.distribution.mahalanobis(
            np.vstack(validation_id_embeddings)
        )
        validation_ood_scores = self.distribution.mahalanobis(
            np.vstack(validation_ood_embeddings)
        )

        y_true = np.hstack(
            [
                np.zeros(len(validation_id_embeddings)),
                np.ones(len(validation_ood_embeddings)),
            ]
        )
        y_score = np.hstack([validation_id_scores, validation_ood_scores])

        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = (2 * precision * recall) / (precision + recall)
        best_f1_idx = np.argmax(f1_scores)
        logging.info(
            f"Best F1 cutoff={pr_thresholds[best_f1_idx]:.2f}, F1-Score={f1_scores[best_f1_idx]:.3f}"
        )
        return (
            float(pr_thresholds[best_f1_idx]),
            validation_id_scores,
            validation_ood_scores,
        )

    def _embed_file(
        self, file_id: str, registry: DataFilesRegistry
    ) -> Iterable[NDArray]:
        for sentence in _file_sentences(file_id, registry):
            yield self.embedding(sentence)

    def _calibrated_score(self, scores: float | NDArray) -> float | NDArray:
        return np.clip(100 * (scores - self.r_id) / (self.r_ood - self.r_id), 0, 100)

    def export_weights(self) -> dict[str, Any]:
        return {
            "cutoff_distance": self.cutoff_distance,
            "r_id": self.r_id,
            "r_ood": self.r_ood,
            "mean": self.distribution.location_,
            "covariance": self.distribution.covariance_,
        }

    def load_weights(self, weights: dict[str, Any]) -> None:
        self.cutoff_distance = weights["cutoff_distance"]
        self.r_id = weights["r_id"]
        self.r_ood = weights["r_ood"]
        self.distribution.location_ = weights["mean"]
        self.distribution._set_covariance(weights["covariance"])


def _file_sentences(file_id: str, registry: DataFilesRegistry) -> Iterable[str]:
    sentences = set()
    paragraphs = registry.load_items(file_id)
    for para in paragraphs:
        for sent in tk.to_sentences(para):
            sentence = _standardize_sentence(sent)
            if len(sentence):
                sentences.add(sentence)
    return sentences


def _standardize_sentence(sentence: str) -> str:
    words = []
    for word in nltk.word_tokenize(sentence):
        if word.isalpha():  # Maybe just punctuation?
            words.append(word.strip())
    return sentence.lower() if len(words) > 1 else ""
