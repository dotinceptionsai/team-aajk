import logging
import math
from collections import defaultdict
from typing import Iterable, Mapping, Collection, MutableMapping, Any

import pipelines.impl.preprocessing as tk
from dataload.dataloading import DataFilesRegistry
from pipelines.filtering import FilterPipeline, FilteredSentence

logger = logging.getLogger(__name__)


class _OkapiBM25:
    """
    Implementation of the Okapi BM25 algorithm as described here: https://en.wikipedia.org/wiki/Okapi_BM25
    Algorythm only uses (relative) frequency of words in documents to determine the most relevant one for a word:
    - the more often a word appears in a document, the more the document is relevant to the word
    - the more documents a word appears in, the less specific is the word and hence lower the relevance
    """

    def __init__(self, k1=1.5, b=0.75):
        self.avg_doc_len: float = 0.0
        self.doc_len: MutableMapping[str, int] = defaultdict(
            int
        )  # all document lengths per doc name
        self.n: MutableMapping[str, int] = defaultdict(int)  # doc count per word
        self.f: MutableMapping[str, MutableMapping[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # word counts per doc
        self.idf = defaultdict(float)  # IDF computation per word
        self.k1 = k1
        self.b = b

    def __repr__(self):
        return f"OkapiBM25(k1={self.k1}, b={self.b})"

    def __str__(self):
        return f"OkapiBM25(k1={self.k1}, b={self.b}) - Included documents: {sorted(self.doc_names())}"

    def doc_names(self):
        return self.doc_len.keys()

    def doc_count(self):
        return len(self.doc_len)

    def add_docs(self, docs: Mapping[str, Collection[str]]) -> None:
        for doc_name, doc_words in docs.items():
            self.add_doc(doc_name, doc_words)
        self._recompute_idfs()

    def add_doc(self, doc_name: str, doc_words: Collection[str]) -> None:
        self._add_single_doc(doc_name, doc_words)
        self._recompute_idfs()

    def _add_single_doc(self, doc_name: str, doc_words: Collection[str]) -> None:
        docs_count = self.doc_count() + 1
        doc = doc_words if isinstance(doc_words, list) else list(doc_words)

        self.avg_doc_len = (self.avg_doc_len * (docs_count - 1) + len(doc)) / docs_count
        self.doc_len[doc_name] = len(doc)

        fs = defaultdict(int)
        for word in set(doc):
            occurrences = doc.count(word)
            fs[word] = occurrences
            if occurrences > 0:
                self.n[word] += 1
        self.f[doc_name] = fs

    def _recompute_idfs(self) -> None:
        for word in self.n.keys():
            self.idf[word] = self._idf(word)

    def _idf(self, word: str) -> float:
        return math.log(
            (self.doc_count() - self.n[word] + 0.5) / (self.n[word] + 0.5) + 1
        )

    def get_score(self, doc_name: str, words_query: Iterable[str]) -> float:
        score = 0
        for wq in words_query:
            if wq in self.f[doc_name]:
                idf = self.f[doc_name][wq]
                relative_len = self.doc_len[doc_name] / self.avg_doc_len
                rel_len_adj = 1 - self.b + self.b * relative_len
                denom = idf + self.k1 * rel_len_adj
                word_score = self.idf[wq] * idf * (self.k1 + 1) / denom
                score += word_score
        return score

    def find_best_doc(
        self, words_query: Iterable[str], top_n: int = 1
    ) -> list[tuple[str, float]]:
        document_scores = [
            (doc_name, self.get_score(doc_name, words_query))
            for doc_name in self.doc_names()
        ]
        ranked_docs = sorted(document_scores, key=lambda x: x[1], reverse=True)
        return ranked_docs[:top_n]

    def ready(self):
        if self.doc_count() < 2:
            logger.error(
                f"There should be at least 2 documents, only: {self.doc_names()}"
            )
        return self.doc_count() > 1

    def export_weights(self) -> dict[str, Any]:
        return {
            "avg_doc_len": self.avg_doc_len,
            "doc_len": dict(self.doc_len),
            "n": dict(self.n),
            "f": {doc_name: dict(d_val) for doc_name, d_val in self.f.items()},
            "idf": dict(self.idf),
        }

    def import_weights(self, params: dict[str, Any]) -> None:
        self.avg_doc_len = params["avg_doc_len"]
        self.doc_len.update(params["doc_len"])
        self.n.update(params["n"])
        for doc_name, d_values in params["f"].items():
            self.f[doc_name].update(d_values)
        self.idf = defaultdict(float, params["idf"])


class Bm25FilterPipeline(FilterPipeline):
    sentence_parser: tk.TextTransformer = tk.pipelined(
        tk.split_in_sentences, tk.chunker(20)
    )
    tokenizer: tk.TextTransformer = tk.pipelined(
        tk.to_words, tk.no_stop_words, tk.lemmatize
    )
    model: _OkapiBM25 = None

    def _post_init(self, k1: float, b: float) -> None:
        self.model = _OkapiBM25(k1, b)

    def filter_sentences(self, text: str) -> Iterable[FilteredSentence]:
        if not self.model or not self.model.ready():
            raise RuntimeError(
                "Keyword Bm25 service is not ready. Please train or load it before."
            )

        for sentence in Bm25FilterPipeline.sentence_parser(text):
            in_domain, score = self._tag(Bm25FilterPipeline.tokenizer(sentence))
            yield FilteredSentence(sentence, in_domain, score)

    def _tag(self, sentence_tokens: Iterable[str]) -> tuple[bool, float]:
        top_2 = self.model.find_best_doc(sentence_tokens, top_n=2)
        best_doc, best_score = top_2[0]
        second_best_doc, second_best_score = top_2[1]
        relative_score_diff = (
            (best_score - second_best_score) / best_score if best_score > 0 else 0
        )
        in_domain = (
            best_doc.startswith("ID") and best_score > 0 and relative_score_diff > 0.01
        )
        return in_domain, best_score

    def fit(self, registry: DataFilesRegistry) -> None:
        self._train_on_document("ID", self.datasets.train_id, registry)
        self._train_on_document("OOD", self.datasets.train_ood, registry)

    def _train_on_document(
        self,
        category: str,
        source_file_ids: Collection[str],
        registry: DataFilesRegistry,
    ) -> None:
        words_by_filename: dict[str, list[str]] = {}

        for file_id in source_file_ids:
            file_content = registry.load_items(file_id)
            words_by_filename[file_id] = []
            for s in file_content:
                words_by_filename[file_id].extend(Bm25FilterPipeline.tokenizer(s))
        for filename, tokens in words_by_filename.items():
            self.model.add_doc(f"{category}_{filename}", tokens)

    def export_weights(self) -> dict[str, Any]:
        return self.model.export_weights()

    def load_weights(self, weights: dict[str, Any]) -> None:
        self.model.import_weights(weights)
