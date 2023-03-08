from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

""" Transforms a string to a sequence of strings. """
TextTransformer = Callable[[str], Iterable[str]]


def identity_transform(text: str) -> Iterable[str]:
    yield text


class ParagraphTransform(BaseEstimator, TransformerMixin):
    """A pipeline of text transformers chained together"""

    def __init__(
        self,
        text_transformers: Iterable[TextTransformer] = (identity_transform,),
        unique_sentences: bool = True,
    ):
        self.text_transformers = list(text_transformers)
        self.unique_sentences = unique_sentences
        self.lazy_func = None
        self.eager_func = None

    def __do_init(self):
        if not self.lazy_func:
            self.lazy_func = pipelined(*self.text_transformers)
            transform = to_scikit_transformer(self.lazy_func)
            deduplicate = FunctionTransformer(unique_texts)
            self.eager_func = (
                make_pipeline(transform, deduplicate)
                if self.unique_sentences
                else transform
            )

    def transform_one(self, text) -> Iterable[str]:
        self.__do_init()
        return self.lazy_func(text)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.__do_init()
        return self.eager_func.transform(X)


def unique_texts(texts: Iterable[str]) -> Iterable[str]:
    uniq_texts = dict.fromkeys(texts)
    return _cast_to_best_type(texts, list(uniq_texts.keys()))


def _cast_to_best_type(
    iterable_param: Iterable[str], iterable_result: Iterable[str]
) -> Iterable[str]:
    # Try to keep output type same as input type
    if isinstance(iterable_param, np.ndarray):
        return np.array(iterable_result)
    elif isinstance(iterable_param, pd.Series):
        return pd.Series(iterable_result)
    else:
        return iterable_result


def to_scikit_transformer(transform: TextTransformer) -> FunctionTransformer:
    """Converts a text transformer to a scikit-learn transformer."""

    def _batched(texts: Iterable[str]) -> Iterable[str]:
        results = [transformed for text in texts for transformed in transform(text)]
        return _cast_to_best_type(texts, results)

    return FunctionTransformer(_batched)


def pipelined(*text_transformers: TextTransformer) -> TextTransformer:
    """Chains text transformations together."""

    def _chain_rec(s: str, i: int) -> Iterable[str]:
        if i == len(text_transformers):
            yield s
        else:
            for t in text_transformers[i](s):
                yield from _chain_rec(t, i + 1)

    def _chained(s: str) -> Iterable[str]:
        return _chain_rec(s, 0)

    return _chained
