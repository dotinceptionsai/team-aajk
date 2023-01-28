from typing import Iterable, Callable

import nltk

nltk.download("wordnet")
nltk.download("stopwords")

stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.WordNetLemmatizer()

""" Transforms a string to a sequence of strings. """
TextTransformer = Callable[[str], Iterable[str]]


#
# Provides simple parsing routines for text and a way to chain them together.
# Chained text transformations are lazy, i.e. they do not parse the whole sentence until it is needed.
#

# todo may be use the sklearn abstraction instead (at the risk of loosing laziness the lazy feature) ?
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


def to_sentences(text) -> Iterable[str]:
    """Splits a text into sentences."""
    for sentence in nltk.sent_tokenize(text):
        for s in sentence.splitlines():
            if len(s) > 1:
                yield s.replace("ex.", "for instance ").replace(
                    "i.e.", "for instance "
                ).strip()


def chunker(max_words: int) -> TextTransformer:
    """Splits a sentence into chunks of at most max_words words."""

    def _chunk(sentence: str) -> Iterable[str]:
        words = sentence.split()
        return [
            " ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)
        ]

    return _chunk


def to_words(text: str) -> Iterable[str]:
    """Splits a sentence into words."""
    for word in nltk.word_tokenize(text):
        yield word


def to_lowercase(text: str) -> Iterable[str]:
    """Converts a text to lowercase."""
    yield text.lower()


def no_stop_words(token: str) -> Iterable[str]:
    """Removes stop words."""
    if token.isalpha() and not token.lower() in stop_words:
        yield token


def lemmatize(token: str) -> Iterable[str]:
    """Lemmatizes a word."""
    yield lemmatizer.lemmatize(token.lower())
