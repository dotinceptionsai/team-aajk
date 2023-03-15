import logging
import re
from typing import Callable, Iterable, Optional, Sequence

import nltk

from pipelines.impl.paragraph import TextTransformer

logger = logging.getLogger(__name__)

nltk.download("wordnet")
nltk.download("stopwords")

stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.WordNetLemmatizer()

SPACE_PATTERN = re.compile(r"\s+")
SENTENCE_START_PATTERN = re.compile(r"([.?!])([A-Z])([ a-z])")
URL_PATTERN = re.compile(
    "(?:https?:\\/\\/)?(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
)
EMAIL_PATTERN = re.compile(
    r"([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+"
)
HYPHENED_WORD_PATTERN = re.compile(r"\b([a-zA-Z0-9]+)(-[a-zA-Z0-9]+)+\b")
EMAIL_PLACEHOLDER = "WEBMAIL"
URL_PLACEHOLDER = "WEBLINK"
PLACEHOLDERS = [URL_PLACEHOLDER, EMAIL_PLACEHOLDER]

# Common abbreviations that may fool sentence splitting
COMMON_ABBR = {
    "e.g.": "for example",
    "i.e.": "for instance",
    "cf.": "confer",
    "def.": "definition",
    "fig.": "figure",
    "et al.": "and others",
    "ex.": "for example",
    "etc.": "and so on",
    "vs.": "versus",
}

TextNormalizer = Callable[[str], str]

_available_normalizers: dict[str, TextNormalizer] = {}


def get_available_normalizers() -> list[str]:
    return list(_available_normalizers.keys())


def normalizer(name: Optional[str] = None):
    """Decorator for registering a normalizer function."""

    def _register(func):
        fn = name or func.__name__
        _available_normalizers[fn] = func
        return func

    return _register


# Normalizers for a big text document
def make_text_normalizer(
    normalizer_keys: Sequence[str] | None = None,
) -> TextTransformer:
    """Create a text normalizer from a list of normalizers."""
    if normalizer_keys is None:
        to_use = list(_available_normalizers.values())
    else:
        for k in normalizer_keys:
            if k not in _available_normalizers:
                raise ValueError(f"Unknown normalizer with key: {k}")
        to_use = [_available_normalizers[n] for n in normalizer_keys]

    def _normalizer(text: str) -> Iterable[str]:
        for norm in to_use:
            text = norm(text)
        yield text

    return _normalizer


@normalizer("spaces")
def normalize_spaces(text: str) -> str:
    return SPACE_PATTERN.sub(" ", text)


@normalizer("sentences_start")
def normalize_sentences_start(text: str) -> str:
    return SENTENCE_START_PATTERN.sub(r"\1 \2\3", text)


@normalizer("uri")
def normalize_uri(text: str) -> str:
    return URL_PATTERN.sub(URL_PLACEHOLDER, text)


@normalizer("email")
def normalize_email(text: str) -> str:
    return EMAIL_PATTERN.sub(EMAIL_PLACEHOLDER, text)


@normalizer("common_abbr")
def normalize_common_abbr(text: str) -> str:
    for k, v in COMMON_ABBR.items():
        text = text.replace(k, v)
    return text


@normalizer("hyphens")
def normalize_hyphenated_words(text: str) -> str:
    words = []
    for m in HYPHENED_WORD_PATTERN.finditer(text):
        words.append(text[m.start() : m.end()])
    for word in words:
        text = text.replace(word, word.replace("-", " "))
    return text


# Break down routines
def split_into_sentences(text) -> Iterable[str]:
    """Splits a text into sentences."""
    for sentence in nltk.sent_tokenize(text):
        for s in sentence.splitlines():
            yield s.strip()


def chunker(max_words: int) -> TextTransformer:
    """Splits a sentence into chunks of at most max_words words."""

    def _chunk(sentence: str) -> Iterable[str]:
        words = sentence.split()
        return [
            " ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)
        ]

    return _chunk


# Inside sentence cleanups


def normalize_sent_min_words(sentence: str) -> Iterable[str]:
    word_count = 0
    for word in nltk.word_tokenize(sentence):
        if word in PLACEHOLDERS:
            word_count += 1
        elif word.replace(".", "").isalpha():  # keep abbreviations
            word_count += 1
        elif (
            word.replace("-", "").isalnum() and word[0].isalpha()
        ):  # keep hyphenated and normal words
            word_count += 1
    if word_count > 1:
        yield sentence


# Legacy stuff below
def to_words(text: str) -> Iterable[str]:
    """Splits a sentence into words."""
    for word in nltk.word_tokenize(text):
        yield word


def to_lowercase(text: str) -> Iterable[str]:
    """Converts a text to lowercase."""
    yield text.lower()


def no_stop_words(token: str) -> Iterable[str]:
    """Removes stop words."""
    if len(token) > 1 and token.lower() not in stop_words:
        yield token


@normalizer("no_stop_words")
def no_stop_words_normalizer(text: str) -> str:
    """Removes stop words."""
    WORD_PATTERN = re.compile(r"\b\w\w+\b")
    parts = []
    pos = 0
    for m in WORD_PATTERN.finditer(text):
        w = text[m.start() : m.end()]
        parts.append(text[pos : m.start()])
        if len(w) > 1 and w.lower() not in stop_words:
            parts.append(w)
        pos = m.end()

    return "".join(parts)


def lemmatize(token: str) -> Iterable[str]:
    """Lemmatizes a word."""
    yield lemmatizer.lemmatize(token.lower())


porter_stemmer = nltk.stem.porter.PorterStemmer()


def stem(token: str) -> Iterable[str]:
    """Stems a word."""
    yield porter_stemmer.stem(token.lower())
