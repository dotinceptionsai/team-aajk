from pipelines.impl.paragraph import ParagraphTransform
from pipelines.impl.preprocessing import (
    normalize_sentences_start,
    normalize_spaces,
    normalize_uri,
    normalize_common_abbr,
    split_into_sentences,
    normalize_sent_min_words,
    normalize_hyphenated_words,
    make_text_normalizer,
)


def test_tidy_sentence_start():
    cleaned = normalize_sentences_start(
        "This phrase is good.This should start with space."
    )
    assert cleaned == "This phrase is good. This should start with space."


def test_tidy_spaces():
    cleaned = normalize_spaces("This phrase    too much.")
    assert cleaned == "This phrase too much."


def test_normalize_uri():
    assert (
        normalize_uri(
            "An ugly http://www.europcar.com/station-finder/ in it.",
        )
        == "An ugly WEBLINK in it."
    )
    assert (
        normalize_uri(
            "An ugly https://www.europcar.com/station-finder in it.",
        )
        == "An ugly WEBLINK in it."
    )
    assert (
        normalize_uri(
            "An ugly www.europcar.com/ in it.",
        )
        == "An ugly WEBLINK in it."
    )


def test_normalize_common_abbr():
    cleaned = normalize_common_abbr("This sentence has an i.e. in it.")
    assert cleaned == "This sentence has an for instance in it."


def test_normalize_hyphenated_words():
    cleaned = normalize_hyphenated_words("This sentence is best-in-class.")
    assert cleaned == "This sentence is best in class."


def test_text_normalizer():
    data = [
        "This sentence has an i.e. in it.But a second sentence also!",
        "This sentence has a https://www.europcar.com/station-finder/ in it.",
        "But a second sentence also!",
    ]

    lazy_pipe = ParagraphTransform(
        [
            ("text_normalizer", make_text_normalizer()),
            ("sentences", split_into_sentences),
            ("normalized_sentences", normalize_sent_min_words),
        ]
    )

    scikit_pipe = lazy_pipe.as_scikit_transformer(unique=True)

    lazy_results = list(lazy_pipe(data))
    scikit_results = scikit_pipe.transform(data)

    assert len(scikit_results) == 3  # Remove duplicates in batch mode

    assert lazy_results[:3] == scikit_results
