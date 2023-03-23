from collections import Counter
from typing import Iterable

import nltk
import pandas as pd

from pipelines.impl.paragraph import ParagraphTransform
from pipelines.impl.preprocessing import no_stop_words, to_lowercase, to_words


def find_named_enties(paragraphs: Iterable[str]):
    """Lists named entites as a dataframe with columns entity_name and entity_type"""
    entities = {}
    for text in paragraphs:
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text))):
            # print if the entity has tag NNP (proper noun)
            if hasattr(chunk, "label") and chunk[0][1] == "NNP":
                entity_name = " ".join(c[0] for c in chunk)
                entity_type = chunk.label()
                entities[entity_name] = entity_type
            elif type(chunk) == tuple:
                word, tag = chunk
                if tag == "NNP" and len(word) > 1:
                    entity_name = word
                    entities[entity_name] = "OTHER"

    return pd.DataFrame(
        {"entity_name": entities.keys(), "entity_type": entities.values()}
    )


def find_most_frequent_words(paragraphs: Iterable[str], n: int = 10):
    """Returns a dataframe with the most frequent words in the paragraphs: columns are named word and count"""
    lp = ParagraphTransform(
        [to_words, to_lowercase, no_stop_words], unique_sentences=False
    )

    top_words = Counter(lp.transform(paragraphs)).most_common(n)
    return pd.DataFrame(top_words, columns=["word", "count"])


def find_top_named_entities(paragraphs: Iterable[str], n: int = 10):
    """Returns a dataframe of the top n named entities: columns are entity_name, entity_type and count"""
    df_entities = find_named_enties(paragraphs)
    df_top_counts = find_most_frequent_words(paragraphs, n)

    # Join together the top words and the named entities on the lower case version of the word
    df_top_counts["word_lower"] = df_top_counts["word"].str.lower()
    df_entities["word_lower"] = df_entities["entity_name"].str.lower()
    df_joined = df_top_counts.merge(df_entities, on="word_lower", how="inner")
    return df_joined[["entity_name", "entity_type", "count"]]
