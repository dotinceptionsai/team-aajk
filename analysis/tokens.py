from typing import Collection, Iterable, Protocol

import nltk
import pandas as pd

from analysis.words import find_top_named_entities


class WordTokenizer(Protocol):
    def tokenize(self, word: str) -> Collection[str]:
        ...


def find_important_split_words(
    texts: Collection[str], tokenizer: WordTokenizer
) -> pd.DataFrame:
    """Returns a dataframe of words that are both important named entities and a split-word int the Huggingface tokenizer.
    The dataframe has a single column named word.
    """
    df_important_words = find_top_named_entities(texts)
    df_split_words = find_vocab_split_words(texts, tokenizer)
    te = df_important_words["entity_name"].values
    sw = df_split_words["word"].values
    inter = _intersect(te, sw)
    return pd.DataFrame({"word": list(inter)})


def find_vocab_split_words(
    texts: Collection[str], tokenizer: WordTokenizer
) -> pd.DataFrame:
    """A split word is a word that is split into multiple sub-tokens by the tokenizer.
    Sub-tokenization works by putting most frequent words (so understood words) in the vocabulary, and then splitting the rest of the words in sub-tokens.
    A model that has a tokenizer that has our important words split into sub-tokens means it has not seen much those words in the training data, and is not likely to be able to understand them well.
    Returns a dataframe with two columns: word and tokens.
    """
    split_words = dict()
    for sent in texts:
        for word in nltk.word_tokenize(sent):
            tokens = tokenizer.tokenize(word)
            if len(tokens) > 1:
                split_words[word] = ", ".join(tokens)
    # return a dataframe
    return pd.DataFrame({"word": split_words.keys(), "tokens": split_words.values()})


def _intersect(words_list1: Iterable[str], words_list2: Iterable[str]) -> set[str]:
    return set(words_list1).intersection(set(words_list2))
