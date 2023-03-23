import nltk
from typing import Callable, Collection, Iterable, Protocol
from analysis.words import find_top_named_entities
import pandas as pd


class WordTokenizer(Protocol):
    def tokenize(self, word: str) -> Collection[str]:
        ...


def find_important_split_words(texts: Collection[str], tokenizer: WordTokenizer):
    df_important_words = find_top_named_entities(texts)
    df_split_words = find_vocab_split_words(texts, tokenizer)
    te = df_important_words["entity_name"].values
    sw = df_split_words["word"].values
    inter = _intersect(te, sw)
    return pd.DataFrame({"word": list(inter)})


def find_vocab_split_words(texts: Collection[str], tokenizer: WordTokenizer):
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
