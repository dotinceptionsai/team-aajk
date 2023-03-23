import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from analysis import words
from analysis.tokens import find_important_split_words
from backoffice import shared
from backoffice.session import SessionKey
from pipelines.impl import preprocessing
from pipelines.impl.paragraph import ParagraphTransform


def find_dangerous_words(para_series: pd.Series):
    pt = ParagraphTransform(
        [
            preprocessing.make_text_normalizer(
                ["spaces", "sentences_start", "uri", "email", "common_abbr"]
            ),
            preprocessing.split_into_sentences,
            preprocessing.normalize_sent_min_words,
        ]
    )
    tokenizer = SentenceTransformer("all-MiniLM-L6-v2").tokenizer
    return find_important_split_words(pt.transform(para_series), tokenizer)


def display():
    datasets = shared.get_ds_registry()
    if SessionKey.SELECTED_DATASET in st.session_state:
        ds = st.session_state[SessionKey.SELECTED_DATASET]
        st.subheader(f"Paragraphs of dataset {ds}")

        with st.expander("Show raw data"):
            paragraphs = datasets.load_items(ds)
            se_paragraphs = pd.Series(paragraphs)
            st.write(f"Found {len(paragraphs)} paragraphs")
            st.dataframe(se_paragraphs, use_container_width=True)

        st.subheader("Top named entities")
        with st.spinner("Computing top named entities"):
            df_important_words = words.find_top_named_entities(se_paragraphs)
            st.dataframe(df_important_words, use_container_width=False)

        st.subheader("Important words that the model may not well understand")
        st.caption(
            "This is typical specialized words that pre-trained models think are rare and do not understand well."
        )

        with st.spinner("Checking for vocabulary coverage..."):
            dangerous_splits = find_dangerous_words(se_paragraphs)
            st.dataframe(dangerous_splits, use_container_width=False)
            if len(dangerous_splits) > 0:
                st.warning(
                    "The model may not understand these words. Consider triggering a retraining of the model to extend its vocabulary understanding."
                )
            else:
                st.success(
                    "The model seems to understand well all the important words in the dataset."
                )

    else:
        shared.go_back_to("Data Selection", "Select dataset first")
