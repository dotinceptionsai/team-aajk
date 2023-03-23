import pandas as pd
import streamlit as st

from backoffice import shared
from backoffice.session import SessionKey
from pipelines.impl.paragraph import ParagraphTransform
from pipelines.impl.preprocessing import (
    make_text_normalizer,
    split_into_sentences,
    get_available_normalizers,
    normalize_sent_min_words,
)

datasets = shared.get_ds_registry()


def display():
    st.write(
        "Our filter relies on proper sentence splitting of sentences. Sentence splitting is not as trivial as you may think. This page helps you to check that your dataset is properly split into sentences."
    )

    if SessionKey.SELECTED_DATASET in st.session_state:
        ds = st.session_state[SessionKey.SELECTED_DATASET]
        paragraphs = datasets.load_items(ds)
        se_paragraphs = pd.Series(paragraphs)

        # Normalizers
        st.subheader("Preprocessing steps to apply")
        st.caption(
            "Following cleanups are applied to help the sentence splitter to work better"
        )

        all_cleanups = get_available_normalizers()
        default_cleanups = ["spaces", "sentences_start", "uri", "email", "common_abbr"]
        active_cleanups = st.multiselect(
            "Select normalizers to see effects on sentence splits:",
            all_cleanups,
            default_cleanups,
        )
        st.caption(":blue[spaces] replaces series of spaces and tabs by a single space")
        st.caption(
            ":blue[sentences_starts] Ensures a sentence end is followed by a space and a Capital letter"
        )
        st.caption(":blue[uri] Replaces URLs by the placeholder WEBLINK")
        st.caption(":blue[email] Removes all emails by the placeholder WEBMAIL")
        st.caption(
            ":blue[common_abbr] Expands common abbreviations so that no dot remains"
        )
        st.caption(":blue[hyphens] Replaces hyphens in hyphenated words by spaces")

        if active_cleanups and len(active_cleanups):
            preprocessing_pipe = ParagraphTransform(
                [
                    make_text_normalizer(active_cleanups),
                    split_into_sentences,
                    normalize_sent_min_words,
                ],
                unique_sentences=True,
            )  # only unique sentences
            df_sentences = preprocessing_pipe.transform(se_paragraphs)

            st.write(f"Unique Sentences count: {len(df_sentences)}")
            st.write(f"Paragraphs (or sections) count: {len(se_paragraphs)}")
            st.dataframe(df_sentences, use_container_width=True)
    else:
        shared.go_back_to("Data Selection", "Select dataset first")
