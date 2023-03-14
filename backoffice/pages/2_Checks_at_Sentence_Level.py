import pandas as pd
import streamlit as st

from backoffice import shared
from backoffice.shared import make_sidebar
from pipelines.impl.paragraph import ParagraphTransform
from pipelines.impl.preprocessing import (
    make_text_normalizer,
    split_into_sentences,
    get_available_normalizers,
    normalize_sent_min_words,
)

datasets = shared.get_ds_registry()

st.title("Sentence Level Checks 🔎")

st.write(
    "Our filter relies on proper sentence splitting of sentences. Sentence splitting is not as trivial as you may think. This page helps you to check that your dataset is properly split into sentences."
)


make_sidebar()

if "selected_dataset" in st.session_state:
    ds = st.session_state.selected_dataset
    paragraphs = datasets.load_items(ds)
    se_paragraphs = pd.Series(paragraphs)

    # Normalizers
    st.subheader("Sentence splitting analysis")
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
    st.caption(":blue[common_abbr] Expands common abbreviations so that no dot remains")
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
    st.subheader("Select a dataset to start")
