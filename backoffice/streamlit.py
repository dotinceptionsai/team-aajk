import base64
import streamlit as st
import pandas as pd
from pathlib import Path
from dataload.dataloading import DataFilesRegistry
import nltk
import svgling


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)


st.title("Initial analysis")


def find_named_enties(texts):
    entities = {}
    for text in texts:
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
    return entities


# Check datasets path
st.subheader("Datasets")
dataset_dir = Path(Path.cwd() / "datasets")
if not dataset_dir.exists():
    message = f"Datasets dir: {dataset_dir} , exists {dataset_dir.exists()}"
    st.write(message)

# List datasets and display in dataframe
st.write("Those are the datasets available:")
datasets = DataFilesRegistry(dataset_dir)
existing_datasets = {
    item: path
    for item, path in datasets.items()
    if "chatterbox" not in str(path) and "validation" not in str(path)
}
df_datasets = pd.DataFrame.from_dict(
    existing_datasets, orient="index", columns=["path"]
)

st.dataframe(df_datasets, use_container_width=True)

# Selectbox
option = st.selectbox("Select your dataset?", existing_datasets.keys())


if option:
    st.subheader(f"Analysis of selected dataset: {option}")
    paragraphs = datasets.load_items(option)
    print(f"Loaded {len(paragraphs)} paragraphs")

    df_raw = pd.Series(paragraphs)
    st.dataframe(df_raw, use_container_width=True)

    st.subheader("Single sentence named entities")
    line_number = st.number_input(
        "Insert a line number from the dataframe",
        min_value=0,
        max_value=len(df_raw) - 1,
        step=1,
    )
    st.write("The current number is ", line_number)

    if line_number is not None:
        chunked = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(df_raw[line_number])))

        svgling.disable_nltk_png()

        # Display the image in streamlit
        tag_tree = svgling.draw_tree(chunked)
        svg = str(tag_tree.get_svg().tostring())
        render_svg(svg)

    entities = find_named_enties(df_raw)
    df_entities = pd.DataFrame(
        {"entity_name": entities.keys(), "entity_type": entities.values()}
    )

    from collections import Counter
    from pipelines.impl.paragraph import ParagraphTransform
    from pipelines.impl.preprocessing import no_stop_words, to_lowercase, to_words, stem

    lp = ParagraphTransform(
        [
            ("to_words", to_words),
            ("lower", to_lowercase),
            ("no_stop_words", no_stop_words),
        ]
    ).as_scikit_transformer(unique=False)

    top_words = Counter(lp.transform(df_raw)).most_common(10)
    df_top_counts = pd.DataFrame(top_words, columns=["word", "count"])

    # Join together the top words and the named entities on the lower case version of the word
    df_top_counts["word_lower"] = df_top_counts["word"].str.lower()
    df_entities["word_lower"] = df_entities["entity_name"].str.lower()
    df_joined = df_top_counts.merge(df_entities, on="word_lower", how="inner")
    df_ners = df_joined[["entity_name", "entity_type", "count"]]

    st.subheader("Top named entities")
    st.dataframe(df_ners, use_container_width=True)

    # Normalizers
    st.subheader("Sentence splitting analysis")
    st.caption(
        "Following cleanups are applied to help the sentence splitter to work better"
    )

    from pipelines.impl.preprocessing import (
        make_text_normalizer,
        split_into_sentences,
        get_available_normalizers,
        normalize_sent_min_words,
    )
    from pipelines.impl.paragraph import ParagraphTransform

    all_cleanups = get_available_normalizers()
    default_cleanups = ["spaces", "sentences_start", "uri", "email", "common_abbr"]
    active_cleanups = st.multiselect("Select cleanups:", all_cleanups, default_cleanups)
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
                ("text_normalizer", make_text_normalizer(active_cleanups)),
                ("sentences", split_into_sentences),
                ("normalized_sentences", normalize_sent_min_words),
            ]
        ).as_scikit_transformer(
            unique=True
        )  # only unique sentences

        df_sentences = preprocessing_pipe.transform(df_raw)

        # print Sentences length vs df_raw length
        st.write(f"Unique Sentences count: {len(df_sentences)}")
        st.write(f"Paragraphs (or sections) count: {len(df_raw)}")

        st.dataframe(df_sentences, use_container_width=True)
