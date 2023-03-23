import pandas as pd
import streamlit as st

import backoffice.shared as shared
from backoffice.session import SessionKey


st.session_state.update(st.session_state)

st.write("# Data Selection📚")

shared.make_sidebar()
shared.make_header(0)
datasets = shared.get_ds_registry()

# List datasets and display in dataframe
st.write(
    "Those are the datasets available. They have been scraped from the internet. In the future you will get the option to upload your FAQ:"
)


@st.cache_data
def existing_datasets():
    dss = shared.get_ds_registry()
    return [
        item
        for item, path in dss.items()
        if "chatterbox" not in str(path) and "validation" not in str(path)
    ]


existing_datasets = existing_datasets()

df_datasets = pd.DataFrame({"dataset": existing_datasets})

# existing_datasets as list
st.write("## List of available FAQs")
for x in existing_datasets:
    st.markdown(f"- {x}")

st.selectbox(
    "Select your FAQ?",
    existing_datasets,
    key=SessionKey.SELECTED_DATASET,
)

shared.go_back_to("Checks at Word level", "Continue to next step")
