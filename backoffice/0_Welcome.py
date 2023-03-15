import pandas as pd
import streamlit as st

import backoffice.shared as shared

st.set_page_config(
    page_title="Fantastic Filter",
    page_icon="👋",
    layout="wide",
)

st.write("# Fantastic Filter🚀 Setup Wizard")

st.markdown(
    """
    The Fantastic Filter🚀is an application that helps Call-Centers to filter live-conversations between an employee and a customer. It tells
    which sentences are relevant to their domain (their FAQs) and which are likely not. 
    
    In general, Call-Centers already own a question-answering software that searches their FAQs or Knowledge Base. However, those systems are slow and cannot afford to run on each and every utterance spoken by the customer or the employee. It is here that the Fantastic Filter comes in:
    it is a fast pre-filtering on the conversation that keeps only sentences that are likely to be relevant to the downstream question-answering system.
    
    This backoffice application will guide you through the setup of the Fantastic Filter for your Call-Center.
    
    For this demo app, we only allow a pre-defined set of FAQs that are listed below. Each of those have been scraped from the internet. In the future, we will allow you to upload your own knowledge base.
    
    **👈 Select a dataset on which you want to work!**
"""
)

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

# Selectbox
with st.sidebar:
    pre_selection = shared.preselected_index("selected_dataset", existing_datasets)
    st.session_state.selected_dataset = st.selectbox(
        "Select your FAQ?", existing_datasets, index=pre_selection
    )
    print(f"Selected FAQ: {st.session_state.selected_dataset}")
