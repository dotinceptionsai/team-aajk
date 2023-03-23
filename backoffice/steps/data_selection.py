import streamlit as st

import backoffice.shared as shared
from backoffice.session import SessionKey


def display():
    st.write(
        "Those are the datasets available. They have been scraped from the internet. In the future you will get the option to upload your FAQ:"
    )

    existing_datasets = shared.existing_datasets()
    dataset_descriptions = {
        "europcar": "EUROPCAR, faq of a car rental company from https://faq.europcar.com/",
        "nih": "NIH, helps innovators turn discovery into health, from https://seed.nih.gov/faqs",
        "olympics": "OLYMPICS, Knowledge base on Olympics from https://olympics.com/ioc/faq",
        "fda": "FDA faq about COVID, from https://www.fda.gov/emergency-preparedness-and-response/coronavirus-disease-2019-covid-19/covid-19-frequently-asked-questions",
        "wwf": "WWF, World Wild Life Foundations faq, from https://www.wwf.org.uk/faqs",
    }

    st.radio(
        "Select one from following knowledge bases: ",
        existing_datasets,
        key=SessionKey.SELECTED_DATASET,
        format_func=lambda x: dataset_descriptions.get(x, x),
        on_change=lambda: st.session_state.pop(SessionKey.SELECTED_BASE_MODEL, None),
    )
