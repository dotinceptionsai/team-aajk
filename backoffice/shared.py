import base64
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import RerunData, RerunException
from streamlit.source_util import get_pages

from backoffice.session import SessionKey
from dataload.dataloading import DataFilesRegistry


def selected_index(state_key: SessionKey, keys: list[str], default_idx=0) -> int:
    if state_key in st.session_state:
        try:
            return keys.index(st.session_state[state_key])
        except ValueError:
            return default_idx
    else:
        return default_idx


def make_header(progress):
    import extra_streamlit_components as stx

    val = stx.stepper_bar(
        steps=["Import data", "Verify data", "Select Model", "Calibrate", "Verify"]
    )
    return val


def make_sidebar():
    with st.sidebar:
        ds = (
            st.session_state[SessionKey.SELECTED_DATASET]
            if SessionKey.SELECTED_DATASET in st.session_state
            else "None"
        )
        if ds:
            st.markdown(f"Selected dataset: **{ds}**")

        model = (
            st.session_state[SessionKey.SELECTED_BASE_MODEL]
            if SessionKey.SELECTED_BASE_MODEL in st.session_state
            else "None"
        )
        if model:
            st.markdown(f"Selected model: **{model}**")


# checks if datasets dir exists
def get_ds_registry():
    dataset_dir = Path(Path.cwd() / "datasets")
    if not dataset_dir.exists():
        message = f"Datasets dir: {dataset_dir} , exists {dataset_dir.exists()}"
        st.write(message)

    return DataFilesRegistry(dataset_dir)


@st.cache_data
def existing_datasets():
    dss = get_ds_registry()
    return [
        item
        for item, path in dss.items()
        if "chatterbox" not in str(path) and "validation" not in str(path)
    ]


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)


def link_to(link_text: str, page_name: str):
    st.markdown(
        f'<a href="/{page_name}" target="_self">{link_text}</a>', unsafe_allow_html=True
    )


def go_back_to(page_name: str, message: str):
    st.subheader(message)
    redirect_button = st.button(f"Go back to {page_name}")
    if redirect_button:
        switch_page(page_name)


# Copied from streamlit extras to avoid adding big dependency
def switch_page(page_name: str):
    def standardize_name(name: str) -> str:
        return name.lower().replace("_", " ")

    page_name = standardize_name(page_name)

    pages = get_pages("streamlit_app.py")  # OR whatever your main page is called

    for page_hash, config in pages.items():
        if standardize_name(config["page_name"]) == page_name:
            raise RerunException(
                RerunData(
                    page_script_hash=page_hash,
                    page_name=page_name,
                )
            )

    page_names = [standardize_name(config["page_name"]) for config in pages.values()]

    raise ValueError(f"Could not find page {page_name}. Must be one of {page_names}")
