import base64
import streamlit as st
from pathlib import Path

from dataload.dataloading import DataFilesRegistry


def preselected_index(state_key, keys, default_idx=0):
    if state_key in st.session_state:
        for i, k in enumerate(keys):
            if k == st.session_state.selected_dataset:
                return i
    return default_idx


def make_sidebar():
    with st.sidebar:
        ds = (
            st.session_state.selected_dataset
            if "selected_dataset" in st.session_state
            else "None"
        )
        if ds:
            st.markdown(f"Selected dataset: **{ds}**")

        model = (
            st.session_state.selected_model
            if "selected_model" in st.session_state
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


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)
