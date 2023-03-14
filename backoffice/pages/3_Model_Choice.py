from pathlib import Path

import mlflow
import streamlit as st

from analysis import experiments
from analysis.experiments import to_dataframe
from backoffice import shared
from backoffice.shared import make_sidebar

datasets = shared.get_ds_registry()

st.title("Model Choice 🪄")

st.write(
    "Choose between a faster model that is less accurate or a slower one but more accurate."
)

make_sidebar()


if "selected_dataset" in st.session_state:
    mlruns = str(str(Path("train/mlruns").absolute()))
    st.write("Models are served from: ", mlruns)
    mlflow.set_tracking_uri("file://" + mlruns)

    ds = st.session_state.selected_dataset
    existing_models = experiments.get_all_runs(experiment_name=ds)

    # display a dataframe from existing_models and allow the user to select one. Display the selected one in the sidebar
    st.subheader(f"Existing models for dataset: {ds}")

    selected_model_idx = shared.preselected_index("selected_model", existing_models)
    st.session_state.selected_model = st.selectbox(
        "Which model would you like to use?",
        existing_models["id"].values,
        index=selected_model_idx,
    )
    st.dataframe(existing_models, use_container_width=True)
