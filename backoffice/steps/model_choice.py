import streamlit as st

from analysis.experiments import ExperimentRegistry
from backoffice import shared
from backoffice.session import SessionKey

datasets = shared.get_ds_registry()


def display():
    st.write(
        "Choose between a faster model that is less accurate or a slower one but more accurate."
    )

    if SessionKey.SELECTED_DATASET in st.session_state:
        ds = st.session_state[SessionKey.SELECTED_DATASET]

        experiments = ExperimentRegistry()
        df_existing_models = experiments.get_all_runs(experiment_name=ds)
        visible_columns = [
            "id",
            "comparison_score",
            "speed_ms",
            "log_loss",
            "f1",
            "start_time",
            "embedder_name",
            "support_fraction",
            "robust_covariance",
            "train_id",
            "val_id",
            "val_ood",
        ]
        df_existing_models = df_existing_models[visible_columns]
        existing_model_ids = df_existing_models["id"].values.tolist()

        # display a dataframe from existing_models and allow the user to select one. Display the selected one in the sidebar
        st.subheader(f"Existing models for dataset: {ds}")

        st.selectbox(
            "Which model would you like to use?",
            existing_model_ids,
            key=SessionKey.SELECTED_BASE_MODEL,
        )
        st.dataframe(df_existing_models, use_container_width=True)
        st.write("Selected model: ", st.session_state[SessionKey.SELECTED_BASE_MODEL])
    else:
        shared.go_back_to("Data Selection", "Select dataset first")
