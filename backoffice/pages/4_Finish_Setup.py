from pathlib import Path

import mlflow
import streamlit as st

from analysis import experiments
from analysis.experiments import to_dataframe
from backoffice import shared
from backoffice.shared import make_sidebar

datasets = shared.get_ds_registry()

st.title("Setup Complete 🎉")

st.write(
    "Bravo! You have successfully setup your environment. You can now start using the filter."
)

make_sidebar()


st.balloons()
