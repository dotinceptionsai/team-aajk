from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

from analysis.experiments import ExperimentRegistry
from backoffice import shared
from backoffice.session import SessionKey
from pipelines import persistence


def display():
    st.write(
        "Bravo! You have successfully setup your environment. You can now start using the filter."
    )

    if SessionKey.SELECTED_DATASET not in st.session_state:
        shared.go_back_to("Data Selection", "Select dataset first")
    elif SessionKey.SELECTED_BASE_MODEL not in st.session_state:
        shared.go_back_to("Model Choice", "Select base model first")
    else:
        ds = st.session_state[SessionKey.SELECTED_DATASET]
        base_model_name = st.session_state[SessionKey.SELECTED_BASE_MODEL]

        experiments = ExperimentRegistry()
        run_info = experiments.get_run_info(
            experiment_name=ds, run_name=base_model_name
        )

        pipeline_dir = Path(run_info.artifact_dir.replace("file:/", "/"))
        model = persistence.load_pipeline(pipeline_dir)

        downloadable_archive = experiments.build_archive(
            experiment_name=ds, run_name=base_model_name
        )

        with open(downloadable_archive, "rb") as f:
            st.caption(
                '<i class="fa-solid fa-cloud-arrow-down" style="font-size:25px;color:purple"></i> Model can be imported in front-office app',
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download model",
                f.read(),
                file_name=f"{run_info.id}-{datetime.now().isoformat()}.tar.gz",
                mime="application/x-gzip-compressed",
            )

        with st.expander("Base model details"):
            st.write(persistence._create_model_dict(model))

            def predict_one(sentences):
                embeddings = model.embedder.transform(sentences)
                raw_scores = model.distribution.transform(embeddings)
                ood_probas = model.calibrator.predict_proba(raw_scores)
                # Concatenate iid and ood as a 2D array
                new_var = np.vstack([1 - ood_probas, ood_probas]).T
                return new_var

        with st.expander("I want to see some explanations"):
            from lime.lime_text import LimeTextExplainer

            explainer = LimeTextExplainer(class_names=["no", "yes"])

            def explain(sentence, num_features=4):
                return explainer.explain_instance(
                    sentence, predict_one, num_features=num_features
                )

            def scale(lst):
                # Find the maximum absolute value in the list
                max_val = max(abs(max(lst)), abs(min(lst)))

                # If the list contains only zeros, return the original list
                if max_val == 0:
                    return lst

                # Scale each value in the list by the maximum absolute value
                scaled_lst = [x / max_val for x in lst]

                return scaled_lst

            btn2 = st.button("Explain")
            sentence = st.text_input(
                "Sentence to explain",
                value="I would like to rent a car",
            )
            with st.spinner("Explaining..."):
                if btn2:
                    pred = predict_one([sentence])
                    st.write("OOD probability: ", round(float(pred[0][1]), 2))

                    fig, ax = plt.subplots(figsize=(6, 2))
                    exp = explain(sentence)
                    explist = exp.as_list()
                    vals = [x[1] for x in explist]
                    names = [x[0] for x in explist]
                    vals.reverse()
                    names.reverse()
                    colors = ["red" if x > 0 else "lightblue" for x in vals]
                    pos = np.arange(len(explist)) + 0.5
                    ax.barh(pos, vals, align="center", color=colors, alpha=0.5)
                    ax.set_yticks(pos)
                    ax.set_yticklabels(names)
                    #
                    # ax1.barh(
                    #     [0.5, -0.5],
                    #     pred[0],
                    #     align="center",
                    #     color=["lightblue", "red"],
                    #     alpha=0.5,
                    # )
                    # ax1.set_yticks([0.5, -0.5])
                    # ax1.set_yticklabels(["OOD", "ID"])

                    st.caption(
                        "Explanation: :red[OOD] words push sentence towards out-of-domain :blue[ID] towards in-domain"
                    )
                    st.pyplot(fig)
