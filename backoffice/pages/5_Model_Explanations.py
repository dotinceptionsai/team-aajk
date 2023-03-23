from pathlib import Path

import mlflow
import nltk
import numpy as np
import pandas as pd
import streamlit as st

from analysis import experiments
from backoffice import shared
from backoffice.session import SessionKey
from backoffice.shared import make_sidebar
from pipelines import persistence

st.session_state.update(st.session_state)

datasets = shared.get_ds_registry()

st.title("Model Calibration 🪄")

st.write(
    "Choose between a faster model that is less accurate or a slower one but more accurate."
)

make_sidebar()

if SessionKey.SELECTED_DATASET not in st.session_state:
    shared.go_back_to("Data Selection", "Select dataset first")
elif SessionKey.SELECTED_BASE_MODEL not in st.session_state:
    shared.go_back_to("Model Choice", "Select base model first")
else:
    ds = st.session_state[SessionKey.SELECTED_DATASET]
    base_model_name = st.session_state[SessionKey.SELECTED_BASE_MODEL]

    mlruns = str(str(Path("train/mlruns").absolute()))
    mlflow.set_tracking_uri("file://" + mlruns)
    run_info = experiments.get_run_info(experiment_name=ds, run_name=base_model_name)

    pipeline_dir = Path(run_info.artifact_dir.replace("file:/", "/"))
    st.write("Pipeline dir exists: ", pipeline_dir.exists())
    st.write("Pipeline dir: ", pipeline_dir)
    model = persistence.load_pipeline(pipeline_dir)

    with st.expander("Base model details"):
        st.write(persistence._create_model_dict(model))

    from pipelines.impl.anomaly_detection import OnInvalidSentence

    default_id_sentences = datasets.load_items(model.datasets.validation_id)
    default_ood_sentences = datasets.load_items(model.datasets.validation_ood)

    with st.expander(
        "Enter your sentences to help the model understand the limits of what is ID and OOD"
    ):
        st.caption(
            "Try to enter sentences that are just at the limit of being in-domain."
        )
        id_sentences_ta = st.text_area(
            "ID sentences",
            value="\n".join(default_id_sentences),
            help="Enter some In-Domain sentences here.",
            height=200,
        )
        st.caption(
            "Try to enter sentences that are just at the limit of being out-of-domain."
        )
        ood_sentences_ta = st.text_area(
            "OOD sentences",
            value="\n".join(default_ood_sentences),
            help="Enter some Out-of-Domain sentences here.",
            height=200,
        )

    ood_sentences = ood_sentences_ta.splitlines()
    id_sentences = id_sentences_ta.splitlines()

    def recalibrate():
        model.recalibrate(
            id_sentences,
            ood_sentences,
            registry=datasets,
            on_invalid_sentence=OnInvalidSentence.WARN,
        )

    b = st.button("Recalibrate model")

    if b:

        model.recalibrate(
            id_sentences,
            ood_sentences,
            registry=datasets,
            on_invalid_sentence=OnInvalidSentence.WARN,
        )
        st.write("Model recalibrated")
        st.write("Cutoff: ", model.calibrator.cutoff_)
        st.write(model.calibrator.cutoff_)

        raw_id_scores = model.train_pipe.transform(id_sentences)
        raw_ood_scores = model.train_pipe.transform(ood_sentences)

        df_id = pd.DataFrame({"score": raw_id_scores, "origin": "ID"})
        df_ood = pd.DataFrame({"score": raw_ood_scores, "origin": "OOD"})
        df = pd.concat([df_id, df_ood])
        df["sentence"] = [*id_sentences, *ood_sentences]

        # get the top 10 rows where abs(delta) is closest to 0
        df = df.sort_values(by="score", ascending=False)
        st.write(df)

        st.write(
            model.calibrator.r_id_, model.calibrator.r_ood_, model.calibrator.cutoff_
        )

        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_style("darkgrid")
        fig, ax = plt.subplots()
        graph = sns.histplot(
            data=df,
            x="score",
            hue="origin",
            legend=True,
            stat="count",
            palette=["red", "lightblue"],
            alpha=0.5,
            kde=True,
            ax=ax,
        )
        graph.axvline(
            model.calibrator.cutoff_, color="grey", linestyle="--", label="Cutoff"
        )
        st.pyplot(fig)

        def predict_one(sentences):
            embeddings = model.embedder.transform(sentences)
            raw_scores = model.distribution.transform(embeddings)
            ood_probas = model.calibrator.predict_proba(raw_scores)
            # Concatenate iid and ood as a 2D array
            new_var = np.vstack([1 - ood_probas, ood_probas]).T
            return new_var

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
                words = nltk.word_tokenize(sentence)
                words = [word for word in words if word.isalpha()]
                explainers = {tok: imp for tok, imp in explain(sentence).as_list()}
                word_importances = [explainers.get(w, 0) for w in words]

                from streamlit_extras.word_importances import format_word_importances

                text = (
                    "Streamlit Extras is a library to help you discover, learn, share and"
                    " use Streamlit bits of code!"
                )
                html = format_word_importances(
                    words=words,
                    importances=scale(word_importances),  # fmt: skip
                )
                st.write(html, unsafe_allow_html=True)
