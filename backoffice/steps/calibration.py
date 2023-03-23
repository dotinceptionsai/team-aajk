from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import evaluate
from analysis.experiments import ExperimentRegistry
from backoffice import shared
from backoffice.session import SessionKey
from pipelines import persistence
from pipelines.impl.anomaly_detection import OnInvalidSentence


def display():
    datasets = shared.get_ds_registry()

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

        default_id_sentences = datasets.load_items(model.datasets.validation_id)
        default_ood_sentences = datasets.load_items(model.datasets.validation_ood)

        with st.form("form"):
            st.caption(
                "Enter your sentences to help the model understand the limits of what is :blue[ID] and :red[OOD]. Try to enter sentences that are just at the frontier of being out-of-domain or in-domain."
            )

            id_sentences_ta = st.text_area(
                "**:blue[ID sentences]**",
                value="\n".join(default_id_sentences),
                help="Enter some In-Domain sentences here.",
                height=200,
            )

            ood_sentences_ta = st.text_area(
                "**:red[OOD sentences]**",
                value="\n".join(default_ood_sentences),
                help="Enter some Out-of-Domain sentences here.",
                height=200,
            )

            recalibration_btn = st.form_submit_button("Refresh distances and metrics")

            if recalibration_btn:
                with st.spinner("Recomputing..."):
                    id_sentences = id_sentences_ta.splitlines()
                    ood_sentences = ood_sentences_ta.splitlines()
                    model.recalibrate(
                        id_sentences,
                        ood_sentences,
                        registry=None,
                        on_invalid_sentence=OnInvalidSentence.WARN,
                    )
                    st.write(
                        "Cutoff distance that maximizes F1: ",
                        round(model.calibrator.cutoff_, 2),
                    )

                    manual_cutoff = st.slider(
                        "Adjust cutoff distance for more precision or recall. Click button again to refresh metrics.",
                        model.calibrator.r_id_,
                        model.calibrator.r_ood_,
                        model.calibrator.cutoff_,
                    )

                    if manual_cutoff:
                        model.calibrator.cutoff_ = manual_cutoff

                    with st.expander(
                        "Sentence distances to center of In-Domain (ID)", expanded=True
                    ):
                        st.caption(
                            "Every sentence with a distance below this cutoff will be considered ID, above will be considered OOD."
                        )

                        raw_id_scores = model.train_pipe.transform(id_sentences)
                        raw_ood_scores = model.train_pipe.transform(ood_sentences)

                        df_id = pd.DataFrame(
                            {"distance": raw_id_scores, "origin": "ID"}
                        )
                        df_ood = pd.DataFrame(
                            {"distance": raw_ood_scores, "origin": "OOD"}
                        )
                        df = pd.concat([df_id, df_ood])
                        df["sentence"] = [*id_sentences, *ood_sentences]

                        df = df.sort_values(by="distance", ascending=True)

                        import seaborn as sns
                        import matplotlib.pyplot as plt

                        metrics_cutoff = (
                            model.calibrator.cutoff_
                            if manual_cutoff is None
                            else manual_cutoff
                        )
                        sns.set_style("darkgrid")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        # plt.xlim(model.calibrator.r_id_, model.calibrator.r_ood_)
                        graph = sns.histplot(
                            data=df,
                            x="distance",
                            hue="origin",
                            legend=True,
                            stat="count",
                            palette=["lightblue", "red"],
                            alpha=0.5,
                            kde=True,
                            ax=ax,
                        )
                        graph.axvline(
                            metrics_cutoff,
                            color="grey",
                            linestyle="--",
                            label="Cutoff",
                        )
                        st.caption(
                            "Histogram shows sentences distance distribution. Check :blue[ID] and :red[OOD] are well separated."
                        )
                        st.pyplot(fig)

                        st.caption(
                            "Table shows each sentences distance to center of the ID. It helps you see where you want the cutoff to be."
                        )
                        st.dataframe(
                            df.style.format(formatter={"distance": "{:.2f}"}),
                            use_container_width=True,
                        )

                    with st.expander(expanded=True, label="Evaluation metrics"):
                        metrics = evaluate.evaluate_model(
                            model, id_sentences, ood_sentences
                        )
                        from sklearn.metrics import (
                            precision_score,
                            recall_score,
                            f1_score,
                        )

                        f1 = round(f1_score(metrics.y_true, metrics.y_pred), 2)
                        prec = round(precision_score(metrics.y_true, metrics.y_pred), 2)
                        recall = round(recall_score(metrics.y_true, metrics.y_pred), 2)

                        previous_f1 = st.session_state.get("f1", f1)
                        previous_prec = st.session_state.get("prec", prec)
                        previous_recall = st.session_state.get("recall", recall)

                        st.session_state["f1"] = f1
                        st.session_state["prec"] = prec
                        st.session_state["recall"] = recall

                        col1, col2, col3 = st.columns(3)
                        col1.metric("F1", f1, round(f1 - previous_f1, 2))
                        col2.metric(
                            "OOD-Precision", prec, round(prec - previous_prec, 2)
                        )
                        col3.metric(
                            "OOD-Recall", recall, round(recall - previous_recall, 2)
                        )

                        st.write("False positives: :blue[ID] but predicted :red[OOD]")
                        st.write(
                            pd.DataFrame(
                                {
                                    "sentence": [
                                        id_sentences[i] for i in metrics.fp_indices
                                    ]
                                }
                            )
                        )
                        st.write("False negatives: :red[OOD] but predicted :blue[ID]")
                        st.write(
                            pd.DataFrame(
                                {
                                    "sentence": [
                                        ood_sentences[i] for i in metrics.fn_indices
                                    ]
                                }
                            )
                        )

                        # cm = confusion_matrix(metrics.y_true, metrics.y_pred)
        # ConfusionMatrixDisplay(cm).plot()
        # plt.savefig(output_dir / "confusion_matrix.png")
